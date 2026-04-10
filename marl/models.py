from typing import Any, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEME_KEY_TO_NAME = {
    "A": "Concatenative Query Network",
    "B": "Gated Query Network",
    "C": "Point-Wise Scoring Network",
}

_SCHEME_ALIAS_TO_KEY = {
    "a": "A",
    "concatenative query network": "A",
    "concatenative_query_network": "A",
    "cqn": "A",
    "concat": "A",
    "b": "B",
    "gated query network": "B",
    "gated_query_network": "B",
    "gqn": "B",
    "soft-gating": "B",
    "soft_gating": "B",
    "c": "C",
    "point-wise scoring network": "C",
    "point_wise_scoring_network": "C",
    "pointwise scoring network": "C",
    "pwsn": "C",
}


def resolve_design_mode(design_mode: Optional[str] = None, use_soft_gating: Optional[bool] = None) -> str:
    """Resolve model design alias to canonical key: A/B/C."""
    if design_mode is None:
        if use_soft_gating is None:
            return "A"
        return "B" if bool(use_soft_gating) else "A"

    normalized = str(design_mode).strip().lower()
    if normalized in _SCHEME_ALIAS_TO_KEY:
        return _SCHEME_ALIAS_TO_KEY[normalized]

    valid_names = ", ".join(SCHEME_KEY_TO_NAME.values())
    raise ValueError(
        f"Unknown design_mode='{design_mode}'. Use one of A/B/C or: {valid_names}."
    )


def design_mode_to_name(design_mode: str) -> str:
    """Get canonical human-readable scheme name from design mode alias/key."""
    return SCHEME_KEY_TO_NAME[resolve_design_mode(design_mode)]


def build_actor_critic_schemes(
    schemes: Optional[Union[str, Sequence[str]]] = None,
    *,
    hidden_dim: int = 128,
    num_heads: int = 4,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, "TODCHeteroActorCritic"]:
    if schemes is None:
        requested: List[str] = ["A", "B", "C"]
    elif isinstance(schemes, str):
        requested = [schemes]
    else:
        requested = list(schemes)

    unique_keys: List[str] = []
    seen = set()
    for item in requested:
        key = resolve_design_mode(item)
        if key not in seen:
            unique_keys.append(key)
            seen.add(key)

    models: Dict[str, TODCHeteroActorCritic] = {}
    for key in unique_keys:
        name = SCHEME_KEY_TO_NAME[key]
        model = TODCHeteroActorCritic(
            hidden_dim=hidden_dim,
            design_mode=key,
            num_heads=num_heads,
        )
        if device is not None:
            model = model.to(device)
        models[name] = model
    return models


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(max(1, num_layers - 1)):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ReLU())
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TODCHeteroActorCritic(nn.Module):
    """Wrapper: actor head + MAPPO centralized critic."""

    def __init__(
        self,
        hidden_dim: int = 128,
        design_mode: Optional[str] = None,
        num_heads: int = 4,
        use_soft_gating: Optional[bool] = None,
    ):
        super().__init__()
        self.model = UAVInterceptionNetwork(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            design_mode=design_mode,
            use_soft_gating=use_soft_gating,
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(obs)

    def actor_forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model.actor_forward(obs)

    def critic_forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.model.critic_forward(obs)


class UAVInterceptionNetwork(nn.Module):
    """
    多分支异构注意力决策网络（A/B/C 三方案）。
    观测含：自身、友机、分配敌机与对应攻击目标（重要设施平面坐标）、候选拦截点等共八路上下文。
    """

    N_BRANCHES = 8

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        design_mode: Optional[str] = None,
        use_soft_gating: Optional[bool] = None,
        **_: Any,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.design_mode = resolve_design_mode(design_mode, use_soft_gating=use_soft_gating)
        self.design_name = SCHEME_KEY_TO_NAME[self.design_mode]

        self.enc_self = _mlp(3, hidden_dim, hidden_dim)
        self.enc_ally = _mlp(3, hidden_dim, hidden_dim)
        self.enc_self_pts = _mlp(8, hidden_dim, hidden_dim)
        self.enc_ally_pts = _mlp(8, hidden_dim, hidden_dim)
        self.enc_enemy = _mlp(3, hidden_dim, hidden_dim)
        self.enc_asset = _mlp(2, hidden_dim, hidden_dim)

        self.mha_ally = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mha_ally_pts = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mha_enemy_self = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mha_enemy_per_ally = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mha_asset_self = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mha_asset_per_ally = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        if self.design_mode in ["A", "B"]:
            self.mha_self_pts = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        elif self.design_mode == "C":
            self.mha_self = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        if self.design_mode == "A":
            self.fusion_mlp = _mlp(hidden_dim * self.N_BRANCHES, hidden_dim, hidden_dim, num_layers=2)
            self.q_opt = nn.Linear(hidden_dim, hidden_dim)
            self.k_opt = nn.Linear(hidden_dim, hidden_dim)
        elif self.design_mode == "B":
            self.gate_ally = nn.Linear(hidden_dim * 2, 1)
            self.gate_self_pts = nn.Linear(hidden_dim * 2, 1)
            self.gate_ally_pts = nn.Linear(hidden_dim * 2, 1)
            self.gate_enemy_self = nn.Linear(hidden_dim * 2, 1)
            self.gate_enemy_ally = nn.Linear(hidden_dim * 2, 1)
            self.gate_asset_self = nn.Linear(hidden_dim * 2, 1)
            self.gate_asset_ally = nn.Linear(hidden_dim * 2, 1)
            self.q_opt = nn.Linear(hidden_dim, hidden_dim)
            self.k_opt = nn.Linear(hidden_dim, hidden_dim)
        elif self.design_mode == "C":
            self.scoring_mlp = _mlp(hidden_dim * self.N_BRANCHES, hidden_dim, 1, num_layers=2)

        self._critic_net: Optional[nn.Module] = None
        self._critic_in_dim: Optional[int] = None

    def _apply_mha(self, mha_module, query, kv, mask=None):
        all_masked = None
        if mask is not None:
            key_padding_mask = ~mask
            all_masked = key_padding_mask.all(dim=-1)
            key_padding_mask[all_masked, :] = False
        else:
            key_padding_mask = None

        attn_out, _ = mha_module(
            query=query,
            key=kv,
            value=kv,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        if mask is not None and all_masked is not None and all_masked.any():
            attn_out[all_masked] = 0.0

        return attn_out

    def _encode_obs(self, obs: Dict[str, torch.Tensor]):
        e_self = self.enc_self(obs["self_uav"])
        e_ally = self.enc_ally(obs["allies_local"])
        e_self_pts = self.enc_self_pts(obs["self_pts"])
        e_ally_pts = self.enc_ally_pts(obs["ally_pts"])
        e_eself = self.enc_enemy(obs["enemy_assigned_self"])
        e_eally = self.enc_enemy(obs["enemy_assigned_per_ally"])
        e_ast_s = self.enc_asset(obs["asset_target_self"])
        e_ast_a = self.enc_asset(obs["asset_target_per_ally"])
        return e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a

    def _branch_tensors_ab(
        self,
        obs: Dict[str, torch.Tensor],
        e_self,
        e_ally,
        e_self_pts,
        e_ally_pts,
        e_eself,
        e_eally,
        e_ast_s,
        e_ast_a,
    ):
        es_mask = obs["enemy_self_mask"].bool()
        h_ally = self._apply_mha(self.mha_ally, e_self, e_ally, obs["ally_mask"])
        h_spts = self._apply_mha(self.mha_self_pts, e_self, e_self_pts, obs["self_pts_mask"])
        h_apts = self._apply_mha(self.mha_ally_pts, e_self, e_ally_pts, obs["ally_pts_mask"])
        h_eself = self._apply_mha(self.mha_enemy_self, e_self, e_eself, es_mask)
        h_eally = self._apply_mha(self.mha_enemy_per_ally, e_self, e_eally, obs["ally_enemy_mask"])
        h_ast_s = self._apply_mha(self.mha_asset_self, e_self, e_ast_s, es_mask)
        h_ast_a = self._apply_mha(self.mha_asset_per_ally, e_self, e_ast_a, obs["ally_enemy_mask"])

        e_self_sq = e_self.squeeze(1)
        h_ally_sq = h_ally.squeeze(1)
        h_spts_sq = h_spts.squeeze(1)
        h_apts_sq = h_apts.squeeze(1)
        h_eself_sq = h_eself.squeeze(1)
        h_eally_sq = h_eally.squeeze(1)
        h_ast_s_sq = h_ast_s.squeeze(1)
        h_ast_a_sq = h_ast_a.squeeze(1)
        return (
            e_self_sq,
            h_ally_sq,
            h_spts_sq,
            h_apts_sq,
            h_eself_sq,
            h_eally_sq,
            h_ast_s_sq,
            h_ast_a_sq,
        )

    def actor_forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a = self._encode_obs(obs)

        if self.design_mode in ["A", "B"]:
            (
                e_self_sq,
                h_ally_sq,
                h_spts_sq,
                h_apts_sq,
                h_eself_sq,
                h_eally_sq,
                h_ast_s_sq,
                h_ast_a_sq,
            ) = self._branch_tensors_ab(
                obs, e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a
            )

            if self.design_mode == "A":
                concat_feat = torch.cat(
                    [
                        e_self_sq,
                        h_ally_sq,
                        h_spts_sq,
                        h_apts_sq,
                        h_eself_sq,
                        h_eally_sq,
                        h_ast_s_sq,
                        h_ast_a_sq,
                    ],
                    dim=-1,
                )
                h_env = self.fusion_mlp(concat_feat)
            else:
                s_ally = F.leaky_relu(self.gate_ally(torch.cat([e_self_sq, h_ally_sq], dim=-1)))
                s_spts = F.leaky_relu(self.gate_self_pts(torch.cat([e_self_sq, h_spts_sq], dim=-1)))
                s_apts = F.leaky_relu(self.gate_ally_pts(torch.cat([e_self_sq, h_apts_sq], dim=-1)))
                s_es = F.leaky_relu(self.gate_enemy_self(torch.cat([e_self_sq, h_eself_sq], dim=-1)))
                s_ea = F.leaky_relu(self.gate_enemy_ally(torch.cat([e_self_sq, h_eally_sq], dim=-1)))
                s_as = F.leaky_relu(self.gate_asset_self(torch.cat([e_self_sq, h_ast_s_sq], dim=-1)))
                s_aa = F.leaky_relu(self.gate_asset_ally(torch.cat([e_self_sq, h_ast_a_sq], dim=-1)))
                scores = torch.stack([s_ally, s_spts, s_apts, s_es, s_ea, s_as, s_aa], dim=2)
                alphas = torch.softmax(scores, dim=2).squeeze(1)
                h_env = e_self_sq + (
                    alphas[:, 0:1] * h_ally_sq
                    + alphas[:, 1:2] * h_spts_sq
                    + alphas[:, 2:3] * h_apts_sq
                    + alphas[:, 3:4] * h_eself_sq
                    + alphas[:, 4:5] * h_eally_sq
                    + alphas[:, 5:6] * h_ast_s_sq
                    + alphas[:, 6:7] * h_ast_a_sq
                )

            q_action = self.q_opt(h_env).unsqueeze(1)
            k_action = self.k_opt(e_self_pts)
            logits = torch.sum(q_action * k_action, dim=-1) / (self.hidden_dim**0.5)

        else:
            es_mask = obs["enemy_self_mask"].bool()
            h_self = self._apply_mha(self.mha_self, query=e_self_pts, kv=e_self, mask=None)
            h_ally = self._apply_mha(self.mha_ally, query=e_self_pts, kv=e_ally, mask=obs["ally_mask"])
            h_apts = self._apply_mha(self.mha_ally_pts, query=e_self_pts, kv=e_ally_pts, mask=obs["ally_pts_mask"])
            h_eself = self._apply_mha(self.mha_enemy_self, query=e_self_pts, kv=e_eself, mask=es_mask)
            h_eally = self._apply_mha(self.mha_enemy_per_ally, query=e_self_pts, kv=e_eally, mask=obs["ally_enemy_mask"])
            h_ast_s = self._apply_mha(self.mha_asset_self, query=e_self_pts, kv=e_ast_s, mask=es_mask)
            h_ast_a = self._apply_mha(self.mha_asset_per_ally, query=e_self_pts, kv=e_ast_a, mask=obs["ally_enemy_mask"])
            concat_pts_feat = torch.cat(
                [e_self_pts, h_self, h_ally, h_apts, h_eself, h_eally, h_ast_s, h_ast_a], dim=-1
            )
            logits = self.scoring_mlp(concat_pts_feat).squeeze(-1)

        self_pts_mask = obs["self_pts_mask"]
        masked_logits = logits.masked_fill(~self_pts_mask, -1e9)
        probs = torch.softmax(masked_logits, dim=-1)
        best_candidate_idx = torch.argmax(probs, dim=-1)

        return {
            "action_logits": masked_logits,
            "action_probs": probs,
            "best_candidate_idx": best_candidate_idx,
        }

    def _ensure_critic(self, p: int, device: torch.device):
        in_dim = p * self.hidden_dim * self.N_BRANCHES
        if self._critic_net is None or self._critic_in_dim != in_dim:
            self._critic_in_dim = in_dim
            self._critic_net = nn.Sequential(
                nn.Linear(in_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, p),
            ).to(device)

    def critic_forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """集中 critic：各智能体八路融合特征 [P,8H] 展平为联合状态后输出每机价值。"""
        e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a = self._encode_obs(obs)
        device = e_self.device
        p = int(e_self.shape[0])
        self._ensure_critic(p, device)

        if self.design_mode in ["A", "B"]:
            (
                e_self_sq,
                h_ally_sq,
                h_spts_sq,
                h_apts_sq,
                h_eself_sq,
                h_eally_sq,
                h_ast_s_sq,
                h_ast_a_sq,
            ) = self._branch_tensors_ab(
                obs, e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a
            )
            per_agent = torch.cat(
                [
                    e_self_sq,
                    h_ally_sq,
                    h_spts_sq,
                    h_apts_sq,
                    h_eself_sq,
                    h_eally_sq,
                    h_ast_s_sq,
                    h_ast_a_sq,
                ],
                dim=-1,
            )
        else:
            es_mask = obs["enemy_self_mask"].bool()
            spm = obs["self_pts_mask"]
            h_self = self._apply_mha(self.mha_self, query=e_self_pts, kv=e_self, mask=None)
            h_ally = self._apply_mha(self.mha_ally, query=e_self_pts, kv=e_ally, mask=obs["ally_mask"])
            h_apts = self._apply_mha(self.mha_ally_pts, query=e_self_pts, kv=e_ally_pts, mask=obs["ally_pts_mask"])
            h_eself = self._apply_mha(self.mha_enemy_self, query=e_self_pts, kv=e_eself, mask=es_mask)
            h_eally = self._apply_mha(self.mha_enemy_per_ally, query=e_self_pts, kv=e_eally, mask=obs["ally_enemy_mask"])
            h_ast_s = self._apply_mha(self.mha_asset_self, query=e_self_pts, kv=e_ast_s, mask=es_mask)
            h_ast_a = self._apply_mha(self.mha_asset_per_ally, query=e_self_pts, kv=e_ast_a, mask=obs["ally_enemy_mask"])

            def pool_pm(t, m):
                mf = m.unsqueeze(-1).float()
                return (t * mf).sum(dim=1) / mf.sum(dim=1).clamp_min(1.0)

            per_agent = torch.cat(
                [
                    pool_pm(e_self_pts, spm),
                    pool_pm(h_self, spm),
                    pool_pm(h_ally, spm),
                    pool_pm(h_apts, spm),
                    pool_pm(h_eself, spm),
                    pool_pm(h_eally, spm),
                    pool_pm(h_ast_s, spm),
                    pool_pm(h_ast_a, spm),
                ],
                dim=-1,
            )

        central = per_agent.reshape(1, -1)
        assert self._critic_net is not None
        values = self._critic_net(central).squeeze(0)
        return values

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = self.actor_forward(obs)
        out["value"] = self.critic_forward(obs)
        return out
