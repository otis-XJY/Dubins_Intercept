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
    p_dim: int = 6,
    e_dim: int = 8,
    c_dim: int = 6,
    hidden_dim: int = 128,
    num_heads: int = 4,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, "TODCHeteroActorCritic"]:
    """
    Build one or multiple scheme models for training in one run.

    Args:
        schemes: None -> all 3 schemes, string -> one scheme, sequence -> multiple schemes.
                 Accepts A/B/C and full names.
    Returns:
        Dict keyed by canonical scheme names.
    """
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
            p_dim=p_dim,
            e_dim=e_dim,
            c_dim=c_dim,
            hidden_dim=hidden_dim,
            design_mode=key,
            num_heads=num_heads,
        )
        if device is not None:
            model = model.to(device)
        models[name] = model
    return models


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    """Helper function to build a standard Multi-Layer Perceptron."""
    layers =[]
    d = in_dim
    for _ in range(max(1, num_layers - 1)):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.ReLU())
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class TODCHeteroActorCritic(nn.Module):
    """
    Backward-compatible wrapper of UAVInterceptionNetwork.
    
    Args:
        design_mode (str): 
            - "A": Design 1 (Ego-centric + Concat Fusion + Pointer Network)
            - "B": Design 2 (Ego-centric + Soft-Gating Fusion + Pointer Network)
            - "C": Design 3 (Point-centric + Independent Eval, No Point Self-Attention)
    """

    def __init__(
        self,
        p_dim: int = 6,
        e_dim: int = 8,
        c_dim: int = 6,
        hidden_dim: int = 128,
        design_mode: Optional[str] = None,
        num_heads: int = 4,
        use_soft_gating: Optional[bool] = None,
    ):
        super().__init__()
        _ = (p_dim, e_dim, c_dim)  # Keep signature compatibility.
        self.model = UAVInterceptionNetwork(
            hidden_dim=hidden_dim, 
            num_heads=num_heads,
            design_mode=design_mode,
            use_soft_gating=use_soft_gating,
        )

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(obs)


class UAVInterceptionNetwork(nn.Module):
    """
    UAV Interception Decision Network supporting 3 distinct MARL Action-Selection Designs.
    """
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

        # ==========================================
        # 1. Independent Encoders (独立特征嵌入)
        # ==========================================
        self.enc_self = _mlp(3, hidden_dim, hidden_dim)
        self.enc_ally = _mlp(3, hidden_dim, hidden_dim)
        self.enc_self_pts = _mlp(8, hidden_dim, hidden_dim)
        self.enc_ally_pts = _mlp(8, hidden_dim, hidden_dim)
        self.enc_enemy = _mlp(3, hidden_dim, hidden_dim)
        self.enc_target = _mlp(2, hidden_dim, hidden_dim)

        # ==========================================
        # 2. Heterogeneous Multi-Head Attention Branches (异构多头注意力分支)
        # ==========================================
        # Shared MHAs across all designs
        self.mha_ally = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mha_ally_pts = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mha_enemy = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.mha_target = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # Design Specific MHAs:
        if self.design_mode in ["A", "B"]:
            # Ego queries Self-Points
            self.mha_self_pts = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        elif self.design_mode == "C":
            # Points query Ego (Self)
            self.mha_self = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

        # ==========================================
        # 3 & 4. Fusion Module & Action Heads (融合组件与动作输出头)
        # ==========================================
        if self.design_mode == "A":
            # [Design A] Concat + Pointer Network
            self.fusion_mlp = _mlp(hidden_dim * 6, hidden_dim, hidden_dim, num_layers=2)
            self.q_opt = nn.Linear(hidden_dim, hidden_dim)
            self.k_opt = nn.Linear(hidden_dim, hidden_dim)

        elif self.design_mode == "B":
            # [Design B] Soft-Gating + Pointer Network
            self.gate_ally = nn.Linear(hidden_dim * 2, 1)
            self.gate_self_pts = nn.Linear(hidden_dim * 2, 1)
            self.gate_ally_pts = nn.Linear(hidden_dim * 2, 1)
            self.gate_enemy = nn.Linear(hidden_dim * 2, 1)
            self.gate_target = nn.Linear(hidden_dim * 2, 1)
            self.q_opt = nn.Linear(hidden_dim, hidden_dim)
            self.k_opt = nn.Linear(hidden_dim, hidden_dim)

        elif self.design_mode == "C":
            # [Design C] Point-Centric Independent Evaluation (No Pointer, Direct Scoring)
            # Input: self_pts(1) + self(1) + ally(1) + apts(1) + enemy(1) + target(1) = 6 Contexts
            self.scoring_mlp = _mlp(hidden_dim * 6, hidden_dim, 1, num_layers=2)

        # ==========================================
        # 5. Critic Network (Centralized V-value)
        # ==========================================
        self.critic = _mlp(hidden_dim * 6, hidden_dim, 1, num_layers=3)

    def _apply_mha(self, mha_module, query, kv, mask=None):
        """
        Safely applies MultiHeadAttention.
        - mask: Tensor of shape [B, SeqLen] where True means VALID, False means PADDED/MASKED.
        """
        all_masked = None
        if mask is not None:
            key_padding_mask = ~mask  # MHA expects True for elements to IGNORE
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

    def forward(self, obs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # -- 1. Extract and Encode Features --
        e_self = self.enc_self(obs["self_uav"])          # [B, 1, D]
        e_ally = self.enc_ally(obs["ally_uavs"])         # [B, A, D]
        e_self_pts = self.enc_self_pts(obs["self_pts"])  # [B, M, D]
        e_ally_pts = self.enc_ally_pts(obs["ally_pts"])  # [B, M_A, D]
        e_enemy = self.enc_enemy(obs["enemies"])         # [B, N, D]
        e_target = self.enc_target(obs["targets"])       # [B, V, D]

        # =====================================================================
        # DESIGN A & B: EGO-CENTRIC QUERY (以我为主的决策)
        # =====================================================================
        if self.design_mode in ["A", "B"]:
            # Query is e_self[B, 1, D]
            h_ally = self._apply_mha(self.mha_ally, e_self, e_ally, obs.get("ally_mask"))
            h_spts = self._apply_mha(self.mha_self_pts, e_self, e_self_pts, obs.get("self_pts_mask"))
            h_apts = self._apply_mha(self.mha_ally_pts, e_self, e_ally_pts, obs.get("ally_pts_mask"))
            h_enemy = self._apply_mha(self.mha_enemy, e_self, e_enemy, obs.get("enemy_mask"))
            h_target = self._apply_mha(self.mha_target, e_self, e_target, obs.get("target_mask"))

            # Squeeze sequence dim -> [B, D]
            e_self_sq = e_self.squeeze(1)
            h_ally_sq = h_ally.squeeze(1)
            h_spts_sq = h_spts.squeeze(1)
            h_apts_sq = h_apts.squeeze(1)
            h_enemy_sq = h_enemy.squeeze(1)
            h_target_sq = h_target.squeeze(1)

            # Fusion
            if self.design_mode == "A":
                concat_feat = torch.cat([e_self_sq, h_ally_sq, h_spts_sq, h_apts_sq, h_enemy_sq, h_target_sq], dim=-1)
                h_env = self.fusion_mlp(concat_feat)  # [B, D]
            else: # Design B
                s_ally = F.leaky_relu(self.gate_ally(torch.cat([e_self_sq, h_ally_sq], dim=-1)))
                s_spts = F.leaky_relu(self.gate_self_pts(torch.cat([e_self_sq, h_spts_sq], dim=-1)))
                s_apts = F.leaky_relu(self.gate_ally_pts(torch.cat([e_self_sq, h_apts_sq], dim=-1)))
                s_enemy = F.leaky_relu(self.gate_enemy(torch.cat([e_self_sq, h_enemy_sq], dim=-1)))
                s_target = F.leaky_relu(self.gate_target(torch.cat([e_self_sq, h_target_sq], dim=-1)))

                scores = torch.stack([s_ally, s_spts, s_apts, s_enemy, s_target], dim=2)
                alphas = torch.softmax(scores, dim=2).squeeze(1)  # [B, 5]

                h_env = e_self_sq + (
                    alphas[:, 0:1] * h_ally_sq + alphas[:, 1:2] * h_spts_sq +
                    alphas[:, 2:3] * h_apts_sq + alphas[:, 3:4] * h_enemy_sq +
                    alphas[:, 4:5] * h_target_sq
                )

            # Pointer Network
            q_action = self.q_opt(h_env).unsqueeze(1)  #[B, 1, D]
            k_action = self.k_opt(e_self_pts)          # [B, M, D]
            logits = torch.sum(q_action * k_action, dim=-1) / (self.hidden_dim**0.5)  # [B, M]

        # =====================================================================
        # DESIGN C: POINT-CENTRIC INDEPENDENT EVALUATION (基于拦截点的独立评估)
        # =====================================================================
        elif self.design_mode == "C":
            # Query is e_self_pts [B, M, D]. M points ask environment independently.
            # No self-attention among points to ensure independent scoring.
            
            # Points query Self (No mask needed as self is length 1)
            h_self = self._apply_mha(self.mha_self, query=e_self_pts, kv=e_self, mask=None)  #[B, M, D]
            
            # Points query the rest of the environment
            h_ally = self._apply_mha(self.mha_ally, query=e_self_pts, kv=e_ally, mask=obs.get("ally_mask"))
            h_apts = self._apply_mha(self.mha_ally_pts, query=e_self_pts, kv=e_ally_pts, mask=obs.get("ally_pts_mask"))
            h_enemy = self._apply_mha(self.mha_enemy, query=e_self_pts, kv=e_enemy, mask=obs.get("enemy_mask"))
            h_target = self._apply_mha(self.mha_target, query=e_self_pts, kv=e_target, mask=obs.get("target_mask"))
            
            # Each point now has 5 contextual embeddings from the environment.
            # Concat them along feature dim: [B, M, 6 * D]
            concat_pts_feat = torch.cat([e_self_pts, h_self, h_ally, h_apts, h_enemy, h_target], dim=-1)
            
            # Directly score each point. MLP maps[B, M, 6D] -> [B, M, 1]
            logits = self.scoring_mlp(concat_pts_feat).squeeze(-1)  # [B, M]

        # ==========================================
        # Action Masking and Probability (通用处理)
        # ==========================================
        self_pts_mask = obs.get("self_pts_mask")
        if self_pts_mask is not None:
            masked_logits = logits.masked_fill(~self_pts_mask, -1e9)
        else:
            masked_logits = logits

        probs = torch.softmax(masked_logits, dim=-1)
        best_candidate_idx = torch.argmax(probs, dim=-1)

        # ==========================================
        # Critic (Global Value Estimation - 保持一致)
        # ==========================================
        def safe_mean(tensor, mask):
            if mask is None:
                return tensor.mean(dim=1)
            mask_float = mask.unsqueeze(-1).float()
            return (tensor * mask_float).sum(dim=1) / mask_float.sum(dim=1).clamp_min(1.0)

        e_ally_pool = safe_mean(e_ally, obs.get("ally_mask"))
        e_spts_pool = safe_mean(e_self_pts, obs.get("self_pts_mask"))
        e_apts_pool = safe_mean(e_ally_pts, obs.get("ally_pts_mask"))
        e_enemy_pool = safe_mean(e_enemy, obs.get("enemy_mask"))
        e_target_pool = safe_mean(e_target, obs.get("target_mask"))
        e_self_sq = e_self.squeeze(1)

        critic_input = torch.cat([e_self_sq, e_ally_pool, e_spts_pool, e_apts_pool, e_enemy_pool, e_target_pool], dim=-1)
        value = self.critic(critic_input).squeeze(-1)

        return {
            "action_logits": masked_logits,
            "action_probs": probs,
            "best_candidate_idx": best_candidate_idx,
            "value": value,
        }