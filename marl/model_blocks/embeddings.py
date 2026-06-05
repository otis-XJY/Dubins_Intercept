from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 2) -> nn.Sequential:
    layers = []
    d = in_dim
    for _ in range(max(1, num_layers - 1)):
        layers.append(nn.Linear(d, hidden_dim))
        layers.append(nn.GELU())
        d = hidden_dim
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


# === ego-centric 相对坐标变换 ===================================================
# 所有实体在进入编码 MLP 之前，统一变换到“本机参考系”：
#   - 平移：以本机位置为原点
#   - 旋转：把本机航向对齐到 +x 轴（旋转 -theta_e）
#   - 朝向：用相对航向的 (sin, cos) 编码，避免 ±pi 跳变
#   - 尺度：几何距离通道（rel_x/rel_y/r）除以 pos_scale；质量通道采用无参压缩（asinh/log1p/sin-cos）
#           把长尾量（path_L/Delta_V）压到可学习范围，减少手工调参依赖。
# 本机 self_uav 在该参考系下退化为 (0,0,0)，因此不再当普通实体编码，改用可学习 ego token。


def _ego_frame(self_uav: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """从 self_uav(P,1,3)=[x,y,theta] 取本机参考系：返回 (x_e,y_e,cos_e,sin_e,theta_e)，均为 (P,)。"""
    x_e = self_uav[:, 0, 0]
    y_e = self_uav[:, 0, 1]
    th_e = self_uav[:, 0, 2]
    return x_e, y_e, torch.cos(th_e), torch.sin(th_e), th_e


def _rel_xy(
    x: torch.Tensor,
    y: torch.Tensor,
    x_e: torch.Tensor,
    y_e: torch.Tensor,
    cos_e: torch.Tensor,
    sin_e: torch.Tensor,
    pos_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """把世界坐标 (x,y)(形状 (P,L)) 变换到本机系并按 pos_scale 归一化，返回 (rel_x, rel_y, r)。"""
    dx = x - x_e[:, None]
    dy = y - y_e[:, None]
    rel_x = cos_e[:, None] * dx + sin_e[:, None] * dy
    rel_y = -sin_e[:, None] * dx + cos_e[:, None] * dy
    r = torch.sqrt(rel_x * rel_x + rel_y * rel_y + 1e-12)
    return rel_x / pos_scale, rel_y / pos_scale, r / pos_scale


def _transform_pose(ent: torch.Tensor, frame, pos_scale: float) -> torch.Tensor:
    """位姿实体 (P,L,3)=[x,y,theta] -> (P,L,5)=[rel_x,rel_y,r,sin dtheta,cos dtheta]。"""
    x_e, y_e, cos_e, sin_e, th_e = frame
    rel_x, rel_y, r = _rel_xy(ent[..., 0], ent[..., 1], x_e, y_e, cos_e, sin_e, pos_scale)
    dth = ent[..., 2] - th_e[:, None]
    return torch.stack([rel_x, rel_y, r, torch.sin(dth), torch.cos(dth)], dim=-1)


def _transform_point(ent: torch.Tensor, frame, pos_scale: float) -> torch.Tensor:
    """纯位置实体 (P,L,2)=[x,y] -> (P,L,3)=[rel_x,rel_y,r]。"""
    x_e, y_e, cos_e, sin_e, th_e = frame
    rel_x, rel_y, r = _rel_xy(ent[..., 0], ent[..., 1], x_e, y_e, cos_e, sin_e, pos_scale)
    return torch.stack([rel_x, rel_y, r], dim=-1)


def _signed_log1p(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def _to_radians(angle: torch.Tensor) -> torch.Tensor:
    """角度自动统一到弧度：若数值明显超出弧度范围则按度数转弧度。"""
    return torch.where(angle.abs() > (2.0 * torch.pi), torch.deg2rad(angle), angle)


def _transform_candidate(ent: torch.Tensor, frame, pos_scale: float) -> torch.Tensor:
    """候选点 (P,L,8)=[c_x,c_y,theta,Delta_t,Delta_d,Delta_theta,path_L,Delta_V]
    -> (P,L,11)= 几何5 + 质量6。

    质量通道使用无参变换：
    - Delta_t: asinh 压缩（保留正负）
    - Delta_d/path_L: -log1p(x/pos_scale)（越小越好统一为“越大越好”）
    - Delta_theta: sin/cos 编码（自动度/弧度兼容）
    - Delta_V: signed log1p 压缩（保留方向）
    """
    x_e, y_e, cos_e, sin_e, th_e = frame
    rel_x, rel_y, r = _rel_xy(ent[..., 0], ent[..., 1], x_e, y_e, cos_e, sin_e, pos_scale)
    dth = ent[..., 2] - th_e[:, None]
    geo = torch.stack([rel_x, rel_y, r, torch.sin(dth), torch.cos(dth)], dim=-1)  # (P,L,5)

    delta_t = torch.asinh(ent[..., 3])
    delta_d = -torch.log1p(torch.clamp(ent[..., 4], min=0.0) / pos_scale)
    delta_theta = _to_radians(ent[..., 5])
    delta_theta_sin = torch.sin(delta_theta)
    delta_theta_cos = torch.cos(delta_theta)
    path_l = -torch.log1p(torch.clamp(ent[..., 6], min=0.0) / pos_scale)
    delta_v = _signed_log1p(ent[..., 7] / pos_scale)
    qual = torch.stack([delta_t, delta_d, delta_theta_sin, delta_theta_cos, path_l, delta_v], dim=-1)  # (P,L,6)
    return torch.cat([geo, qual], dim=-1)


class StructuredCandidateEncoder(nn.Module):
    """候选点结构化编码：几何分支(5维) 与 质量分支(6维) 各自 MLP -> 拼接 -> 投影到 D。

    几何分支回答“候选点在哪/朝向”，质量分支回答“这个拦截计划有多好”，两者尺度/语义不同，
    分开编码避免相互淹没，再融合到统一 D 维。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        d = int(hidden_dim)
        d_geo = d // 2
        d_qual = d - d_geo
        self.geo = _mlp(5, d, d_geo)
        self.qual = _mlp(6, d, d_qual)
        self.proj = nn.Linear(d, d)

    def forward(self, cand: torch.Tensor) -> torch.Tensor:
        g = self.geo(cand[..., :5])
        q = self.qual(cand[..., 5:])
        return self.proj(torch.cat([g, q], dim=-1))


class TODCEmbeddings(nn.Module):
    """8 路观测编码（输入先变换到本机参考系并按 pos_scale 归一化，再分类型编码）。

    obs 键与 shape 保持不变。self_pts 与 ally_pts 共享同一个结构化候选编码器（同语义类型），
    用可学习 role embedding 区分“自/友”来源；本机用可学习 ego token。
    """

    def __init__(self, hidden_dim: int, pos_scale: float = 1000.0):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.pos_scale = float(pos_scale)

        # 本机在本机系下退化为常数，改用可学习 ego token 作为 query token
        self.ego_token = nn.Parameter(torch.randn(hidden_dim) * 0.02)

        self.enc_ally = _mlp(5, hidden_dim, hidden_dim)
        # 共享候选编码器：self_pts / ally_pts 同类型共享权重，用 role embedding 区分来源
        self.enc_cand = StructuredCandidateEncoder(hidden_dim)
        self.role_self_pts = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.role_ally_pts = nn.Parameter(torch.randn(hidden_dim) * 0.02)
        self.enc_enemy = _mlp(5, hidden_dim, hidden_dim)
        self.enc_asset = _mlp(3, hidden_dim, hidden_dim)

        self.norm_self = nn.LayerNorm(hidden_dim)
        self.norm_ally = nn.LayerNorm(hidden_dim)
        self.norm_self_pts = nn.LayerNorm(hidden_dim)
        self.norm_ally_pts = nn.LayerNorm(hidden_dim)
        self.norm_enemy = nn.LayerNorm(hidden_dim)
        self.norm_asset = nn.LayerNorm(hidden_dim)

    def forward(
        self, obs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ps = self.pos_scale
        frame = _ego_frame(obs["self_uav"])
        num_p = obs["self_uav"].shape[0]

        # 本机 token：相对系下位姿恒为 0，使用可学习 token 广播到每个 pursuer
        e_self = self.norm_self(self.ego_token.view(1, 1, -1)).expand(num_p, 1, -1)

        e_ally = self.norm_ally(self.enc_ally(_transform_pose(obs["allies_local"], frame, ps)))

        cand_self = self.enc_cand(_transform_candidate(obs["self_pts"], frame, ps))
        cand_ally = self.enc_cand(_transform_candidate(obs["ally_pts"], frame, ps))
        e_self_pts = self.norm_self_pts(cand_self + self.role_self_pts)
        e_ally_pts = self.norm_ally_pts(cand_ally + self.role_ally_pts)

        e_eself = self.norm_enemy(self.enc_enemy(_transform_pose(obs["enemy_assigned_self"], frame, ps)))
        e_eally = self.norm_enemy(self.enc_enemy(_transform_pose(obs["enemy_assigned_per_ally"], frame, ps)))
        e_ast_s = self.norm_asset(self.enc_asset(_transform_point(obs["asset_target_self"], frame, ps)))
        e_ast_a = self.norm_asset(self.enc_asset(_transform_point(obs["asset_target_per_ally"], frame, ps)))
        return e_self, e_ally, e_self_pts, e_ally_pts, e_eself, e_eally, e_ast_s, e_ast_a
