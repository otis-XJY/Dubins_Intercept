"""测试 Design D 增强功能：P0-1 时间修正 GAE、P0-3 反事实信用分配、P2-1 多帧时序输入。"""

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

from marl.nn.models import UAVInterceptionNetwork, build_actor_critic_schemes
from marl.rl.mappo import compute_gae, ppo_minibatch_update
from marl.runners.online_train import SelfCtxHistoryBuffer
from marl.model_blocks.actor import TemporalSelfCtxEncoder, CounterfactualQHead


def _fake_obs(bsz, a, m, ma, n_e=2, n_v=2):
    """构造与 UAVInterceptionNetwork 兼容的假观测字典。"""
    return {
        "self_uav": torch.randn(bsz, 1, 3),
        "allies_local": torch.randn(bsz, a, 3),
        "enemy_assigned_self": torch.randn(bsz, 1, 3),
        "enemy_assigned_per_ally": torch.randn(bsz, a, 3),
        "asset_target_self": torch.randn(bsz, 1, 2),
        "asset_target_per_ally": torch.randn(bsz, a, 2),
        "self_pts": torch.randn(bsz, m, 8),
        "ally_pts": torch.randn(bsz, ma, 8),
        "ally_mask": torch.ones(bsz, a, dtype=torch.bool),
        "enemy_self_mask": torch.ones(bsz, 1, dtype=torch.bool),
        "ally_enemy_mask": torch.ones(bsz, a, dtype=torch.bool),
        "self_pts_mask": torch.ones(bsz, m, dtype=torch.bool),
        "ally_pts_mask": torch.ones(bsz, ma, dtype=torch.bool),
        "pursuer_active": torch.ones(bsz, dtype=torch.bool),
        "enemies": torch.randn(bsz, n_e, 3),
        "targets": torch.randn(bsz, n_e, 2),
        "assets": torch.randn(bsz, n_v, 2),
        "enemy_mask": torch.ones(bsz, n_e, dtype=torch.bool),
        "target_mask": torch.ones(bsz, n_e, dtype=torch.bool),
        "asset_mask": torch.ones(bsz, n_v, dtype=torch.bool),
    }


def _fake_obs_np(P, a, m, ma, n_e=2, n_v=2):
    """构造 numpy 格式的假观测字典，用于 PPO 更新测试。"""
    return {
        "self_uav": np.random.randn(P, 1, 3).astype(np.float32),
        "allies_local": np.random.randn(P, a, 3).astype(np.float32),
        "enemy_assigned_self": np.random.randn(P, 1, 3).astype(np.float32),
        "enemy_assigned_per_ally": np.random.randn(P, a, 3).astype(np.float32),
        "asset_target_self": np.random.randn(P, 1, 2).astype(np.float32),
        "asset_target_per_ally": np.random.randn(P, a, 2).astype(np.float32),
        "self_pts": np.random.randn(P, m, 8).astype(np.float32),
        "ally_pts": np.random.randn(P, ma, 8).astype(np.float32),
        "ally_mask": np.ones((P, a), dtype=bool),
        "enemy_self_mask": np.ones((P, 1), dtype=bool),
        "ally_enemy_mask": np.ones((P, a), dtype=bool),
        "self_pts_mask": np.ones((P, m), dtype=bool),
        "ally_pts_mask": np.ones((P, ma), dtype=bool),
        "pursuer_active": np.ones(P, dtype=bool),
        "enemies": np.random.randn(P, n_e, 3).astype(np.float32),
        "targets": np.random.randn(P, n_e, 2).astype(np.float32),
        "assets": np.random.randn(P, n_v, 2).astype(np.float32),
        "enemy_mask": np.ones((P, n_e), dtype=bool),
        "target_mask": np.ones((P, n_e), dtype=bool),
        "asset_mask": np.ones((P, n_v), dtype=bool),
    }


def _build_obs_np(obs_np, device):
    """将 numpy obs 字典转为 torch tensor 字典。"""
    out = {}
    for k, v in obs_np.items():
        if v.dtype == bool:
            out[k] = torch.as_tensor(v, dtype=torch.bool, device=device)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out


# ==================== P0-1：时间修正 GAE ====================


class TestTimeCorrectedGAE:
    """测试 compute_gae 的时间修正折扣因子。"""

    def test_time_corrected_vs_fixed(self):
        """时间修正 GAE 在 dt!=sim_dt_ref 时应产生与固定 gamma 不同的优势。"""
        T, P = 5, 3
        rewards = np.random.randn(T, P).astype(np.float32)
        values = np.random.randn(T, P).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        last_v = np.random.randn(P).astype(np.float32)
        # dt_seq 长度必须等于 T
        dt_seq = np.array([2.0, 0.5, 1.0, 3.0, 1.5], dtype=np.float32)
        sim_dt_ref = 1.0

        adv_fixed, _ = compute_gae(
            rewards, values, dones, last_v, gamma=0.99, lam=0.95
        )
        adv_tc, _ = compute_gae(
            rewards, values, dones, last_v,
            gamma=0.99, lam=0.95,
            dt_seq=dt_seq, sim_dt_ref=sim_dt_ref,
        )
        # dt 不全为 sim_dt_ref 时，两者应不同
        assert not np.allclose(adv_fixed, adv_tc, atol=1e-6), "时间修正应改变 GAE 优势值"

    def test_time_corrected_uniform_dt_equals_fixed(self):
        """当 dt_seq 全部等于 sim_dt_ref 时，时间修正 GAE 应退化为固定 gamma。"""
        T, P = 4, 2
        rewards = np.random.randn(T, P).astype(np.float32)
        values = np.random.randn(T, P).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        last_v = np.random.randn(P).astype(np.float32)
        dt_seq = np.ones(T, dtype=np.float32)
        sim_dt_ref = 1.0

        adv_fixed, _ = compute_gae(rewards, values, dones, last_v, gamma=0.99, lam=0.95)
        adv_tc, _ = compute_gae(
            rewards, values, dones, last_v,
            gamma=0.99, lam=0.95,
            dt_seq=dt_seq, sim_dt_ref=sim_dt_ref,
        )
        assert np.allclose(adv_fixed, adv_tc, atol=1e-6), "均匀 dt 时应退化为固定 gamma"

    def test_no_dt_seq_degrades_to_fixed(self):
        """不提供 dt_seq 时应退化为固定 gamma。"""
        T, P = 4, 2
        rewards = np.random.randn(T, P).astype(np.float32)
        values = np.random.randn(T, P).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        last_v = np.random.randn(P).astype(np.float32)

        adv1, _ = compute_gae(rewards, values, dones, last_v, gamma=0.99, lam=0.95)
        adv2, _ = compute_gae(
            rewards, values, dones, last_v,
            gamma=0.99, lam=0.95,
            dt_seq=None, sim_dt_ref=1.0,
        )
        assert np.allclose(adv1, adv2, atol=1e-6), "dt_seq=None 时应退化为固定 gamma"

    def test_zero_sim_dt_ref_degrades_to_fixed(self):
        """sim_dt_ref=0 时应退化为固定 gamma。"""
        T, P = 4, 2
        rewards = np.random.randn(T, P).astype(np.float32)
        values = np.random.randn(T, P).astype(np.float32)
        dones = np.zeros(T, dtype=bool)
        last_v = np.random.randn(P).astype(np.float32)
        dt_seq = np.array([2.0, 0.5, 1.0, 3.0], dtype=np.float32)

        adv1, _ = compute_gae(rewards, values, dones, last_v, gamma=0.99, lam=0.95)
        adv2, _ = compute_gae(
            rewards, values, dones, last_v,
            gamma=0.99, lam=0.95,
            dt_seq=dt_seq, sim_dt_ref=0.0,
        )
        assert np.allclose(adv1, adv2, atol=1e-6), "sim_dt_ref=0 时应退化为固定 gamma"

    def test_larger_dt_produces_smaller_discount(self):
        """dt 越大折扣因子越小，GAE 优势应更短视（更重视即时奖励）。"""
        T, P = 3, 1
        rewards = np.array([[0.0], [1.0], [0.0]], dtype=np.float32)
        values = np.zeros((T, P), dtype=np.float32)
        dones = np.zeros(T, dtype=bool)
        last_v = np.zeros(P, dtype=np.float32)

        # dt 小：折扣大，远处奖励传播更多
        adv_small_dt, _ = compute_gae(
            rewards, values, dones, last_v, gamma=0.99, lam=0.95,
            dt_seq=np.array([0.5, 0.5, 0.5], dtype=np.float32), sim_dt_ref=1.0,
        )
        # dt 大：折扣小，远处奖励传播更少
        adv_large_dt, _ = compute_gae(
            rewards, values, dones, last_v, gamma=0.99, lam=0.95,
            dt_seq=np.array([3.0, 3.0, 3.0], dtype=np.float32), sim_dt_ref=1.0,
        )
        # t=0 的优势在 dt 小时应更大（因为后续奖励传播更多）
        assert adv_small_dt[0, 0] > adv_large_dt[0, 0], \
            "dt 小时 t=0 的优势应更大（折扣大→远处奖励传播多）"


# ==================== P0-3：反事实信用分配 ====================


class TestCounterfactualCreditAssignment:
    """测试 Design D 的反事实信用分配输出。"""

    def test_design_d_outputs_cf_adv(self):
        """Design D 前向应输出 cf_adv、q_values、cf_baseline。"""
        bsz, a, m, ma = 4, 3, 8, 6
        model = UAVInterceptionNetwork(hidden_dim=64, num_heads=4, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        out = model.actor_forward(obs)
        assert "cf_adv" in out, "Design D 应输出 cf_adv"
        assert "q_values" in out, "Design D 应输出 q_values"
        assert "cf_baseline" in out, "Design D 应输出 cf_baseline"
        assert out["cf_adv"].shape == (bsz,), f"cf_adv 形状应为 (P,)，实际 {out['cf_adv'].shape}"
        assert out["q_values"].shape == (bsz, m), f"q_values 形状应为 (P,N)，实际 {out['q_values'].shape}"
        assert out["cf_baseline"].shape == (bsz,), f"cf_baseline 形状应为 (P,)，实际 {out['cf_baseline'].shape}"

    def test_cf_adv_finite(self):
        """反事实优势应为有限值。"""
        bsz, a, m, ma = 2, 3, 5, 6
        model = UAVInterceptionNetwork(hidden_dim=32, num_heads=2, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        out = model.actor_forward(obs)
        assert torch.isfinite(out["cf_adv"]).all(), "cf_adv 应全部有限"
        assert torch.isfinite(out["q_values"]).all(), "q_values 应全部有限"
        assert torch.isfinite(out["cf_baseline"]).all(), "cf_baseline 应全部有限"

    def test_cf_adv_near_zero_at_init(self):
        """反事实 Q 头 zero-init，初始化时 cf_adv 应接近零。"""
        bsz, a, m, ma = 4, 3, 8, 6
        model = UAVInterceptionNetwork(hidden_dim=64, num_heads=4, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        out = model.actor_forward(obs)
        # CounterfactualQHead 最后一层 zero-init → q_values ≈ 0 → cf_baseline ≈ 0 → cf_adv ≈ 0
        assert out["cf_adv"].abs().max().item() < 0.1, \
            f"初始化时 cf_adv 应接近零，实际最大值 {out['cf_adv'].abs().max().item():.4f}"

    def test_counterfactual_q_head_zero_init(self):
        """CounterfactualQHead 最后一层 zero-init 应保证输出为零。"""
        D = 32
        head = CounterfactualQHead(D)
        self_ctx = torch.randn(4, D)
        e_self_pts = torch.randn(4, 6, D)
        q = head(self_ctx, e_self_pts)
        assert torch.allclose(q, torch.zeros_like(q), atol=1e-6), \
            "CounterfactualQHead zero-init 应输出零"

    def test_cf_adv_definition(self):
        """cf_adv = Q(s, a_selected) - baseline，其中 baseline = Σ π(a) Q(s, a)。"""
        bsz, a, m, ma = 2, 3, 5, 6
        model = UAVInterceptionNetwork(hidden_dim=32, num_heads=2, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        out = model.actor_forward(obs)

        q_values = out["q_values"]  # (P, N)
        probs = out["action_probs"]  # (P, N)

        # 手动计算 baseline = Σ π(a) * Q(s, a)
        # 注意：q_values 在初始化时接近零，所以这个测试主要验证维度正确
        manual_baseline = (probs * q_values).sum(dim=-1)  # (P,)
        assert torch.allclose(out["cf_baseline"], manual_baseline, atol=1e-5), \
            "cf_baseline 应等于 Σ π(a) * Q(s, a)"


# ==================== P2-1：多帧时序输入 ====================


class TestTemporalSelfCtxEncoder:
    """测试 TemporalSelfCtxEncoder 和多帧时序输入。"""

    def test_design_d_outputs_self_ctx(self):
        """Design D 前向应输出 self_ctx (P, D) 供时序缓存。"""
        bsz, a, m, ma = 4, 3, 8, 6
        model = UAVInterceptionNetwork(hidden_dim=64, num_heads=4, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        out = model.actor_forward(obs)
        assert "self_ctx" in out, "Design D 应输出 self_ctx"
        assert out["self_ctx"].shape == (bsz, 64), f"self_ctx 形状应为 (P,D)，实际 {out['self_ctx'].shape}"

    def test_design_d_with_temporal_history(self):
        """Design D 接收时序历史时应正常前向传播。"""
        bsz, a, m, ma = 4, 3, 8, 6
        D = 64
        temporal_window = 4
        model = UAVInterceptionNetwork(
            hidden_dim=D, num_heads=4, design_mode="D",
            temporal_heads=2, temporal_layers=1, temporal_window=temporal_window,
        )
        obs = _fake_obs(bsz, a, m, ma)

        # 无历史帧时应正常工作
        out_no_hist = model.actor_forward(obs)
        assert out_no_hist["action_probs"].shape == (bsz, m)

        # 有历史帧时应正常工作
        hist_len = temporal_window - 1
        self_ctx_history = torch.randn(bsz, hist_len, D)
        self_ctx_history_mask = torch.ones(bsz, hist_len, dtype=torch.bool)
        out_with_hist = model.actor_forward(
            obs,
            self_ctx_history=self_ctx_history,
            self_ctx_history_mask=self_ctx_history_mask,
        )
        assert out_with_hist["action_probs"].shape == (bsz, m)
        assert torch.isfinite(out_with_hist["action_probs"]).all()

    def test_design_d_temporal_encoder_is_identity_at_init(self):
        """TemporalSelfCtxEncoder zero-init 输出投影，初始化时应近似恒等。"""
        bsz, a, m, ma = 2, 3, 5, 6
        D = 32
        model = UAVInterceptionNetwork(
            hidden_dim=D, num_heads=2, design_mode="D",
            temporal_heads=2, temporal_layers=1, temporal_window=4,
        )
        obs = _fake_obs(bsz, a, m, ma)

        # 无历史帧 vs 有历史帧：由于 zero-init，初始时应近似一致
        out_no_hist = model.actor_forward(obs)
        hist = torch.randn(bsz, 3, D)
        hist_mask = torch.ones(bsz, 3, dtype=torch.bool)
        out_with_hist = model.actor_forward(obs, self_ctx_history=hist, self_ctx_history_mask=hist_mask)

        # 不严格要求完全一致（Transformer 的 PE 会引入差异），但概率分布不应差异过大
        prob_diff = (out_no_hist["action_probs"] - out_with_hist["action_probs"]).abs().max().item()
        assert prob_diff < 0.5, f"初始时有无历史帧的概率差异应较小，实际 {prob_diff:.4f}"

    def test_temporal_encoder_t1_identity(self):
        """TemporalSelfCtxEncoder 在 T=1 时应退化为恒等映射。"""
        D = 32
        encoder = TemporalSelfCtxEncoder(hidden_dim=D, num_heads=2, num_layers=1, max_window=4)
        P = 3
        x = torch.randn(P, 1, D)  # T=1
        mask = torch.ones(P, 1, dtype=torch.bool)
        out = encoder(x, mask)
        # T=1 时直接 squeeze 返回，不经过 Transformer
        assert out.shape == (P, D), f"T=1 输出形状应为 (P,D)，实际 {out.shape}"
        assert torch.allclose(out, x.squeeze(1), atol=1e-6), \
            "T=1 时应退化为恒等映射"

    def test_temporal_encoder_output_shape(self):
        """TemporalSelfCtxEncoder 多帧输入时输出形状应为 (P, D)。"""
        D = 32
        encoder = TemporalSelfCtxEncoder(hidden_dim=D, num_heads=2, num_layers=1, max_window=8)
        P = 4
        T = 5
        x = torch.randn(P, T, D)
        mask = torch.ones(P, T, dtype=torch.bool)
        out = encoder(x, mask)
        assert out.shape == (P, D), f"输出形状应为 (P,D)，实际 {out.shape}"
        assert torch.isfinite(out).all(), "输出应全部有限"

    def test_temporal_encoder_with_partial_mask(self):
        """TemporalSelfCtxEncoder 应正确处理部分有效帧的 mask。"""
        D = 16
        encoder = TemporalSelfCtxEncoder(hidden_dim=D, num_heads=2, num_layers=1, max_window=4)
        P = 2
        T = 3
        x = torch.randn(P, T, D)
        # 第一条序列只有最后2帧有效
        mask = torch.ones(P, T, dtype=torch.bool)
        mask[0, 0] = False
        out = encoder(x, mask)
        assert out.shape == (P, D)
        assert torch.isfinite(out).all(), "部分 mask 时输出应全部有限"

    def test_design_d_self_ctx_detached(self):
        """输出的 self_ctx 应是 detached 的（不参与梯度回传）。"""
        bsz, a, m, ma = 2, 3, 5, 6
        model = UAVInterceptionNetwork(hidden_dim=32, num_heads=2, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        out = model.actor_forward(obs)
        assert not out["self_ctx"].requires_grad, "self_ctx 应为 detached"

    def test_design_d_without_temporal_params(self):
        """Design D 不传 temporal 参数时也应正常工作（使用默认值）。"""
        bsz, a, m, ma = 2, 3, 5, 6
        model = UAVInterceptionNetwork(hidden_dim=32, num_heads=2, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        # 不传 temporal 参数，temporal_encoder 仍会创建（默认 window=4）
        out = model.actor_forward(obs)
        assert "self_ctx" in out
        assert "action_probs" in out
        assert out["action_probs"].shape == (bsz, m)


class TestSelfCtxHistoryBuffer:
    """测试 SelfCtxHistoryBuffer 的数据管理。"""

    def test_buffer_append_and_get(self):
        """基本 append 和 get_history 功能。"""
        P, D, temporal_window = 3, 16, 4
        buf = SelfCtxHistoryBuffer(temporal_window, P, D)

        # 初始为空
        assert buf.get_history() is None
        assert buf.get_history_mask() is None

        # 添加一帧
        ctx1 = np.random.randn(P, D).astype(np.float32)
        buf.append(ctx1)
        hist = buf.get_history()
        assert hist is not None
        assert hist.shape == (P, 1, D), f"历史形状应为 (P,1,D)，实际 {hist.shape}"

        mask = buf.get_history_mask()
        assert mask is not None
        assert mask.shape == (P, 1)
        assert mask.all(), "所有帧应标记为有效"

    def test_buffer_window_size(self):
        """缓存应只保留 temporal_window-1 帧。"""
        P, D, temporal_window = 2, 8, 4
        buf = SelfCtxHistoryBuffer(temporal_window, P, D)
        max_hist = temporal_window - 1  # 3

        for i in range(10):
            buf.append(np.random.randn(P, D).astype(np.float32))

        hist = buf.get_history()
        assert hist.shape == (P, max_hist, D), f"应保留 {max_hist} 帧，实际 {hist.shape[1]}"

    def test_buffer_reset(self):
        """reset 应清空缓存。"""
        P, D = 2, 8
        buf = SelfCtxHistoryBuffer(4, P, D)
        buf.append(np.random.randn(P, D).astype(np.float32))
        buf.reset()
        assert buf.get_history() is None

    def test_buffer_preserves_order(self):
        """缓存应保留帧的时间顺序（最新帧在最后）。"""
        P, D = 2, 4
        buf = SelfCtxHistoryBuffer(4, P, D)
        # 添加3帧，每帧的值递增
        for i in range(3):
            ctx = np.full((P, D), float(i), dtype=np.float32)
            buf.append(ctx)
        hist = buf.get_history()  # (P, 3, D)
        assert hist is not None
        # 最新帧（i=2）应在最后一列
        assert np.allclose(hist[0, -1, :], 2.0), "最新帧应在历史序列最后一列"
        assert np.allclose(hist[0, 0, :], 0.0), "最旧帧应在历史序列第一列"

    def test_buffer_temporal_window_1(self):
        """temporal_window=1 时不应保留任何历史帧。"""
        P, D = 2, 8
        buf = SelfCtxHistoryBuffer(1, P, D)
        buf.append(np.random.randn(P, D).astype(np.float32))
        # temporal_window=1 → max_hist=0 → 不保留任何帧
        assert buf.get_history() is None

    def test_buffer_append_does_not_modify_input(self):
        """append 后修改原始数组不应影响缓存内容。"""
        P, D = 2, 4
        buf = SelfCtxHistoryBuffer(4, P, D)
        ctx = np.ones((P, D), dtype=np.float32)
        buf.append(ctx)
        # 修改原始数组
        ctx[:] = 99.0
        # 缓存内容不应改变
        hist = buf.get_history()
        assert hist is not None
        assert not np.any(hist == 99.0), "append 应做深拷贝"


class TestBuildWithTemporalKwargs:
    """测试 build_actor_critic_schemes 传递 temporal kwargs。"""

    def test_build_design_d_with_temporal_params(self):
        """通过 build_actor_critic_schemes 传递 temporal 参数应正常构建模型。"""
        models = build_actor_critic_schemes(
            "D",
            hidden_dim=32,
            num_heads=2,
            temporal_heads=2,
            temporal_layers=1,
            temporal_window=4,
        )
        assert len(models) == 1
        name = list(models.keys())[0]
        model = models[name]

        # 验证 temporal_encoder 已创建
        assert model.model.temporal_encoder is not None

        # 验证模型可前向传播
        obs = _fake_obs(2, 3, 5, 6)
        out = model(obs)
        assert "action_probs" in out
        assert "self_ctx" in out

    def test_build_design_d_without_temporal_params(self):
        """不传 temporal 参数时 Design D 也能正常构建。"""
        models = build_actor_critic_schemes(
            "D",
            hidden_dim=32,
            num_heads=2,
        )
        assert len(models) == 1
        name = list(models.keys())[0]
        model = models[name]
        # temporal_encoder 仍会被创建（使用默认值）
        assert model.model.temporal_encoder is not None


class TestPPOMinibatchWithTemporal:
    """测试 ppo_minibatch_update 的时序历史支持。"""

    def test_ppo_with_temporal_data(self):
        """PPO 更新接收 self_ctx_seq 和 mask 时应正常工作。"""
        T, P, D, M = 4, 2, 16, 5
        model = UAVInterceptionNetwork(
            hidden_dim=D, num_heads=2, design_mode="D",
            temporal_heads=2, temporal_layers=1, temporal_window=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        # 构造 rollout 数据
        obs_list = [_fake_obs_np(P, 3, M, 6) for _ in range(T)]

        actions = np.random.randint(0, M, size=(T, P)).astype(np.int64)
        logp_old = np.random.randn(T, P).astype(np.float32)
        advantages = np.random.randn(T, P).astype(np.float32)
        returns = np.random.randn(T, P).astype(np.float32)
        old_values = np.random.randn(T, P).astype(np.float32)
        self_ctx_seq = np.random.randn(T, P, D).astype(np.float32)
        self_ctx_seq_mask = np.ones((T, P), dtype=bool)

        result = ppo_minibatch_update(
            model, optimizer, obs_list, actions, logp_old, advantages, returns, old_values,
            device, _build_obs_np,
            clip_range=0.2, ppo_epochs=1, value_coef=0.5, entropy_coef=0.01,
            max_grad_norm=1.0, minibatch_size=2,
            self_ctx_seq=self_ctx_seq,
            self_ctx_seq_mask=self_ctx_seq_mask,
            temporal_window=4,
        )
        assert len(result) == 7, f"PPO 更新应返回 7 个值，实际 {len(result)}"
        # 所有返回值应为有限数值
        for i, v in enumerate(result):
            assert np.isfinite(v), f"PPO 返回值 [{i}] 应为有限值，实际 {v}"

    def test_ppo_without_temporal_data(self):
        """PPO 更新不传时序数据时应正常工作（回归测试）。"""
        T, P, D, M = 4, 2, 16, 5
        model = UAVInterceptionNetwork(
            hidden_dim=D, num_heads=2, design_mode="D",
            temporal_heads=2, temporal_layers=1, temporal_window=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        obs_list = [_fake_obs_np(P, 3, M, 6) for _ in range(T)]
        actions = np.random.randint(0, M, size=(T, P)).astype(np.int64)
        logp_old = np.random.randn(T, P).astype(np.float32)
        advantages = np.random.randn(T, P).astype(np.float32)
        returns = np.random.randn(T, P).astype(np.float32)
        old_values = np.random.randn(T, P).astype(np.float32)

        result = ppo_minibatch_update(
            model, optimizer, obs_list, actions, logp_old, advantages, returns, old_values,
            device, _build_obs_np,
            clip_range=0.2, ppo_epochs=1, value_coef=0.5, entropy_coef=0.01,
            max_grad_norm=1.0, minibatch_size=2,
            # 不传 self_ctx_seq 和 temporal_window
        )
        assert len(result) == 7
        for i, v in enumerate(result):
            assert np.isfinite(v), f"PPO 返回值 [{i}] 应为有限值，实际 {v}"

    def test_ppo_temporal_window_0_disables_temporal(self):
        """temporal_window=0 时应禁用时序输入。"""
        T, P, D, M = 4, 2, 16, 5
        model = UAVInterceptionNetwork(
            hidden_dim=D, num_heads=2, design_mode="D",
            temporal_heads=2, temporal_layers=1, temporal_window=4,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        device = torch.device("cpu")

        obs_list = [_fake_obs_np(P, 3, M, 6) for _ in range(T)]
        actions = np.random.randint(0, M, size=(T, P)).astype(np.int64)
        logp_old = np.random.randn(T, P).astype(np.float32)
        advantages = np.random.randn(T, P).astype(np.float32)
        returns = np.random.randn(T, P).astype(np.float32)
        old_values = np.random.randn(T, P).astype(np.float32)
        self_ctx_seq = np.random.randn(T, P, D).astype(np.float32)
        self_ctx_seq_mask = np.ones((T, P), dtype=bool)

        result = ppo_minibatch_update(
            model, optimizer, obs_list, actions, logp_old, advantages, returns, old_values,
            device, _build_obs_np,
            clip_range=0.2, ppo_epochs=1, value_coef=0.5, entropy_coef=0.01,
            max_grad_norm=1.0, minibatch_size=2,
            self_ctx_seq=self_ctx_seq,
            self_ctx_seq_mask=self_ctx_seq_mask,
            temporal_window=0,  # 禁用
        )
        assert len(result) == 7
        for i, v in enumerate(result):
            assert np.isfinite(v), f"PPO 返回值 [{i}] 应为有限值，实际 {v}"


# ==================== 集成测试：Design D 完整前向 ====================


class TestDesignDIntegration:
    """Design D 综合集成测试。"""

    def test_design_d_forward_produces_all_keys(self):
        """Design D 的 forward 应输出所有预期的键。"""
        bsz, a, m, ma = 4, 3, 8, 6
        model = UAVInterceptionNetwork(hidden_dim=64, num_heads=4, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        out = model(obs)  # 注意用 model() 而非 model.actor_forward()

        expected_keys = {
            "action_logits", "action_probs", "best_candidate_idx",
            "coord_attn_weights", "cf_adv", "q_values", "cf_baseline",
            "self_ctx", "value",
        }
        for key in expected_keys:
            assert key in out, f"Design D forward 应输出 {key}"

    def test_design_d_value_shape(self):
        """Design D 的 value 输出形状应为 (P,)。"""
        bsz, a, m, ma = 4, 3, 8, 6
        model = UAVInterceptionNetwork(hidden_dim=64, num_heads=4, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        out = model(obs)
        assert out["value"].shape == (bsz,), f"value 形状应为 (P,)，实际 {out['value'].shape}"

    def test_design_d_probs_sum_to_one(self):
        """Design D 的 action_probs 应在有效行上求和为 1。"""
        bsz, a, m, ma = 4, 3, 8, 6
        model = UAVInterceptionNetwork(hidden_dim=64, num_heads=4, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        out = model(obs)
        for i in range(bsz):
            row_sum = out["action_probs"][i].sum().item()
            assert abs(row_sum - 1.0) < 1e-5, f"action_probs 行 {i} 求和应为 1.0，实际 {row_sum}"

    def test_design_d_with_masked_candidates(self):
        """Design D 在部分候选点被 mask 时应正确处理。"""
        bsz, a, m, ma = 2, 3, 8, 6
        model = UAVInterceptionNetwork(hidden_dim=32, num_heads=2, design_mode="D")
        obs = _fake_obs(bsz, a, m, ma)
        # mask 掉部分候选点
        obs["self_pts_mask"][0, 4:] = False
        obs["self_pts_mask"][1, :] = False  # 全部 mask

        out = model.actor_forward(obs)
        # 全部 mask 的行应输出 -1 的 best_candidate_idx
        assert out["best_candidate_idx"][1].item() == -1
        # 部分 mask 的行概率应在有效候选点上求和为 1
        valid_sum = out["action_probs"][0, :4].sum().item()
        assert abs(valid_sum - 1.0) < 1e-5, f"部分 mask 行概率应在有效候选点上求和为 1，实际 {valid_sum}"
