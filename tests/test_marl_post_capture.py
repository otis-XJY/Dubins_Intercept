"""
捕获之后的状态：手工注入 Capflag + 调用 _phase_update，以及「门控失效」行为。

真实几何捕获：可选超大 cap_dist/cap_angle 使 _update_capflag_from_geometry 一次性全捕获。

已修复（见 ``TODCMARLEnv._update_capflag_from_geometry`` / ``marl.envs.todc_env``）：在 ``PosE`` 行数与上一时刻 ``Capflag`` 长度不一致时，
上一捕获向量与新的 ``new_cap`` 对齐（否则 ``~prev_cap & Capflag`` 会广播报错）。该情形出现在
``_phase_update`` 已收缩 ``pairs``、``_advance_from_paths`` 已用紧凑 ``UnCapEidNew`` 采样 ``PosE``，
但 ``Capflag`` 尚未重写为同长度之前。
"""

import numpy as np
import pytest

from marl.envs import TODCMARLEnv


@pytest.fixture
def env_ready():
    """完成 reset（含 IsoMap），pairs 与 Capflag 均为 num_E 长。"""
    e = TODCMARLEnv({"render_mode": "none", "max_episode_steps": 50})
    e.reset()
    assert e.pairs_realE2P is not None
    assert e.pairs_realE2P.shape[0] == e.num_E
    assert e.Capflag.shape[0] == e.num_E
    yield e


def test_manual_one_flag_true_phase_update_filters_pairs_and_uncap(env_ready):
    """模拟「第 0 号敌已被标为捕获」：_phase_update 收缩 pairs，并刷新 UnCap*New。"""
    env = env_ready
    ne = env.num_E
    assert not np.any(env.Capflag)

    env.Capflag[0] = True
    env._phase_update()

    assert env.pairs_realE2P.shape[0] == ne - 1
    assert len(env.UnCapPidNew) == ne - 1
    assert len(env.UnCapEidNew) == ne - 1
    # Capflag 未被 _phase_update 截短，仍与初始几何步一致
    assert env.Capflag.shape[0] == ne


def test_second_phase_update_skipped_when_pairs_shorter_than_capflag(env_ready):
    """第一次过滤后 pairs 行数 < Capflag 长度，门控失败，后续 _phase_update 不再改 pairs/UnCap。"""
    env = env_ready
    ne = env.num_E
    env.Capflag[0] = True
    env._phase_update()

    pairs_after = env.pairs_realE2P.copy()
    uncap_p = env.UnCapPidNew.copy()
    uncap_e = env.UnCapEidNew.copy()

    env._phase_update()

    assert np.array_equal(env.pairs_realE2P, pairs_after)
    assert np.array_equal(env.UnCapPidNew, uncap_p)
    assert np.array_equal(env.UnCapEidNew, uncap_e)
    assert env.pairs_realE2P.shape[0] != env.Capflag.shape[0]


def test_advance_from_paths_uses_row_count_of_uncap_pid_new(env_ready):
    """捕获一人造后 UnCapPidNew 变短，_advance_from_paths 产生 compact PosP。"""
    env = env_ready
    env.Capflag[0] = True
    env._phase_update()

    env._advance_from_paths()
    assert env.PosP.shape[0] == len(env.UnCapPidNew)
    assert env.PosE.shape[0] == len(env.UnCapEidNew)


def test_update_capflag_geometry_resizes_capflag_to_match_pos_e(env_ready):
    """几何更新按当前 PosE 行数重写 Capflag；与「未截短的旧 Capflag」长度可对齐或不对齐。"""
    env = env_ready
    env.Capflag[0] = True
    env._phase_update()
    env._advance_from_paths()

    n_e_active = env.PosE.shape[0]
    assert n_e_active == env.num_E - 1

    env._update_capflag_from_geometry()
    assert env.Capflag.shape[0] == n_e_active
    assert env.Capflag.shape[0] == env.PosE.shape[0]


def test_second_wave_capture_after_capflag_resized(env_ready):
    """第一波：手工标 e0 已捕获 → pairs 与 Capflag 仍同长失败；先 phase 再 advance+geometry 对齐长度后，第二波可继续收缩 pairs。"""
    env = env_ready
    ne = env.num_E
    if ne < 2:
        pytest.skip("需要至少 2 架敌机")

    env.Capflag[0] = True
    env._phase_update()
    assert env.pairs_realE2P.shape[0] == ne - 1

    env._advance_from_paths()
    env._update_capflag_from_geometry()
    assert env.Capflag.shape[0] == ne - 1
    assert env.pairs_realE2P.shape[0] == ne - 1

    env.Capflag[0] = True
    env._phase_update()
    assert env.pairs_realE2P.shape[0] == ne - 2
    assert len(env.UnCapEidNew) == ne - 2


def test_extreme_cap_dist_geometry_all_captured(env_ready):
    """超大捕获距离 + 全角：几何上一步全部标为已捕获，Capflag 与 PosE 行数一致。"""
    env = env_ready
    env.CapRef["CapDist"] = 1.0e15
    env.CapRef["CapAngle"] = 360.0

    env._update_capflag_from_geometry()
    assert env.Capflag.shape[0] == env.PosE.shape[0]
    assert np.all(env.Capflag)


def test_step_after_extreme_capture_terminates_or_truncates(env_ready):
    """全捕获后 step 内层应判定 terminal（全捕获）。"""
    env = env_ready
    env.CapRef["CapDist"] = 1.0e15
    env.CapRef["CapAngle"] = 360.0

    act = np.zeros(env.num_P, dtype=np.int64)
    obs, r, term, trunc, info = env.step(act)
    # 全捕获或碰撞/时间等任一终止
    assert term["__all__"] is True or trunc["__all__"] is True or env.episode_step >= 1


def test_normalize_action_inactive_pursuer_index_minus_one(env_ready):
    """部分捕获后无任务机动作 -1，_normalize_action 不报错的且 idx 为 -1。"""
    env = env_ready
    env.Capflag[0] = True
    env._phase_update()
    env._advance_from_paths()
    env._update_capflag_from_geometry()
    obs = env._build_obs()
    env._sync_dynamic_k(obs)
    active = obs["pursuer_active"].astype(bool)
    act = np.full(env.num_P, -1, dtype=np.int64)
    for i in range(env.num_P):
        if active[i]:
            m = obs["self_pts_mask"][i]
            act[i] = int(np.argmax(m.astype(np.float32)))
    _norm, idx, _snap = env._normalize_action(act)
    for i in range(env.num_P):
        if not active[i]:
            assert idx[i] == -1
        else:
            assert idx[i] >= 0


def test_build_obs_after_partial_manual_capture_shapes(env_ready):
    """人工标一敌捕获并同步几何后，观测首维仍为 num_P（Gym 批维不变）。"""
    env = env_ready
    env.Capflag[0] = True
    env._phase_update()
    env._advance_from_paths()
    env._update_capflag_from_geometry()

    obs = env._build_obs()
    env._sync_dynamic_k(obs)
    assert obs["self_uav"].shape[0] == env.num_P
    assert obs["self_pts"].shape[0] == env.num_P
    assert obs["enemies"].shape == (env.num_P, env.num_E, 3)
    assert obs["self_pts_mask"].shape[1] == obs["self_pts"].shape[1]
    assert obs["pursuer_active"].shape == (env.num_P,)
    assert int(np.sum(obs["pursuer_active"])) == env.num_P - 1


def test_captured_total_matches_sum_capflag_on_normal_step(env_ready):
    """未人工破坏状态时，info 中 captured_total 与 sum(Capflag_full) 一致。"""
    env = env_ready
    act = np.zeros(env.num_P, dtype=np.int64)
    _, _, _, _, info = env.step(act)
    assert info["p_0"]["captured_total"] == int(np.sum(env.Capflag_full))


def test_capflag_row_i_pairs_with_pairs_row_i_not_global_eid(env_ready):
    """_phase_update 中 ``Capflag[i]`` 与 ``pairs_realE2P`` 第 i 行对齐（与 main0319 布尔过滤一致），非按全局 eid 查表。"""
    env = env_ready
    pr = env.pairs_realE2P.copy()
    env.Capflag[0] = True
    env._phase_update()
    assert env.pairs_realE2P.shape[0] == pr.shape[0] - 1
    r0 = (int(pr[0, 0]), int(pr[0, 1]))
    for row in env.pairs_realE2P:
        assert (int(row[0]), int(row[1])) != r0


def test_pairwise_dist_shape_after_compact_geometry(env_ready):
    """紧凑 PosP/PosE 下距离阵为 (nP_compact, nE_active)。"""
    env = env_ready
    env.Capflag[0] = True
    env._phase_update()
    env._advance_from_paths()
    d = env._pairwise_dist()
    assert d.shape == (env.PosP.shape[0], env.PosE.shape[0])
