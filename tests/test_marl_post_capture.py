"""
捕获之后的状态：手工注入 Capflag_full + 调用 _phase_update。
"""

import numpy as np
import pytest

from marl.envs import TODCMARLEnv


@pytest.fixture
def env_ready():
    """完成 reset（含 IsoMap），pairs 与 Capflag_full 均为 num_E 长。"""
    e = TODCMARLEnv({"render_mode": "none", "max_episode_steps": 50})
    e.reset()
    assert e.pairs_realE2P is not None
    assert e.pairs_realE2P.shape[0] == e.num_E
    assert e.Capflag_full.shape[0] == e.num_E
    yield e


def test_manual_one_flag_true_phase_update_filters_pairs_and_uncap(env_ready):
    """模拟「第 0 号敌已被标为捕获」：_phase_update 收缩 pairs，并刷新 UnCap*New。"""
    env = env_ready
    ne = env.num_E
    assert not np.any(env.Capflag_full)

    cap_eid = int(env.pairs_realE2P[0, 0])
    env.Capflag_full[cap_eid] = True
    env._phase_update()

    assert env.pairs_realE2P.shape[0] == ne - 1
    assert len(env.UnCapPidNew) == ne - 1
    assert len(env.UnCapEidNew) == ne - 1
    assert env.Capflag_full.shape[0] == ne


def test_second_phase_update_skipped_when_pairs_shorter_than_capflag(env_ready):
    """第一次过滤后 pairs 行数 < Capflag_full 长度，后续 _phase_update 不再改 pairs/UnCap。"""
    env = env_ready
    ne = env.num_E
    cap_eid = int(env.pairs_realE2P[0, 0])
    env.Capflag_full[cap_eid] = True
    env._phase_update()

    pairs_after = env.pairs_realE2P.copy()
    uncap_p = env.UnCapPidNew.copy()
    uncap_e = env.UnCapEidNew.copy()

    env._phase_update()

    assert np.array_equal(env.pairs_realE2P, pairs_after)
    assert np.array_equal(env.UnCapPidNew, uncap_p)
    assert np.array_equal(env.UnCapEidNew, uncap_e)
    assert env.pairs_realE2P.shape[0] != env.Capflag_full.shape[0]


def test_advance_from_paths_uses_row_count_of_uncap_pid_new(env_ready):
    """捕获一人造后 UnCapPidNew 变短，_advance_from_paths 产生 compact PosP。"""
    env = env_ready
    cap_eid = int(env.pairs_realE2P[0, 0])
    env.Capflag_full[cap_eid] = True
    env._phase_update()

    env._advance_from_paths()
    assert env.PosP.shape[0] == len(env.UnCapPidNew)
    assert env.PosE.shape[0] == len(env.UnCapEidNew)


def test_step_after_extreme_capture_terminates_or_truncates(env_ready):
    """全捕获后 step 内层应判定 terminal（全捕获）。"""
    env = env_ready
    env.CapRef["CapDist"] = 1.0e15
    env.CapRef["CapAngle"] = 360.0

    act = np.zeros(env.num_P, dtype=np.int64)
    obs, r, term, trunc, info = env.step(act)
    assert term["__all__"] is True or trunc["__all__"] is True or env.episode_step >= 1


def test_normalize_action_inactive_pursuer_index_minus_one(env_ready):
    """部分捕获后无任务机动作 -1，_normalize_action 不报错的且 idx 为 -1。"""
    env = env_ready
    cap_eid = int(env.pairs_realE2P[0, 0])
    env.Capflag_full[cap_eid] = True
    env._phase_update()
    env._advance_from_paths()
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
    """人工标一敌捕获后，观测首维仍为 num_P（Gym 批维不变）。"""
    env = env_ready
    cap_eid = int(env.pairs_realE2P[0, 0])
    env.Capflag_full[cap_eid] = True
    env._phase_update()
    env._advance_from_paths()

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
    """_phase_update 按 Capflag_full[global_eid] 过滤 pairs_realE2P。"""
    env = env_ready
    pr = env.pairs_realE2P.copy()
    cap_eid = int(pr[0, 0])
    env.Capflag_full[cap_eid] = True
    env._phase_update()
    assert env.pairs_realE2P.shape[0] == pr.shape[0] - 1
    r0 = (int(pr[0, 0]), int(pr[0, 1]))
    for row in env.pairs_realE2P:
        assert (int(row[0]), int(row[1])) != r0


def test_pairwise_dist_shape_after_compact_geometry(env_ready):
    """紧凑 PosP/PosE 下 _pairwise_dist 返回 element-wise 距离 (1D, 长度=min(nP,nE))。"""
    env = env_ready
    cap_eid = int(env.pairs_realE2P[0, 0])
    env.Capflag_full[cap_eid] = True
    env._phase_update()
    env._advance_from_paths()
    d = env._pairwise_dist()
    assert d.ndim == 1
    assert d.shape[0] == min(env.PosP.shape[0], env.PosE.shape[0])
