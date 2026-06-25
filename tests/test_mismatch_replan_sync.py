import numpy as np


def _setup_env_3v3(seed: int = 0):
    from marl.envs import TODCMARLEnv

    env = TODCMARLEnv({"render_mode": "none", "max_episode_steps": 20})
    env.reset(seed=seed)
    if env.num_P < 3 or env.num_E < 3:
        return None
    return env


def test_drop_captured_pairs_before_mismatch_replan():
    """验证非匹配捕获重规划前会先剔除已捕获 evader，避免进入 Hungarian。"""
    env = _setup_env_3v3(seed=7)
    if env is None:
        return

    # 强制 3v3 对角配对
    env.pairs_realE2P = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env._pairs_ic_compact = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env.UnCapEid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapPid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapEidNew = np.asarray([0, 1, 2], dtype=int)
    env.UnCapPidNew = np.asarray([0, 1, 2], dtype=int)
    env._sync_assigned_eid_full_from_pairs()
    env.Capflag_full[:] = False

    # 触发非匹配捕获：eid=0 被 pid=1 捕获（pid=0 才是 assigned）
    env.PosP = np.array([
        [2000.0, 1000.0, np.pi],
        [1100.0, 1000.0, np.pi],
        [3000.0, 3000.0, 0.0],
    ])
    env.PosE = np.array([
        [1000.0, 1000.0, 0.0],
        [5000.0, 5000.0, 0.0],
        [6000.0, 6000.0, 0.0],
    ])

    newly, mismatch = env._update_capflag_full_from_geometry()
    assert newly == 1
    assert mismatch == 1
    assert bool(env.Capflag_full[0])

    dropped = env._drop_captured_pairs(log_context="TEST_MISMATCH")
    assert dropped == 1

    # 关键断言：已捕获 eid=0 不再出现在配对/存活集合中
    remaining_pairs = np.asarray(env.pairs_realE2P, dtype=np.int64)
    assert remaining_pairs.shape[0] == 2
    assert 0 not in set(int(x) for x in remaining_pairs[:, 0].tolist())
    assert 0 not in set(int(x) for x in np.asarray(env.UnCapEidNew, dtype=int).tolist())

    # 与完整性校验保持一致
    env._validate_pair_data_integrity(context="test:post_drop")


def test_mismatch_replan_preamble_wont_reinclude_captured_eid():
    """验证 mismatch 重规划前置同步不会把已捕获 eid 带回 UnCap 集合。"""
    env = _setup_env_3v3(seed=8)
    if env is None:
        return

    env.pairs_realE2P = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env._pairs_ic_compact = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env.UnCapEid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapPid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapEidNew = np.asarray([0, 1, 2], dtype=int)
    env.UnCapPidNew = np.asarray([0, 1, 2], dtype=int)
    env._sync_assigned_eid_full_from_pairs()
    env.Capflag_full[:] = False

    # 构造非匹配捕获
    env.PosP = np.array([
        [2000.0, 1000.0, np.pi],
        [1100.0, 1000.0, np.pi],
        [3000.0, 3000.0, 0.0],
    ])
    env.PosE = np.array([
        [1000.0, 1000.0, 0.0],
        [5000.0, 5000.0, 0.0],
        [6000.0, 6000.0, 0.0],
    ])

    newly, mismatch = env._update_capflag_full_from_geometry()
    assert newly == 1
    assert mismatch == 1
    assert bool(env.Capflag_full[0])

    # 对齐 step 中 mismatch 分支的新逻辑
    env._drop_captured_pairs(log_context="TEST_MISMATCH_PRE")

    # 关键：重规划前用于构造候选的集合来自 UnCapEidNew/UnCapPidNew。
    # 只要这里已剔除 captured eid，后续 _compute_isomap_intercept_candidates 就不会再包含它。
    assert 0 not in set(int(x) for x in np.asarray(env.UnCapEidNew, dtype=int).tolist())

    env.UnCapEid = env.UnCapEidNew.copy()
    env.UnCapPid = env.UnCapPidNew.copy()
    assert 0 not in set(int(x) for x in np.asarray(env.UnCapEid, dtype=int).tolist())
