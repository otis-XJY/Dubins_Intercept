import numpy as np


def _setup_env_3v3(seed: int = 0):
    from marl.envs import TODCMARLEnv

    env = TODCMARLEnv({"render_mode": "none", "max_episode_steps": 20})
    env.reset(seed=seed)
    if env.num_P < 3 or env.num_E < 3:
        return None
    return env


def _assert_pos_mapping_consistent(env):
    """
    捕获后最容易错的是“紧凑 PosP/PosE 的行”与 “UnCapPidNew/UnCapEidNew 的全局 id”错位。

    我们用稳定的全局映射函数断言：
    - 对每个存活 pid: env._pos_p_xyz_for_global_pid(pid) 与 env.PosP[对应行] 一致
    - 对每个存活 eid: env._pos_e_xyz_for_global_eid(eid) 与 env.PosE[对应行] 一致
    """
    assert env.PosP.shape[0] == len(env.UnCapPidNew)
    assert env.PosE.shape[0] == len(env.UnCapEidNew)
    for row_i, pid in enumerate(np.asarray(env.UnCapPidNew, dtype=int).ravel()):
        px, py, pth = env._pos_p_xyz_for_global_pid(int(pid))
        got = np.asarray(env.PosP[row_i], dtype=float).reshape(-1)
        assert np.allclose(got[:3], np.asarray([px, py, pth], dtype=float), atol=1e-6)
    for row_i, eid in enumerate(np.asarray(env.UnCapEidNew, dtype=int).ravel()):
        ex, ey, eth = env._pos_e_xyz_for_global_eid(int(eid))
        got = np.asarray(env.PosE[row_i], dtype=float).reshape(-1)
        assert np.allclose(got[:3], np.asarray([ex, ey, eth], dtype=float), atol=1e-6)


def test_phase_update_and_advance_keep_global_eid_mapping_after_capture():
    """
    复现用户场景：
    - 全局 P=0,1,2；全局 E=0,1,2
    - pairs_realE2P = [[0,2],[1,0],[2,1]]
    - 捕获第一行 (eid=0,pid=2) 后，应剩余 [[1,0],[2,1]]（全局索引）

    同时刻意把 UnCapEid 置为非 0..n 顺序，检查 _advance_from_paths 中 PathEpre 的索引是否仍按全局 eid 对齐。
    """
    env = _setup_env_3v3(seed=0)
    if env is None:
        return

    # 强制设置配对为用户给出的全局配对顺序
    env.pairs_realE2P = np.asarray([[0, 2], [1, 0], [2, 1]], dtype=np.int64)
    # _phase_update 会同步 shrink _pairs_ic_compact；这里给一个同 shape 占位即可
    env._pairs_ic_compact = env.pairs_realE2P.copy()

    # 刻意让 UnCapEid/UnCapPid 与全局顺序不一致（模拟经历过重规划后被覆盖为 New 列表）
    env.UnCapEid = np.asarray([0, 2, 1], dtype=int)
    env.UnCapPid = np.asarray([2, 0, 1], dtype=int)
    env.UnCapEidNew = env.UnCapEid.copy()
    env.UnCapPidNew = env.UnCapPid.copy()

    # 记录 reset 后的 PathE（按全局 eid 构造的列表），用于校验 PathEpre 的引用是否对应正确全局 eid
    pathE_global = list(env.PathE)

    # 模拟“捕获 pairs 第 0 行”的布尔过滤语义（Capflag[i] 与 pairs_realE2P 第 i 行对齐）
    env.Capflag = np.asarray([True, False, False], dtype=bool)
    env._phase_update()

    assert env.pairs_realE2P.shape == (2, 2)
    assert np.array_equal(env.pairs_realE2P, np.asarray([[1, 0], [2, 1]], dtype=np.int64))
    assert np.array_equal(env.UnCapEidNew, np.asarray([1, 2], dtype=int))
    assert np.array_equal(env.UnCapPidNew, np.asarray([0, 1], dtype=int))

    env._advance_from_paths()

    # 关键：PathEpre 必须按 UnCapEidNew 的全局 eid 对齐引用 PathE[eid]
    assert len(env.PathEpre) == 2
    assert env.PathEpre[0] is pathE_global[1]
    assert env.PathEpre[1] is pathE_global[2]

    _assert_pos_mapping_consistent(env)


def test_capture_two_rows_keeps_uncap_sets_and_pos_mapping():
    """
    一次捕获两行：pairs=[[0,2],[1,0],[2,1]]，捕获行 0 与 2，剩余应为 [[1,0]]。
    同时 UnCapPid/UnCapEid 使用非全局顺序，验证 _advance_from_paths 后 PosP/PosE 不错位。
    """
    env = _setup_env_3v3(seed=1)
    if env is None:
        return

    env.pairs_realE2P = np.asarray([[0, 2], [1, 0], [2, 1]], dtype=np.int64)
    env._pairs_ic_compact = env.pairs_realE2P.copy()
    env.UnCapEid = np.asarray([2, 0, 1], dtype=int)
    env.UnCapPid = np.asarray([1, 2, 0], dtype=int)
    env.UnCapEidNew = env.UnCapEid.copy()
    env.UnCapPidNew = env.UnCapPid.copy()

    env.Capflag = np.asarray([True, False, True], dtype=bool)
    env._phase_update()

    assert env.pairs_realE2P.shape == (1, 2)
    assert np.array_equal(env.pairs_realE2P, np.asarray([[1, 0]], dtype=np.int64))
    assert np.array_equal(env.UnCapEidNew, np.asarray([1], dtype=int))
    assert np.array_equal(env.UnCapPidNew, np.asarray([0], dtype=int))

    env._advance_from_paths()
    _assert_pos_mapping_consistent(env)


def test_pairs_row_order_shuffled_capture_first_row_still_filters_correct_pair():
    """
    pairs 行顺序被打乱时，Capflag[i] 仍是“与 pairs_realE2P 第 i 行对齐”的过滤语义。
    该测试确保捕获后剩余对与 UnCap*New 一致，且 _advance_from_paths 后 PosP/PosE 映射正确。
    """
    env = _setup_env_3v3(seed=2)
    if env is None:
        return

    # 打乱行顺序：把 (eid=0,pid=2) 放到中间行
    env.pairs_realE2P = np.asarray([[2, 1], [0, 2], [1, 0]], dtype=np.int64)
    env._pairs_ic_compact = env.pairs_realE2P.copy()
    env.UnCapEid = np.asarray([1, 0, 2], dtype=int)
    env.UnCapPid = np.asarray([0, 2, 1], dtype=int)
    env.UnCapEidNew = env.UnCapEid.copy()
    env.UnCapPidNew = env.UnCapPid.copy()

    # 捕获第 1 行（即 (0,2)）
    env.Capflag = np.asarray([False, True, False], dtype=bool)
    env._phase_update()

    assert env.pairs_realE2P.shape == (2, 2)
    assert np.array_equal(env.pairs_realE2P, np.asarray([[2, 1], [1, 0]], dtype=np.int64))
    assert np.array_equal(env.UnCapEidNew, np.asarray([2, 1], dtype=int))
    assert np.array_equal(env.UnCapPidNew, np.asarray([1, 0], dtype=int))

    env._advance_from_paths()
    _assert_pos_mapping_consistent(env)


def test_step_forced_single_capture_keeps_posp_pose_mapping():
    """
    更贴近真实 step()：
    - 不直接调用 _phase_update/_advance_from_paths，而是跑一次 env.step()
    - 用 monkeypatch 的方式强制 _update_capflag_from_geometry 在 step 内只捕获一个“紧凑槽位”的敌机
    - step 后检查：
      1) info 中 captured_delta_full == 1
      2) UnCapEidNew/UnCapPidNew 与 PosE/PosP 行数一致
      3) PosE 每行都对应 UnCapEidNew 的全局 eid（用 PathE2Val_true 的当前位置采样核对）
    """
    env = _setup_env_3v3(seed=3)
    if env is None:
        return

    # Ensure Capflag_full exists and starts with all False
    assert env.Capflag_full is not None
    env.Capflag_full[:] = False

    # Force exactly one compact-slot capture inside step.
    orig = env._update_capflag_from_geometry
    orig_full = env._update_capflag_full_from_geometry
    did_capture = {"done": False, "eid": None}

    def _forced_single_capture():
        n = int(env.PosE.shape[0])
        if n <= 0:
            env.Capflag = np.zeros((0,), dtype=bool)
            return 0
        cap = np.zeros((n,), dtype=bool)
        if not did_capture["done"]:
            cap[0] = True
            # Remember which global eid was at compact slot 0 on the capture tick.
            did_capture["eid"] = int(np.asarray(env.UnCapEidNew, dtype=int).ravel()[0])
            did_capture["done"] = True
            env.Capflag = cap
            return 1
        env.Capflag = cap
        return 0

    def _forced_single_capture_full():
        # Mirror the legacy compact-slot capture into stable global Capflag_full.
        if env.Capflag_full is None:
            raise RuntimeError("Capflag_full is None")
        if did_capture["done"] is False or did_capture["eid"] is None:
            # full should be updated on the same tick as legacy capture;
            # if legacy has not captured yet, do nothing.
            return 0
        eid = int(did_capture["eid"])
        prev = bool(env.Capflag_full[eid])
        env.Capflag_full[eid] = True
        return 0 if prev else 1

    env._update_capflag_from_geometry = _forced_single_capture  # type: ignore[assignment]
    env._update_capflag_full_from_geometry = _forced_single_capture_full  # type: ignore[assignment]

    try:
        act = np.zeros(env.num_P, dtype=np.int64)
        _obs, _rew, _term, _trunc, info = env.step(act)
    finally:
        env._update_capflag_from_geometry = orig  # type: ignore[assignment]
        env._update_capflag_full_from_geometry = orig_full  # type: ignore[assignment]

    # step 之后的 info 应该提供稳定捕获增量
    assert int(info["p_0"]["captured_delta_full"]) == 1
    assert int(info["p_0"]["captured_total_full"]) == 1

    # 检查 PosP/PosE 行数与 UnCap*New 一致
    assert env.PosP.shape[0] == len(env.UnCapPidNew)
    assert env.PosE.shape[0] == len(env.UnCapEidNew)
    _assert_pos_mapping_consistent(env)

    # 额外核对：PosE[i] 确实来自 PathE2Val_true[eid] 在当前时刻的采样（这就是 _advance_from_paths 的定义）
    curr_idx_e = int(round(float(env.t_all) * float(env.v_E)))
    for row_i, eid in enumerate(np.asarray(env.UnCapEidNew, dtype=int).ravel()):
        e_path = np.asarray(env.PathE2Val_true[int(eid)], dtype=float)
        j = min(curr_idx_e, int(e_path.shape[0] - 1))
        expected = e_path[j, :3]
        got = np.asarray(env.PosE[row_i], dtype=float).reshape(-1)[:3]
        assert np.allclose(got, expected, atol=1e-6)

