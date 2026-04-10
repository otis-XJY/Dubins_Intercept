"""
MARL 变量语义探测（Capflag / UnCapPid / UnCapPidNew / pairs_realE2P / captured）

结论摘要（由代码路径 + 一次 reset/step 快照验证；长回合捕获需另行仿真）：

1. **不存在** ``env.captured`` 或 ``self.captured`` 数组。
   - 步内新增捕获数在 ``StepStats.captured`` / ``stats.captured``；
   - ``info[p_i][\"captured_total\"]`` = ``int(np.sum(env.Capflag))``（全为已捕获敌机数之和的计数语义）。

2. **Capflag**
   - ``reset`` 时初始化为长度 **num_E** 的全 False；
   - 每次 ``_update_capflag_from_geometry`` **整表重写**为长度 **len(PosE.shape[0])** 的布尔向量；
   - ``PosE`` 在 ``_advance_from_paths`` 中按 ``UnCapEidNew`` 逐行采样，故 **len(Capflag) == len(UnCapEidNew)**（紧凑敌机列表长度），
     在全员存活时等于 **num_E**；若有未捕获子集收缩，则会 **变短**（与当前 ``PosE`` 行数一致）。
   - 语义：**当前几何步里，每个「活跃敌机槽位」是否满足距离+扇区捕获条件**（对最近追捕者），不是单独的持久 ``self.captured`` 存档。

3. **UnCapPid / UnCapEid**
   - 初始为 ``np.arange(num_P)`` / ``np.arange(num_E)``：**全局机号 / 敌号** 的「注册表」；
   - 重规划入口 ``_compute_isomap_intercept_candidates`` 内会执行 ``self.UnCapEid = self.UnCapEidNew``、``self.UnCapPid = self.UnCapPidNew``，
     用当前 **New** 列表覆盖，顺序可与 ``0..n-1`` 不同（例如排列 ``[1,2,0]``），表示 **活跃集合 + 列举顺序**，不一定是紧凑 0..k-1。

4. **UnCapPidNew / UnCapEidNew**
   - 初值与 UnCapPid/UnCapEid 相同；
   - 当 ``_phase_update`` 中门控成立时：
     ``UnCapPidNew = pairs_realE2P[~Capflag, 1]``，``UnCapEidNew = pairs_realE2P[~Capflag, 0]``，
     即取 **仍未被 Capflag 标为捕获** 的分配行上的 **全局 pid / eid**；
   - **不是**从 0 起的紧凑下标，除非碰巧全局 id 为 0..k-1。
   - ``_advance_from_paths`` 用 ``UnCapPidNew`` 列举更新 ``PosP``：**PosP 行数 == len(UnCapPidNew)**（每行对应一个当前参与推进的全局 pursuer）。

5. **pairs_realE2P**
   - 每行 ``[eid, pid]``（经 Hungarian 与 UnCapEid/UnCapPid 映射后的全局 id）；
   - 与 ``IC`` 中 ``(eid_ref, pid_ref)`` 对齐；**不必对角**，例如可出现 ``[[1,0],[2,1],[0,2]]``；
   - ``_phase_update`` 要求 ``pairs_realE2P.shape[0] == Capflag.shape[0]`` 才能用 ``~Capflag`` 行过滤，
     故二者须 **同长**；该长度在全员存活时常为 **num_E**。

6. **_build_c_nodes**（obs_generator）用 ``subsets[pid]`` 与 ``pairs`` 行顺序耦合：仅当 ``pairs`` 按 **pid 列升序** 且行数等于 num_P 时与「按全局 pid 填行」一致；否则存在错行风险（见计划文档）。
"""

import numpy as np
import pytest

from marl.MARL_env import TODCMARLEnv


@pytest.fixture
def env():
    e = TODCMARLEnv({"render_mode": "none", "max_episode_steps": 50})
    yield e


def test_no_self_captured_array_on_env(env):
    # 当前实现：环境实例上无 self.captured 向量；捕获状态由 Capflag + info 派生
    assert not hasattr(env, "captured")


def test_after_reset_capflag_un_cap_and_pairs_shapes(env):
    obs, _ = env.reset()
    assert env.Capflag.dtype == bool or env.Capflag.dtype == np.bool_
    assert env.Capflag.shape[0] == env.num_E
    assert len(env.UnCapPid) == env.num_P and len(env.UnCapEid) == env.num_E
    assert np.array_equal(env.UnCapPid, np.arange(env.num_P))
    assert np.array_equal(env.UnCapEid, np.arange(env.num_E))
    assert len(env.UnCapPidNew) == env.num_P and len(env.UnCapEidNew) == env.num_E

    pr = env.pairs_realE2P
    assert pr is not None
    assert pr.shape[1] == 2
    assert pr.shape[0] == env.num_E
    # 第二列为全局 pursuer 下标
    pids = pr[:, 1].astype(int)
    assert np.all((pids >= 0) & (pids < env.num_P))
    assert len(np.unique(pids)) == env.num_P
    eids = pr[:, 0].astype(int)
    assert np.all((eids >= 0) & (eids < env.num_E))
    assert len(np.unique(eids)) == env.num_E

    assert env.PosP.shape[0] == len(env.UnCapPidNew)
    assert env.PosE.shape[0] == len(env.UnCapEidNew)
    assert env.PosP.shape[0] == env.num_P and env.PosE.shape[0] == env.num_E


def test_capflag_length_equals_pos_e_rows_after_geometry(env):
    env.reset()
    assert env.Capflag.shape[0] == env.PosE.shape[0]


def test_one_step_keeps_dict_terminations_false(env):
    env.reset()
    act = np.zeros(env.num_P, dtype=np.int64)
    _, _, term, trunc, _ = env.step(act)
    assert term["__all__"] is False
    assert trunc["__all__"] is False


def test_captured_total_info_equals_sum_capflag(env):
    env.reset()
    act = np.zeros(env.num_P, dtype=np.int64)
    _, _, _, _, info = env.step(act)
    total = info["p_0"]["captured_total"]
    assert total == int(np.sum(env.Capflag))


def test_phase_update_gate_requires_matching_lengths(env):
    """门控：pairs 行数与 Capflag 长度一致时才能布尔过滤 pairs。"""
    env.reset()
    if env.pairs_realE2P is not None:
        assert env.pairs_realE2P.shape[0] == env.Capflag.shape[0]


def test_uncap_pid_new_values_are_global_indices(env):
    env.reset()
    for pid in env.UnCapPidNew:
        assert 0 <= int(pid) < env.num_P


def test_pairs_second_column_unique_global_pids_when_square(env):
    """当 num_E == num_P 时，匈牙利常给出每机恰好一行；第二列为全局 pid 的一个排列。"""
    env.reset()
    if env.num_E != env.num_P:
        pytest.skip("需要 num_E == num_P 方验证全排列")
    pr = env.pairs_realE2P
    pids_sorted = np.sort(pr[:, 1].astype(int))
    assert np.array_equal(pids_sorted, np.arange(env.num_P))
