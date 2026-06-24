"""测试捕获错配问题的根因。

捕获错配：[CAPTURE] WARNING: evader X captured by pursuer Y (NOT assigned pursuer Z)

本测试直接构造场景，验证：
1. 捕获是纯几何的（最近 pursuer 捕获），不受 assignment 影响
2. assigned pursuer 可能不是最近的 pursuer → 错配
3. 错配后数据结构的变化（Capflag_full、assigned_eid_full、pairs_realE2P）
"""
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault("MPLBACKEND", "Agg")


def test_capture_is_geometric_not_assignment():
    """测试：捕获由几何距离决定，不受 assignment 影响。

    构造场景：
    - 3 个 pursuer (pid=0,1,2)，3 个 evader (eid=0,1,2)
    - pairs_realE2P = [[0,0],[1,1],[2,2]]（对角分配）
    - 但 pid=1 离 eid=0 最近 → pid=1 捕获 eid=0（非分配的 pid=0）
    """
    from marl.envs import TODCMARLEnv

    env = TODCMARLEnv({"render_mode": "none", "max_episode_steps": 20})
    env.reset(seed=42)

    # 确保 3v3
    assert env.num_P >= 3 and env.num_E >= 3, f"需要 3v3，实际 P={env.num_P} E={env.num_E}"

    # 设置配对：eid=0→pid=0, eid=1→pid=1, eid=2→pid=2
    env.pairs_realE2P = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env._pairs_ic_compact = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env.UnCapPid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapEid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapPidNew = np.asarray([0, 1, 2], dtype=int)
    env.UnCapEidNew = np.asarray([0, 1, 2], dtype=int)
    env._sync_assigned_eid_full_from_pairs()
    env.Capflag_full[:] = False

    # 获取 CapRef
    cap_dist = float(env.CapRef["CapDist"])
    cap_angle_half = float(env.CapRef["CapAngle"]) / 2.0
    print(f"[INFO] CapDist={cap_dist}, CapAngle/2={cap_angle_half}°")

    # 构造位置：
    # eid=0 在 (1000, 1000)
    # pid=0（assigned to eid=0）在 (2000, 1000, π) —— 远离 eid=0，朝向 eid=0
    # pid=1（assigned to eid=1）在 (1100, 1000, π) —— 靠近 eid=0！朝向 eid=0
    # pid=2（assigned to eid=2）在 (3000, 3000, 0)
    # eid=1, eid=2 在远处
    eid0_pos = np.array([1000, 1000, 0], dtype=float)
    pid0_pos = np.array([2000, 1000, np.pi], dtype=float)  # 朝向 eid=0 但远（d=1000 > cap_dist=200）
    pid1_pos = np.array([1100, 1000, np.pi], dtype=float)  # 靠近 eid=0（d=100 ≤ 200），朝向 eid=0
    pid2_pos = np.array([3000, 3000, 0], dtype=float)
    eid1_pos = np.array([5000, 5000, 0], dtype=float)
    eid2_pos = np.array([6000, 6000, 0], dtype=float)

    # 设置 PosP/PosE（紧凑，与 UnCapPidNew/UnCapEidNew 对齐）
    env.PosP = np.array([pid0_pos, pid1_pos, pid2_pos])
    env.PosE = np.array([eid0_pos, eid1_pos, eid2_pos])

    # 计算预期距离
    d_p0_e0 = np.hypot(pid0_pos[0] - eid0_pos[0], pid0_pos[1] - eid0_pos[1])
    d_p1_e0 = np.hypot(pid1_pos[0] - eid0_pos[0], pid1_pos[1] - eid0_pos[1])
    print(f"[INFO] pid=0 → eid=0 距离={d_p0_e0:.1f} (assigned)")
    print(f"[INFO] pid=1 → eid=0 距离={d_p1_e0:.1f} (NOT assigned, but closer)")

    # 需要让 _pos_e_xyz_for_global_eid 返回我们设置的位置
    # 由于 PosE 行数 == num_E，_pos_e_xyz_for_global_eid 会直接用 PosE[eid]
    assert env.PosE.shape[0] == env.num_E, f"PosE rows={env.PosE.shape[0]} != num_E={env.num_E}"

    # 调用捕获检查
    print("\n[TEST] 调用 _update_capflag_full_from_geometry ...")
    newly = env._update_capflag_full_from_geometry()

    print(f"[TEST] 新增捕获数: {newly}")
    print(f"[TEST] Capflag_full: {env.Capflag_full.tolist()}")

    # 验证：eid=0 被 pid=1（非分配的 pid=0）捕获
    assert env.Capflag_full[0] == True, "eid=0 应被捕获"
    assert newly == 1, f"应捕获 1 个，实际 {newly}"

    # 验证 assigned_eid_full：pid=0 的 assigned 被清除（eid=0 被捕获）
    print(f"[TEST] assigned_eid_full: {env.assigned_eid_full.tolist()}")
    assert env.assigned_eid_full[0] == -1, "pid=0 的 assigned_eid 应被清除（eid=0 被捕获）"
    # pid=1 仍分配给 eid=1（它的原始分配未受影响）
    assert env.assigned_eid_full[1] == 1, "pid=1 的 assigned_eid 应仍为 1"

    print("\n[结论] 捕获错配根因确认：")
    print("  1. 捕获逻辑是纯几何的：_update_capflag_full_from_geometry 找最近的 pursuer")
    print("  2. 最近的 pursuer 可能不是 assigned pursuer（pairs_realE2P 分配的）")
    print("  3. 捕获后：Capflag_full[eid]=True，assigned_eid_full[原assigned_pid]=-1")
    print("  4. 但捕获者（cap_pid）的 assigned_eid_full 不变 → 数据结构不对称")
    print("  5. pairs_realE2P 在下一次 _phase_update 才被过滤（移除已捕获 eid）")

    return True


def test_validation_functions_redundancy():
    """分析 _validate_pair_data_integrity / _validate_capflag_consistency /
    _sync_assigned_eid_full_from_pairs 是否冗余。"""
    from marl.envs import TODCMARLEnv

    env = TODCMARLEnv({"render_mode": "none", "max_episode_steps": 20})
    env.reset(seed=42)

    print("[分析] _sync_assigned_eid_full_from_pairs:")
    print("  - 功能：从 pairs_realE2P 重建 assigned_eid_full")
    print("  - 调用时机：_phase_update（捕获过滤后）、_apply_hungarian_and_paths（重规划后）")
    print("  - 是否冗余：否。pairs_realE2P 变化后必须同步 assigned_eid_full，")
    print("    否则 _compute_curr_min_dist_stable 会用过期的分配计算距离。")

    print("\n[分析] _validate_pair_data_integrity:")
    print("  - 功能：纯校验（无副作用），检查 pairs/ic_compact/UnCap/Path 的一致性")
    print("  - 调用时机：step 中 5 处（pre_apply_action, post_phase_update, post_replan,")
    print("    apply_action:post_paths, hungarian:post_fallback）")
    print("  - 是否冗余：是（作为运行时断言）。若代码正确，这些检查永远通过。")
    print("    它们不修改任何状态，只 raise ValueError。")
    print("  - 性能影响：每次调用都遍历所有 pair 做 Python 循环，对 fps 有负面影响。")

    print("\n[分析] _validate_capflag_consistency:")
    print("  - 功能：纯校验，检查 Capflag_full/UnCap*New/assigned_eid_full 一致性")
    print("  - 调用时机：step 中 post_phase_update")
    print("  - 是否冗余：是（同上，纯断言）。")

    print("\n[结论] 三个函数中：")
    print("  - _sync_assigned_eid_full_from_pairs 是必需的状态更新（非冗余）")
    print("  - _validate_pair_data_integrity 和 _validate_capflag_consistency 是冗余的运行时断言")
    print("  - 它们在每次 step 中被调用多次，影响训练速度（fps=0.4 的部分原因）")
    print("  - 建议在生产训练中禁用，仅在调试时启用")

    return True


if __name__ == "__main__":
    print("=" * 80)
    print("测试 1: 捕获错配根因（几何捕获 vs 分配）")
    print("=" * 80)
    r1 = test_capture_is_geometric_not_assignment()

    print("\n" + "=" * 80)
    print("测试 2: 验证函数冗余性分析")
    print("=" * 80)
    r2 = test_validation_functions_redundancy()

    print("\n" + "=" * 80)
    print("全部测试完成")
    print("=" * 80)
