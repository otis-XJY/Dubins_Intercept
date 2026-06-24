"""验证三个修复方案的测试。

方案一: obtainPath 超时不再崩溃（超时 break 包含当前代理 i）
方案二: _update_capflag_full_from_geometry 返回 (newly, mismatch_count)，
        step 检测到非匹配捕获后立即重规划
方案三: apply_terminal_rewards 区分匹配/非匹配捕获，非匹配给半额奖励
"""
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault("MPLBACKEND", "Agg")


# ===== 方案一测试 =====

def test_obtainpath_timeout_no_crash():
    """方案一：超时后 obtainPE2IsoPath 不再崩溃，返回含 None 的路径列表。"""
    import time
    import intercept.IsoPair.obtainIsoPath as oip_mod
    import A_dubins.A_dubins_nocircle_swarm as ads_mod

    orig_timeout = ads_mod._PLANNER_TIMEOUT_SEC
    orig_ads = oip_mod.A_dubins_nocircle_swarm

    try:
        # 设置极短超时（1ms），mock 函数 sleep 10ms 确保超时
        ads_mod._PLANNER_TIMEOUT_SEC = 0.001
        oip_mod._PLANNER_TIMEOUT_SEC = 0.001

        def _mock_ads(*args, **kwargs):
            time.sleep(0.01)  # 超过 1ms deadline
            # 返回非 None 的 path_segments，使代码进入超时检查而非直线兜底
            return "dummy_path", np.array([]), np.array([]), None

        oip_mod.A_dubins_nocircle_swarm = _mock_ads

        Map = {
            'obs': None, 'sure': None, 'r': 50, 'obs_no_circle': [],
            'outline_all': np.array([]), 'Stepsize': 1.0, 'resolution': 5.0,
            'numTime': 10, 'timePlot': np.arange(10, dtype=float),
        }
        PosP = np.array([[0, 0, 0], [100, 100, 0]], dtype=float)
        PIsoPos = np.array([[500, 500], [600, 600]], dtype=float)

        path_list, iso_map, end_time = oip_mod.obtainPE2IsoPath(PosP, PIsoPos, Map, v=50.0)

        # 修复后：不再崩溃，超时代理使用直线兜底（非 None）
        assert path_list is not None, "path_list 不应为 None"
        assert len(path_list) == 2, f"path_list 长度应为 2，实际 {len(path_list)}"
        # 超时代理的路径应为直线兜底（3×N 矩阵），而非 None
        assert path_list[0] is not None, f"超时代理 0 应有直线兜底路径，实际 None"
        assert path_list[0].shape[0] == 3, f"直线兜底路径应为 3 行，实际 {path_list[0].shape}"
        assert path_list[1] is not None, f"超时代理 1 应有直线兜底路径，实际 None"
        assert path_list[1].shape[0] == 3, f"直线兜底路径应为 3 行，实际 {path_list[1].shape}"
        print(f"[方案一] PASS: 超时后返回直线兜底路径, shape={path_list[0].shape}")
    finally:
        ads_mod._PLANNER_TIMEOUT_SEC = orig_timeout
        oip_mod.A_dubins_nocircle_swarm = orig_ads


# ===== 方案二测试 =====

def test_mismatch_capture_returns_count():
    """方案二：_update_capflag_full_from_geometry 返回 (newly, mismatch_count)。"""
    from marl.envs import TODCMARLEnv

    env = TODCMARLEnv({"render_mode": "none", "max_episode_steps": 20})
    env.reset(seed=42)
    assert env.num_P >= 3 and env.num_E >= 3

    env.pairs_realE2P = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env._pairs_ic_compact = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env.UnCapPid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapEid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapPidNew = np.asarray([0, 1, 2], dtype=int)
    env.UnCapEidNew = np.asarray([0, 1, 2], dtype=int)
    env._sync_assigned_eid_full_from_pairs()
    env.Capflag_full[:] = False

    cap_dist = float(env.CapRef["CapDist"])
    # pid=1 靠近 eid=0（非匹配），pid=0 远离
    env.PosP = np.array([
        [2000, 1000, np.pi],   # pid=0 assigned to eid=0, 但远
        [1100, 1000, np.pi],   # pid=1 assigned to eid=1, 但靠近 eid=0
        [3000, 3000, 0],
    ])
    env.PosE = np.array([
        [1000, 1000, 0],       # eid=0
        [5000, 5000, 0],       # eid=1
        [6000, 6000, 0],       # eid=2
    ])

    newly, mismatch = env._update_capflag_full_from_geometry()

    assert newly == 1, f"应捕获 1 个，实际 {newly}"
    assert mismatch == 1, f"应非匹配捕获 1 个，实际 {mismatch}"
    assert env.Capflag_full[0] == True
    assert hasattr(env, '_last_mismatch_capture_eids')
    assert 0 in env._last_mismatch_capture_eids
    print("[方案二] PASS: _update_capflag_full_from_geometry 正确返回 (newly=1, mismatch=1)")


def test_matched_capture_returns_zero_mismatch():
    """方案二：匹配捕获时 mismatch_count=0。"""
    from marl.envs import TODCMARLEnv

    env = TODCMARLEnv({"render_mode": "none", "max_episode_steps": 20})
    env.reset(seed=42)
    assert env.num_P >= 3 and env.num_E >= 3

    env.pairs_realE2P = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env._pairs_ic_compact = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)
    env.UnCapPid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapEid = np.asarray([0, 1, 2], dtype=int)
    env.UnCapPidNew = np.asarray([0, 1, 2], dtype=int)
    env.UnCapEidNew = np.asarray([0, 1, 2], dtype=int)
    env._sync_assigned_eid_full_from_pairs()
    env.Capflag_full[:] = False

    # pid=0 靠近 eid=0（匹配）
    env.PosP = np.array([
        [1100, 1000, np.pi],   # pid=0 assigned to eid=0, 靠近
        [3000, 3000, 0],
        [4000, 4000, 0],
    ])
    env.PosE = np.array([
        [1000, 1000, 0],       # eid=0
        [5000, 5000, 0],       # eid=1
        [6000, 6000, 0],       # eid=2
    ])

    newly, mismatch = env._update_capflag_full_from_geometry()

    assert newly == 1, f"应捕获 1 个，实际 {newly}"
    assert mismatch == 0, f"匹配捕获应 mismatch=0，实际 {mismatch}"
    print("[方案二] PASS: 匹配捕获时 mismatch_count=0")


# ===== 方案三测试 =====

def test_reward_matched_full_mismatch_half():
    """方案三：匹配捕获给全额奖励，非匹配捕获给半额奖励。"""
    from marl.rewards.todc_reward import TODCRewardFunction

    reward_fn = TODCRewardFunction()
    bonus = reward_fn.config.terminal_capture_bonus  # 默认 100.0
    num_e = 3

    # 场景 1: 1 个匹配捕获
    rewards_1m = {f"p_{i}": 0.0 for i in range(3)}
    reward_fn.apply_terminal_rewards(
        rewards_1m, captured_delta=1, num_e=num_e,
        asset_breached=False, matched_captured=1,
    )
    expected_1m = bonus * 1.0 / num_e  # 全额
    for k in rewards_1m:
        assert abs(rewards_1m[k] - expected_1m) < 1e-6, \
            f"匹配捕获应给全额 {expected_1m}，实际 {rewards_1m[k]}"

    # 场景 2: 1 个非匹配捕获
    rewards_1mis = {f"p_{i}": 0.0 for i in range(3)}
    reward_fn.apply_terminal_rewards(
        rewards_1mis, captured_delta=1, num_e=num_e,
        asset_breached=False, matched_captured=0,
    )
    expected_1mis = bonus * 0.5 / num_e  # 半额
    for k in rewards_1mis:
        assert abs(rewards_1mis[k] - expected_1mis) < 1e-6, \
            f"非匹配捕获应给半额 {expected_1mis}，实际 {rewards_1mis[k]}"

    # 场景 3: 1 匹配 + 1 非匹配 = 2 个总捕获
    rewards_mix = {f"p_{i}": 0.0 for i in range(3)}
    reward_fn.apply_terminal_rewards(
        rewards_mix, captured_delta=2, num_e=num_e,
        asset_breached=False, matched_captured=1,
    )
    expected_mix = bonus * (1.0 + 0.5) / num_e  # 全额 + 半额
    for k in rewards_mix:
        assert abs(rewards_mix[k] - expected_mix) < 1e-6, \
            f"混合捕获应给 {expected_mix}，实际 {rewards_mix[k]}"

    # 验证半额 < 全额
    assert expected_1mis < expected_1m, "半额应小于全额"

    print(f"[方案三] PASS: 匹配={expected_1m:.2f}, 非匹配={expected_1mis:.2f}, "
          f"混合={expected_mix:.2f} (bonus={bonus}, num_e={num_e})")


if __name__ == "__main__":
    print("=" * 60)
    print("方案一: obtainPath 超时不再崩溃")
    print("=" * 60)
    test_obtainpath_timeout_no_crash()

    print("\n" + "=" * 60)
    print("方案二: 非匹配捕获返回 mismatch_count")
    print("=" * 60)
    test_mismatch_capture_returns_count()
    test_matched_capture_returns_zero_mismatch()

    print("\n" + "=" * 60)
    print("方案三: 奖励区分匹配/非匹配捕获")
    print("=" * 60)
    test_reward_matched_full_mismatch_half()

    print("\n" + "=" * 60)
    print("全部测试通过!")
    print("=" * 60)
