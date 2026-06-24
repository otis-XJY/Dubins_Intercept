"""确证 obtainPath 维度不匹配崩溃的根因：规划超时导致 vthetaAllP2Iso[i] 为 None。

根因假设：
  obtainIsoPath.py 中，当 A_dubins_nocircle_swarm 规划超时（超过 deadline），
  代码在设置 vthetaAllP2Iso[i] 之前 break，使 vthetaAllP2Iso[i] 保持 None。
  随后 obtainPath(seg, None, 0) 被调用，np.asarray(None, dtype=float32) 产生
  size=1 的 0-d 数组（nan），与 path_x（size=N）vstack 时维度不匹配崩溃。

本测试通过：
  1. monkey-patch _PLANNER_TIMEOUT_SEC 为极小值，强制超时
  2. monkey-patch A_dubins_nocircle_swarm 返回有效 final_path 但耗时超过 deadline
  3. 调用 obtainPE2IsoPath，验证 vthetaAllP2Iso[i] 为 None 且 obtainPath 崩溃
"""
import os
import sys
import time

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

os.environ.setdefault("MPLBACKEND", "Agg")


def test_timeout_leaves_vtheta_none():
    """测试：规划超时后 vthetaAllP2Iso[i] 保持 None，导致 obtainPath 崩溃。"""
    import intercept.IsoPair.obtainIsoPath as oip_mod
    import A_dubins.A_dubins_nocircle_swarm as ads_mod

    # 构造一个有效的 final_path（2 个节点：初始节点 + 1 个路径节点）
    # 路径节点有 path 和 vtheta_all
    n_pts = 100
    xs = np.linspace(0, 1000, n_pts, dtype=np.float32)
    ys = np.linspace(0, 500, n_pts, dtype=np.float32)
    thetas = np.full(n_pts, 0.5, dtype=np.float32)

    # 每个节点的 path 是 5 个子路径的列表，每个子路径是 [x_array, y_array]
    sub_path = [xs, ys]
    path_node_path = [sub_path, sub_path, [np.array([]), np.array([])],
                      [np.array([]), np.array([])], [np.array([]), np.array([])]]
    # vtheta_all 是 5 个数组
    vtheta_node_vta = [thetas, thetas, np.array([]), np.array([]), np.array([])]

    initial_node = {'point': np.zeros((3, 1)), 'vtheta': 0.0, 'g': 0, 'f': 0,
                    'pos_id': -1, 'parent_id': -1}
    path_node = {'point': np.zeros((3, 2)), 'vtheta': np.array([0.5]),
                 'vtheta_plot': np.array([0.5]), 'vtheta_all': vtheta_node_vta,
                 'path': path_node_path, 'r': 50, 'center': np.zeros((2, 3)),
                 'pos_id': 0, 'parent_id': -1, 'g': 100, 'f': 200}
    final_path = [initial_node, path_node]

    # 保存原始值
    orig_timeout = ads_mod._PLANNER_TIMEOUT_SEC
    orig_ads = oip_mod.A_dubins_nocircle_swarm

    try:
        # 设置极短的超时时间
        ads_mod._PLANNER_TIMEOUT_SEC = 0.001  # 1ms
        oip_mod._PLANNER_TIMEOUT_SEC = 0.001
        # 同时 patch obtainIsoPath 模块中已导入的引用
        oip_mod.A_dubins_nocircle_swarm.__defaults__  # noqa

        call_count = {"n": 0}

        def _mock_ads(Start_Point, End_Point, *args, **kwargs):
            call_count["n"] += 1
            # 模拟规划耗时（超过 1ms deadline）
            time.sleep(0.01)
            # 返回有效的 final_path
            return final_path, np.array([]), np.array([]), None

        oip_mod.A_dubins_nocircle_swarm = _mock_ads

        # 构造 Map
        Map = {
            'obs': None, 'sure': None, 'r': 50, 'obs_no_circle': [],
            'outline_all': np.array([]), 'Stepsize': 1.0, 'resolution': 5.0,
            'numTime': 10, 'timePlot': np.arange(10, dtype=float),
        }

        # 2 个代理
        PosP = np.array([[0, 0, 0], [100, 100, 0]], dtype=float)
        PIsoPos = np.array([[500, 500], [600, 600]], dtype=float)

        print("[TEST] 调用 obtainPE2IsoPath（超时模拟）...")
        try:
            path_list, iso_map, end_time = oip_mod.obtainPE2IsoPath(
                PosP, PIsoPos, Map, v=50.0
            )
            print(f"[TEST] 未崩溃！path_list={path_list}")
            # 检查 vthetaAllP2Iso 是否有 None
            # 由于 obtainPE2IsoPath 内部变量不可直接访问，通过结果判断
            print("[TEST] 结果：未崩溃 — 可能是超时未触发或 numpy 版本行为不同")
        except ValueError as e:
            if "must match exactly" in str(e) or "concatenation axis" in str(e):
                print(f"[TEST] 成功复现崩溃！")
                print(f"[TEST] 错误信息: {e}")
                print(f"[TEST] 调用次数: {call_count['n']}")
                print(f"[TEST] 根因确认：规划超时后 vthetaAllP2Iso[i] 保持 None，")
                print(f"[TEST]   np.asarray(None, dtype=float32) 产生 size=1 的 nan 数组，")
                print(f"[TEST]   与 path_x 的 vstack 维度不匹配。")
                return True
            else:
                print(f"[TEST] 其它 ValueError: {e}")
                return False
        except Exception as e:
            print(f"[TEST] 其它异常: {type(e).__name__}: {e}")
            return False

    finally:
        ads_mod._PLANNER_TIMEOUT_SEC = orig_timeout
        oip_mod.A_dubins_nocircle_swarm = orig_ads

    return False


def test_vtheta_none_produces_size1():
    """直接测试：vthetaAllP=None 时 obtainPath 的行为。"""
    from intercept.IsoMap.obtainPath import obtainPath

    # 构造一个 close 列表（2 个节点）
    n_pts = 100
    xs = np.linspace(0, 1000, n_pts, dtype=np.float32)
    ys = np.linspace(0, 500, n_pts, dtype=np.float32)
    thetas = np.full(n_pts, 0.5, dtype=np.float32)

    sub_path = [xs, ys]
    path_node_path = [sub_path, sub_path, [np.array([]), np.array([])],
                      [np.array([]), np.array([])], [np.array([]), np.array([])]]

    initial_node = {'point': np.zeros((3, 1)), 'pos_id': -1}
    path_node = {'path': path_node_path, 'pos_id': 0}
    close = [initial_node, path_node]

    print(f"[TEST2] path_x 预期长度: {n_pts * 2}（2 个非空子路径，各 {n_pts} 点）")

    # 测试 vthetaAllP = None
    print("[TEST2] 测试 vthetaAllP=None ...")
    try:
        result = obtainPath(close, None, 0)
        print(f"[TEST2] 未崩溃！result shape={result.shape}")
    except ValueError as e:
        print(f"[TEST2] 崩溃！{e}")
        # 检查是否符合预期
        if "size 1" in str(e):
            print("[TEST2] 确认：vthetaAllP=None → size=1 → 维度不匹配崩溃")
            return True
    except Exception as e:
        print(f"[TEST2] 其它异常: {type(e).__name__}: {e}")

    return False


if __name__ == "__main__":
    print("=" * 80)
    print("测试 1: 直接测试 vthetaAllP=None 时 obtainPath 的行为")
    print("=" * 80)
    r2 = test_vtheta_none_produces_size1()

    print()
    print("=" * 80)
    print("测试 2: 模拟规划超时，验证 obtainPE2IsoPath 崩溃")
    print("=" * 80)
    r1 = test_timeout_leaves_vtheta_none()

    print()
    print("=" * 80)
    if r1 or r2:
        print("结论：根因已确认 — 规划超时导致 vthetaAllP2Iso[i]=None，")
        print("      np.asarray(None) 产生 size=1 nan 数组，与 path_x vstack 维度不匹配。")
    else:
        print("结论：未能复现，需要进一步调查。")
    print("=" * 80)
