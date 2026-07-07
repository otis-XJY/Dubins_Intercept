"""分析 TODCMARLEnv.step 中关键函数的调用次数，用于定位冗余调用 / 冗余功能。

用法：
    export MPLBACKEND=Agg
    python scripts/analyze_step_redundancy.py

输出每次 step 中 _build_obs / _sync_dynamic_k / _check_collision 等函数的调用次数，
据此判断 step 流程中是否存在冗余重复调用。
"""
import importlib.util
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, "marl", "envs", "todc_env.py")

# 按文件路径加载模块，避免触发 marl/__init__.py → torch 依赖
spec = importlib.util.spec_from_file_location("todc_env_module", env_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载环境模块：{env_path}")
mod = importlib.util.module_from_spec(spec)
sys.modules["todc_env_module"] = mod
spec.loader.exec_module(mod)
TODCMARLEnv = getattr(mod, "TODCMARLEnv")


# 要统计调用次数的方法名
TRACKED = [
    "_build_obs",
    "_sync_dynamic_k",
    "_normalize_action",
    "_apply_assignment_from_action",
    "_check_collision",
    "_check_asset_breach",
    "_compute_isomap_intercept_candidates",
    "_apply_hungarian_and_paths",
    "_sync_assigned_eid_full_from_pairs",
    "_update_capflag_full_from_geometry",
    "_drop_captured_pairs",
    "_advance_from_paths",
    "_compute_rewards",
    "_compute_curr_min_dist_stable",
]


def install_counters():
    """给 TRACKED 中的方法套上调用计数器，返回 (计数字典, 原方法字典)。"""
    counts = {name: 0 for name in TRACKED}
    for name in TRACKED:
        orig = getattr(TODCMARLEnv, name)

        def make_wrapper(n, fn):
            def wrapper(self, *args, **kwargs):
                counts[n] += 1
                return fn(self, *args, **kwargs)
            return wrapper

        setattr(TODCMARLEnv, name, make_wrapper(name, orig))
    return counts


def first_valid_actions(obs, num_p):
    mask = np.asarray(obs["self_pts_mask"]).astype(bool)
    a = np.zeros((num_p,), dtype=np.int64)
    for i in range(num_p):
        valid = np.where(mask[i])[0]
        a[i] = int(valid[0]) if valid.size > 0 else 0
    return a


def main():
    env = TODCMARLEnv(
        {
            "render_mode": "none",
            "max_episode_steps": 30,
            "debug_print": False,
        }
    )
    counts = install_counters()

    obs, info = env.reset(seed=0)
    print(f"[RESET] num_P={env.num_P} num_E={env.num_E} "
          f"k={obs['self_pts'].shape[1]} t_all={info['t_all']:.2f}")

    done = False
    trunc = False
    step_i = 0
    while not done and not trunc and step_i < 8:
        action = first_valid_actions(obs, env.num_P)
        # 只统计本次 step 的调用
        for k in counts:
            counts[k] = 0
        obs, rewards, terminations, truncations, infos = env.step(action)
        step_i += 1
        done = bool(terminations["__all__"])
        trunc = bool(truncations["__all__"])
        p0 = infos.get("p_0", {})
        print(f"\n[STEP {step_i}] replanned={p0.get('replanned')} "
              f"done={done} trunc={trunc} "
              f"K={p0.get('num_candidates')} t_all={infos.get('global_t_all', 0):.2f}")
        print("  调用次数:")
        for name in TRACKED:
            c = counts[name]
            if c > 0:
                print(f"    {name}: {c}")

    print("\n=== 优化后预期（与上面统计对照）===")
    print("R1. _build_obs: 优化后 2 次 = _normalize_action(1) + step 内 curr_obs(1)")
    print("    _compute_rewards 复用 curr_obs，step 末尾复用 curr_obs，各省 1 次。")
    print("R2. _sync_dynamic_k: 优化后 2 次 = _normalize_action(1) + step末尾(1)")
    print("    _apply_assignment_from_action 在 obs 非 None 时跳过冗余同步。")
    print("R3. _check_collision: 优化后非 terminal 步仍 3 次（内层）；terminal 步省 1（_phase_check_decision 返回 collision）。")
    print("R4. _check_asset_breach: 仍 4 次（保留 step 600 行独立检测以服务于奖励正确性，未优化）。")
    print("R5. _normalize_action: step 传 return_norm=False，跳过 one-hot 矩阵分配（调用次数不变，但省分配）。")
    print("R6. need_replan / mismatch 分支统一走 _do_replan()，代码去重（调用次数不变）。")


if __name__ == "__main__":
    main()
