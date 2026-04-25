import importlib.util
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, "marl", "envs", "todc_env.py")

# 说明：直接 `import marl...` 会触发 `marl/__init__.py`，其会 import nn/models 并依赖 torch。
# 这里用“按文件路径加载模块”的方式，只验证环境本身（不需要 torch）。
spec = importlib.util.spec_from_file_location("todc_env_module", env_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"无法加载环境模块：{env_path}")
mod = importlib.util.module_from_spec(spec)
sys.modules["todc_env_module"] = mod
spec.loader.exec_module(mod)
TODCMARLEnv = getattr(mod, "TODCMARLEnv")


def main():
    # 你可以通过环境变量指定要验证的 TimeMap
    time_map = os.environ.get("TIME_MAP", "").strip() or None
    env = TODCMARLEnv(
        {
            "render_mode": "none",
            "time_map": time_map,
            "max_episode_steps": 5,
            "debug_print": True,
            "evader_profile_mode": "cycle",
        }
    )
    obs, info = env.reset(seed=0)
    print("[SMOKE] reset info:", {k: info.get(k) for k in ("time_map", "profile", "num_candidates", "t_all")})
    # action_space.sample() 不考虑动态 mask，可能抽到非法候选；这里按 mask 选第一个合法项做冒烟。
    mask = np.asarray(obs["self_pts_mask"]).astype(bool)
    a = np.zeros((mask.shape[0],), dtype=np.int64)
    for i in range(mask.shape[0]):
        valid = np.where(mask[i])[0]
        if valid.size == 0:
            a[i] = 0
        else:
            a[i] = int(valid[0])
    obs2, rew, term, trunc, infos = env.step(a)
    rmean = float(np.mean([rew[k] for k in rew if k.startswith("p_")]))
    print("[SMOKE] step r_mean=", rmean, "done=", bool(term.get("__all__")), "trunc=", bool(trunc.get("__all__")))


if __name__ == "__main__":
    main()

