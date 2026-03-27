#!/usr/bin/env python3
"""
在无视频/无 matplotlib 的情况下跑若干步 DubinsInterceptMARLEnv，用于自检。

请在项目根目录执行:
  conda activate dubins
  cd /path/to/Dubins_Intercept
  python marl0327/run_env_demo.py

依赖: map/<time_map> 与 map/<time_iso> 下存在与 main0319forRL 相同的 .jbl 文件。
"""

from __future__ import annotations

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from marl0327.env import DubinsInterceptEnvConfig, DubinsInterceptMARLEnv


def main() -> None:
    cfg = DubinsInterceptEnvConfig()
    try:
        env = DubinsInterceptMARLEnv(cfg)
    except FileNotFoundError as e:
        print("无法加载地图文件，请将 .jbl 放到项目 map/ 目录下（与 main0319forRL 一致）。")
        print(e)
        sys.exit(1)

    _, info0 = env.reset(seed=0)
    print("reset OK, t_all=", info0.get("t_all"))

    max_steps = 5000
    for it in range(max_steps):
        obs, reward, term, trunc, info = env.step(None)
        if info.get("replanned"):
            print(f"step {it}: replanned decision_step={info.get('decision_step')} t_all={info['t_all']:.3f}")
        if term or trunc:
            print(
                "done:",
                "terminated" if term else "truncated",
                "t_all=",
                info["t_all"],
                "reward=",
                reward,
            )
            break
    else:
        print("reached max_steps without terminal")


if __name__ == "__main__":
    main()
