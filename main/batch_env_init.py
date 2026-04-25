import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np

# 无显示服务器时用 Agg，确保 plt.savefig 可用
import matplotlib

if os.environ.get("DISPLAY", "") == "" and os.name != "nt":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from intercept.Map.obtainMap import obtainMap


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "缺少 PyYAML 依赖。请先安装：pip install PyYAML（或按 requirements.txt 安装）。"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML 顶层必须是 dict，实际为 {type(data)}")
    return data


def _time_tag(prefix: str = "", *, with_seconds: bool = True) -> str:
    fmt = "%m%d_%H%M%S" if with_seconds else "%m%d_%H%M"
    base = datetime.now().strftime(fmt)
    return f"{prefix}{base}" if prefix else base


def _ensure_unique_dir(base_dir: str) -> str:
    if not os.path.exists(base_dir):
        return base_dir
    # 秒级时间戳在批量时仍可能碰撞，这里加后缀避免覆盖
    for k in range(1, 10000):
        cand = f"{base_dir}_{k}"
        if not os.path.exists(cand):
            return cand
    raise RuntimeError(f"无法为目录分配唯一名称：{base_dir}")


def _fast_save(obj: Any, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.jbl")
    joblib.dump(obj, path, compress=3)
    return path


def _build_value_pos(map_params: Dict[str, Any]) -> np.ndarray:
    if "ValuePos" in map_params:
        arr = np.asarray(map_params["ValuePos"], dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("map_params.ValuePos 需为 Nx3（x,y,theta）")
        return arr[:, :3]

    sure = float(map_params["sure"])
    mapsize_x = float(map_params["mapsize_x"])
    num_value = int(map_params.get("numValue", 3))
    xs = np.linspace(10 * sure, mapsize_x - 10 * sure, num_value)
    return np.column_stack([xs, 2 * sure * np.ones(num_value), -np.ones(num_value) * np.pi / 2])


def _build_pstart(map_params: Dict[str, Any]) -> np.ndarray:
    if "PStart_Point" in map_params:
        arr = np.asarray(map_params["PStart_Point"], dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("map_params.PStart_Point 需为 Nx3（x,y,theta）")
        return arr[:, :3]

    sure = float(map_params["sure"])
    mapsize_x = float(map_params["mapsize_x"])
    num_uav = int(map_params.get("numUav", 3))
    xs = np.linspace(5 * sure, mapsize_x - 5 * sure, num_uav)
    return np.column_stack([xs, 2.5 * sure * np.ones(num_uav), np.ones(num_uav) * np.pi / 2])


def _build_trans_point(map_params: Dict[str, Any]) -> np.ndarray:
    if "Trans_Point" in map_params:
        arr = np.asarray(map_params["Trans_Point"], dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 3:
            raise ValueError("map_params.Trans_Point 需为 Nx3（x,y,theta）")
        return arr[:, :3]

    mapsize_x = float(map_params["mapsize_x"])
    mapsize_y = float(map_params["mapsize_y"])
    res = map_params.get("resolution_map_pos", [5, 5])
    if not (isinstance(res, (list, tuple)) and len(res) == 2):
        raise ValueError("map_params.resolution_map_pos 需为 [nx, ny]")
    nx, ny = int(res[0]), int(res[1])
    tx = np.linspace(mapsize_x / nx / 2, mapsize_x - mapsize_x / nx / 2, nx)
    ty = np.linspace(mapsize_y / ny / 2, mapsize_y - mapsize_y / ny / 2, ny)
    X, Y = np.meshgrid(tx, ty)
    angles = np.arctan2(mapsize_y - Y.ravel(), mapsize_x / 2 - X.ravel())
    return np.column_stack([X.ravel(), Y.ravel(), angles])


def _build_map_dict(map_params: Dict[str, Any]) -> Dict[str, Any]:
    required = [
        "numTime",
        "r",
        "Stepsize",
        "sure",
        "obs_mapsize_x",
        "obs_mapsize_y",
        "mapsize_x",
        "mapsize_y",
        "R",
        "num_obs_nocircle",
        "num_steps",
        "resolution_map_pos",
    ]
    for k in required:
        if k not in map_params:
            raise ValueError(f"env_init.map_params 缺少必填字段：{k}")

    m: Dict[str, Any] = {}
    for k in required:
        m[k] = map_params[k]

    m["ValuePos"] = _build_value_pos(map_params)
    m["PStart_Point"] = _build_pstart(map_params)
    m["Trans_Point"] = _build_trans_point(map_params)
    return m


def main():
    parser = argparse.ArgumentParser(description="批量环境初始化：生成 map/<TimeMap>/Map.jbl")
    parser.add_argument("--config", required=True, help="env_init.yaml 路径")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    n_maps = int(cfg.get("n_maps", 1))
    out_root = str(cfg.get("out_root", "./map"))
    prefix = str(cfg.get("time_prefix", "")).strip()
    with_seconds = bool(cfg.get("time_with_seconds", True))
    save_png = bool(cfg.get("save_png", True))
    png_dpi = int(cfg.get("png_dpi", 300))

    map_params = cfg.get("map_params")
    if not isinstance(map_params, dict):
        raise ValueError("env_init.yaml 必须包含 map_params: {...}")

    map_base = _build_map_dict(map_params)
    obtain_cfg = cfg.get("obtain_map", {})
    if not isinstance(obtain_cfg, dict):
        obtain_cfg = {}
    n_topo = int(obtain_cfg.get("n_topo", 15))
    n_blank = int(obtain_cfg.get("n_blank", 15))
    safety = float(obtain_cfg.get("safety", 20.0))

    created = []
    for _ in range(n_maps):
        tag = _time_tag(prefix, with_seconds=with_seconds)
        time_map = tag
        save_dir = _ensure_unique_dir(os.path.join(out_root, time_map))
        time_map = os.path.basename(save_dir)

        m = dict(map_base)
        m = obtainMap(m, n_topo=n_topo, n_blank=n_blank, safety=safety)
        _fast_save(m, save_dir, "Map")
        if save_png:
            # obtainMap 内部通常会绘制当前地图；这里直接保存当前 figure
            png_path = os.path.join(save_dir, "Map.png")
            plt.savefig(png_path, dpi=png_dpi)
            plt.close("all")
        created.append(time_map)
        print(f"[INIT] created TimeMap={time_map} -> {os.path.join(save_dir, 'Map.jbl')}")

    print("[INIT] done. created:", created)


if __name__ == "__main__":
    main()

