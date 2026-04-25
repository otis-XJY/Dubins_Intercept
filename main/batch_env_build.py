import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from intercept.IsoMap.WH_main_obtainMap import (
    WH_main_obtainIso,
    WH_main_obtainMapRef,
    WH_update_time_fields_from_diag,
)


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


def _fast_save(obj: Any, save_dir: str, name: str) -> str:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.jbl")
    joblib.dump(obj, path, compress=3)
    return path


def _list_time_maps(map_root: str) -> List[str]:
    if not os.path.isdir(map_root):
        return []
    out = []
    for name in sorted(os.listdir(map_root)):
        base = os.path.join(map_root, name)
        if os.path.isdir(base) and os.path.exists(os.path.join(base, "Map.jbl")):
            out.append(name)
    return out


def _required_static_assets() -> List[str]:
    # 与 TODCMARLEnv._load_assets 的 required_map_files 对齐
    return [
        "Map.jbl",
        "IsoMapPTP2Iso_i_tt.jbl",
        "IsoMapPIso2TP_i_tt.jbl",
        "pathFinalMapPTP2Iso.jbl",
        "IsoMapTP2Val_i_tt.jbl",
        "pathFinalTP2Val.jbl",
        "IsoMapETP2Iso_i_tt.jbl",
        "IsoMapEIso2TP_i_tt.jbl",
        "pathFinalMapETP2Iso.jbl",
    ]


def _has_required_assets(time_map_dir: str) -> bool:
    for fn in _required_static_assets():
        if not os.path.exists(os.path.join(time_map_dir, fn)):
            return False
    return True


def _build_one(time_map_dir: str, *, v_p: float, v_e: float, keep_intermediates: bool) -> None:
    map_path = os.path.join(time_map_dir, "Map.jbl")
    m = joblib.load(map_path)
    if not isinstance(m, dict):
        raise TypeError(f"Map.jbl 必须是 dict，实际为 {type(m)}")

    # 速度写入 Map（后续 WH_* 会用）
    m["v_P"] = float(v_p)
    m["v_E"] = float(v_e)

    # 1) 仅更新 Map 的时间字段（numTime/timePlot/time/timeIsoRes）
    # 原先通过 WH_main_obtainMapP 生成 IsoMapP_i_tt/pathFinalP 来间接更新 Map，这些中间产物 MARL 不使用，已移除。
    m = WH_update_time_fields_from_diag(m)

    # 2) Trans_Point 反向版本（与 main/mainObtainIsoMap0901.py 一致）
    trans = np.asarray(m["Trans_Point"], dtype=float).copy()
    angles = np.arctan2(0 - trans[:, 1], float(m["obs_mapsize_x"][1]) / 2 - trans[:, 0])
    if trans.shape[1] < 3:
        trans_verse = np.column_stack((trans, angles))
    else:
        trans_verse = trans
        trans_verse[:, 2] = angles

    # 3) TP -> Val
    m, final_pathTP2Val, vthetaAllTP2Val = WH_main_obtainMapRef(m, trans_verse, m["ValuePos"], 1)
    pathFinalTP2Val, IsoMapTP2Val_i_tt, _ = WH_main_obtainIso(m, m["v_E"], final_pathTP2Val, vthetaAllTP2Val, 0)
    _fast_save(IsoMapTP2Val_i_tt, time_map_dir, "IsoMapTP2Val_i_tt")
    _fast_save(pathFinalTP2Val, time_map_dir, "pathFinalTP2Val")
    del IsoMapTP2Val_i_tt, pathFinalTP2Val, final_pathTP2Val

    # 4) PTP -> Iso & PIso -> TP
    m, final_pathPTP2Iso, vthetaAllPTP2Iso = WH_main_obtainMapRef(m, m["Trans_Point"], m["Trans_Point"], 0)
    pathFinalMapPTP2Iso, IsoMapPTP2Iso_i_tt, IsoMapPIso2TP_i_tt = WH_main_obtainIso(
        m, m["v_E"], final_pathPTP2Iso, vthetaAllPTP2Iso, 1
    )
    _fast_save(IsoMapPTP2Iso_i_tt, time_map_dir, "IsoMapPTP2Iso_i_tt")
    _fast_save(IsoMapPIso2TP_i_tt, time_map_dir, "IsoMapPIso2TP_i_tt")
    _fast_save(pathFinalMapPTP2Iso, time_map_dir, "pathFinalMapPTP2Iso")
    del IsoMapPTP2Iso_i_tt, IsoMapPIso2TP_i_tt, pathFinalMapPTP2Iso, final_pathPTP2Iso

    # 5) ETP -> Iso & EIso -> TP
    m, final_pathETP2Iso, vthetaAllETP2Iso = WH_main_obtainMapRef(m, trans_verse, trans_verse, 0)
    pathFinalMapETP2Iso, IsoMapETP2Iso_i_tt, IsoMapEIso2TP_i_tt = WH_main_obtainIso(
        m, m["v_E"], final_pathETP2Iso, vthetaAllETP2Iso, 1
    )
    _fast_save(IsoMapEIso2TP_i_tt, time_map_dir, "IsoMapEIso2TP_i_tt")
    _fast_save(IsoMapETP2Iso_i_tt, time_map_dir, "IsoMapETP2Iso_i_tt")
    _fast_save(pathFinalMapETP2Iso, time_map_dir, "pathFinalMapETP2Iso")
    del IsoMapETP2Iso_i_tt, IsoMapEIso2TP_i_tt, pathFinalMapETP2Iso, final_pathETP2Iso

    # 最后保存回 Map（含 v_P/v_E 等更新）
    _fast_save(m, time_map_dir, "Map")

    # 硬校验：必须齐全，否则视为失败（便于后续 env._load_assets 读取）
    if not _has_required_assets(time_map_dir):
        missing = [fn for fn in _required_static_assets() if not os.path.exists(os.path.join(time_map_dir, fn))]
        raise FileNotFoundError(f"静态资产缺失：{missing}")


def main():
    parser = argparse.ArgumentParser(description="批量环境构建：为每个 map/<TimeMap>/ 生成 IsoMap* 与 pathFinal*")
    parser.add_argument("--config", required=True, help="env_build.yaml 路径")
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    map_root = str(cfg.get("map_root", "./map"))
    time_maps_cfg = cfg.get("time_maps")
    if time_maps_cfg is None:
        time_maps = _list_time_maps(map_root)
    elif isinstance(time_maps_cfg, list):
        time_maps = [str(x) for x in time_maps_cfg]
    else:
        raise ValueError("env_build.time_maps 必须是列表或省略（自动扫描）")

    v_p = float(cfg.get("v_P", 20.0))
    v_e = float(cfg.get("v_E", 20.0))
    cleanup_on_fail = bool(cfg.get("cleanup_on_fail", True))
    # keep_intermediates 已不再生效：已移除 IsoMapP_i_tt/pathFinalP 中间产物生成（MARL 不使用）
    keep_intermediates = bool(cfg.get("keep_intermediates", False))
    log_path = str(cfg.get("failure_log", os.path.join(map_root, "_build_failures.jsonl")))

    os.makedirs(map_root, exist_ok=True)
    ok = []
    failed = []

    for tm in time_maps:
        tm_dir = os.path.join(map_root, tm)
        t0 = time.perf_counter()
        try:
            if not os.path.exists(os.path.join(tm_dir, "Map.jbl")):
                raise FileNotFoundError(f"缺少 {tm_dir}/Map.jbl")
            _build_one(tm_dir, v_p=v_p, v_e=v_e, keep_intermediates=keep_intermediates)
            ok.append(tm)
            dt = time.perf_counter() - t0
            print(f"[BUILD] ok TimeMap={tm} elapsed={dt:.2f}s")
        except Exception as e:
            dt = time.perf_counter() - t0
            failed.append(tm)
            rec = {
                "time": datetime.now().isoformat(timespec="seconds"),
                "TimeMap": tm,
                "elapsed_s": float(dt),
                "error": repr(e),
            }
            os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"[BUILD] FAIL TimeMap={tm} elapsed={dt:.2f}s error={repr(e)}")
            if cleanup_on_fail:
                try:
                    shutil.rmtree(tm_dir)
                    print(f"[BUILD] rollback removed {tm_dir}")
                except Exception as re:
                    print(f"[BUILD] rollback failed for {tm_dir}: {repr(re)}")

    print("[BUILD] done. ok:", ok)
    print("[BUILD] done. failed:", failed)


if __name__ == "__main__":
    main()

