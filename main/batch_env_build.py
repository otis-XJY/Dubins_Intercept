import argparse
import json
import logging
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import get_context
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


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


def _build_one(time_map_dir: str, *, v_p: float, v_e: float) -> None:
    from intercept.IsoMap.WH_main_obtainMap import (
        WH_main_obtainIso,
        WH_main_obtainMapRef,
        WH_update_time_fields_from_diag,
    )

    map_path = os.path.join(time_map_dir, "Map.jbl")
    m = joblib.load(map_path)
    if not isinstance(m, dict):
        raise TypeError(f"Map.jbl 必须是 dict，实际为 {type(m)}")

    m["v_P"] = float(v_p)
    m["v_E"] = float(v_e)

    m = WH_update_time_fields_from_diag(m)

    trans = np.asarray(m["Trans_Point"], dtype=float).copy()
    angles = np.arctan2(0 - trans[:, 1], float(m["obs_mapsize_x"][1]) / 2 - trans[:, 0])
    if trans.shape[1] < 3:
        trans_verse = np.column_stack((trans, angles))
    else:
        trans_verse = trans
        trans_verse[:, 2] = angles

    m, final_pathTP2Val, vthetaAllTP2Val = WH_main_obtainMapRef(m, trans_verse, m["ValuePos"], 1)
    pathFinalTP2Val, IsoMapTP2Val_i_tt, _ = WH_main_obtainIso(m, m["v_E"], final_pathTP2Val, vthetaAllTP2Val, 0)
    _fast_save(IsoMapTP2Val_i_tt, time_map_dir, "IsoMapTP2Val_i_tt")
    _fast_save(pathFinalTP2Val, time_map_dir, "pathFinalTP2Val")
    del IsoMapTP2Val_i_tt, pathFinalTP2Val, final_pathTP2Val

    m, final_pathPTP2Iso, vthetaAllPTP2Iso = WH_main_obtainMapRef(m, m["Trans_Point"], m["Trans_Point"], 0)
    pathFinalMapPTP2Iso, IsoMapPTP2Iso_i_tt, IsoMapPIso2TP_i_tt = WH_main_obtainIso(
        m, m["v_E"], final_pathPTP2Iso, vthetaAllPTP2Iso, 1
    )
    _fast_save(IsoMapPTP2Iso_i_tt, time_map_dir, "IsoMapPTP2Iso_i_tt")
    _fast_save(IsoMapPIso2TP_i_tt, time_map_dir, "IsoMapPIso2TP_i_tt")
    _fast_save(pathFinalMapPTP2Iso, time_map_dir, "pathFinalMapPTP2Iso")
    del IsoMapPTP2Iso_i_tt, IsoMapPIso2TP_i_tt, pathFinalMapPTP2Iso, final_pathPTP2Iso

    m, final_pathETP2Iso, vthetaAllETP2Iso = WH_main_obtainMapRef(m, trans_verse, trans_verse, 0)
    pathFinalMapETP2Iso, IsoMapETP2Iso_i_tt, IsoMapEIso2TP_i_tt = WH_main_obtainIso(
        m, m["v_E"], final_pathETP2Iso, vthetaAllETP2Iso, 1
    )
    _fast_save(IsoMapEIso2TP_i_tt, time_map_dir, "IsoMapEIso2TP_i_tt")
    _fast_save(IsoMapETP2Iso_i_tt, time_map_dir, "IsoMapETP2Iso_i_tt")
    _fast_save(pathFinalMapETP2Iso, time_map_dir, "pathFinalMapETP2Iso")
    del IsoMapETP2Iso_i_tt, IsoMapEIso2TP_i_tt, pathFinalMapETP2Iso, final_pathETP2Iso

    _fast_save(m, time_map_dir, "Map")

    if not _has_required_assets(time_map_dir):
        missing = [fn for fn in _required_static_assets() if not os.path.exists(os.path.join(time_map_dir, fn))]
        raise FileNotFoundError(f"静态资产缺失：{missing}")


def _worker_build_one(args: Tuple) -> Dict[str, Any]:
    """单个 worker：构建一个 TimeMap 的 IsoMap 资产。在子进程中执行。"""
    tm, tm_dir, v_p, v_e = args
    t0 = time.perf_counter()
    try:
        if os.path.exists(os.path.join(tm_dir, "IsoMapEIso2TP_i_tt.jbl")):
            dt = time.perf_counter() - t0
            return {"tm": tm, "elapsed": dt, "error": None, "skipped": True}
        if not os.path.exists(os.path.join(tm_dir, "Map.jbl")):
            raise FileNotFoundError(f"缺少 {tm_dir}/Map.jbl")
        _build_one(tm_dir, v_p=v_p, v_e=v_e)
        dt = time.perf_counter() - t0
        return {"tm": tm, "elapsed": dt, "error": None, "skipped": False}
    except Exception as e:
        dt = time.perf_counter() - t0
        return {"tm": tm, "elapsed": dt, "error": repr(e), "skipped": False}


def main():
    parser = argparse.ArgumentParser(description="批量环境构建：为每个 map/<TimeMap>/ 生成 IsoMap* 与 pathFinal*")
    parser.add_argument("--config", required=True, help="env_build.yaml 路径")
    parser.add_argument("--num-workers", type=int, default=None, help="并行 worker 数（覆盖 YAML 配置）")
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
    log_path_jsonl = str(cfg.get("failure_log", os.path.join(map_root, "_build_failures.jsonl")))

    num_workers = args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 1))
    if num_workers <= 0:
        num_workers = min(os.cpu_count() or 1, len(time_maps))

    # 启动信息（立即输出）
    print(f"[BUILD] time_maps={len(time_maps)} num_workers={num_workers} map_root={map_root}", flush=True)
    if not time_maps:
        print("[BUILD] 未发现任何 TimeMap，退出。", flush=True)
        return

    os.makedirs(map_root, exist_ok=True)

    # 设置日志
    log_path = os.path.join(map_root, "_batch_build.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    logger = logging.getLogger("batch_env_build")

    work_items = [(tm, os.path.join(map_root, tm), v_p, v_e) for tm in time_maps]

    ok = []
    skipped = []
    failed = []
    t_start = time.perf_counter()

    if num_workers == 1:
        # 串行模式
        print(f"[BUILD] 开始串行处理 ...", flush=True)
        for item in work_items:
            result = _worker_build_one(item)
            if result.get("skipped"):
                skipped.append(result["tm"])
                logger.info(f"[BUILD] skip TimeMap={result['tm']} (IsoMapEIso2TP_i_tt.jbl already exists)")
                print(f"[BUILD] skip TimeMap={result['tm']} (IsoMapEIso2TP_i_tt.jbl already exists)", flush=True)
            elif result["error"] is None:
                ok.append(result["tm"])
                logger.info(f"[BUILD] ok TimeMap={result['tm']} elapsed={result['elapsed']:.2f}s")
                print(f"[BUILD] ok TimeMap={result['tm']} elapsed={result['elapsed']:.2f}s", flush=True)
            else:
                failed.append(result["tm"])
                logger.error(f"[BUILD] FAIL TimeMap={result['tm']} elapsed={result['elapsed']:.2f}s error={result['error']}")
                print(f"[BUILD] FAIL TimeMap={result['tm']} elapsed={result['elapsed']:.2f}s error={result['error']}", flush=True)
                rec = {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "TimeMap": result["tm"],
                    "elapsed_s": result["elapsed"],
                    "error": result["error"],
                }
                os.makedirs(os.path.dirname(log_path_jsonl) or ".", exist_ok=True)
                with open(log_path_jsonl, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if cleanup_on_fail:
                    tm_dir = os.path.join(map_root, result["tm"])
                    try:
                        shutil.rmtree(tm_dir)
                        logger.info(f"[BUILD] rollback removed {tm_dir}")
                        print(f"[BUILD] rollback removed {tm_dir}", flush=True)
                    except Exception as re:
                        logger.error(f"[BUILD] rollback failed for {tm_dir}: {repr(re)}")
                        print(f"[BUILD] rollback failed for {tm_dir}: {repr(re)}", flush=True)
    else:
        # 并行模式：使用 fork（Linux 默认），避免 spawn 重复导入
        ctx = get_context("fork")
        print(f"[BUILD] 启动 {num_workers} 个 worker ...", flush=True)
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            futures = {executor.submit(_worker_build_one, item): item for item in work_items}
            done_count = 0
            for future in as_completed(futures):
                done_count += 1
                result = future.result()
                if result.get("skipped"):
                    skipped.append(result["tm"])
                    logger.info(f"[BUILD] skip TimeMap={result['tm']} (IsoMapEIso2TP_i_tt.jbl already exists)")
                elif result["error"] is None:
                    ok.append(result["tm"])
                    logger.info(f"[BUILD] ok TimeMap={result['tm']} elapsed={result['elapsed']:.2f}s")
                else:
                    failed.append(result["tm"])
                    logger.error(f"[BUILD] FAIL TimeMap={result['tm']} elapsed={result['elapsed']:.2f}s error={result['error']}")
                    rec = {
                        "time": datetime.now().isoformat(timespec="seconds"),
                        "TimeMap": result["tm"],
                        "elapsed_s": result["elapsed"],
                        "error": result["error"],
                    }
                    os.makedirs(os.path.dirname(log_path_jsonl) or ".", exist_ok=True)
                    with open(log_path_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if cleanup_on_fail:
                        tm_dir = os.path.join(map_root, result["tm"])
                        try:
                            shutil.rmtree(tm_dir)
                            logger.info(f"[BUILD] rollback removed {tm_dir}")
                        except Exception as re:
                            logger.error(f"[BUILD] rollback failed for {tm_dir}: {repr(re)}")
                if result.get("skipped"):
                    status = "skip"
                elif result["error"] is None:
                    status = "ok"
                else:
                    status = "FAIL"
                print(f"[BUILD] [{done_count}/{len(time_maps)}] {status} TimeMap={result['tm']} elapsed={result['elapsed']:.2f}s", flush=True)

    # 汇总
    elapsed = time.perf_counter() - t_start
    logger.info(f"[BUILD] done. ok={len(ok)}, skipped={len(skipped)}, failed={len(failed)}, elapsed={elapsed:.1f}s")
    print(f"\n[BUILD] 完成。成功: {len(ok)}, 跳过: {len(skipped)}, 失败: {len(failed)}, 耗时: {elapsed:.1f}s", flush=True)
    if failed:
        print(f"[BUILD] 失败列表: {failed}", flush=True)
    print(f"[BUILD] 详细日志: {log_path}", flush=True)
    if failed:
        print(f"[BUILD] 失败记录: {log_path_jsonl}", flush=True)


if __name__ == "__main__":
    main()
