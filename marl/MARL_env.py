import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.animation import FFMpegWriter
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Point, Polygon


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Draw.Draw_map import Draw_map
from intercept.IsoPair.obtainIsoPairsAll import obtainPTP2TP_IsoPos_timeShift2
from intercept.IsoPair.obtainIsoPath import insertIsoMapP2TP
from intercept.IsoPair.obtainNearTPall import obtainNearETP, obtainNearTP
from intercept.IsoPair.obtainPE2TP import obtainPE2TP
from intercept.IsoPair.obtainTaskAll import obtainTask_timeShift2
from intercept.Prediction.obtainDWAprePath import obtainDWAprePath
from intercept.Prediction.predictLikelyTarget import predictLikelyTargetNew
from marl.obs_generator import TODCObservationGenerator
from marl.rewards import TODCRewardFunction


@dataclass
class StepStats:
    replanned: bool = False
    collision: bool = False
    captured: int = 0


class TODCMARLEnv(gym.Env):
    metadata = {"render_modes": ["none", "human", "rgb_array"], "render_fps": 20}

    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or {}
        self.data_dir = self.config.get("data_dir", project_root)
        self.map_root = self.config.get("map_root", os.path.join(self.data_dir, "map"))
        self.time_map = self.config.get("time_map", None)
        self.evader_profile_mode = self.config.get("evader_profile_mode", "random")
        self.evader_profile_id = self.config.get("evader_profile_id", None)
        self.time_res = float(self.config.get("time_res", 1.0))
        # Dynamic candidate length: k_max is inferred from current candidates per step.
        self.k_max = 1
        self.render_mode = self.config.get("render_mode", "none")
        self.enable_dwa_replan = bool(self.config.get("enable_dwa_replan", True))
        # step_mode="decision" makes one RL step align with one online decision event (replan).
        self.step_mode = str(self.config.get("step_mode", "decision")).lower()
        self.max_episode_steps = int(self.config.get("max_episode_steps", 600))
        self.collision_dist = float(self.config.get("collision_dist", 25.0))
        self.reward_fn = TODCRewardFunction.from_env_config(self.config)

        self._load_assets()
        self.num_V = int(self.ValuePos.shape[0]) if hasattr(self, "ValuePos") else 0
        self.obs_generator = TODCObservationGenerator()
        self._build_spaces()

        self.fig = None
        self.ax = None
        self.writer = None
        self._video_ctx = None

        self.episode_step = 0
        self.last_min_dist = None
        self.last_action_weights = np.zeros((self.num_P, self.k_max), dtype=np.float32)
        self.last_action_indices = np.zeros((self.num_P,), dtype=np.int64)
        self._profile_cursor = -1
        self.current_profile = None
        self.decision_step = 0
        # Cached geometry from last _compute_isomap_intercept_candidates (for path stitching)
        self._path_e2tp_cache = None
        self._path_p2tp_cache = None
        self._e2tp_tp_idx_cache = None
        self._p2tp_tp_idx_cache = None
        self._pairs_e2val_cache = None
        self._ic_best_per_pair = None
        # 当前路径段起始时的全局仿真时间 t_all；t 置 0 时同步更新，使得 t_all = _t_all_at_path_start + t
        self._t_all_at_path_start = 0.0

    def set_runtime_params(
        self,
        *,
        step_mode: Optional[str] = None,
    ):
        """Update online decision-step runtime parameters safely."""
        if step_mode is not None:
            mode = str(step_mode).lower()
            if mode not in {"decision", "time"}:
                raise ValueError(f"Unsupported step_mode={step_mode}. Use 'decision' or 'time'.")
            self.step_mode = mode

    def set_reward_params(self, **kwargs):
        """Runtime reward tuning without touching environment dynamics."""
        self.reward_fn.update_config(**kwargs)

    @property
    def sim_dt(self) -> float:
        """单步推进对应的仿真时间增量（与 main0319 中 t、t_all 同步增长量一致）。"""
        return float(self.time_res) / float(self.Stepsize)

    def _build_spaces(self):
        ally_slots = max(1, self.num_P - 1)
        ally_pts_slots = max(1, (self.num_P - 1) * self.k_max)
        asset_slots = max(1, self.num_V)
        self.action_space = spaces.MultiDiscrete(np.full((self.num_P,), self.k_max, dtype=np.int64))
        self.observation_space = spaces.Dict(
            {
                "self_uav": spaces.Box(-np.inf, np.inf, shape=(self.num_P, 1, 3), dtype=np.float32),
                "ally_uavs": spaces.Box(-np.inf, np.inf, shape=(self.num_P, ally_slots, 3), dtype=np.float32),
                "self_pts": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.k_max, 8), dtype=np.float32),
                "reward_nodes": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.k_max, 8), dtype=np.float32),
                "ally_pts": spaces.Box(-np.inf, np.inf, shape=(self.num_P, ally_pts_slots, 8), dtype=np.float32),
                "enemies": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.num_E, 3), dtype=np.float32),
                "targets": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.num_E, 2), dtype=np.float32),
                "assets": spaces.Box(-np.inf, np.inf, shape=(self.num_P, asset_slots, 2), dtype=np.float32),
                "ally_mask": spaces.Box(0, 1, shape=(self.num_P, ally_slots), dtype=np.int8),
                "self_pts_mask": spaces.Box(0, 1, shape=(self.num_P, self.k_max), dtype=np.int8),
                "ally_pts_mask": spaces.Box(0, 1, shape=(self.num_P, ally_pts_slots), dtype=np.int8),
                "enemy_mask": spaces.Box(0, 1, shape=(self.num_P, self.num_E), dtype=np.int8),
                "target_mask": spaces.Box(0, 1, shape=(self.num_P, self.num_E), dtype=np.int8),
                "asset_mask": spaces.Box(0, 1, shape=(self.num_P, asset_slots), dtype=np.int8),
            }
        )

    def _sync_dynamic_k(self, obs: Dict[str, np.ndarray]):
        k_new = int(obs["self_pts"].shape[1])
        if k_new != self.k_max:
            self.k_max = max(1, k_new)
            self._build_spaces()
            self.last_action_weights = np.zeros((self.num_P, self.k_max), dtype=np.float32)
            self.last_action_indices = np.zeros((self.num_P,), dtype=np.int64)

    @staticmethod
    def _normalize_time_folder(tag: Optional[str]) -> Optional[str]:
        if tag is None:
            return None
        t = str(tag).strip()
        if not t:
            return None
        return t.lstrip("_")

    def _time_folders(self) -> List[str]:
        if not os.path.isdir(self.map_root):
            return []
        folders = [
            d
            for d in os.listdir(self.map_root)
            if os.path.isdir(os.path.join(self.map_root, d))
        ]
        folders.sort(reverse=True)
        return folders

    def _folder_has_files(self, base: str, filenames: List[str]) -> bool:
        return all(os.path.exists(os.path.join(base, f)) for f in filenames)

    def _discover_evader_profile_dirs(self, map_base: str) -> List[str]:
        required = ["pathFinalE2ValIn.jbl", "PathE2Val_true.jbl"]
        slots = []
        for name in sorted(os.listdir(map_base)):
            slot_dir = os.path.join(map_base, name)
            if os.path.isdir(slot_dir) and self._folder_has_files(slot_dir, required):
                slots.append(slot_dir)

        # 兼容旧目录结构：若不存在子目录，则允许直接使用 map/TimeMap 下的单组轨迹
        if not slots and self._folder_has_files(map_base, required):
            slots = [map_base]
        return slots

    def _load_evader_profile(self, profile_dir: str):
        self.pathFinalE2ValIn = joblib.load(os.path.join(profile_dir, "pathFinalE2ValIn.jbl"))
        self.PathE2Val_true = joblib.load(os.path.join(profile_dir, "PathE2Val_true.jbl"))

        if isinstance(self.PathE2Val_true, dict):
            self.PathE2Val_true = [self.PathE2Val_true[i] for i in sorted(self.PathE2Val_true.keys())]

        self.length_E_max = 0
        for i in range(len(self.pathFinalE2ValIn)):
            for j in range(len(self.pathFinalE2ValIn[i])):
                self.length_E_max = max(self.length_E_max, len(self.pathFinalE2ValIn[i][j][0]))

    def _select_profile_dir(self, options: Optional[Dict] = None) -> str:
        if len(self.evader_profile_dirs) == 1:
            return self.evader_profile_dirs[0]

        if options is not None and "profile_id" in options:
            idx = int(options["profile_id"]) % len(self.evader_profile_dirs)
            return self.evader_profile_dirs[idx]

        if self.evader_profile_id is not None:
            idx = int(self.evader_profile_id) % len(self.evader_profile_dirs)
            return self.evader_profile_dirs[idx]

        mode = str(self.evader_profile_mode).lower()
        if mode == "cycle":
            self._profile_cursor = (self._profile_cursor + 1) % len(self.evader_profile_dirs)
            return self.evader_profile_dirs[self._profile_cursor]

        idx = int(self.np_random.integers(0, len(self.evader_profile_dirs)))
        return self.evader_profile_dirs[idx]

    def _load_assets(self):
        map_names = {
            "Map": "Map.jbl",
            "IsoMapPTP2Iso_i_tt": "IsoMapPTP2Iso_i_tt.jbl",
            "IsoMapPIso2TP_i_tt": "IsoMapPIso2TP_i_tt.jbl",
            "pathFinalMapPTP2Iso": "pathFinalMapPTP2Iso.jbl",
            "IsoMapTP2Val_i_tt": "IsoMapTP2Val_i_tt.jbl",
            "pathFinalTP2Val": "pathFinalTP2Val.jbl",
            "IsoMapETP2Iso_i_tt": "IsoMapETP2Iso_i_tt.jbl",
            "IsoMapEIso2TP_i_tt": "IsoMapEIso2TP_i_tt.jbl",
            "pathFinalMapETP2Iso": "pathFinalMapETP2Iso.jbl",
        }
        required_map_files = list(map_names.values())

        folders = self._time_folders()
        explicit_map = self._normalize_time_folder(self.time_map)
        map_candidates = [explicit_map] if explicit_map is not None else folders
        map_candidates = [f for f in map_candidates if f is not None]

        selected_map_folder = None
        for folder in map_candidates:
            folder_base = os.path.join(self.map_root, folder)
            if self._folder_has_files(folder_base, required_map_files):
                selected_map_folder = folder
                break

        if selected_map_folder is None:
            raise FileNotFoundError(
                f"Cannot resolve static map assets in {self.map_root}. "
                f"Required files: {required_map_files}"
            )

        map_base = os.path.join(self.map_root, selected_map_folder)
        resolved = {k: os.path.join(map_base, v) for k, v in map_names.items()}
        self.time_map = selected_map_folder
        self.evader_profile_dirs = self._discover_evader_profile_dirs(map_base)

        if len(self.evader_profile_dirs) == 0:
            raise FileNotFoundError(
                f"No evader profile found under {map_base}. Expected map/<TimeMap>/<slot>/PathE2Val_true.jbl"
            )

        self.Map = joblib.load(resolved["Map"])
        self.IsoMapPTP2Iso_i_tt = joblib.load(resolved["IsoMapPTP2Iso_i_tt"])
        self.IsoMapPIso2TP_i_tt = joblib.load(resolved["IsoMapPIso2TP_i_tt"])
        self.pathFinalMapPTP2Iso = joblib.load(resolved["pathFinalMapPTP2Iso"])
        self.IsoMapTP2Val_i_tt = joblib.load(resolved["IsoMapTP2Val_i_tt"])
        self.pathFinalTP2Val = joblib.load(resolved["pathFinalTP2Val"])
        self.IsoMapETP2Iso_i_tt = joblib.load(resolved["IsoMapETP2Iso_i_tt"])
        self.IsoMapEIso2TP_i_tt = joblib.load(resolved["IsoMapEIso2TP_i_tt"])
        self.pathFinalMapETP2Iso = joblib.load(resolved["pathFinalMapETP2Iso"])

        self.obs = self.Map["obs"]
        self.sure = self.Map["sure"]
        self.obs_no_circle = self.Map["obs_no_circle"]
        self.obs_no_circle_in = self.Map["obs_no_circle_in"]
        self.Stepsize = self.Map["Stepsize"]
        self.v_P = float(self.Map["v_P"])
        self.v_E = float(self.Map["v_E"])
        self.Trans_Point = self.Map["Trans_Point"]
        self.ValuePos = self.Map["ValuePos"]
        self.PStart_Point = self.Map["PStart_Point"]
        self.Evader = self.Map.get("Evader")
        if self.Evader is None:
            # 兼容无 Evader 字段的 Map.jbl（测试桩或旧资源），与 main0319forRL 默认一致
            self.Evader = np.array(
                [[200.0, 1950.0, -np.pi / 2], [1000.0, 1950.0, -np.pi / 2], [1800.0, 1950.0, -np.pi / 2]],
                dtype=float,
            )
        else:
            self.Evader = np.asarray(self.Evader, dtype=float)
        self.timeIsoRes = self.Map["timeIsoRes"]



        self._load_evader_profile(self.evader_profile_dirs[0])

        self.num_E = int(self.Evader.shape[0])
        self.num_P = int(self.PStart_Point.shape[0])
        self.agents = [f"p_{i}" for i in range(self.num_P)]

        self.E_PreRef = {
            "num_v": 10,
            "num_w": 10,
            "v_range": [100, 150],
            "w_range": [-np.pi / 6, np.pi / 6],
            "Stepsize": self.Stepsize,
            "T_pred": 1,
        }

        cap_dist = float(self.config.get("cap_dist", 200.0))
        self.CapRef = {
            "CapDist": cap_dist,
            "CapAngle": float(self.config.get("cap_angle", 87.0)),
            "Stepsize": self.Stepsize,
            "CapDistRef": float(self.config.get("cap_dist_ref", 350.0)),
            "v_P": self.v_P,
            "v_E": self.v_E,
            "timeIsoRes": self.timeIsoRes,
        }

        self.obs_polygons = [
            Polygon(np.asarray(poly, dtype=float)[:, :2]) for poly in self.obs_no_circle if poly is not None
        ]
        self.CapRef["obs_polygons"] = self.obs_polygons

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        options = options or {}

        # Support per-episode runtime override for online control knobs.
        if "step_mode" in options:
            self.set_runtime_params(step_mode=options.get("step_mode"))

        reward_opts = options.get("reward")
        if isinstance(reward_opts, dict) and len(reward_opts) > 0:
            self.set_reward_params(**reward_opts)

        profile_dir = self._select_profile_dir(options)
        self._load_evader_profile(profile_dir)
        self.current_profile = os.path.relpath(profile_dir, self.map_root)

        self.episode_step = 0
        self.decision_step = 0
        self.t = 0.0
        self.t_all = 0.0
        self._t_all_at_path_start = 0.0

        self.PosE = self.Evader.copy()
        self.PosP = self.PStart_Point.copy()

        self.PathPtrue = [p.reshape(-1, 1) for p in self.PosP]
        self.PathP = [np.tile(self.PosP[i].reshape(-1, 1), (1, self.length_E_max)) for i in range(self.num_P)]

        self.PathE = [
            self.PathE2Val_true[eid][: max(2, int(self.time_res / self.Stepsize * self.v_E)), :].T
            for eid in range(self.num_E)
        ]

        self.Capflag = np.array([False] * self.num_E)
        self.pairs_realE2P = None
        self.UnCapPid = np.arange(self.num_P, dtype=int)
        self.UnCapEid = np.arange(self.num_E, dtype=int)
        self.UnCapPidNew = self.UnCapPid.copy()
        self.UnCapEidNew = self.UnCapEid.copy()

        # self.IC = np.empty((0, 18), dtype=float)
        self.IC_candidates = np.empty((0, 18), dtype=float)
        self.ICFinalAssign = np.empty((0, 18), dtype=float)
        self.ICFinalAction = np.empty((0, 18), dtype=float)
        self.ICFinalActionCandidates = np.empty((0, 18), dtype=float)
        self._path_e2tp_cache = None
        self._ic_best_per_pair = None

        # main0319: 内层 _phase_update -> _advance_from_paths -> _phase_check_decision 直至 DWA 需重规划 (line 264)，再构建候选
        while not np.all(self.Capflag) and (self.t_all < self.length_E_max /self.v_E):
            self._phase_update()
            self._advance_from_paths()
            need_replan, terminal = self._phase_check_decision()
            if terminal:
                break
            if need_replan:
                self._replan_with_isomap()
                break
            self._update_capflag_from_geometry()



        self.last_min_dist = self._pairwise_dist().min(axis=1)
        obs = self._build_obs()
        self._sync_dynamic_k(obs)
        info = {
            "real_mode": True,
            "replanned": True,
            "num_candidates": int(self.IC_candidates.shape[0]),
            "time_map": self.time_map,
            "profile": self.current_profile,
            "t": float(self.t),
            "t_all": float(self.t_all),
            "sim_dt": float(self.sim_dt),
            "step_mode": self.step_mode,
            "reward": {
                "dist_progress_scale": float(self.reward_fn.config.dist_progress_scale),
                "w_global": float(self.reward_fn.config.w_global),
                "step_cost": float(self.reward_fn.config.step_cost),
                "terminal_capture_bonus": float(self.reward_fn.config.terminal_capture_bonus),
                "terminal_asset_loss_penalty": float(self.reward_fn.config.terminal_asset_loss_penalty),
            },
        }
        return obs, info

    def step(self, action_dict):
        """RL 对外一步。decision 模式内层按 main0319：update -> _advance_from_paths -> _phase_check_decision 循环直至重规划或终止。"""
        action_weights, action_indices = self._normalize_action(action_dict)
        self.last_action_weights = action_weights
        self.last_action_indices = action_indices

        stats = StepStats(replanned=False, collision=False, captured=0)
        delta_t_all = 0.0

        decision_mode = self.step_mode == "decision"
        prev_cap = self.Capflag.copy()

        if decision_mode:
            # One RL step: 先应用动作，再内层循环 main0319：update -> _advance_from_paths -> check，
            # 直至 need_replan、全局回合时间到、或其它终止。
            self._apply_assignment_from_action(action_indices)

            t_all_before = float(self.t_all)

            while not np.all(self.Capflag) and (self.t_all < self.length_E_max /self.v_E):
                self._phase_update()
                self._advance_from_paths()
                need_replan, terminal = self._phase_check_decision()
                if terminal:
                    stats.collision = bool(self._check_collision())
                    break
                if need_replan:
                    self._replan_with_isomap()
                    stats.replanned = True
                    break
                self._update_capflag_from_geometry()

            stats.captured = int(np.sum(~prev_cap & self.Capflag))
            self.decision_step += 1
            self.episode_step += 1
            delta_t_all = float(self.t_all) - t_all_before

        rewards, reward_details = self._compute_rewards(action_indices, sim_time_elapsed=delta_t_all)
        asset_breached = self._check_asset_breach()
        self.reward_fn.apply_terminal_rewards(
            rewards,
            captured_delta=stats.captured,
            num_e=self.num_E,
            asset_breached=asset_breached,
            details=reward_details,
        )

        max_t = float(self.length_E_max) / float(self.v_E)
        time_exceeded = self.t_all >= max_t - 1e-9
        done_all = bool(
            np.all(self.Capflag) or stats.collision or asset_breached or time_exceeded
        )
        truncated_all = bool(self.episode_step >= self.max_episode_steps)

        obs = self._build_obs()
        self._sync_dynamic_k(obs)
        terminations = {f"p_{i}": done_all for i in range(self.num_P)}
        truncations = {f"p_{i}": truncated_all for i in range(self.num_P)}
        terminations["__all__"] = done_all
        truncations["__all__"] = truncated_all

        infos = {}
        for i in range(self.num_P):
            entry = {
                "replanned": stats.replanned,
                "collision": stats.collision,
                "captured_total": int(np.sum(self.Capflag)),
                "asset_breached": bool(asset_breached),
                "num_candidates": int(self.IC_candidates.shape[0]),
                "delta_t_all": float(delta_t_all),
                "t": float(self.t),
                "t_all": float(self.t_all),
                "sim_dt": float(self.sim_dt),
                "decision_step": int(self.decision_step),
                "step_mode": self.step_mode,
            }
            # attach per-agent reward breakdown if available
            try:
                if reward_details is not None and f"p_{i}" in reward_details:
                    entry["reward_details"] = reward_details[f"p_{i}"]
            except Exception:
                pass
            infos[f"p_{i}"] = entry
        # expose full reward breakdown mapping for external use (logging, analysis)
        try:
            infos["_reward_details_all"] = reward_details
        except Exception:
            infos["_reward_details_all"] = None
        # 全局仿真时间 t_all / 路径段执行时间 t：训练脚本主记 t_all，env 内层终止条件用 t（见 decision 内层 while）
        infos["global_t_all"] = float(self.t_all)
        infos["path_exec_t"] = float(self.t)
        return obs, rewards, terminations, truncations, infos

    def _advance_from_paths(self):
        curr_idx_p = int(round(self.t * self.v_P))
        seg_start = int(round((self.t - self.time_res / self.Stepsize) * self.v_P))
        seg_end = curr_idx_p

        new_pos_p = []
        for pid in range(self.num_P):
            path = self.PathP[pid]
            idx = min(curr_idx_p, path.shape[1] - 1)
            new_pos_p.append(path[:, idx])
            seg = path[:, max(0, seg_start) : max(max(0, seg_start), seg_end)]
            if seg.size > 0:
                self.PathPtrue[pid] = np.hstack((self.PathPtrue[pid], seg))
        self.PosP = np.asarray(new_pos_p)

        curr_idx_e = int(round(self.t_all * self.v_E))
        self.PosE = np.array(
            [
                self.PathE2Val_true[eid][min(curr_idx_e, self.PathE2Val_true[eid].shape[0] - 1), :]
                for eid in range(self.num_E)
            ]
        )

        self.PathEpre = [None] * len(self.UnCapEidNew)
        for _id, eid in enumerate(self.UnCapEidNew):
            self.PathEpre[_id] = self.PathE[int(np.where(eid == self.UnCapEid)[0][0])]

    def _replan_with_isomap(self):
        """Full replan: build candidates + Hungarian assignment + paths (legacy / smoke tests)."""
        if not self._compute_isomap_intercept_candidates():
            return
            # TODO:应该是model获得的action
        self._apply_hungarian_and_paths()

    def _compute_isomap_intercept_candidates(self) -> bool:
        """Build iso maps and intercept table; cache geometry for _apply_paths_from_assigned_rows. Raises if no valid IsoPairs."""
        traj = [self.PathE2Val_true[i][: max(2, int(self.t_all * self.v_E)), :] for i in range(self.num_E)]
        targets = self.ValuePos[:, :2]
        res = predictLikelyTargetNew(traj, targets)
        validnew = res["rank_idx"][:, 0]
        pairs_e2val = np.column_stack((np.arange(len(validnew)), validnew))

        _, near_tpid_e = obtainNearETP(self.Trans_Point[:, :2], self.PosE, self.ValuePos, pairs_e2val)

        path_e2tp, iso_e2tp, e2tp_results = obtainPE2TP(
            self.IsoMapEIso2TP_i_tt,
            near_tpid_e,
            self.PosE,
            self.Trans_Point,
            self.pathFinalMapETP2Iso,
            self.Map,
            self.Map["v_E"],
        )
        e2tp_tp_idx = np.array([r[1] for r in e2tp_results])

        _, near_tpid_p = obtainNearTP(self.Trans_Point[:, :2], self.PosP, self.PosE)
        path_p2tp, iso_p2tp, p2tp_results = obtainPE2TP(
            self.IsoMapPIso2TP_i_tt,
            near_tpid_p,
            self.PosP,
            self.Trans_Point,
            self.pathFinalMapPTP2Iso,
            self.Map,
            self.Map["v_P"],
        )
        p2tp_tp_idx = np.array([r[1] for r in p2tp_results])

        iso_p_raw = insertIsoMapP2TP(self.IsoMapPTP2Iso_i_tt, iso_p2tp, p2tp_tp_idx, path_p2tp)
        iso_e_raw = insertIsoMapP2TP(self.IsoMapTP2Val_i_tt, iso_e2tp, e2tp_tp_idx, path_e2tp)

        time_l = max(max(len(row) for row in iso_p_raw), max(len(row) for row in iso_e_raw))
        self.IsoMap_i_tt_P2Iso = [row + [None] * (time_l - len(row)) for row in iso_p_raw]
        self.IsoMap_i_tt_E2Iso = [row + [None] * (time_l - len(row)) for row in iso_e_raw]

        eids, pids, te_idx, tp_idx = np.meshgrid(
            np.arange(self.num_E), np.arange(self.num_P), np.arange(time_l), np.arange(time_l), indexing="ij"
        )

        distances = np.min(self._pairwise_dist(), axis=1)
        x1, y1 = self.CapRef["CapDist"], self.CapRef["CapDist"]
        x2, y2 = 2 * self.CapRef["CapDistRef"], self.CapRef["CapDistRef"]
        cap_dist_time = np.where(
            distances < x1,
            np.inf,
            np.where(distances >= x2, y2, y1 + (distances - x1) * (y2 - y1) / (x2 - x1)),
        )
        self.CapRef["CapDistTime"] = cap_dist_time

        x1a, y1a = self.CapRef["CapDist"], self.CapRef["CapAngle"]
        x2a, y2a = 2 * self.CapRef["CapDistRef"], 180.0
        cap_angle_time = np.where(
            distances < x1a,
            360.0,
            np.where(distances >= x2a, y2a, y1a + (distances - x1a) * (y2a - y1a) / (x2a - x1a)),
        )
        self.CapRef["CapAngleTime"] = cap_angle_time

        flat = zip(
            eids.ravel(),
            pids.ravel(),
            te_idx.ravel(),
            tp_idx.ravel(),
            eids.ravel(),
            pids.ravel(),
        )
        results = []
        for eid, pid, te, tp, ide, idp in flat:
            pair_mat = obtainPTP2TP_IsoPos_timeShift2(
                idp,
                ide,
                tp,
                te,
                self.CapRef,
                self.v_E,
                cap_dist_time,
                self.IsoMap_i_tt_P2Iso,
                self.IsoMap_i_tt_E2Iso,
                self.ValuePos,
                pairs_e2val,
            )
            if pair_mat is not None and len(pair_mat) > 0:
                results.append([te, tp, e2tp_tp_idx[ide], p2tp_tp_idx[idp], pair_mat, eid, pid, ide, idp])

        if len(results) == 0:
            self._path_e2tp_cache = None
            raise RuntimeError("No IsoPairs: obtainPTP2TP_IsoPos_timeShift2 produced no valid pair matrices.")

        iso_pairs = np.array(results, dtype=object)
        iso_pairs = iso_pairs[np.argsort(iso_pairs[:, 0].astype(int))]

        intercept = obtainTask_timeShift2(iso_pairs, self.IsoMap_i_tt_P2Iso, self.IsoMap_i_tt_E2Iso, self.CapRef)
        self.IC_candidates = intercept.copy()

        unique_pairs = np.unique(intercept[:, [11, 12]], axis=0)
        best_candidates_list = []
        for pair in unique_pairs:
            eid_val, pid_val = pair
            group = intercept[(intercept[:, 11] == eid_val) & (intercept[:, 12] == pid_val)]
            sort_idx = np.lexsort((group[:, 16], group[:, 8], -group[:, 15]))
            best_candidates_list.append(group[sort_idx[0]])
        ic_candidates = np.array(best_candidates_list)
        self._ic_best_per_pair = ic_candidates

        self._path_e2tp_cache = path_e2tp
        self._path_p2tp_cache = path_p2tp
        self._e2tp_tp_idx_cache = e2tp_tp_idx
        self._p2tp_tp_idx_cache = p2tp_tp_idx
        self._pairs_e2val_cache = pairs_e2val
        return True

    def _apply_hungarian_and_paths(self):
        ic_candidates = self._ic_best_per_pair
        if ic_candidates is None or ic_candidates.size == 0:
            raise RuntimeError("Hungarian assignment has no per-pair intercept candidates (_ic_best_per_pair is empty).")

        e_set, e_inv = np.unique(ic_candidates[:, 11], return_inverse=True)
        p_set, p_inv = np.unique(ic_candidates[:, 12], return_inverse=True)
        cost_mat = np.full((len(p_set), len(e_set)), 1e9)
        idx_mat = np.full((len(p_set), len(e_set)), -1, dtype=int)
        cost_mat[p_inv, e_inv] = ic_candidates[:, 8]
        idx_mat[p_inv, e_inv] = np.arange(len(ic_candidates))
        p_idx, e_idx = linear_sum_assignment(cost_mat)
        valid_mask = cost_mat[p_idx, e_idx] < 1e8
        final_rows = idx_mat[p_idx[valid_mask], e_idx[valid_mask]]
        assigned = ic_candidates[final_rows]
        self.ICFinalAssign = assigned.copy()

        pairs_realE2P_ = assigned[:, [11, 12]]

        self.pairs_realE2P = np.array(
            [
                [
                    self.UnCapEid[int(pairs_realE2P_[i, 0])],
                    self.UnCapPid[int(pairs_realE2P_[i, 1])],
                ]
                for i in range(len(pairs_realE2P_))
            ]
        )

        ep = self.IC_candidates[:, [11, 12]].astype(np.int64, copy=False)
        pr = pairs_realE2P_.astype(np.int64, copy=False)
        pair_dtype = np.dtype([("e", np.int64), ("p", np.int64)])
        ep_keys = np.ascontiguousarray(ep).view(pair_dtype).ravel()
        pr_keys = np.unique(np.ascontiguousarray(pr).view(pair_dtype).ravel())
        mask = np.isin(ep_keys, pr_keys)
        self.ICFinalActionCandidates = self.IC_candidates[mask]

        # TODO:应该是model获得的action,而不是直接使用任务分配的结果
        # self._apply_paths_from_assigned_rows(assigned)

    def _apply_paths_from_assigned_rows(self, assigned: np.ndarray):
        path_e2tp = self._path_e2tp_cache
        path_p2tp = self._path_p2tp_cache
        if path_e2tp is None or path_p2tp is None or assigned.size == 0:
            return

        e_idx = assigned[:, 11].astype(int)
        vp_idx = assigned[:, 9].astype(int)
        iso_idx = assigned[:, 13].astype(int)
        etp_idx = assigned[:, 2].astype(int)
        path_e_active = [
            np.hstack(
                (
                    path_e2tp[idx],
                    self.pathFinalTP2Val[etp_idx[i]][vp_idx[i]][
                        :, : max(0, int(iso_idx[i] - path_e2tp[idx].shape[1]))
                    ],
                )
            )
            if vp_idx[i] >= 0
            else path_e2tp[idx][:, : int(iso_idx[i])]
            for i, idx in enumerate(e_idx)
        ]
        for i, idx in enumerate(np.argsort(e_idx)):
            self.PathE[i] = path_e_active[idx]

        p_idx2 = assigned[:, 12].astype(int)
        tp_to_idx = assigned[:, 10].astype(int)
        iso_p_idx = assigned[:, 14].astype(int)
        tp_from_idx = assigned[:, 3].astype(int)
        path_p_active = [
            np.hstack(
                (
                    path_p2tp[idx],
                    self.pathFinalMapPTP2Iso[tp_from_idx[i]][tp_to_idx[i]][
                        :, : max(0, int(iso_p_idx[i] - path_p2tp[idx].shape[1]))
                    ],
                )
            )
            if tp_to_idx[i] >= 0
            else path_p2tp[idx][:, : int(iso_p_idx[i])]
            for i, idx in enumerate(p_idx2)
        ]
        for i, idx in enumerate(np.argsort(p_idx2)):
            self.PathP[i] = path_p_active[idx]

        # self.pairs_realE2P = np.column_stack((assigned[:, 11].astype(int), assigned[:, 12].astype(int)))
        self._t_all_at_path_start = float(self.t_all)
        self.t = 0.0

    # --- main0319 内层单步分解：update -> _advance_from_paths -> check（Gym 的 step(action) 仍是对外 RL 接口）---

    def _phase_update(self) -> None:
        """main0319:204-216 — 时间推进并同步未捕获 E-P 配对。"""
        dt = self.sim_dt
        self.t += dt
        self.t_all += dt
        if self.pairs_realE2P is not None and self.pairs_realE2P.shape[0] == self.Capflag.shape[0]:
            self.UnCapPidNew = self.pairs_realE2P[~self.Capflag, 1].astype(int)
            self.UnCapEidNew = self.pairs_realE2P[~self.Capflag, 0].astype(int)
            self.pairs_realE2P = self.pairs_realE2P[~self.Capflag]


    def _phase_check_decision(self) -> Tuple[bool, bool]:
        """碰撞/资产/全捕获与 DWA（main0319:258-264）。返回 (need_replan, terminal)。"""
        if self._check_collision():
            return False, True
        if self._check_asset_breach():
            return False, True
        
        if self.num_E > 0 and np.all(self.Capflag):
            return False, True
        need_replan = False
        if self.enable_dwa_replan:
            try:
                # pos_e, path_pre = self._dwa_pos_e_and_path_pre()
                min_dists, _ = obtainDWAprePath(self.PosE, self.PathEpre, self.E_PreRef)
                need_replan = not bool(np.all(min_dists <= 10))
            except Exception:
                need_replan = True
        return need_replan, False

# TODO: 为什么返回新增捕获数
    def _update_capflag_from_geometry(self) -> int:
        """按最近追捕者更新 Capflag；返回本步新增捕获数。"""
        distances_now = self._pairwise_dist()
        assignments_now = np.argmin(distances_now, axis=0)
        p_for_e_now = self.PosP[assignments_now]
        e_now = self.PosE
        dx_now = e_now[:, 0] - p_for_e_now[:, 0]
        dy_now = e_now[:, 1] - p_for_e_now[:, 1]
        ang_pe_now = np.degrees(np.arctan2(dy_now, dx_now))
        angle_diff_now = (ang_pe_now - np.degrees(p_for_e_now[:, 2]) + 180.0) % 360.0 - 180.0
        cap_dist = float(self.CapRef["CapDist"])
        cap_angle_half = float(self.CapRef["CapAngle"]) / 2.0
        prev_cap = self.Capflag.copy()
        self.Capflag = (np.min(distances_now, axis=0) <= cap_dist) & (np.abs(angle_diff_now) <= cap_angle_half)
        return int(np.sum(~prev_cap & self.Capflag))

    def _apply_assignment_from_action(self, action_indices: np.ndarray) -> None:
        """Apply RL-chosen candidate rows instead of Hungarian (requires valid isomap cache).

        `action_indices` 为每个 P 的离散候选下标（与 train 中 Categorical.sample() 一致），非概率向量。
        """
        if self._path_e2tp_cache is None:
            return
        try:
            obs = self._build_obs()
            self._sync_dynamic_k(obs)
            mask = np.asarray(obs["self_pts_mask"], dtype=np.float32)
            assigned_rows = []
            for pid in range(self.num_P):
                subset = self.ICFinalActionCandidates[self.ICFinalActionCandidates[:, 12].astype(int) == pid]
                n = int(subset.shape[0])
                idx = int(action_indices[pid])
                if idx >= n:
                    idx = n - 1
                if idx < 0:
                    idx = 0
                if mask[pid, idx] <= 0:
                    valid = np.where(mask[pid, :n] > 0)[0]
                    idx = int(valid[0]) if valid.size > 0 else 0
                assigned_rows.append(subset[idx])
            assigned = np.vstack(assigned_rows)
            self.ICFinalAction = assigned.copy()
            self._apply_paths_from_assigned_rows(assigned)
        except Exception:
            self._apply_hungarian_and_paths()

    def _pairwise_dist(self):
        pp = self.PosP[:, :2]
        ee = self.PosE[:, :2]
        return np.linalg.norm(pp[:, None, :] - ee[None, :, :], axis=2)

    def _extract_candidate_pos(self, row: np.ndarray, pid: int) -> Tuple[float, float, float]:

        # 尝试使用 IsoMap 中的 IsoPos（与 _extract_iso_points 行为一致）
        if hasattr(self, "IsoMap_i_tt_P2Iso") and row.shape[0] > 6:
            try:
                # 常见布局（见 obtainRLOutput）: te=0, tp=1, IsoIdxE=4, IsoIdxP=5, eid=11, pid=12
                tp_idx = int(row[1]) if row.shape[0] > 1 else 0
                iso_idx = int(row[5]) if row.shape[0] > 5 else (int(row[4]) if row.shape[0] > 4 else 0)

                cols = getattr(self.obs_generator, "cols", None)
                if cols is not None:
                    pid_ref = int(row[cols.pid_ref]) if row.shape[0] > cols.pid_ref else pid
                else:
                    pid_ref = int(row[12]) if row.shape[0] > 12 else pid

                iso_obj = self.IsoMap_i_tt_P2Iso[pid_ref][tp_idx]
                if iso_obj is not None and hasattr(iso_obj, "IsoPos") and iso_obj.IsoPos is not None:
                    iso_pos = np.asarray(iso_obj.IsoPos)
                    if iso_pos.ndim == 2:
                        iso_idx = max(0, min(iso_idx, iso_pos.shape[1] - 1))
                        return float(iso_pos[0, iso_idx]), float(iso_pos[1, iso_idx]), float(iso_pos[2, iso_idx])
            except Exception:
                pass

        # 回退：使用敌方位置与朝向（兼容老格式 / 无 IsoMap 情况）
        cols = getattr(self.obs_generator, "cols", None)
        if cols is not None and row.shape[0] > cols.eid_ref:
            eid = int(row[cols.eid_ref])
        else:
            eid = int(row[11]) if row.shape[0] > 11 else pid % self.num_E
        eid = max(0, min(eid, self.num_E - 1))
        return float(self.PosE[eid, 0]), float(self.PosE[eid, 1]), float(self.PosE[eid, 2])

    def _build_obs(self):
        inferred_targets = np.array([self.ValuePos[i % len(self.ValuePos), :2] for i in range(self.num_E)])
        return self.obs_generator.generate(
            pos_p=self.PosP,
            pos_e=self.PosE,
            v_p=self.v_P,
            v_e=self.v_E,
            value_pos=self.ValuePos,
            num_p=self.num_P,
            num_e=self.num_E,
            # TODO:关注这个ic最后指向的信息是什么？
            ic_candidates=self.ICFinalActionCandidates,
            candidate_pos_fn=self._extract_candidate_pos,
            inferred_targets=inferred_targets,
            pairs_realE2P=self.pairs_realE2P,
        )

    def _normalize_action(self, action_dict) -> Tuple[np.ndarray, np.ndarray]:
        """解析策略输出：返回 (one-hot 权重, 每智能体离散索引)。

        训练脚本通常传入 **采样后的索引** ``(num_P,)`` int；亦兼容 logits/概率矩阵。
        """
        obs = self._build_obs()
        self._sync_dynamic_k(obs)
        mask = np.asarray(obs["self_pts_mask"], dtype=np.float32)
        k_curr = int(mask.shape[1])

        idx = np.zeros((self.num_P,), dtype=np.int64)

        if isinstance(action_dict, dict):
            for i in range(self.num_P):
                key = f"p_{i}"
                if key not in action_dict:
                    continue
                a = np.asarray(action_dict[key]).reshape(-1)
                if a.size == 0:
                    continue
                if a.size == k_curr:
                    w = np.asarray(a, dtype=np.float32) * mask[i]
                    idx[i] = int(np.argmax(w)) if np.any(mask[i] > 0) else 0
                else:
                    idx[i] = int(a[0])
        else:
            arr = np.asarray(action_dict)
            if arr.ndim == 0:
                idx[:] = int(arr)
            elif arr.ndim == 1:
                if arr.shape[0] == self.num_P:
                    idx = arr.astype(np.int64)
                elif arr.shape[0] == k_curr:
                    w = np.tile(arr.reshape(1, -1), (self.num_P, 1)).astype(np.float32)
                    idx = np.argmax(w * mask, axis=1).astype(np.int64)
                else:
                    idx[:] = int(arr[0])
            else:
                w = np.zeros((self.num_P, k_curr), dtype=np.float32)
                n0 = min(self.num_P, arr.shape[0])
                n1 = min(k_curr, arr.shape[1])
                w[:n0, :n1] = np.asarray(arr[:n0, :n1], dtype=np.float32)
                idx = np.argmax(w * mask, axis=1).astype(np.int64)

        norm = np.zeros((self.num_P, k_curr), dtype=np.float32)
        for i in range(self.num_P):
            valid_idx = np.where(mask[i] > 0)[0]
            if valid_idx.size == 0:
                norm[i, 0] = 1.0
                continue
            sel = int(idx[i])
            if sel < 0 or sel >= k_curr or mask[i, sel] <= 0:
                sel = int(valid_idx[0])
            norm[i, sel] = 1.0
        return norm, idx

    def _compute_rewards(self, action_indices: np.ndarray, sim_time_elapsed: float):
        """`action_indices` 为形状 (num_P,) 的离散候选下标，与 `TODCRewardFunction.compute_step_rewards` 的 ndim==1 分支一致。"""
        curr_min_dist = self._pairwise_dist().min(axis=1)
        obs = self._build_obs()
        sd = float(self.sim_dt)
        # Request detailed breakdown from reward function so env can expose it via infos
        try:
            rewards_or_pair = self.reward_fn.compute_step_rewards(
                actions=action_indices,
                obs=obs,
                curr_min_dist=curr_min_dist,
                last_min_dist=self.last_min_dist,
                num_p=self.num_P,
                sim_time_elapsed=float(sim_time_elapsed),
                sim_dt=sd,
                return_details=True,
            )
            # compute_step_rewards may return (rewards, details) when return_details=True
            if isinstance(rewards_or_pair, tuple) and len(rewards_or_pair) == 2:
                rewards, details = rewards_or_pair
            else:
                rewards = rewards_or_pair
                details = None
        except Exception:
            rewards = self.reward_fn.compute_step_rewards(
                actions=action_indices,
                obs=obs,
                curr_min_dist=curr_min_dist,
                last_min_dist=self.last_min_dist,
                num_p=self.num_P,
                sim_time_elapsed=float(sim_time_elapsed),
                sim_dt=sd,
            )
            details = None

        self.last_min_dist = curr_min_dist.copy()
        # store last reward details for external inspection
        self._last_reward_details = details
        return rewards, details

    def _check_collision(self) -> bool:

        d_pp = self._pairwise_self_dist(self.PosP[:, :2])
        if np.any((d_pp > 0) & (d_pp < self.collision_dist)):
            return True
        return False

    def _check_asset_breach(self) -> bool:
        if self.PosE.size == 0 or self.ValuePos.size == 0:
            return False

        dist = np.linalg.norm(self.PosE[:, None, :2] - self.ValuePos[None, :, :2], axis=2)
        return bool(np.any(dist <= float(self.collision_dist)))

    @staticmethod
    def _pairwise_self_dist(pos: np.ndarray) -> np.ndarray:
        diff = pos[:, None, :] - pos[None, :, :]
        return np.linalg.norm(diff, axis=2)

    def render(self):
        if self.render_mode == "none":
            return None

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))

        self.ax.cla()
        Draw_map(
            self.PStart_Point,
            self.Trans_Point,
            self.ValuePos,
            self.obs,
            self.sure,
            self.obs_no_circle,
            self.obs_no_circle_in,
            ax=self.ax,
        )

        color_map = mpl.colormaps["tab20"]
        for idx in range(self.num_E):
            color_p = color_map((idx * 2) % 20)
            color_e = color_map((idx * 2 + 1) % 20)

            self.ax.plot(self.PosP[idx, 0], self.PosP[idx, 1], "o", color=color_p, markersize=8, markeredgecolor="w")
            self.ax.quiver(
                self.PosP[idx, 0],
                self.PosP[idx, 1],
                80 * np.cos(self.PosP[idx, 2]),
                80 * np.sin(self.PosP[idx, 2]),
                color=color_p,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.004,
            )
            self.ax.plot(self.PosE[idx, 0], self.PosE[idx, 1], "d", color=color_e, markersize=8, markeredgecolor="w")

            self.ax.plot(self.PathPtrue[idx][0, :], self.PathPtrue[idx][1, :], "-", color="gray", linewidth=2, alpha=0.4)
            len_e = min(int(self.t_all * self.v_E), self.PathE2Val_true[idx].shape[0] - 1)
            self.ax.plot(
                self.PathE2Val_true[idx][: len_e + 1, 0],
                self.PathE2Val_true[idx][: len_e + 1, 1],
                "-",
                color="gray",
                linewidth=2,
                alpha=0.4,
            )

        self.ax.set_title(f"TODC-MARL Env Time: {self.t_all:.2f}")
        self.ax.grid(True, linestyle="--", alpha=0.5)

        if self.render_mode == "human":
            plt.pause(0.001)
            return None

        self.fig.canvas.draw()
        width, height = self.fig.canvas.get_width_height()
        frame = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
        return frame

    def save_random_rollout_video(self, steps: int = 150, output_dir: str = "output") -> str:
        os.makedirs(output_dir, exist_ok=True)
        current_time = datetime.now().strftime("%m%d_%H%M")
        output_video_name = os.path.join(output_dir, f"uav_interception_env_{current_time}.mp4")

        fig, ax = plt.subplots(figsize=(12, 10))
        writer = FFMpegWriter(fps=20, metadata=dict(artist="TODC-MARL"), bitrate=1800)
        with writer.saving(fig, output_video_name, dpi=100):
            self.fig = fig
            self.ax = ax
            obs, _ = self.reset()
            for _ in range(steps):
                action = self.action_space.sample()
                obs, _, term, trunc, _ = self.step(action)
                self.render_mode = "rgb_array"
                self.render()
                writer.grab_frame()
                if term.get("__all__", False) or trunc.get("__all__", False):
                    break
        plt.close(fig)
        return output_video_name


def smoke_test():
    env = TODCMARLEnv(
        {
            "render_mode": "none",
            "max_episode_steps": 50,
        }
    )
    obs, info = env.reset()
    done = False
    trunc = False
    n = 0
    while not done and not trunc and n < 30:
        action = env.action_space.sample()
        obs, rew, term, trn, inf = env.step(action)
        done = term["__all__"]
        trunc = trn["__all__"]
        n += 1
    print(
        {
            "real_mode": info["real_mode"],
            "steps": n,
            "done": done,
            "truncated": trunc,
            "reward_keys": list(rew.keys()),
            "candidate_shape": obs["self_pts"].shape,
        }
    )


if __name__ == "__main__":
    smoke_test()