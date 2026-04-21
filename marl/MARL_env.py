import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from matplotlib.animation import FFMpegWriter
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon


project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from Draw.Draw_map import Draw_map
from intercept.IsoPair.obtainIsoPairsAll import obtainPTP2TP_IsoPos_timeShift2
from intercept.IsoPair.obtainIsoPath import insertIsoMapP2TP
from intercept.IsoPair.obtainIsoPath import obtainPE2IsoPath
from intercept.IsoPair.obtainNearTPall import obtainNearETP, obtainNearTP
from intercept.IsoPair.obtainPE2TP import obtainPE2TP
from intercept.IsoPair.obtainTaskAll import obtainTask_timeShift2
from intercept.Prediction.obtainDWAprePath import obtainDWAprePath
from intercept.Prediction.predictLikelyTarget import predictLikelyTargetNew
from marl.obs_generator import TODCObservationGenerator
from marl.rewards import TODCRewardFunction


# 本模块实现 Gymnasium 接口的 Dubins 拦截环境：底层为 IsoMap/匈牙利分配与 main0319 风格内层仿真；
# 策略输出离散候选索引，观测由 ``TODCObservationGenerator`` 生成。


@dataclass
class StepStats:
    replanned: bool = False
    collision: bool = False
    captured: int = 0


class TODCMARLEnv(gym.Env):
    """多追捕者 vs 固定轨迹逃逸者的决策环境。

    - 单次 ``step`` 对齐一次重规划/决策（内层可推进多物理 tick）。
    - 动作：每机对当前候选集 ``k_max`` 的离散索引；动态 ``K`` 时 ``_sync_dynamic_k`` 重建空间。
    - 观测：见 ``TODCObservationGenerator``；奖励见 ``TODCRewardFunction``。
    """

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
        self.max_episode_steps = int(self.config.get("max_episode_steps", 600))
        self.collision_dist = float(self.config.get("collision_dist", 25.0))
        self.reward_fn = TODCRewardFunction.from_env_config(self.config)

        self._load_assets()
        self.num_V = int(self.ValuePos.shape[0]) if hasattr(self, "ValuePos") else 0
        obs_cfg = self.config.get("obs")
        if not isinstance(obs_cfg, dict):
            obs_cfg = {}
        ally_r = obs_cfg.get("ally_perception_radius", self.config.get("ally_perception_radius"))
        self.obs_generator = TODCObservationGenerator(
            ally_perception_radius=float(ally_r) if ally_r is not None else None
        )
        self._build_spaces()

        self.fig = None
        self.ax = None
        self.writer = None
        self._video_ctx = None

        # Optional tick callback for step-internal visualization/logging.
        # Called from inside `step()`'s inner while-loop after `_advance_from_paths()`.
        self._on_tick: Optional[Callable[["TODCMARLEnv", int], None]] = None
        self._tick_counter: int = 0

        self.episode_step = 0
        self.last_min_dist = None
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
        self._pairs_ic_compact: Optional[np.ndarray] = None
        # 每全局 eid 的推断攻击目标 (x,y)，来自 ``predictLikelyTargetNew``；供观测与 v_e_nodes 一致
        self._inferred_targets_e: Optional[np.ndarray] = None
        # 当前路径段起始时的全局仿真时间 t_all；t 置 0 时同步更新，使得 t_all = _t_all_at_path_start + t
        self._t_all_at_path_start = 0.0

    def set_on_tick(self, cb: Optional[Callable[["TODCMARLEnv", int], None]]) -> None:
        self._on_tick = cb

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
                "allies_local": spaces.Box(-np.inf, np.inf, shape=(self.num_P, ally_slots, 3), dtype=np.float32),
                "enemy_assigned_self": spaces.Box(-np.inf, np.inf, shape=(self.num_P, 1, 3), dtype=np.float32),
                "enemy_assigned_per_ally": spaces.Box(-np.inf, np.inf, shape=(self.num_P, ally_slots, 3), dtype=np.float32),
                "enemy_self_mask": spaces.Box(0, 1, shape=(self.num_P, 1), dtype=np.int8),
                "asset_target_self": spaces.Box(-np.inf, np.inf, shape=(self.num_P, 1, 2), dtype=np.float32),
                "asset_target_per_ally": spaces.Box(-np.inf, np.inf, shape=(self.num_P, ally_slots, 2), dtype=np.float32),
                "self_pts": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.k_max, 8), dtype=np.float32),
                "reward_nodes": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.k_max, 8), dtype=np.float32),
                "ally_pts": spaces.Box(-np.inf, np.inf, shape=(self.num_P, ally_pts_slots, 8), dtype=np.float32),
                "enemies": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.num_E, 3), dtype=np.float32),
                "targets": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.num_E, 2), dtype=np.float32),
                "assets": spaces.Box(-np.inf, np.inf, shape=(self.num_P, asset_slots, 2), dtype=np.float32),
                "ally_mask": spaces.Box(0, 1, shape=(self.num_P, ally_slots), dtype=np.int8),
                "ally_enemy_mask": spaces.Box(0, 1, shape=(self.num_P, ally_slots), dtype=np.int8),
                "self_pts_mask": spaces.Box(0, 1, shape=(self.num_P, self.k_max), dtype=np.int8),
                "ally_pts_mask": spaces.Box(0, 1, shape=(self.num_P, ally_pts_slots), dtype=np.int8),
                "enemy_mask": spaces.Box(0, 1, shape=(self.num_P, self.num_E), dtype=np.int8),
                "target_mask": spaces.Box(0, 1, shape=(self.num_P, self.num_E), dtype=np.int8),
                "asset_mask": spaces.Box(0, 1, shape=(self.num_P, asset_slots), dtype=np.int8),
                "pursuer_active": spaces.Box(0, 1, shape=(self.num_P,), dtype=np.int8),
            }
        )

    def _sync_dynamic_k(self, obs: Dict[str, np.ndarray]):
        k_new = int(obs["self_pts"].shape[1])
        if k_new != self.k_max:
            self.k_max = max(1, k_new)
            self._build_spaces()

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
        self.num_E_ = self.num_E
        self.num_P = int(self.PStart_Point.shape[0])
        self.num_P_ = self.num_P
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

        reward_opts = options.get("reward")
        if isinstance(reward_opts, dict) and len(reward_opts) > 0:
            self.set_reward_params(**reward_opts)

        profile_dir = self._select_profile_dir(options)
        self._load_evader_profile(profile_dir)
        self.current_profile = os.path.relpath(profile_dir, self.map_root)

        self.episode_step = 0
        self.decision_step = 0
        self._tick_counter = 0
        self.t = 0.0
        self.t_all = 0.0
        self._t_all_at_path_start = 0.0

        self.PosE = self.Evader.copy() #动态的存活的敌机
        self.PosP = self.PStart_Point.copy() #动态的存活的追捕者

        self.PathPtrue = [p.reshape(-1, 1) for p in self.PosP]
        self.PathP = [np.tile(self.PosP[i].reshape(-1, 1), (1, self.length_E_max)) for i in range(self.num_P)]

        self.PathE = [
            self.PathE2Val_true[eid][: max(2, int(self.time_res / self.Stepsize * self.v_E)), :].T
            for eid in range(self.num_E)
        ]

        self.Capflag = np.array([False] * self.num_E) #连续的
        self.pairs_realE2P = None #实际的id，构造观测的标准
        self.UnCapPid = np.arange(self.num_P, dtype=int) #实际的id
        self.UnCapEid = np.arange(self.num_E, dtype=int) #实际的id
        self.UnCapPidNew = self.UnCapPid.copy() #实际的id
        self.UnCapEidNew = self.UnCapEid.copy() #实际的id

        # self.IC = np.empty((0, 18), dtype=float)
        self.IC_candidates = np.empty((0, 18), dtype=float)
        self.ICFinalAssign = np.empty((0, 18), dtype=float)
        self.ICFinalAction = np.empty((0, 18), dtype=float)
        self.ICFinalActionCandidates = np.empty((0, 18), dtype=float)
        self._path_e2tp_cache = None
        self._ic_best_per_pair = None
        self._inferred_targets_e = None
        self._pairs_ic_compact = None  # 与 pairs_realE2P 逐行对齐的 IC 表 11–12 列紧凑 (ide,idp)
        self._fallback_paths = {}  # (pid_global, eid_global) -> 3xL path, only for TPid == -2 fallback rows

        # main0319: 内层 _phase_update -> _advance_from_paths -> _phase_check_decision 直至 DWA 需重规划 (line 264)，再构建候选
        while not np.all(self.Capflag) and (self.t_all < self.length_E_max /self.v_E):
            self._phase_update()
            self._advance_from_paths()
            need_replan, terminal = self._phase_check_decision()
            if terminal:
                break
            if need_replan:
                self._compute_isomap_intercept_candidates()
                self._apply_hungarian_and_paths()
                break
            self._update_capflag_from_geometry()

        if self._inferred_targets_e is None:
            inferred, _ = self._predict_facility_ranks_for_e()
            self._inferred_targets_e = inferred

        self.last_min_dist = self._pairwise_dist()
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
        """RL 对外一步：内层按 main0319，update -> _advance_from_paths -> _phase_check_decision 循环直至重规划或终止。"""
        _, action_indices, obs_at_action = self._normalize_action(action_dict)

        stats = StepStats(replanned=False, collision=False, captured=0)
        delta_t_all = 0.0

        prev_cap = self.Capflag.copy()

        # One RL step: 先应用动作，再内层循环 main0319：update -> _advance_from_paths -> check，
        # 直至 need_replan、全局回合时间到、或其它终止。
        self._apply_assignment_from_action(action_indices)

        t_all_before = float(self.t_all)

        while not np.all(self.Capflag) and (self.t_all < self.length_E_max / self.v_E):
            self._phase_update()
            self._advance_from_paths()
            self._tick_counter += 1
            if self._on_tick is not None:
                self._on_tick(self, self._tick_counter)
            need_replan, terminal = self._phase_check_decision()
            if terminal:
                stats.collision = bool(self._check_collision())
                break
            if need_replan:
                self._compute_isomap_intercept_candidates()
                self._apply_hungarian_and_paths()
                stats.replanned = True
                break
            self._update_capflag_from_geometry()

        stats.captured = int(np.sum(~prev_cap & self.Capflag))
        self.decision_step += 1
        self.episode_step += 1
        delta_t_all = float(self.t_all) - t_all_before

        rewards, reward_details = self._compute_rewards(
            action_indices, sim_time_elapsed=delta_t_all, obs_at_action=obs_at_action
        )
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
            }
            entry["reward_details"] = reward_details[f"p_{i}"]
            infos[f"p_{i}"] = entry
        infos["_reward_details_all"] = reward_details
        # 全局仿真时间 t_all / 路径段执行时间 t：训练脚本主记 t_all，env 内层终止条件用 t（见 decision 内层 while）
        infos["global_t_all"] = float(self.t_all)
        infos["path_exec_t"] = float(self.t)
        return obs, rewards, terminations, truncations, infos

    def _advance_from_paths(self):
        curr_idx_p = int(round(self.t * self.v_P))
        seg_start = int(round((self.t - self.time_res / self.Stepsize) * self.v_P))
        seg_end = curr_idx_p

        self.PosP = np.array([self.PathP[np.where(pid==self.UnCapPid)[0][0]][:,min(curr_idx_p, self.PathP[np.where(pid==self.UnCapPid)[0][0]].shape[1] - 1)]
        for pid in self.UnCapPidNew])

        for id,pid in enumerate(self.UnCapPidNew):
        # 提取并排序 (替代最后一个 cellfun + cell2mat + sort)
            self.PathPtrue[pid] = np.hstack((self.PathPtrue[pid], self.PathP[np.where(pid==self.UnCapPid)[0][0]][:, max(0, seg_start) : seg_end]))



        curr_idx_e = int(round(self.t_all * self.v_E))
        self.PosE = np.array(
            [
                self.PathE2Val_true[eid][min(curr_idx_e, self.PathE2Val_true[eid].shape[0] - 1), :]
                for eid in self.UnCapEidNew
            ]
        )

        self.PathEpre = [None] * len(self.UnCapEidNew)
        for _id, eid in enumerate(self.UnCapEidNew):
            self.PathEpre[_id] = self.PathE[int(np.where(eid == self.UnCapEid)[0][0])]

    def _predict_facility_ranks_for_e(self) -> Tuple[np.ndarray, np.ndarray]:
        """单次 predictLikelyTargetNew：返回 (每机推断目标 xy float32 (num_E,2), rank_idx)。"""
        traj = [self.PathE2Val_true[i][: max(2, int(self.t_all * self.v_E)), :] for i in range(self.num_E)]
        res = predictLikelyTargetNew(traj, self.ValuePos[:, :2])
        rank_idx = res["rank_idx"]

        if rank_idx.ndim != 2 or rank_idx.shape[0] != self.num_E or rank_idx.shape[1] < 1:
            raise ValueError(
                f"predictLikelyTargetNew rank_idx 形状异常: {getattr(rank_idx, 'shape', None)}，"
                f"期望第一维为 num_E={self.num_E} 且至少一列目标"
            )
        best_fac = rank_idx[:, 0].astype(np.int64, copy=False)
        inferred = self.ValuePos[best_fac, :2].astype(np.float32)
        return inferred, rank_idx

    def _compute_isomap_intercept_candidates(self) -> bool:
        """Build iso maps and intercept table; cache geometry for _apply_paths_from_assigned_rows. Raises if no valid IsoPairs."""
        self._inferred_targets_e, rank_idx = self._predict_facility_ranks_for_e()
        validnew = rank_idx[:, 0]
        pairs_e2val_ = np.column_stack((np.arange(len(validnew)), validnew))
        pairs_e2val=pairs_e2val_[self.UnCapEidNew, :]

        _, near_tpid_e = obtainNearETP(self.Trans_Point[:, :2], self.PosE, self.ValuePos, pairs_e2val)

        self.UnCapEid=self.UnCapEidNew
        self.UnCapPid=self.UnCapPidNew


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


        self.num_E_ = len(self.IsoMap_i_tt_E2Iso)   
        self.num_P_ = len(self.IsoMap_i_tt_P2Iso)

        eids, pids, te_idx, tp_idx = np.meshgrid(
            self.UnCapEid, self.UnCapPid, np.arange(time_l), np.arange(time_l), indexing="ij"
        )

        idxE, idxP, te_idx, tp_idx = np.meshgrid(
            np.arange(self.num_E_),np.arange(self.num_P_),  np.arange(time_l), np.arange(time_l),
            indexing='ij'
        )

        distances = self._pairwise_dist()
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
            idxE.ravel(),
            idxP.ravel(),
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
        self._pairs_ic_compact = np.asarray(pairs_realE2P_, dtype=np.int64).copy()

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

        #TODO:如果pairs_realE2P与UncapPid和UncapEid不一致，则需要补充对应的兜底策略，
        # 兜底：对 Hungarian 未覆盖的 UnCapPid/UnCapEid 生成“追捕者到敌机当前位置”的候选与配对。
        # 兜底候选使用 -2 标记（TPid=-2/cost=-2 等），reward 中会按 -2 触发每步惩罚。
        if self.pairs_realE2P is not None:
            assigned_pids = {int(x) for x in np.asarray(self.pairs_realE2P[:, 1], dtype=np.int64).ravel()}
            assigned_eids = {int(x) for x in np.asarray(self.pairs_realE2P[:, 0], dtype=np.int64).ravel()}
            no_ans_pids = sorted({int(x) for x in np.asarray(self.UnCapPid, dtype=np.int64).ravel()} - assigned_pids)
            no_ans_eids = sorted({int(x) for x in np.asarray(self.UnCapEid, dtype=np.int64).ravel()} - assigned_eids)
        else:
            no_ans_pids = []
            no_ans_eids = []

        if len(no_ans_pids) > 0 and len(no_ans_eids) > 0:
            pos_p_full, pos_e_full = self._positions_full_for_obs()
            pairs = [(pid, eid) for pid in no_ans_pids for eid in no_ans_eids]
            posp_batch = np.stack([pos_p_full[pid] for pid, _ in pairs], axis=0)
            pose_batch = np.stack([pos_e_full[eid] for _, eid in pairs], axis=0)

            path_list, _iso_map, _end_time = obtainPE2IsoPath(posp_batch, pose_batch, self.Map, float(self.v_P))
            if path_list is None or len(path_list) != len(pairs):
                raise RuntimeError(f"fallback obtainPE2IsoPath returned {0 if path_list is None else len(path_list)} paths for {len(pairs)} pairs")

            path_lens = np.zeros((len(pairs),), dtype=np.int64)
            fallback_paths = {}
            for i, (pid, eid) in enumerate(pairs):
                path = path_list[i]
                if path is None:
                    raise RuntimeError(f"fallback path is None for pid={pid}, eid={eid}")
                path = np.asarray(path, dtype=float)
                if path.ndim != 2 or path.shape[0] < 3 or path.shape[1] < 2:
                    raise RuntimeError(f"fallback path shape invalid for pid={pid}, eid={eid}: {path.shape}")
                path_lens[i] = int(path.shape[1])
                fallback_paths[(int(pid), int(eid))] = path

            cost_mat = path_lens.reshape(len(no_ans_pids), len(no_ans_eids)).astype(np.float64)
            p_idx, e_idx = linear_sum_assignment(cost_mat)

            # 兜底配对（全局 pid/eid）
            fb_pairs = [(int(no_ans_pids[i]), int(no_ans_eids[j])) for i, j in zip(p_idx, e_idx)]

            # 合并 pairs_realE2P：replace_all = 原 Hungarian + 兜底配对
            merged_pairs = []
            merged_compact = []
            if self.pairs_realE2P is not None and self.pairs_realE2P.size > 0:
                for row_i in range(self.pairs_realE2P.shape[0]):
                    merged_pairs.append((int(self.pairs_realE2P[row_i, 0]), int(self.pairs_realE2P[row_i, 1])))
                    if self._pairs_ic_compact is None:
                        raise RuntimeError("_pairs_ic_compact is None while pairs_realE2P is not None")
                    merged_compact.append((int(self._pairs_ic_compact[row_i, 0]), int(self._pairs_ic_compact[row_i, 1])))

            for pid, eid in fb_pairs:
                if (eid, pid) in merged_pairs:
                    continue
                # compact 索引：在当前 UnCap 列表中的位置
                cpid = int(np.where(np.asarray(self.UnCapPid, dtype=int) == pid)[0][0])
                ceid = int(np.where(np.asarray(self.UnCapEid, dtype=int) == eid)[0][0])
                merged_pairs.append((eid, pid))
                merged_compact.append((ceid, cpid))

            self.pairs_realE2P = np.asarray(merged_pairs, dtype=np.int64)
            self._pairs_ic_compact = np.asarray(merged_compact, dtype=np.int64)

            # 兜底候选行追加到 ICFinalActionCandidates；保持列数与 obtainTask_timeShift2 对齐（至少 25 列）
            ncols = int(self.IC_candidates.shape[1]) if self.IC_candidates is not None and self.IC_candidates.ndim == 2 else 0
            if ncols < 25:
                raise RuntimeError(f"IC_candidates has {ncols} columns (<25); cannot append fallback candidates")

            num_time = int(self.Map.get("numTime"))
            tp_max = max(0, num_time - 1)
            fb_rows = []
            for pid, eid in fb_pairs:
                cpid = int(np.where(np.asarray(self.UnCapPid, dtype=int) == pid)[0][0])
                ceid = int(np.where(np.asarray(self.UnCapEid, dtype=int) == eid)[0][0])
                path = fallback_paths[(pid, eid)]
                L = int(path.shape[1])
                tp = int(round((float(L) / float(self.v_P)) / float(self.timeIsoRes)))
                tp = max(0, min(tp, tp_max))
                dist_pe = float(np.hypot(pos_p_full[pid, 0] - pos_e_full[eid, 0], pos_p_full[pid, 1] - pos_e_full[eid, 1]))

                row = np.zeros((ncols,), dtype=float)
                row[:] = 0.0
                row[0] = 0.0  # te
                row[1] = float(tp)  # tp
                row[2] = -2.0  # ETPid
                row[3] = -2.0  # PTPid
                row[4] = 0.0  # IsoPosidxE
                row[5] = 0.0  # IsoPosidxP
                row[6] = dist_pe  # Iso_dist
                row[7] = float(L)  # path_len
                row[8] = -2.0  # cost
                row[9] = -2.0  # ValPosid
                row[10] = -2.0  # TPid (fallback path)
                row[11] = float(ceid)  # Eid_ref (compact)
                row[12] = float(cpid)  # Pid_ref (compact)
                row[13] = 0.0  # Eiso
                row[14] = 0.0  # Piso (unused)
                # costAll / rewards fields use -2 sentinel (reward will penalize)
                if ncols >= 25:
                    row[15:25] = -2.0
                fb_rows.append(row)

            if len(fb_rows) > 0:
                self.ICFinalActionCandidates = np.vstack([self.ICFinalActionCandidates, np.vstack(fb_rows)])
                self._fallback_paths = dict(fallback_paths)

        # 应该是model获得的action,而不是直接使用任务分配的结果
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
        # fallback: 若该行来自兜底候选（ETPid/ValPosid 使用 -2 标记），则保持上一轮的 PathE 不变
        order_e = np.argsort(e_idx)
        for out_i, row_i in enumerate(order_e):
            if int(etp_idx[row_i]) == -2 or int(vp_idx[row_i]) == -2:
                continue
            self.PathE[out_i] = path_e_active[row_i]

        p_idx2 = assigned[:, 12].astype(int)
        tp_to_idx = assigned[:, 10].astype(int)
        iso_p_idx = assigned[:, 14].astype(int)
        tp_from_idx = assigned[:, 3].astype(int)
        path_p_active = [
            (
                self._fallback_paths[
                    (
                        int(self.UnCapPid[int(p_idx2[i])]),
                        int(self.UnCapEid[int(assigned[i, 11])]),
                    )
                ]
                if tp_to_idx[i] == -2
                else np.hstack(
                    (
                        path_p2tp[idx],
                        self.pathFinalMapPTP2Iso[tp_from_idx[i]][tp_to_idx[i]][
                            :, : max(0, int(iso_p_idx[i] - path_p2tp[idx].shape[1]))
                        ],
                    )
                )
                if tp_to_idx[i] >= 0
                else path_p2tp[idx][:, : int(iso_p_idx[i])]
            )
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
        # 如果有捕获，但是分配不改变，则会
        dt = self.sim_dt
        self.t += dt
        self.t_all += dt
        if self.pairs_realE2P is not None and self.pairs_realE2P.shape[0] == self.Capflag.shape[0]:
            keep = ~self.Capflag
            self.UnCapPidNew = self.pairs_realE2P[keep, 1].astype(int)
            self.UnCapEidNew = self.pairs_realE2P[keep, 0].astype(int)
            self.pairs_realE2P = self.pairs_realE2P[keep]
            self._pairs_ic_compact = self._pairs_ic_compact[keep]


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
            # pos_e, path_pre = self._dwa_pos_e_and_path_pre()
            min_dists, _ = obtainDWAprePath(self.PosE, self.PathEpre, self.E_PreRef)
            # TODO:测试这个数值
            need_replan = not bool(np.all(min_dists <= 50))
        return need_replan, False

    def _update_capflag_from_geometry(self) -> int:
        """按最近追捕者更新 Capflag；返回本步新增捕获数。"""
        distances_now = self._pairwise_dist()
        assignments_now = np.argmin(distances_now, axis=0)
        p_for_e_now = self.PosP
        e_now = self.PosE
        dx_now = e_now[:, 0] - p_for_e_now[:, 0]
        dy_now = e_now[:, 1] - p_for_e_now[:, 1]
        ang_pe_now = np.degrees(np.arctan2(dy_now, dx_now))
        angle_diff_now = (ang_pe_now - np.degrees(p_for_e_now[:, 2]) + 180.0) % 360.0 - 180.0
        cap_dist = float(self.CapRef["CapDist"])
        cap_angle_half = float(self.CapRef["CapAngle"]) / 2.0
        self.Capflag = (np.min(distances_now, axis=0) <= cap_dist) & (np.abs(angle_diff_now) <= cap_angle_half)

    def _apply_assignment_from_action(self, action_indices: np.ndarray) -> None:
        """Apply RL-chosen candidate rows instead of Hungarian (requires valid isomap cache).

        `action_indices` 为每个全局 pid 的离散候选下标（与 train 中 Categorical.sample() 一致），非概率向量。
        仅 ``pairs_realE2P`` 中仍存活的 (eid, pid) 会取候选并应用；无配对的 pid 忽略。
        """
        if self._path_e2tp_cache is None:
            raise ValueError("path_e2tp_cache is None")
        if self.pairs_realE2P is None or self.pairs_realE2P.size == 0:
            raise ValueError("pairs_realE2P is None")
        obs = self._build_obs()
        self._sync_dynamic_k(obs)
        mask = np.asarray(obs["self_pts_mask"], dtype=np.float32)
        cols = self.obs_generator.cols
        assigned_rows = []
        pr = np.asarray(self.pairs_realE2P, dtype=np.int64)
        pic = self._pairs_ic_compact
        for row_i in range(pr.shape[0]):
            eid, pid = int(pr[row_i, 0]), int(pr[row_i, 1])
            ceid, c_pid = int(pic[row_i, 0]), int(pic[row_i, 1])
            subset = self.ICFinalActionCandidates[
                (self.ICFinalActionCandidates[:, cols.eid_ref].astype(int) == ceid)
                & (self.ICFinalActionCandidates[:, cols.pid_ref].astype(int) == c_pid)
            ]
            n = int(subset.shape[0])
            idx = int(action_indices[pid])
            if idx < 0 or idx >= n:
                raise IndexError(f"action index {idx} out of range for pid {pid} (eid={eid}, n_candidates={n})")
            if mask[pid, idx] <= 0:
                raise ValueError(f"invalid action: mask[{pid}, {idx}] is zero for masked candidate")
            assigned_rows.append(subset[idx])
        assigned = np.vstack(assigned_rows)
        self.ICFinalAction = assigned.copy()
        self._apply_paths_from_assigned_rows(assigned)

    def _pairwise_dist(self):
        pp = self.PosP[:, :2]
        ee = self.PosE[:, :2]
        return np.linalg.norm(pp - ee, axis=1)

    def _extract_candidate_pos(self, row: np.ndarray, pid: int) -> Tuple[float, float, float]:
        # 兜底候选：直接取“敌机当前位置”作为拦截点位置（与兜底策略定义一致）
        # 标志位：TPid == -2 或 cost == -2
        if row is not None and row.shape[0] > 12:
            if int(row[10]) == -2 or int(row[8]) == -2:
                cols = self.obs_generator.cols
                ceid = int(row[cols.eid_ref]) if row.shape[0] > cols.eid_ref else int(row[11])
                ceid = max(0, min(ceid, int(len(self.UnCapEid) - 1)))
                eid_global = int(self.UnCapEid[ceid])
                x, y, th = self._pos_e_xyz_for_global_eid(eid_global)
                return float(x), float(y), float(th)

        # 尝试使用 IsoMap 中的 IsoPos（与 _extract_iso_points 行为一致）
        if hasattr(self, "IsoMap_i_tt_P2Iso") and row.shape[0] > 6:
            # 常见布局（见 obtainRLOutput）: te=0, tp=1, IsoIdxE=4, IsoIdxP=5, eid=11, pid=12
            tp_idx = int(row[1]) if row.shape[0] > 1 else 0
            iso_idx = int(row[5]) if row.shape[0] > 5 else (int(row[4]) if row.shape[0] > 4 else 0)

            cols = self.obs_generator.cols
            pid_ref = int(row[cols.pid_ref]) if row.shape[0] > cols.pid_ref else pid

            iso_obj = self.IsoMap_i_tt_P2Iso[pid_ref][tp_idx]
            if iso_obj is not None and hasattr(iso_obj, "IsoPos") and iso_obj.IsoPos is not None:
                iso_pos = np.asarray(iso_obj.IsoPos)
                if iso_pos.ndim == 2:
                    iso_idx = max(0, min(iso_idx, iso_pos.shape[1] - 1))
                    return float(iso_pos[0, iso_idx]), float(iso_pos[1, iso_idx]), float(iso_pos[2, iso_idx])

        # 回退：使用敌方位置与朝向（兼容老格式 / 无 IsoMap 情况）
        cols = self.obs_generator.cols
        if row.shape[0] > cols.eid_ref:
            eid = int(row[cols.eid_ref])
        else:
            eid = int(row[11]) if row.shape[0] > 11 else pid % self.num_E
        eid = max(0, min(eid, self.num_E - 1))
        x, y, th = self._pos_e_xyz_for_global_eid(eid)
        return float(x), float(y), float(th)

    def _pos_e_xyz_for_global_eid(self, eid: int):
        """全局 eid → 平面位置+朝向；紧凑 PosE 时用 UnCapEidNew 行映射，已移除敌用 Evader 初值。"""
        eid = int(max(0, min(int(eid), self.num_E - 1)))
        if self.PosE.shape[0] == self.num_E:
            pe = self.PosE[eid]
            return float(pe[0]), float(pe[1]), float(pe[2])
        idx = np.where(np.asarray(self.UnCapEidNew, dtype=int) == eid)[0]
        if idx.size > 0:
            pe = self.PosE[int(idx[0])]
            return float(pe[0]), float(pe[1]), float(pe[2])
        ev = self.Evader[eid]
        return float(ev[0]), float(ev[1]), float(ev[2])

    def _positions_full_for_obs(self):
        """scatter 到全局槽位后，将非存活 id 的槽位置零，避免已拦截机几何进入 v_p/v_e 中间量。"""
        if self.PosP.shape[0] == self.num_P and self.PosE.shape[0] == self.num_E:
            out_p = np.asarray(self.PosP, dtype=float).copy()
            out_e = np.asarray(self.PosE, dtype=float).copy()
        else:
            out_p = np.asarray(self.PStart_Point, dtype=float).copy()
            for i in range(self.PosP.shape[0]):
                pid = int(self.UnCapPidNew[i])
                out_p[pid] = self.PosP[i]
            out_e = np.asarray(self.Evader[:, :3], dtype=float).copy()
            for i in range(self.PosE.shape[0]):
                eid = int(self.UnCapEidNew[i])
                out_e[eid] = self.PosE[i]
        alive_p = {int(x) for x in np.asarray(self.UnCapPidNew, dtype=np.int64).ravel()}
        alive_e = {int(x) for x in np.asarray(self.UnCapEidNew, dtype=np.int64).ravel()}
        for pid in range(self.num_P):
            if pid not in alive_p:
                out_p[pid] = 0.0
        for eid in range(self.num_E):
            if eid not in alive_e:
                out_e[eid] = 0.0
        return out_p, out_e

    def _build_obs(self):
        pos_p_obs, pos_e_obs = self._positions_full_for_obs()
        if self._inferred_targets_e is None:
            raise ValueError("_inferred_targets_e 为 None，构建观测前必须先完成目标推断")

        inferred_targets = np.asarray(self._inferred_targets_e, dtype=np.float32)
        
        if inferred_targets.shape != (self.num_E, 2):
            raise ValueError(
                f"_inferred_targets_e 形状应为 ({self.num_E}, 2)，实际为 {inferred_targets.shape}"
            )
        return self.obs_generator.generate(
            pos_p=pos_p_obs,
            pos_e=pos_e_obs,
            v_p=self.v_P,
            v_e=self.v_E,
            value_pos=self.ValuePos,
            num_p=self.num_P,
            num_e=self.num_E,
            ic_candidates=self.ICFinalActionCandidates,
            candidate_pos_fn=self._extract_candidate_pos,
            inferred_targets=inferred_targets,
            pairs_realE2P=self.pairs_realE2P,
            pairs_ic_ref=self._pairs_ic_compact,
            capflag=self.Capflag,
        )

    def _normalize_action(self, action_dict) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """解析策略输出：返回 (one-hot 权重, 每智能体离散索引, 决策时刻观测快照)。

        训练脚本通常传入 **采样后的索引** ``(num_P,)`` int；亦兼容 logits/概率矩阵。
        无任务机（``pursuer_active[i]==0``）：**动作索引为 -1**，``norm`` 对应行为全零。
        快照用于计奖：仿真内层可能重规划并改变候选数 K，必须与选动作时的 mask 一致。
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

        active = np.asarray(obs["pursuer_active"], dtype=np.int8).reshape(-1)
        if active.shape[0] != self.num_P:
            raise ValueError(f"pursuer_active 长度应为 {self.num_P}，实际 {active.shape}")

        norm = np.zeros((self.num_P, k_curr), dtype=np.float32)
        for i in range(self.num_P):
            if int(active[i]) == 0:
                idx[i] = -1
                continue
            valid_idx = np.where(mask[i] > 0)[0]
            if valid_idx.size == 0:
                raise ValueError(f"no valid action candidates for pursuer {i} (self_pts_mask row is all zero)")
            sel = int(idx[i])
            if sel < 0 or sel >= k_curr or mask[i, sel] <= 0:
                raise ValueError(f"invalid normalized action index {sel} for pursuer {i} (k={k_curr})")
            norm[i, sel] = 1.0
        return norm, idx, obs

    def _compute_rewards(
        self,
        action_indices: np.ndarray,
        sim_time_elapsed: float,
        obs_at_action: Dict[str, np.ndarray],
    ):
        """`action_indices` 相对于 ``obs_at_action`` 中的候选掩码；几何类项用当前态势。"""
        curr_min_dist = self._pairwise_dist()
        curr_obs = self._build_obs()
        obs = dict(curr_obs)
        for key in ("reward_nodes", "self_pts_mask", "self_pts", "pursuer_active"):
            obs[key] = obs_at_action[key]
        sd = float(self.sim_dt)
        rewards, details = self.reward_fn.compute_step_rewards(
            actions=action_indices,
            obs=obs,
            curr_min_dist=curr_min_dist,
            last_min_dist=self.last_min_dist,
            num_p=self.num_P,
            sim_time_elapsed=float(sim_time_elapsed),
            sim_dt=sd,
            return_details=True,
        )

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
        """与 ``main/main0319.py`` 563–632 行一致：地图 + Evader 起点 + 未捕获对的姿态/预测轨迹 + DWA BestPaths + 灰色真实历史轨迹。"""
        if self.render_mode == "none":
            return None

        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))

        ax = self.ax
        ax.cla()
        Draw_map(
            self.PStart_Point,
            self.Trans_Point,
            self.ValuePos,
            self.obs,
            self.sure,
            self.obs_no_circle,
            self.obs_no_circle_in,
            ax=ax,
        )

        color_map = mpl.colormaps["tab20"]
        ax.scatter(
            self.Evader[:, 0],
            self.Evader[:, 1],
            c="gray",
            marker="d",
            s=20,
            alpha=0.5,
            label="Evader Start",
        )

        n_active = len(self.UnCapEidNew)
        if n_active > 0:
            if self.PosP.shape[0] != n_active or self.PosE.shape[0] != n_active:
                raise RuntimeError(
                    f"render: PosP/PosE 行数应与未捕获对数一致， got PosP={self.PosP.shape[0]}, "
                    f"PosE={self.PosE.shape[0]}, len(UnCapEidNew)={n_active}"
                )
            if len(self.PathEpre) != n_active:
                raise RuntimeError(
                    f"render: PathEpre 长度应为 {n_active}, got {len(self.PathEpre)}"
                )
            _, best_paths = obtainDWAprePath(self.PosE, self.PathEpre, self.E_PreRef)
            if len(best_paths) != n_active:
                raise RuntimeError(
                    f"obtainDWAprePath 返回 BestPaths 长度 {len(best_paths)} != {n_active}"
                )

            for idx, eid in enumerate(self.UnCapEidNew):
                eid = int(eid)
                pid = int(self.UnCapPidNew[idx])
                color_p = color_map((idx * 2) % 20)
                color_e = color_map((idx * 2 + 1) % 20)

                ax.plot(
                    self.PosP[idx, 0],
                    self.PosP[idx, 1],
                    "o",
                    color=color_p,
                    markersize=8,
                    markeredgecolor="w",
                    alpha=1,
                )
                ax.quiver(
                    self.PosP[idx, 0],
                    self.PosP[idx, 1],
                    150 * np.cos(self.PosP[idx, 2]),
                    150 * np.sin(self.PosP[idx, 2]),
                    color=color_p,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.004,
                    alpha=0.8,
                )
                ax.plot(
                    self.PosE[idx, 0],
                    self.PosE[idx, 1],
                    "d",
                    color=color_e,
                    markersize=8,
                    markeredgecolor="w",
                    alpha=1,
                )
                ax.quiver(
                    self.PosE[idx, 0],
                    self.PosE[idx, 1],
                    150 * np.cos(self.PosE[idx, 2]),
                    150 * np.sin(self.PosE[idx, 2]),
                    color=color_e,
                    angles="xy",
                    scale_units="xy",
                    scale=1,
                    width=0.004,
                    alpha=0.8,
                )

                path_p = self.PathP[pid]
                path_e = self.PathE[eid]
                ax.plot(path_p[0, :], path_p[1, :], "-", color=color_p, linewidth=2, alpha=1)
                ax.plot(path_e[0, :], path_e[1, :], "--", color=color_e, linewidth=2, alpha=1)

                bp = best_paths[idx]
                ax.plot(bp[0, :], bp[1, :], "-", color=color_p, linewidth=2.5, zorder=3)

        len_e_true = int(self.t_all * self.v_E)
        len_p_true = int(self.t_all * self.v_P)
        for i_plot in range(len(self.PathE2Val_true)):
            ax.plot(
                self.PathE2Val_true[i_plot][:len_e_true, 0],
                self.PathE2Val_true[i_plot][:len_e_true, 1],
                "-",
                color="gray",
                linewidth=2.5,
                alpha=0.3,
                zorder=1,
            )
            ax.plot(
                self.PathPtrue[i_plot][0, :len_p_true],
                self.PathPtrue[i_plot][1, :len_p_true],
                "-",
                color="gray",
                linewidth=2.5,
                alpha=0.3,
                zorder=1,
            )

        ax.set_title(f"Simulation Time: {self.t_all:.2f}")
        ax.grid(True, linestyle="--", alpha=0.5)

        if self.render_mode == "human":
            plt.pause(0.001)
            return None

        self.fig.canvas.draw()
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return rgba[:, :, :3].copy()

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