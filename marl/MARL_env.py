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


@dataclass
class StepStats:
    replanned: bool = False
    collision: bool = False
    captured: int = 0
    inner_ticks: int = 0
    forced_decision: bool = False


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
        self.k_max = int(self.config.get("k_max", 32))
        self.render_mode = self.config.get("render_mode", "none")
        self.allow_dummy_if_missing = bool(self.config.get("allow_dummy_if_missing", True))
        self.enable_dwa_replan = bool(self.config.get("enable_dwa_replan", True))
        # step_mode="decision" makes one RL step align with one online decision event (replan).
        self.step_mode = str(self.config.get("step_mode", "decision")).lower()
        self.max_inner_ticks = int(self.config.get("max_inner_ticks", 200))
        self.max_episode_steps = int(self.config.get("max_episode_steps", 600))
        self.entropy_lambda = float(self.config.get("entropy_lambda", 0.1))
        self.dist_reward_scale = float(self.config.get("dist_reward_scale", 0.05))
        self.collision_dist = float(self.config.get("collision_dist", 25.0))

        self._load_assets()
        self.obs_generator = TODCObservationGenerator(k_max=self.k_max)
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

    def _build_spaces(self):
        self.action_space = spaces.MultiDiscrete(np.full((self.num_P,), self.k_max, dtype=np.int64))
        self.observation_space = spaces.Dict(
            {
                "pursuers": spaces.Box(-np.inf, np.inf, shape=(self.num_P, 6), dtype=np.float32),
                "evaders": spaces.Box(-np.inf, np.inf, shape=(self.num_E, 8), dtype=np.float32),
                "candidates": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.k_max, 6), dtype=np.float32),
                "candidate_mask": spaces.Box(0.0, 1.0, shape=(self.num_P, self.k_max), dtype=np.float32),
                "V_P": spaces.Box(-np.inf, np.inf, shape=(self.num_P, 6), dtype=np.float32),
                "V_E": spaces.Box(-np.inf, np.inf, shape=(self.num_E, 8), dtype=np.float32),
                "V_C": spaces.Box(-np.inf, np.inf, shape=(self.num_P, self.k_max, 6), dtype=np.float32),
                "V_C_mask": spaces.Box(0.0, 1.0, shape=(self.num_P, self.k_max), dtype=np.float32),
            }
        )

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
            if not self.allow_dummy_if_missing:
                raise FileNotFoundError(
                    f"Cannot resolve static map assets in {self.map_root}. map_folder={selected_map_folder}"
                )
            self.real_mode = False
            self._init_dummy_world()
            return

        map_base = os.path.join(self.map_root, selected_map_folder)
        resolved = {k: os.path.join(map_base, v) for k, v in map_names.items()}
        self.time_map = selected_map_folder
        self.evader_profile_dirs = self._discover_evader_profile_dirs(map_base)

        if len(self.evader_profile_dirs) == 0:
            if not self.allow_dummy_if_missing:
                raise FileNotFoundError(
                    f"No evader profile found under {map_base}. Expected map/<TimeMap>/<slot>/PathE2Val_true.jbl"
                )
            self.real_mode = False
            self._init_dummy_world()
            return

        self.real_mode = True
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
        self.timeIsoRes = self.Map["timeIsoRes"]

        self.Evader0 = np.asarray(
            self.config.get(
                "evader_init",
                [[200, 1950, -np.pi / 2], [1000, 1950, -np.pi / 2], [1800, 1950, -np.pi / 2]],
            ),
            dtype=float,
        )

        self._load_evader_profile(self.evader_profile_dirs[0])

        self.num_E = int(self.Evader0.shape[0])
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

    def _init_dummy_world(self):
        self.Stepsize = 0.05
        self.v_P = 25.0
        self.v_E = 18.0
        self.timeIsoRes = 1.0
        self.PStart_Point = np.array(
            [[200.0, 120.0, np.pi / 2], [500.0, 120.0, np.pi / 2], [800.0, 120.0, np.pi / 2]], dtype=float
        )
        self.Evader0 = np.array(
            [[220.0, 900.0, -np.pi / 2], [520.0, 900.0, -np.pi / 2], [820.0, 900.0, -np.pi / 2]], dtype=float
        )
        self.Trans_Point = np.array(
            [[150.0, 500.0, 0.0], [500.0, 500.0, 0.0], [850.0, 500.0, 0.0]], dtype=float
        )
        self.ValuePos = np.array(
            [[200.0, 60.0, -np.pi / 2], [500.0, 60.0, -np.pi / 2], [800.0, 60.0, -np.pi / 2]], dtype=float
        )
        self.obs = []
        self.sure = 20
        self.obs_no_circle = []
        self.obs_no_circle_in = []
        self.obs_polygons = []

        horizon = 800
        self.PathE2Val_true = []
        for i in range(self.Evader0.shape[0]):
            xs = np.full(horizon, self.Evader0[i, 0])
            ys = np.linspace(self.Evader0[i, 1], self.ValuePos[i, 1], horizon)
            yaw = -np.pi / 2 * np.ones(horizon)
            self.PathE2Val_true.append(np.column_stack((xs, ys, yaw)))

        self.length_E_max = horizon
        self.num_E = int(self.Evader0.shape[0])
        self.num_P = int(self.PStart_Point.shape[0])
        self.agents = [f"p_{i}" for i in range(self.num_P)]

        cap_dist = float(self.config.get("cap_dist", 120.0))
        self.CapRef = {
            "CapDist": cap_dist,
            "CapAngle": float(self.config.get("cap_angle", 90.0)),
            "Stepsize": self.Stepsize,
            "CapDistRef": float(self.config.get("cap_dist_ref", 250.0)),
            "v_P": self.v_P,
            "v_E": self.v_E,
            "timeIsoRes": self.timeIsoRes,
            "obs_polygons": self.obs_polygons,
        }
        self.E_PreRef = {
            "num_v": 10,
            "num_w": 10,
            "v_range": [50, 100],
            "w_range": [-np.pi / 6, np.pi / 6],
            "Stepsize": self.Stepsize,
            "T_pred": 1,
        }

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if self.real_mode:
            profile_dir = self._select_profile_dir(options)
            self._load_evader_profile(profile_dir)
            self.current_profile = os.path.relpath(profile_dir, self.map_root)

        self.episode_step = 0
        self.decision_step = 0
        self.t = 0.0
        self.t_all = 0.0

        self.PosE = self.Evader0.copy()
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

        self.IC = np.empty((0, 18), dtype=float)
        self.IC_candidates = np.empty((0, 18), dtype=float)
        self.ICFinal = np.empty((0, 18), dtype=float)

        if self.real_mode:
            self._try_replan()
        else:
            self._dummy_candidates()

        self.last_min_dist = self._pairwise_dist().min(axis=1)
        obs = self._build_obs()
        info = {
            "real_mode": self.real_mode,
            "replanned": bool(self.real_mode),
            "num_candidates": int(self.IC_candidates.shape[0]),
            "time_map": self.time_map,
            "profile": self.current_profile,
        }
        return obs, info

    def step(self, action_dict):
        actions = self._normalize_action(action_dict)
        self.last_action_weights = actions
        self.last_action_indices = np.argmax(actions, axis=1).astype(np.int64)

        stats = StepStats(replanned=False, collision=False, captured=0)

        def should_replan_now() -> bool:
            if not (self.real_mode and self.enable_dwa_replan):
                return False
            try:
                path_e_pre = [self.PathE[eid] for eid in range(self.num_E)]
                min_dists, _ = obtainDWAprePath(self.PosE, path_e_pre, self.E_PreRef)
                return not np.all(min_dists <= 10)
            except Exception:
                return False

        def update_capture_and_collision() -> bool:
            collision_now = self._check_collision()
            stats.collision = bool(collision_now)

            distances_now = self._pairwise_dist()
            assignments_now = np.argmin(distances_now, axis=0)
            p_for_e_now = self.PosP[assignments_now]
            e_now = self.PosE

            dx_now = e_now[:, 0] - p_for_e_now[:, 0]
            dy_now = e_now[:, 1] - p_for_e_now[:, 1]
            ang_pe_now = np.degrees(np.arctan2(dy_now, dx_now))
            angle_diff_now = (ang_pe_now - np.degrees(p_for_e_now[:, 2]) + 180.0) % 360.0 - 180.0

            cap_dist = self.CapRef["CapDist"]
            cap_angle_half = self.CapRef["CapAngle"] / 2.0
            prev_cap = self.Capflag.copy()
            self.Capflag = (np.min(distances_now, axis=0) <= cap_dist) & (np.abs(angle_diff_now) <= cap_angle_half)
            stats.captured += int(np.sum(~prev_cap & self.Capflag))
            return bool(np.all(self.Capflag) or collision_now)

        decision_mode = self.step_mode == "decision"
        inner_limit = max(1, self.max_inner_ticks)

        if decision_mode:
            for _ in range(inner_limit):
                self.t += self.time_res / self.Stepsize
                self.t_all += self.time_res / self.Stepsize
                stats.inner_ticks += 1

                if should_replan_now():
                    self._try_replan()
                    stats.replanned = True
                    self.decision_step += 1

                if self.real_mode:
                    self._advance_from_paths()
                else:
                    self._advance_dummy(actions)

                done_inner = update_capture_and_collision()
                if stats.replanned or done_inner:
                    break

            if (not stats.replanned) and (not np.all(self.Capflag)) and (not stats.collision):
                stats.forced_decision = True

            self.episode_step += 1
        else:
            self.t += self.time_res / self.Stepsize
            self.t_all += self.time_res / self.Stepsize
            stats.inner_ticks = 1

            if should_replan_now():
                self._try_replan()
                stats.replanned = True
                self.decision_step += 1

            if self.real_mode:
                self._advance_from_paths()
            else:
                self._advance_dummy(actions)

            update_capture_and_collision()
            self.episode_step += 1

        rewards = self._compute_rewards(actions)

        if stats.captured > 0:
            for i in range(self.num_P):
                rewards[f"p_{i}"] += 100.0 * stats.captured / max(1, self.num_E)

        done_all = bool(np.all(self.Capflag) or stats.collision)
        truncated_all = bool(self.episode_step >= self.max_episode_steps)

        obs = self._build_obs()
        terminations = {f"p_{i}": done_all for i in range(self.num_P)}
        truncations = {f"p_{i}": truncated_all for i in range(self.num_P)}
        terminations["__all__"] = done_all
        truncations["__all__"] = truncated_all

        infos = {
            f"p_{i}": {
                "replanned": stats.replanned,
                "collision": stats.collision,
                "captured_total": int(np.sum(self.Capflag)),
                "num_candidates": int(self.IC_candidates.shape[0]),
                "inner_ticks": int(stats.inner_ticks),
                "forced_decision": bool(stats.forced_decision),
                "decision_step": int(self.decision_step),
                "step_mode": self.step_mode,
            }
            for i in range(self.num_P)
        }
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

    def _advance_dummy(self, actions: np.ndarray):
        targets = np.zeros((self.num_P, 2), dtype=float)
        obs_now = self._build_obs()
        cand_nodes = obs_now["V_C"]
        cand_mask = obs_now["V_C_mask"]
        for pid in range(self.num_P):
            c_feat, c_mask = cand_nodes[pid], cand_mask[pid]
            valid = c_mask > 0
            if np.any(valid):
                w = actions[pid, :]
                w = w * c_mask
                if np.sum(w) <= 1e-8:
                    w = c_mask / np.sum(c_mask)
                else:
                    w = w / np.sum(w)
                targets[pid] = np.sum(w[:, None] * c_feat[:, :2], axis=0)
            else:
                targets[pid] = self.PosE[pid % self.num_E, :2]

        dt = self.time_res
        for pid in range(self.num_P):
            vec = targets[pid] - self.PosP[pid, :2]
            dist = np.linalg.norm(vec)
            if dist > 1e-6:
                direction = vec / dist
            else:
                direction = np.array([np.cos(self.PosP[pid, 2]), np.sin(self.PosP[pid, 2])])
            step_len = min(dist, self.v_P * dt)
            self.PosP[pid, 0:2] += direction * step_len
            self.PosP[pid, 2] = np.arctan2(direction[1], direction[0])
            self.PathPtrue[pid] = np.hstack((self.PathPtrue[pid], self.PosP[pid].reshape(-1, 1)))

        for eid in range(self.num_E):
            target = self.ValuePos[eid % len(self.ValuePos), :2]
            vec = target - self.PosE[eid, :2]
            dist = np.linalg.norm(vec)
            if dist > 1e-6:
                direction = vec / dist
            else:
                direction = np.array([0.0, -1.0])
            step_len = min(dist, self.v_E * dt)
            self.PosE[eid, 0:2] += direction * step_len
            self.PosE[eid, 2] = np.arctan2(direction[1], direction[0])

    def _try_replan(self):
        try:
            self._replan_with_isomap()
        except Exception:
            self._dummy_candidates()

    def _replan_with_isomap(self):
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
            self._dummy_candidates()
            return

        iso_pairs = np.array(results, dtype=object)
        iso_pairs = iso_pairs[np.argsort(iso_pairs[:, 0].astype(int))]

        intercept = obtainTask_timeShift2(iso_pairs, self.IsoMap_i_tt_P2Iso, self.IsoMap_i_tt_E2Iso, self.CapRef)
        self.IC = intercept.copy()

        unique_pairs = np.unique(intercept[:, [11, 12]], axis=0)
        best = []
        for pair in unique_pairs:
            eid_val, pid_val = pair
            group = intercept[(intercept[:, 11] == eid_val) & (intercept[:, 12] == pid_val)]
            group = group[np.argsort(group[:, 8])]
            n1 = max(1, int(np.ceil(len(group) * 0.5)))
            group = group[:n1]
            group = group[np.argsort(group[:, 15])[::-1]]
            n2 = max(1, int(np.ceil(len(group) * 0.5)))
            group = group[:n2]
            best_idx = np.argmax(group[:, 16])
            best.append(group[best_idx])
        ic_candidates = np.array(best)
        self.IC_candidates = ic_candidates

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
        self.ICFinal = assigned.copy()

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

        self.pairs_realE2P = np.column_stack((assigned[:, 11].astype(int), assigned[:, 12].astype(int)))
        self.t = 0.0

    def _dummy_candidates(self):
        rows = []
        for pid in range(self.num_P):
            for eid in range(self.num_E):
                pos = 0.6 * self.PosE[eid, :2] + 0.4 * self.PosP[pid, :2]
                dist = np.linalg.norm(self.PosP[pid, :2] - self.PosE[eid, :2])
                rows.append(
                    [
                        0,
                        0,
                        -1,
                        -1,
                        0,
                        0,
                        dist,
                        dist,
                        dist,
                        eid,
                        pid,
                        eid,
                        pid,
                        0,
                        0,
                        1,
                        1,
                        1,
                        pos[0],
                        pos[1],
                    ]
                )
        arr = np.array(rows, dtype=float)
        self.IC = arr[:, :18]
        self.IC_candidates = self.IC.copy()
        self.ICFinal = self.IC.copy()

    def _pairwise_dist(self):
        pp = self.PosP[:, :2]
        ee = self.PosE[:, :2]
        return np.linalg.norm(pp[:, None, :] - ee[None, :, :], axis=2)

    def _extract_candidate_pos(self, row: np.ndarray, pid: int) -> Tuple[float, float]:
        if row.shape[0] >= 20:
            return float(row[18]), float(row[19])

        if self.real_mode and hasattr(self, "IsoMap_i_tt_P2Iso") and row.shape[0] > 6:
            try:
                t_p = int(row[1])
                idx_p = int(row[5])
                pid_ref = int(row[12]) if row.shape[0] > 12 else pid
                iso_obj = self.IsoMap_i_tt_P2Iso[pid_ref][t_p]
                if iso_obj is not None and hasattr(iso_obj, "IsoPos") and iso_obj.IsoPos is not None:
                    iso_pos = np.asarray(iso_obj.IsoPos)
                    if iso_pos.ndim == 2:
                        idx_p = max(0, min(idx_p, iso_pos.shape[1] - 1))
                        return float(iso_pos[0, idx_p]), float(iso_pos[1, idx_p])
            except Exception:
                pass

        eid = int(row[11]) if row.shape[0] > 11 else pid % self.num_E
        eid = max(0, min(eid, self.num_E - 1))
        return float(self.PosE[eid, 0]), float(self.PosE[eid, 1])

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
            ic_candidates=self.IC_candidates,
            candidate_pos_fn=self._extract_candidate_pos,
            inferred_targets=inferred_targets,
        )

    def _normalize_action(self, action_dict) -> np.ndarray:
        obs = self._build_obs()
        mask = obs["candidate_mask"]

        # Internally use one-hot action weights, but accept index actions by default.
        idx = np.zeros((self.num_P,), dtype=np.int64)

        if isinstance(action_dict, dict):
            for i in range(self.num_P):
                key = f"p_{i}"
                if key not in action_dict:
                    continue
                a = np.asarray(action_dict[key]).reshape(-1)
                if a.size == 0:
                    continue
                if a.size == self.k_max:
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
                elif arr.shape[0] == self.k_max:
                    w = np.tile(arr.reshape(1, -1), (self.num_P, 1)).astype(np.float32)
                    idx = np.argmax(w * mask, axis=1).astype(np.int64)
                else:
                    idx[:] = int(arr[0])
            else:
                w = np.zeros((self.num_P, self.k_max), dtype=np.float32)
                n0 = min(self.num_P, arr.shape[0])
                n1 = min(self.k_max, arr.shape[1])
                w[:n0, :n1] = np.asarray(arr[:n0, :n1], dtype=np.float32)
                idx = np.argmax(w * mask, axis=1).astype(np.int64)

        norm = np.zeros((self.num_P, self.k_max), dtype=np.float32)
        for i in range(self.num_P):
            valid_idx = np.where(mask[i] > 0)[0]
            if valid_idx.size == 0:
                norm[i, 0] = 1.0
                continue
            sel = int(idx[i])
            if sel < 0 or sel >= self.k_max or mask[i, sel] <= 0:
                sel = int(valid_idx[0])
            norm[i, sel] = 1.0
        return norm

    def _compute_rewards(self, actions: np.ndarray) -> Dict[str, float]:
        rewards = {f"p_{i}": 0.0 for i in range(self.num_P)}
        curr_min_dist = self._pairwise_dist().min(axis=1)

        if self.last_min_dist is None:
            self.last_min_dist = curr_min_dist.copy()

        delta = self.last_min_dist - curr_min_dist
        for i in range(self.num_P):
            rewards[f"p_{i}"] += float(self.dist_reward_scale * delta[i])

        obs = self._build_obs()
        cand = obs["candidates"]
        mask = obs["candidate_mask"]
        eps = 1e-12
        for i in range(self.num_P):
            w = actions[i] * mask[i]
            s = np.sum(w)
            if s <= eps:
                continue
            w = w / s
            entropy = -np.sum(w[w > 0] * np.log(w[w > 0] + eps))
            top_idx = np.argsort(w)[-3:]
            delta_t = cand[i, top_idx, 4]
            feasible = float(np.min(delta_t) > 0.0)
            rewards[f"p_{i}"] += float(self.entropy_lambda * entropy * feasible)

        self.last_min_dist = curr_min_dist.copy()
        return rewards

    def _check_collision(self) -> bool:
        for p in self.PosP[:, :2]:
            pt = Point(float(p[0]), float(p[1]))
            for poly in self.obs_polygons:
                if poly.contains(pt):
                    return True

        d_pp = self._pairwise_self_dist(self.PosP[:, :2])
        if np.any((d_pp > 0) & (d_pp < self.collision_dist)):
            return True
        return False

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
            "allow_dummy_if_missing": True,
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
            "candidate_shape": obs["candidates"].shape,
        }
    )


if __name__ == "__main__":
    smoke_test()