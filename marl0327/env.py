"""
Dubins 拦截仿真环境（与 main/main0319forRL.py 单步循环对齐）。

用法概要:
  - reset(): 重置 episode 状态。
  - step(): 执行一次外层循环迭代（时间推进 + 条件触发时重规划 + 捕获判定）。
  - 策略网络在重规划时由 refine_icfinal_with_model 内部调用（与主脚本一致）。

运行示例见 marl0327/run_env_demo.py 与 DESIGN.md 中「如何运行」。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from intercept.IsoPair.obtainIsoPairsAll import obtainPTP2TP_IsoPos_timeShift2
from intercept.IsoPair.obtainIsoPath import insertIsoMapP2TP
from intercept.IsoPair.obtainNearTPall import obtainNearETP, obtainNearTP
from intercept.IsoPair.obtainPE2TP import obtainPE2TP
from intercept.IsoPair.obtainRLOutput import obtainNeighbour, obtain_output_iso
from intercept.IsoPair.obtainTaskAll import obtainTask_timeShift2
from intercept.Prediction.obtainDWAprePath import obtainDWAprePath
from intercept.Prediction.predictLikelyTarget import predictLikelyTargetNew

from marl0327.config import MARLConfig
from marl0327.intercept_select import build_rl_model, refine_icfinal_with_model


@dataclass
class DubinsInterceptEnvConfig:
    """地图与仿真路径（工作目录一般为项目根目录）。"""

    data_root: str = field(default_factory=lambda: _project_root)
    time_map: str = "0320_0920"
    time_iso: str = "0320_0920"
    time_res: float = 1.0
    cap_dist: float = 200.0
    cap_angle_deg: float = 87.0
    cap_dist_ref: float = 350.0
    evader: Optional[np.ndarray] = None
    e_pref: Optional[Dict[str, Any]] = None
    policy_checkpoint: Optional[str] = None
    device: Optional[str] = None


class DubinsInterceptMARLEnv:
    """
    非 Gym 封装：提供 reset / step，与主脚本逻辑一致。
    step 返回 Gymnasium 风格五元组 (obs, reward, terminated, truncated, info)。
    """

    def __init__(self, config: Optional[DubinsInterceptEnvConfig] = None):
        self.cfg = config or DubinsInterceptEnvConfig()
        self.map_root = os.path.join(self.cfg.data_root, "map")
        self._load_assets()
        self._build_cap_ref()
        self.rl_cfg = MARLConfig()
        dev = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._rl_device = torch.device(dev)
        self._rl_model = build_rl_model(self.rl_cfg, self._rl_device)
        self._rl_model.eval()
        if self.cfg.policy_checkpoint and os.path.isfile(self.cfg.policy_checkpoint):
            sd = torch.load(self.cfg.policy_checkpoint, map_location=self._rl_device)
            self._rl_model.load_state_dict(sd, strict=False)

        self._episode_step = 0
        self._ICFinal: Optional[np.ndarray] = None

    def _jbl(self, sub: str, name: str) -> Any:
        path = os.path.join(self.map_root, sub, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"缺少地图文件: {path}")
        return joblib.load(path)

    def _load_assets(self) -> None:
        tm = self.cfg.time_map
        ti = self.cfg.time_iso
        self.Map = self._jbl(tm, "Map.jbl")
        self.IsoMapPTP2Iso_i_tt = self._jbl(tm, "IsoMapPTP2Iso_i_tt.jbl")
        self.IsoMapPIso2TP_i_tt = self._jbl(tm, "IsoMapPIso2TP_i_tt.jbl")
        self.pathFinalMapPTP2Iso = self._jbl(tm, "pathFinalMapPTP2Iso.jbl")
        self.pathFinalE2ValIn = self._jbl(ti, "pathFinalE2ValIn.jbl")
        self.PathE2Val_true = self._jbl(ti, "PathE2Val_true.jbl")
        self.IsoMapTP2Val_i_tt = self._jbl(tm, "IsoMapTP2Val_i_tt.jbl")
        self.pathFinalTP2Val = self._jbl(tm, "pathFinalTP2Val.jbl")
        self.IsoMapETP2Iso_i_tt = self._jbl(tm, "IsoMapETP2Iso_i_tt.jbl")
        self.IsoMapEIso2TP_i_tt = self._jbl(tm, "IsoMapEIso2TP_i_tt.jbl")
        self.pathFinalMapETP2Iso = self._jbl(tm, "pathFinalMapETP2Iso.jbl")

        p_iso = os.path.join(self.map_root, tm, "IsoMapP2TP_i_tt.jbl")
        if os.path.isfile(p_iso):
            self.IsoMapP2TP_i_tt = joblib.load(p_iso)
        else:
            self.IsoMapP2TP_i_tt = self.IsoMapPIso2TP_i_tt

        self.length_E_max = 0
        for i in range(len(self.pathFinalE2ValIn)):
            for j in range(len(self.pathFinalE2ValIn[i])):
                self.length_E_max = max(self.length_E_max, len(self.pathFinalE2ValIn[i][j][0]))

        M = self.Map
        self.obs = M["obs"]
        self.sure = M["sure"]
        self.obs_no_circle = M["obs_no_circle"]
        self.obs_no_circle_in = M["obs_no_circle_in"]
        self.Stepsize = M["Stepsize"]
        self.v_P = M["v_P"]
        self.v_E = M["v_E"]
        self.Trans_Point = M["Trans_Point"]
        self.ValuePos = M["ValuePos"]
        self.PStart_Point = M["PStart_Point"]
        self.timeIsoRes = M["timeIsoRes"]

        if self.cfg.evader is not None:
            self.Evader = np.asarray(self.cfg.evader, dtype=float)
        else:
            self.Evader = np.array(
                [[200, 1950, -np.pi / 2], [1000, 1950, -np.pi / 2], [1800, 1950, -np.pi / 2]],
                dtype=float,
            )

        if self.cfg.e_pref is not None:
            self.E_PreRef = dict(self.cfg.e_pref)
        else:
            self.E_PreRef = {
                "num_v": 10,
                "num_w": 10,
                "v_range": [100, 150],
                "w_range": [-np.pi / 6, np.pi / 6],
                "Stepsize": self.Stepsize,
                "T_pred": 1,
            }

        self.num_E = self.Evader.shape[0]
        self.num_P = self.PStart_Point.shape[0]

    def _build_cap_ref(self) -> None:
        self.CapDist = float(self.cfg.cap_dist)
        self.CapRef: Dict[str, Any] = {}
        self.CapRef["CapDist"] = self.CapDist
        self.CapRef["CapAngle"] = float(self.cfg.cap_angle_deg)
        self.CapRef["Stepsize"] = self.Stepsize
        self.CapRef["CapDistRef"] = float(self.cfg.cap_dist_ref)
        self.CapRef["v_P"] = self.v_P
        self.CapRef["v_E"] = self.v_E
        self.CapRef["timeIsoRes"] = self.timeIsoRes
        obs_polygons = [
            Polygon(np.asarray(poly, dtype=float)[:, :2])
            for poly in (self.obs_no_circle)
            if poly is not None
        ]
        self.CapRef["obs_polygons"] = obs_polygons
        self.CapRef["CapDistTime"] = np.array([self.CapDist] * self.num_E)
        self.CapRef["CapAngleTime"] = np.array([self.CapRef["CapAngle"]] * self.num_E)

        self.distance_P = float(self.CapRef["CapDistRef"])
        self.distance_E = float(self.CapRef["CapDistRef"])
        self.distance_V = float(self.CapRef["CapDistRef"])

    def _reset_inner_state(self) -> None:
        self.TimeRes = float(self.cfg.time_res)
        self.t = 0.0
        self.t_all = 0.0
        self.PosE = self.Evader.copy()
        self.PosP = self.PStart_Point.copy()
        self.distances = np.linalg.norm(self.PosP[:, :2] - self.PosE[:, :2], axis=1)

        self.PathPtrue = [p.reshape(-1, 1) for p in self.PosP]
        self.PathP = [None] * self.num_P
        for pid in range(self.num_P):
            self.PathP[pid] = np.tile(self.PosP[pid].reshape(-1, 1), (1, self.length_E_max))

        self.PathE = [None] * self.num_E
        for eid in range(self.num_E):
            self.PathE[eid] = self.PathE2Val_true[eid][: int(self.TimeRes / self.Stepsize * self.v_E), :].T

        self.Capflag = np.array([False] * self.num_E)
        self.pairs_realE2P: Optional[np.ndarray] = None

        self.UnCapPid = np.arange(self.num_P, dtype=int)
        self.UnCapPidNew = self.UnCapPid.copy()
        self.UnCapEid = np.arange(self.num_E, dtype=int)
        self.UnCapEidNew = self.UnCapEid.copy()

        self.decision_step = 0
        self.decision_outputs: List[Dict[str, Any]] = []
        self.Valid: Optional[np.ndarray] = None
        self.EfromTPid: Optional[np.ndarray] = None
        self._ICFinal = None

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        self._reset_inner_state()
        self._episode_step = 0
        info: Dict[str, Any] = {
            "t_all": self.t_all,
            "decision_outputs": [],
        }
        return None, info

    def step(self, action: Optional[Any] = None) -> Tuple[Optional[Dict[str, Any]], float, bool, bool, Dict[str, Any]]:
        """
        执行一次与 main0319forRL 外层 while 相同的一次迭代。
        action 占位，当前策略由内部 RL 模型在重规划时选拦截点。
        """
        _ = action
        self._episode_step += 1
        terminated = False
        truncated = False
        replanned = False
        last_decision: Optional[Dict[str, Any]] = None

        self.t += self.TimeRes / self.Stepsize
        self.t_all += self.TimeRes / self.Stepsize

        if self.pairs_realE2P is not None and self.pairs_realE2P.shape[0] == self.Capflag.shape[0]:
            self.UnCapPidNew = self.pairs_realE2P[~self.Capflag, 1].astype(int)
            self.UnCapEidNew = self.pairs_realE2P[~self.Capflag, 0].astype(int)
            self.pairs_realE2P = self.pairs_realE2P[~self.Capflag]

        flag_plot = 1 if len(self.UnCapPidNew) != len(self.UnCapPid) else 0

        curr_idx_p = int(round(self.t * self.v_P))
        seg_start = int(round((self.t - self.TimeRes / self.Stepsize) * self.v_P))
        seg_end = curr_idx_p

        PosP = np.array(
            [
                self.PathP[np.where(pid == self.UnCapPid)[0][0]][
                    :, min(curr_idx_p, self.PathP[np.where(pid == self.UnCapPid)[0][0]].shape[1] - 1)
                ]
                for pid in self.UnCapPidNew
            ]
        )
        for _id, pid in enumerate(self.UnCapPidNew):
            self.PathPtrue[pid] = np.hstack(
                (
                    self.PathPtrue[pid],
                    self.PathP[np.where(pid == self.UnCapPid)[0][0]][:, max(0, seg_start) : seg_end],
                )
            )

        curr_idx_e = int(round(self.t_all * self.v_E))
        PosE = np.array(
            [
                self.PathE2Val_true[eid][min(curr_idx_e, self.PathE2Val_true[eid].shape[0] - 1), :]
                for eid in self.UnCapEidNew
            ]
        )

        PathEpre = [None] * len(self.UnCapEidNew)
        for _id, eid in enumerate(self.UnCapEidNew):
            PathEpre[_id] = self.PathE[np.where(eid == self.UnCapEid)[0][0]]

        min_dists, BestPaths = obtainDWAprePath(PosE, PathEpre, self.E_PreRef)
        flag_in = 1 if np.all(min_dists <= 10) else 0

        if not flag_in:
            self.decision_step += 1
            replanned = True
            traj = [self.PathE2Val_true[i][: int(self.t_all * self.v_E), :] for i in range(self.num_E)]
            targets = self.ValuePos[:, :2]
            res = predictLikelyTargetNew(traj, targets)
            Validnew = res["rank_idx"][:, 0]
            pairsE2Val_ = np.column_stack((np.arange(len(Validnew)), Validnew))
            pairsE2Val = pairsE2Val_[self.UnCapEidNew, :]

            _, NearTPid_E = obtainNearETP(self.Trans_Point[:, :2], PosE, self.ValuePos, pairsE2Val)

            self.UnCapEid = self.UnCapEidNew
            self.UnCapPid = self.UnCapPidNew
            flag_plot = 1
            self.Valid = Validnew
            self.EfromTPid = NearTPid_E

            PathE2TP, IsoMapE2TP, E2TP_results = obtainPE2TP(
                self.IsoMapEIso2TP_i_tt,
                self.EfromTPid,
                PosE,
                self.Trans_Point,
                self.pathFinalMapETP2Iso,
                self.Map,
                self.Map["v_E"],
            )
            E2TP_TPIdx = np.array([res[1] for res in E2TP_results])

            _, NearTPid_P = obtainNearTP(self.Trans_Point[:, :2], PosP, PosE)
            PathP2TP, IsoMapP2TP, P2TP_results = obtainPE2TP(
                self.IsoMapPIso2TP_i_tt,
                NearTPid_P,
                PosP,
                self.Trans_Point,
                self.pathFinalMapPTP2Iso,
                self.Map,
                self.Map["v_P"],
            )
            P2TP_TPIdx = np.array([res[1] for res in P2TP_results])

            IsoMap_i_tt_P2Iso_raw = insertIsoMapP2TP(self.IsoMapPTP2Iso_i_tt, IsoMapP2TP, P2TP_TPIdx, PathP2TP)
            IsoMap_i_tt_E2Iso_raw = insertIsoMapP2TP(self.IsoMapTP2Val_i_tt, IsoMapE2TP, E2TP_TPIdx, PathE2TP)

            TimeP_len = [len(row) for row in IsoMap_i_tt_P2Iso_raw]
            TimeE_len = [len(row) for row in IsoMap_i_tt_E2Iso_raw]
            TimeL = max(max(TimeP_len), max(TimeE_len))

            IsoMap_i_tt_P2Iso = [row + [None] * (TimeL - len(row)) for row in IsoMap_i_tt_P2Iso_raw]
            IsoMap_i_tt_E2Iso = [row + [None] * (TimeL - len(row)) for row in IsoMap_i_tt_E2Iso_raw]

            num_e = len(IsoMap_i_tt_E2Iso)
            num_p = len(IsoMap_i_tt_P2Iso)

            eid_idx, pid_idx, te_idx, tp_idx = np.meshgrid(
                self.UnCapEid, self.UnCapPid, np.arange(TimeL), np.arange(TimeL), indexing="ij"
            )
            idxE, idxP, te_idx2, tp_idx2 = np.meshgrid(
                np.arange(num_e), np.arange(num_p), np.arange(TimeL), np.arange(TimeL), indexing="ij"
            )

            eid_flat = eid_idx.ravel()
            pid_flat = pid_idx.ravel()
            te_flat = te_idx.ravel()
            tp_flat = tp_idx.ravel()
            idxE_flat = idxE.ravel()
            idxP_flat = idxP.ravel()

            x1, y1 = self.CapDist, self.CapDist
            x2, y2 = 2 * self.CapRef["CapDistRef"], self.CapRef["CapDistRef"]
            CapDistTime = np.where(
                self.distances < x1,
                np.inf,
                np.where(
                    self.distances >= x2,
                    y2,
                    y1 + (self.distances - x1) * (y2 - y1) / (x2 - x1),
                ),
            )
            self.CapRef["CapDistTime"] = CapDistTime

            x1, y1 = self.CapDist, self.CapRef["CapAngle"]
            x2, y2 = 2 * self.CapRef["CapDistRef"], 180
            CapAngleTime = np.where(
                self.distances < x1,
                360,
                np.where(
                    self.distances >= x2,
                    y2,
                    y1 + (self.distances - x1) * (y2 - y1) / (x2 - x1),
                ),
            )
            self.CapRef["CapAngleTime"] = CapAngleTime

            results = [
                [
                    te,
                    tp,
                    E2TP_TPIdx[ide],
                    P2TP_TPIdx[idp],
                    obtainPTP2TP_IsoPos_timeShift2(
                        idp,
                        ide,
                        tp,
                        te,
                        self.CapRef,
                        self.v_E,
                        CapDistTime,
                        IsoMap_i_tt_P2Iso,
                        IsoMap_i_tt_E2Iso,
                        self.ValuePos,
                        pairsE2Val,
                    ),
                    eid,
                    pid,
                    ide,
                    idp,
                ]
                for eid, pid, te, tp, ide, idp in zip(
                    eid_flat, pid_flat, te_flat, tp_flat, idxE_flat, idxP_flat
                )
            ]

            flat_results = [r for r in results if r[4] is not None and len(r[4]) > 0]
            IsoPairs_time_ETPid_PTPid_Posid_ = np.array(flat_results, dtype=object)
            sort_idx = np.argsort(IsoPairs_time_ETPid_PTPid_Posid_[:, 0].astype(int))
            IsoPairs_time_ETPid_PTPid_Posid = IsoPairs_time_ETPid_PTPid_Posid_[sort_idx]

            InterceptCandidates = obtainTask_timeShift2(
                IsoPairs_time_ETPid_PTPid_Posid,
                IsoMap_i_tt_P2Iso,
                IsoMap_i_tt_E2Iso,
                self.CapRef,
            )

            IC = InterceptCandidates.copy()

            output_P_state = {int(pid): PosP[idx].copy() for idx, pid in enumerate(self.UnCapPid)}
            output_P_alley_state = obtainNeighbour(
                PosP,
                PosP,
                self.distance_P,
                query_ids=self.UnCapPid,
                target_ids=self.UnCapPid,
                return_mode="dict",
            )
            output_enemy_state = obtainNeighbour(
                PosP,
                PosE,
                self.distance_E,
                query_ids=self.UnCapPid,
                target_ids=self.UnCapEid,
                return_mode="dict",
            )
            output_IsoP = obtain_output_iso(IC, self.IsoMapP2TP_i_tt, None)
            output_E_state = {int(eid): PosE[idx].copy() for idx, eid in enumerate(self.UnCapEid)}
            output_V_state = obtainNeighbour(
                PosP,
                self.ValuePos,
                self.distance_V,
                query_ids=self.UnCapPid,
                target_ids=np.arange(self.ValuePos.shape[0]),
                return_mode="dict",
            )

            last_decision = {
                "step": int(self.decision_step),
                "t_all": float(self.t_all),
                "pids": self.UnCapPid.copy(),
                "eids": self.UnCapEid.copy(),
                "output_P_state": output_P_state,
                "output_P_alley_state": output_P_alley_state,
                "output_enemy_state": output_enemy_state,
                "output_IsoP": output_IsoP,
                "output_E_state": output_E_state,
                "output_V_state": output_V_state,
            }
            self.decision_outputs.append(last_decision)

            unique_pairs = np.unique(IC[:, [11, 12]], axis=0)
            best_candidates_list = []
            for pair in unique_pairs:
                eid_val, pid_val = pair
                mask = (IC[:, 11] == eid_val) & (IC[:, 12] == pid_val)
                group = IC[mask]
                sort_idx = np.lexsort((group[:, 16], group[:, 8], -group[:, 15]))
                best_candidates_list.append(group[sort_idx[0]])

            IC_candidates = np.array(best_candidates_list)

            E_set, e_inv = np.unique(IC_candidates[:, 11], return_inverse=True)
            P_set, p_inv = np.unique(IC_candidates[:, 12], return_inverse=True)

            CostMat = np.full((len(P_set), len(E_set)), 1e9)
            CandidateIdxMat = np.full((len(P_set), len(E_set)), -1, dtype=int)
            CostMat[p_inv, e_inv] = IC_candidates[:, 8]
            CandidateIdxMat[p_inv, e_inv] = np.arange(len(IC_candidates))

            p_indices, e_indices = linear_sum_assignment(CostMat)
            valid_mask = CostMat[p_indices, e_indices] < 1e8
            p_final = p_indices[valid_mask]
            e_final = e_indices[valid_mask]

            final_rows = CandidateIdxMat[p_final, e_final]
            AssignedIntercepts = IC_candidates[final_rows]
            pairs_realE2P_ = AssignedIntercepts[:, [11, 12]]

            self.pairs_realE2P = np.array(
                [
                    [
                        self.UnCapEid[pairs_realE2P_[i, 0].astype(int)],
                        self.UnCapPid[pairs_realE2P_[i, 1].astype(int)],
                    ]
                    for i in range(len(pairs_realE2P_))
                ]
            )

            decision_entry = {
                "output_P_state": output_P_state,
                "output_P_alley_state": output_P_alley_state,
                "output_enemy_state": output_enemy_state,
                "output_V_state": output_V_state,
            }
            ICFinal = refine_icfinal_with_model(
                IC,
                AssignedIntercepts,
                output_IsoP,
                decision_entry,
                self.UnCapPid,
                self._rl_model,
                self._rl_device,
                self.rl_cfg,
            )
            self._ICFinal = ICFinal

            E_idx = ICFinal[:, 11].astype(int)
            VP_idx = ICFinal[:, 9].astype(int)
            Iso_idx = ICFinal[:, 13].astype(int)
            ETP_idx = ICFinal[:, 2].astype(int)

            PathE_active = [
                np.hstack(
                    (
                        PathE2TP[idx],
                        self.pathFinalTP2Val[ETP_idx[i]][VP_idx[i]][
                            :, : int(Iso_idx[i] - PathE2TP[idx].shape[1])
                        ],
                    )
                )
                if VP_idx[i] >= 0
                else PathE2TP[idx][:, : int(Iso_idx[i])]
                for i, idx in enumerate(E_idx)
            ]
            PathE = [PathE_active[i] for i in np.argsort(E_idx)]

            P_idx = ICFinal[:, 12].astype(int)
            TP_to_idx = ICFinal[:, 10].astype(int)
            Iso_p_idx = ICFinal[:, 14].astype(int)
            TP_from_idx = ICFinal[:, 3].astype(int)

            PathP_active = [
                np.hstack(
                    (
                        PathP2TP[idx],
                        self.pathFinalMapPTP2Iso[TP_from_idx[i]][TP_to_idx[i]][
                            :, : int(Iso_p_idx[i] - PathP2TP[idx].shape[1])
                        ],
                    )
                )
                if TP_to_idx[i] >= 0
                else PathP2TP[idx][:, : int(Iso_p_idx[i])]
                for i, idx in enumerate(P_idx)
            ]
            self.PathP = [PathP_active[i] for i in np.argsort(P_idx)]
            self.PathE = PathE
            self.t = 0.0

        self.distances = np.linalg.norm(PosP[:, :2] - PosE[:, :2], axis=1)
        dx = PosE[:, 0] - PosP[:, 0]
        dy = PosE[:, 1] - PosP[:, 1]
        angPE = np.degrees(np.arctan2(dy, dx))
        angleDiff_ = angPE - np.degrees(PosP[:, 2])
        angleDiff = (angleDiff_ + 180) % 360 - 180
        self.Capflag = (self.distances <= self.CapDist) & (np.abs(angleDiff) <= self.CapRef["CapAngle"] / 2)

        if self.Capflag.size > 0 and np.all(self.Capflag):
            terminated = True

        max_t = self.length_E_max / self.v_E
        if self.t_all >= max_t:
            truncated = True

        reward = 1.0 if terminated else 0.0
        info = {
            "t_all": float(self.t_all),
            "replanned": replanned,
            "decision_step": self.decision_step,
            "last_decision": last_decision,
            "flag_plot": flag_plot,
            "min_dists_dwa": min_dists,
            "pairs_realE2P": self.pairs_realE2P,
            "ICFinal": self._ICFinal,
        }
        obs = last_decision if last_decision is not None else None
        return obs, float(reward), terminated, truncated, info
