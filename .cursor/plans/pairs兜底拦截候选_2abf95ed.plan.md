---
name: pairs兜底拦截候选
overview: 在 `pairs_realE2P` 未覆盖全部 `UnCapPid/UnCapEid` 时，为缺失的追捕者/逃避者生成“兜底”候选拦截点与路径，并在 reward 中对选择兜底候选施加每步惩罚（-50），避免观测/动作应用因配对缺失而失真或崩溃。
todos:
  - id: inspect-columns
    content: 确认兜底候选严格对齐 `obtainTask_timeShift2` 的 25 列布局，并确定用哪一列存兜底路径索引（建议复用 `Piso`/col14）
    status: pending
  - id: fallback-generate
    content: 在 `TODCMARLEnv._apply_hungarian_and_paths()` 计算 NoAns 集合、构建 P-E 路径、二次 Hungarian，并把兜底候选 append 到 `ICFinalActionCandidates`，同时更新 `pairs_realE2P/_pairs_ic_compact`
    status: pending
  - id: fallback-path-apply
    content: 在 `_apply_paths_from_assigned_rows()` 识别 `TPid==-2` 并应用 `pathNoAns` 到 `PathP`
    status: pending
  - id: fallback-pos
    content: 在 `_extract_candidate_pos()` 识别兜底行并用 compact->global 映射返回敌机当前位置
    status: pending
  - id: reward-penalty
    content: 在 `marl/rewards.py` 增加 `fallback_penalty` 配置，并在选到兜底候选时每步扣分
    status: pending
isProject: false
---

## 目标
- 当 Hungarian 分配得到的 `pairs_realE2P` 无法覆盖全部未捕获集合（`UnCapPid`/`UnCapEid`）时，补齐缺失配对的候选与路径，保证：
  - `obs_generator` 里的 `alive_pids/alive_eids` 不会因为缺配对而把“尚未捕获”的实体清零
  - `TODCMARLEnv._apply_assignment_from_action()` 始终能在 `ICFinalActionCandidates` 中为存活 pair 找到候选（避免空候选/索引越界）
  - reward 能识别“兜底候选”并施加每步惩罚

## 关键现状梳理（用于对齐列与数据流）
- `IC_candidates` 的列布局来自 `obtainTask_timeShift2()`，其输出列数是 25（0–24），其中 11–12 是紧凑索引（compact ide/idp）。见 [intercept/IsoPair/obtainTaskAll.py](intercept/IsoPair/obtainTaskAll.py) 的 `obtainTask_timeShift2()`。
- `obs_generator` 在 `_build_c_nodes()` 里用 `pairs_realE2P` 与 `pairs_ic_ref`（即 `self._pairs_ic_compact`）逐行对齐来过滤 `ic_candidates` 的 11–12 列；因此兜底行必须遵守同一套 compact 语义。见 [marl/obs_generator.py](marl/obs_generator.py) `_build_c_nodes()`。
- 当前 `marl/rewards.py` 只用 `obs['reward_nodes']` 计算奖励；`reward_nodes` 取自 `ic_candidates` 的 20–24 列（尽管名字叫 cost_*，实际上对应 `obtainCost()` 返回的 reward_*）。因此“用 -2 标记兜底”应通过构造候选行的这些列、并在 reward 中检测实现。

## 实现方案
### 1) 在 `_apply_hungarian_and_paths()` 补齐缺失配对并扩展 `ICFinalActionCandidates`
改动位置：[marl/MARL_env.py](marl/MARL_env.py) `_apply_hungarian_and_paths()`，紧接着现有：
- 生成 `self.pairs_realE2P`
- 生成 `self._pairs_ic_compact`
- 生成 `self.ICFinalActionCandidates`

具体步骤：
- **计算缺失集合**
  - `assigned_pids = set(self.pairs_realE2P[:,1])`、`assigned_eids = set(self.pairs_realE2P[:,0])`
  - `noAnsP = sorted(set(self.UnCapPid) - assigned_pids)`
  - `noAnsE = sorted(set(self.UnCapEid) - assigned_eids)`
  - 若任一为空，直接返回（不触发兜底）。
- **构造 all-to-all 组合并计算代价（用 path_len）**
  - 对所有 `(pid,eid)` 组合，构造 `PosP_batch`（追捕者当前位置）与 `PIsoPos_batch`（敌机当前位置 `PosE`，即“从自身当前位置 PosP 到敌机当前位置 PosE”）。
  - 调用 `obtainPE2IsoPath(PosP_batch, PIsoPos_batch, Map, v_P)` 得到 `pathNoAns`（list of 3×L arrays）。见 [intercept/IsoPair/obtainIsoPath.py](intercept/IsoPair/obtainIsoPath.py) `obtainPE2IsoPath()`。
  - `path_len = L` 作为该 pair 的代价；构造 `cost_mat[p_index,e_index]=path_len`。
- **对缺失集合再跑一次 Hungarian（用 path_len 最小）**
  - `linear_sum_assignment(cost_mat)` 得到一对一配对（最多 `min(len(noAnsP),len(noAnsE))`）。
  - 用户选择了 **replace_all**：将“原 Hungarian 配对 + 兜底配对”合并成新的完整 `pairs_realE2P`（覆盖所有 `UnCapPid/UnCapEid`，不足一侧会剩余未配对者继续视为 inactive）。
- **为每个兜底配对生成 25 列候选行（NoAnsInterceptCandidates）**
  - 生成一条候选即可（K=1），并将其 append 到 `self.ICFinalActionCandidates`。
  - 列赋值对齐 `obtainTask_timeShift2()` 的 0–24：
    - 0 `te=0`
    - 1 `tp`：按路径长度估计到达时间索引（用 `CapRef['timeIsoRes']`、`v_P` 将长度换算为步数并 clamp 到 `[0,numTime-1]`；若不方便获取 `numTime` 则允许先不 clamp）
    - 2 `ETPid=-2`（兜底标记，避免被当成正常 E2TP）
    - 3 `PTPid=-2`（兜底标记）
    - 4 `IsoPosidxE=0`、5 `IsoPosidxP=0`（占位）
    - 6 `dist`：可用当前 `||P-E||` 或 0（但建议填真实几何距离便于 debug）
    - 7 `path_len=L`
    - 8 `cost=-2`（兜底标记）
    - 9 `ValPosid=-2`（占位/标记）
    - 10 `TPid=-2`（你要求的关键标记，用于后续快速取兜底路径）
    - 11–12 `Eid_ref,Pid_ref`：**必须是 compact 索引**（`ceid = index_of(eid in self.UnCapEid)`；`cpid = index_of(pid in self.UnCapPid)`），保证与 `pairs_ic_ref` 的筛选逻辑一致。
    - 13 `Eiso=0`、14 `Piso`：用作兜底路径索引（例如把 `Piso` 写成在一个 `self._fallback_paths` 列表里的 index），供 `_apply_paths_from_assigned_rows()` 读取。
    - 15–19（Delta_t,Delta_d,Delta_theta,path_L,Delta_V）：统一置 `-2`
    - 20–24（reward_t,reward_d,reward_v,reward_L,reward_D）：统一置 `-2`
- **维护与 `pairs_realE2P` 逐行对齐的 `self._pairs_ic_compact`**
  - 对兜底配对追加对应的 `[ceid, cpid]` 行（与新的 `pairs_realE2P` 行顺序一致）。

### 2) 让 `_apply_paths_from_assigned_rows()` 能读取兜底路径
改动位置：[marl/MARL_env.py](marl/MARL_env.py) `_apply_paths_from_assigned_rows()`。
- 当 `assigned[:,10]`（TPid）为 `-2` 时：
  - 不走原先 `path_p2tp` / `pathFinalMapPTP2Iso` 的逻辑
  - 改为从 `self._fallback_paths`（上一步保存的 list/dict）取出 `pathNoAns` 直接写入 `self.PathP` 对应行
- 同时保持非兜底行逻辑不变。

### 3) 让 `_extract_candidate_pos()` 对兜底候选返回“敌机当前位置”
改动位置：[marl/MARL_env.py](marl/MARL_env.py) `_extract_candidate_pos()`。
- 若检测到兜底标记（例如 `row[10]==-2` 或 `row[8]==-2`）：
  - 读取 compact `ceid=row[11]`，再映射到全局 `eid_global = self.UnCapEid[ceid]`
  - 返回该敌机当前 `(x,y,theta)`（通过 `_pos_e_xyz_for_global_eid(eid_global)`）
- 避免当前实现把 compact eid 当成 global eid 导致取错敌机。

### 4) reward 中识别 -2 并施加每步惩罚（-50）
改动位置：[marl/rewards.py](marl/rewards.py) `TODCRewardFunction.compute_step_rewards()`。
- 在得到 `selected = reward_nodes[np.arange(num_p), selected_idx]` 后：
  - 若 `selected[i]` 中检测到标志位 `-2`（例如任意一个 reward/cost 字段为 `-2`），则对该 pursuer 施加 `fallback_penalty`（无需使用“整行 <= -1.5”这类启发式）。
- 你选择了 **每步惩罚**，因此每次选到兜底候选都扣。
- 惩罚数值做成可配置：在 `RewardConfig` 增加 `fallback_penalty: float = 50.0`（实现时做减法 `rewards -= fallback_penalty`，或直接存负数二选一，但要统一），默认值固定为 **-50/步**。

## 已确认参数
- 每步兜底惩罚：**-50**（检测到 `-2` 标志位即触发）。