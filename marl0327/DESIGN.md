# Dubins 拦截 MARL 项目设计（marl0327）

## 1. 与 `main/main0319forRL.py` 的对应关系

主循环在「物理时间推进 → 是否重规划 → 绘图 → 捕获判定」之间轮转。代码里用注释标出了与 **env.update** / **env.step** 的对应关系，建议按下列语义理解（便于封装为 Gym 式 API）：

| 概念 | 主脚本位置 | 职责 |
|------|------------|------|
| **update（时间层）** | 约 L203–206：每次循环开始 `t`、`t_all` 递增 | 离散时间步前进，驱动沿路径采样的索引 |
| **update（状态采样）** | 约 L224–251：`curr_idx_p` / `curr_idx_e` 从 `PathP` / `PathE2Val_true` 取当前 `PosP`、`PosE`，并更新 `PathPtrue` 历史 | **无人机的位置与轨迹历史**随时间更新（不执行新的 Dubins 规划） |
| **step（决策层）** | 约 L264–560：`not flagIn` 时进入；含预测、等时面、候选任务、匈牙利指派、`PathP`/`PathE` 重写、`t=0` | **动作的执行**：在此脚本中等价于「触发一次完整重规划并切换跟踪路径」；RL 训练时此处用策略网络对拦截候选打分/选点，替代或约束原有指派 |
| **update（终止层）** | 约 L639–665：重算 `distances`、`Capflag` | **捕获条件**（距离 + 朝向角窗） |

辅助说明：

- **DWA**（`obtainDWAprePath`）：用于判断逃避者预测是否一致（`flagIn`）；`flagIn==1` 时本轮不进入重规划 `step` 块。
- **决策输出**（`decision_outputs`）：仅在重规划分支内追加，包含每个 `step` 下供 RL 使用的观测字典（`output_P_state`、`obtainNeighbour` 结果、`obtain_output_iso` 等）。

## 2. 算法与模块划分

- **环境（仿真核）**：继续复用 `intercept/` 下等时面、任务生成与路径拼接逻辑；长期可将 `main0319forRL.py` 中循环拆成类方法 `physics_update` / `planning_step` / `check_capture`。
- **智能体观测**：与 `models.py` 中 `UAVInterceptionNetwork` 对齐——`self_uav`(3)、`ally_uavs`(3)、`self_pts`/`ally_pts`(8)、`enemies`(3)、`targets`(2)，以及各序列 mask。
- **动作空间**：对每名 pursuer，在 **拦截候选集合**（`output_IsoP` / `InterceptCandidates` 行）上做离散选择；与网络输出的 `action_logits` / `best_candidate_idx` 一致。
- **学习算法（建议）**：**MAPPO** 或 **IPPO**（每架机独立 critic 亦可）；价值网络已用全局池化特征（`critic`），适合 CTDE 风格的 centralized critic + decentralized actor。
- **多卡 A6000**：使用 `torchrun --nproc_per_node=N train_marl.py`；DDP 包裹 `UAVInterceptionNetwork`，各进程独立跑环境 rollout，梯度 `all_reduce`。环境在 CPU 上跑规划、仅把 batch 观测送 GPU 即可。

## 3. 目录结构

```
marl0327/
  DESIGN.md
  __init__.py
  models.py           # Actor-Critic
  observation.py      # decision → 张量；含按 (E_ref,P_ref) 子集的 encode_single_pursuer_obs_ep
  intercept_select.py # 匈牙利之后用模型在 IC 子集中选拦截行
  train_marl.py
  config.py
```

## 4. 任务分配与拦截点选择（已实现）

- **匈牙利指派**（`linear_sum_assignment` + `IC_candidates` 代价矩阵）：不变，仍决定每架 pursuer 与哪个 evader 配对。
- **拦截点**：在配对确定后，对完整表 `IC` 中该 `(E_ref, P_ref)` 的**所有候选行**构造观测子集，由 `UAVInterceptionNetwork` 在 `self_pts` 上输出 logits，取 argmax 得到最终 `ICFinal` 行（见 `marl0327/intercept_select.py` 与 `main/main0319forRL.py` 中 `refine_icfinal_with_model`）。

## 5. 后续实现顺序

1. MAPPO：存 rollout buffer、GAE、PPO clip、价值损失；对 `refine_icfinal_with_model` 中使用的 logits 计算策略梯度。
2. 可选：多进程并行环境采样。

## 6. 环境与 conda

- 激活：`conda activate dubins`
- 依赖：与主项目一致（`torch`、`numpy`、`joblib` 等）；多卡需 CUDA 与 NCCL 可用。

## 7. 如何运行（独立环境 `marl0327/env.py`）

1. **准备地图数据**  
   将 `map/<time_map>/`、`map/<time_iso>/` 下的 `.jbl` 放在项目根目录对应路径（与 `main/main0319forRL.py` 中 `TimeMap` / `TimeIso` 一致，默认 `0320_0920`）。若缺少 `IsoMapP2TP_i_tt.jbl`，环境会回退使用已加载的 `IsoMapPIso2TP_i_tt` 供 `obtain_output_iso` 使用。

2. **仅跑环境自检（无渲染、无 mp4）**  
   在项目根目录执行：
   ```bash
   conda activate dubins
   cd /path/to/Dubins_Intercept
   python marl0327/run_env_demo.py
   ```
   脚本会 `reset()` 后循环 `step(None)`，在终端打印重规划步与结束原因（全捕获 / 超时）。

3. **代码中自行调用**  
   ```python
   from marl0327.env import DubinsInterceptEnvConfig, DubinsInterceptMARLEnv
   cfg = DubinsInterceptEnvConfig(time_map="0320_0920", time_iso="0320_0920")
   env = DubinsInterceptMARLEnv(cfg)
   obs, info = env.reset(seed=0)
   obs, reward, terminated, truncated, info = env.step(None)
   ```
   `step` 的 `obs` 在发生重规划时为该步的 `decision_outputs` 风格字典，否则为 `None`；`info["last_decision"]` 同步最近一次决策（若有）。

4. **带训练权重的策略**  
   `DubinsInterceptEnvConfig(policy_checkpoint="your.pt")` 指向 `UAVInterceptionNetwork` 的 `state_dict`；未指定则使用随机初始化（与主脚本一致）。

5. **原始仿真 + 视频**  
   仍可使用 `python main/main0319forRL.py`（需配置本机 `ffmpeg` 等）；环境与主循环逻辑对齐，但不负责录屏。
