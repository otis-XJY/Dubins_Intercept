---
name: wandb可视化与网页实时流
overview: 在现有 MARL 训练入口（marl/train_online0325.py + marl/MARL_env.py）基础上，补齐“每个决策 step 的帧图 + 每个 episode 的视频”，并新增一个可通过端口转发访问的网页实时画面流（可选包含 step 内部物理 tick 的过程）。
todos:
  - id: inspect-current-wandb-logging
    content: 梳理 train_online0325 中 step_image/train_episode_video/eval_video 的现有键与频率，确定新增的 episode_summary_image 与 tick 采样接口
    status: pending
  - id: add-tick-frame-hook
    content: 在 TODCMARLEnv.step 内部 while 增加可选 tick 回调/采样机制，支持 step 内过程帧输出
    status: pending
  - id: implement-live-http-stream
    content: 新增 marl/live_server.py：提供 /, /stream.mjpg, /status.json 并支持线程安全更新最新帧与状态
    status: pending
  - id: wire-live-stream-into-training
    content: 在 train_online0325 中按配置启动 live server，并在训练/eval 推进时更新帧与状态；可选从 tick 回调更新
    status: pending
  - id: update-config-and-docs
    content: 在 configs/train_online0324.yaml 增加 live/wandb 相关配置项，README 补充端口转发使用方式
    status: pending
isProject: false
---

## 目标

- 任务1：在 wandb 页面同时看到
  - **逐帧图**：按 step（以及可选的 step 内 tick）持续上传图片，便于逐步回放
  - **整段轨迹视频**：每个 episode（训练/评估）上传 mp4 录像
- 任务2：在服务器上开一个端口（便于 SSH 端口转发），浏览器实时看到运行结果（实时画面流 + 关键数值）。

## 现状核对（已确认）

- `marl/train_online0325.py` 已有
  - `wandb.Image` 的 step 级别日志（`cfg.step_frame_interval` 控制频率）
  - `train_episode_video`（`cfg.wandb_train_video_every`）与 `eval_video`（`cfg.eval_interval`）
  - 通过 `TODCMARLEnv(render_mode="rgb_array")` 获取 `env.render()` 的 RGB 帧
- `marl/MARL_env.py::TODCMARLEnv.render()` 已复刻了你给的 `main0319.py:563-632` 绘图逻辑，返回 `rgb_array`。
- `main/mainObtainPathETrue.py` 已有 `ThreadingHTTPServer + BaseHTTPRequestHandler` 的网页交互服务骨架，可复用其“轻量内置 HTTP 服务 + 前端页面”模式。

## 设计选择

- **wandb**
  - 保留现有 `step_image / train_episode_video / eval_video`。
  - 新增“**轨迹汇总图**”（每个 episode 结束时上传 1 张总览图）：例如把 `PathE2Val_true / PathPtrue` 叠加并标注终点/捕获事件，用于快速对比不同 episode。
  - 增加“**step 内 tick 帧采样（必须实现）**”：通过 `wandb_tick_image_every` 控制 tick 级别图片上传频率，确保能在 wandb 上看到 **每个 time_step（step 内物理推进）的过程**。实现上采用 env 内 tick 回调（见下文方案A），在 `TODCMARLEnv.step()` 的内部 while 推进循环里按 interval 触发渲染并上传/转发。
  - wandb 日志键建议：
    - `{scheme}/step_image`：RL step 粒度（现有能力，保留）
    - `{scheme}/tick_image`：tick 粒度（新加，受 `wandb_tick_image_every` 控制；caption 里带 `episode/global_step/decision_step/t_all/tick_idx`）
    - `{scheme}/train_episode_video`、`{scheme}/eval_video`：整段 mp4（现有能力，保留）
- **网页实时**
  - 采用“实时画面流（MJPEG）+ JSON 状态接口”的最小实现：
    - `/stream.mjpg`：浏览器 `<img>` 直接播放，无需 websockets 依赖，端口转发最省心
    - `/status.json`：返回 `t_all / decision_step / captured / replanned / asset_breached ...`
    - `/`：一个简单页面，左侧 `<img>` 播放，右侧定时轮询 JSON 显示数值
  - 帧来源：优先复用与 wandb 相同的 tick 回调产出的帧（这样网页端也能看到 step 内过程）；若未开启 tick 回调，则退化为 RL step 边界的 `env.render()` 更新。

## 需要改动的文件（核心）

- `[marl/train_online0325.py](/home/xujunyi/Dubins_Intercept/marl/train_online0325.py)`
  - **新增配置项**（YAML/CLI 都支持）：
    - `wandb_step_image_every`（更直观替代/兼容现有 `step_frame_interval`）
    - `wandb_tick_image_every`（step 内 tick 采样频率；用于看到每个 time_step 的过程；0 表示关闭 tick 级图片与 tick 级网页刷新）
    - `live_server_enable` / `live_server_host` / `live_server_port`
    - `live_stream_fps_limit`（防止推太快占满 CPU/带宽）
  - **训练 loop 与 eval loop**：
    - 训练：维持现有 `step_image` 与 `train_episode_video`，补一个 `episode_summary_image`（episode 结束时）
    - 评估：维持现有 `eval_video`，补一个 `eval_episode_summary_image`
    - 若 `wandb_tick_image_every>0`：在 env 内部推进时采样帧并 log 到 wandb 的 `{scheme}/tick_image`；该采样与网页实时流共享同一帧源。
- `[marl/MARL_env.py](/home/xujunyi/Dubins_Intercept/marl/MARL_env.py)`
  - 在 `TODCMARLEnv.step()` 的内部 while 循环里提供一个“**可选 tick 回调（采用方案A）**”机制，用于在 tick 级别产出帧（默认关闭，避免训练变慢）：
    - **方案A（确定采用）**：在 env 上挂 `self._on_tick` 回调（签名建议 `on_tick(env: TODCMARLEnv, tick_idx: int) -> None`），由训练脚本设置；env 在内部 while 循环每推进一次 tick 后按 `wandb_tick_image_every` 的 interval 触发一次回调。\n+      - 触发位置：建议在 `_advance_from_paths()` 更新 `PosP/PosE` 之后立刻触发（此时姿态与轨迹一致），再进入 `_phase_check_decision()`。\n+      - 回调内行为：可调用 `env.render()` 获取 RGB 帧，并交由训练脚本决定是否 `wandb.log` 与是否更新 live server 的最新帧。\n+      - 异常处理：按你的规则“不做过强鲁棒”，回调若抛错直接让训练中止，便于快速暴露问题。
- 新增一个轻量模块（便于复用）
  - `marl/live_server.py`（新文件）
    - 基于 `ThreadingHTTPServer` 实现：
      - 最新 JPEG 帧缓冲区（`bytes`）
      - 最新状态字典缓冲区（`dict`）
      - `/stream.mjpg` 持续输出 multipart/x-mixed-replace
      - `/status.json` 输出状态
      - `/` 返回前端页面 HTML
- 配置文件
  - `[configs/train_online0324.yaml](/home/xujunyi/Dubins_Intercept/configs/train_online0324.yaml)` 增加上述字段（默认关闭 live server；wandb tick 采样默认关闭）。

## 关键实现步骤（执行顺序）

- 在 `marl/live_server.py` 实现一个可启动/停止的后台线程 HTTP 服务（不依赖 Flask/FastAPI）。
- 在 `marl/train_online0325.py`：
  - `wandb`：补上 episode 汇总图的生成与上传（从 env 内已有 `PathE2Val_true / PathPtrue / t_all` 等变量生成一帧总览图，或复用 `env.render()` 在终局时输出）
  - `live server`：
    - 开启时启动服务，训练/评估每次拿到 `env.render()` 就更新最新帧；同时更新 status（`global_step/episode/decision_step/t_all/...`）
  - `tick 级帧`：若启用，则把一个回调函数挂到 `env` 上，使得 env 内层 while 推进时也能间歇性产出帧，既能用于 live stream（看到 step 内过程），也能用于 wandb（低频采样）。
- 在 `marl/MARL_env.py`：按上面选择的回调方式，在内部 while 的合适位置触发（建议 `_advance_from_paths()` 之后、`_phase_check_decision()` 之前或之后）。
  - 由于你已选择方案A：计划中将以 `_advance_from_paths()` 之后、`_phase_check_decision()` 之前触发为准，并让训练脚本通过闭包捕获 `scheme_name/run/global_step/episode` 等上下文完成 wandb 日志与网页推送。

## 验证方式（不做“过强鲁棒”兜底，异常直接报错）

- 本地/服务器单机：
  - `python -m marl.train_online0325 --config configs/train_online0324.yaml --wandb-mode offline` 跑几个 episode
  - 在 wandb（offline 同步前先本地看日志目录）确认出现 `step_image`、`train_episode_video`、`eval_video`、`episode_summary_image`
- 网页实时：
  - 启动训练后终端打印 `http://127.0.0.1:<port>/`
  - SSH 端口转发 `ssh -L <local_port>:127.0.0.1:<port> user@server` 后本地浏览器打开 `http://127.0.0.1:<local_port>/`
  - 观察画面连续刷新，右侧数值随 step/tick 变化。

## 性能与默认值建议

- `wandb_step_image_every`: 10~50（取决于你想多密）
- `wandb_tick_image_every`: 0（默认关闭；开启后建议 50~200）
- `live_stream_fps_limit`: 5~10（避免训练被可视化拖慢）

## 与你给的参考代码的对应关系

- 轨迹绘图逻辑继续以 `TODCMARLEnv.render()` 为唯一来源（其注释已明确“与 main0319.py 563–632 行一致”），wandb/网页都复用同一帧源，避免两套画法不一致。

