---
name: Render候选点可视化
overview: 在 `TODCMARLEnv.render()` 叠加显示每个我方无人机的候选拦截点（来自观测 `self_pts/self_pts_mask`），并按实际执行动作索引高亮选中候选；该叠加会同时出现在网页实时流与保存视频里，因为它们共享 `env.render()` 输出。
todos:
  - id: cache-last-obs
    content: 在 `TODCMARLEnv` 中缓存 `reset/step` 产生的观测快照与动作索引（`_last_obs/_last_obs_at_action/_last_action_indices`）
    status: completed
  - id: render-candidates
    content: 在 `TODCMARLEnv.render()` 叠加绘制每个 pid 的候选点（基于 `self_pts/self_pts_mask`），并按 `action_indices` 高亮选中候选
    status: completed
  - id: add-render-config
    content: 增加渲染开关与样式参数（alpha/size 等）并从 env config 读取
    status: completed
  - id: verify-smoke
    content: 跑环境与训练 smoke 测试并做一次手动可视化检查（live/video）
    status: completed
isProject: false
---

## 目标与范围
- 在环境渲染中可视化“当前决策时刻每个 pursuer 的候选拦截点”。
- 候选点来源使用观测对齐的 `self_pts/self_pts_mask`（与策略输入一致）。
- 选中候选点高亮来源使用环境实际执行动作：`step()` 内 `_normalize_action()` 得到的 `action_indices`。
- 仅改环境侧渲染与状态缓存，不改训练逻辑；网页实时流与视频会自然同步生效（都调用 `TODCMARLEnv.render()`）。

## 关键设计
- **缓存策略（推荐）**：避免在 `render()` 内重新计算观测，防止性能问题与状态偏差。
  - 在 `reset()` 末尾生成 obs 后缓存 `self._last_obs`。
  - 在 `step()` 中：
    - `_normalize_action()` 会构造“决策时刻 obs 快照”（当前实现已返回 `obs_at_action`），将其缓存到 `self._last_obs_at_action`。
    - 同时缓存 `self._last_action_indices = action_indices`（无任务机为 `-1`）。
    - step 结束构造的新 obs 也缓存到 `self._last_obs`（用于非决策时刻的兜底显示）。
- **渲染数据选择**：
  - 优先用 `self._last_obs_at_action` 的 `self_pts/self_pts_mask` 画候选点（严格对应动作选择时刻）。
  - 若缓存不存在（例如直接调用 `render()` 且尚未 reset/step），则退回用 `self._last_obs`；仍不存在则不画候选点。
- **颜色与层级**：
  - 复用 `render()` 中已有 `tab20` 色表。
  - 候选点按 pursuer(pid) 上色：`color_p`（与 UAV 本体/路径一致），以半透明散点叠加。
  - 高亮点：同色但更大 `s`、白色描边 `edgecolors='w'`、更高 `zorder`。
  - 仅画 mask 为 True 的点；避免绘制无效候选。
- **可配置开关**：在 env config 中新增（默认开启或默认关闭需你决定；建议默认开启但可关）：
  - `render_candidates`: bool
  - `render_candidates_alpha`: float（默认 0.5）
  - `render_candidates_size`: int（默认 18）
  - `render_selected_size`: int（默认 60）

## 需要改动的文件
- `marl/envs/todc_env.py`
  - 增加缓存字段：`_last_obs`, `_last_obs_at_action`, `_last_action_indices`。
  - 在 `reset()` 末尾写入 `_last_obs`。
  - 在 `step()`：在 `_normalize_action()` 之后写入 `_last_obs_at_action` 与 `_last_action_indices`；step 末尾写入 `_last_obs`。
  - 在 `render()`：在当前绘制 UAV/路径之后，叠加绘制候选点散点与高亮点（受开关控制）。

## 验证思路
- 运行现有 smoke/训练短跑：
  - `tests/test_marl_env_smoke.py` 确认 reset/step 不受影响。
  - `tests/test_train_online_smoke.py` 确认训练流程仍可跑。
- 手动观察：开启 `live_server_enable` 或保存 eval video，确认候选点与高亮点随决策步变化，且与每个 pursuer 颜色一致。

## 风险与规避
- **K 动态变化**：通过使用 `_normalize_action()` 返回的 `obs_at_action` 来画候选与高亮，避免因 step 内重规划导致候选集变化而“高亮错位”。
- **性能**：render 仅使用缓存数据，不额外调用 `_build_obs()`；散点数量为 `P*K`，通常可控。