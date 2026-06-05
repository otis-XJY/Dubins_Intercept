---
name: batch_env_generation_pipeline
overview: 新增三段式批量环境生成流水线：1) 批量初始化 Map 资产到 map/<TimeMap>/；2) 批量构建 IsoMap 等静态环境资产，失败则删除对应 TimeMap；3) 在每个 TimeMap 下批量生成多条敌方轨迹 profile 到 map/<TimeMap>/<TimeEpath>/。同时修改 TODCMARLEnv 的 profile 发现逻辑以匹配该结构（不做兜底兼容）。
todos:
  - id: spec_dir_conventions
    content: 确认新目录约定：map/<TimeMap>/ 与 map/<TimeMap>/<TimeEpath>/ 的文件清单与命名格式（TimeMap/TimeEpath 到秒级）
    status: completed
  - id: rewrite_env_init
    content: 设计并实现《环境初始化》批量脚本：读取 env_init.yaml，循环生成 n 个 Map 并保存到各 TimeMap
    status: completed
  - id: rewrite_env_build
    content: 设计并实现《环境构建》批量脚本：读取 env_build.yaml，扫描 TimeMap 逐个构建 IsoMap 资产，失败则删除目录并记录日志
    status: completed
  - id: rewrite_evader_paths
    content: 设计并实现《敌方无人机路径构建》批量脚本：读取 evader_paths.yaml，为每个 TimeMap 生成 m 条 profile（交互/回放）并保存到 map/<TimeMap>/<TimeEpath>/
    status: completed
  - id: update_env_loader
    content: 修改 TODCMARLEnv 的 profile 发现逻辑以支持 map/<TimeMap>/<TimeEpath>/（不做兜底兼容）
    status: completed
  - id: smoke_verify
    content: 增加最小验证脚本/步骤：对生成资产执行一次 env.reset/step，确认能发现 profile 并不缺 required 静态文件
    status: completed
isProject: false
---

## 目标与约束
- **目标**：重写 3 个脚本分别完成《环境初始化》《环境构建》《敌方无人机路径构建》，支持循环批量生成，并且**保存格式与现有 joblib (`.jbl`) 一致**。
- **目录约定（新）**：
  - `map/<TimeMap>/`：保存地图与环境静态资产（`Map.jbl`、IsoMap*、pathFinal* 等）。
  - `map/<TimeMap>/<TimeEpath>/`：保存同一地图的多条敌方轨迹 profile（至少含 `PathE2Val_true.jbl`、`pathFinalE2ValIn.jbl`，可附带 `IsoMapE2ValIn_i_tt.jbl`、点选文件与预览图）。
  - **命名格式**：`TimeMap` / `TimeEpath` = `datetime.now().strftime('%m%d_%H%M%S')`（建议带秒，避免批量碰撞；当前脚本用到分钟级 `%m%d_%H%M` 可能重复）。
- **不做兜底/兼容**：你保证环境均按该结构构建；若缺文件直接报错，便于快速暴露数据问题。

## 现状对齐（从现有脚本抽取的“资产清单”）
- 地图初始化目前只写：`map/<TimeMap>/Map.jbl`（见 `main/mainObtainMap.py`）。
- 环境构建（IsoMap 等）目前写入 `map/<TimeMap>/`：
  - `IsoMapPTP2Iso_i_tt.jbl`
  - `IsoMapPIso2TP_i_tt.jbl`
  - `pathFinalMapPTP2Iso.jbl`
  - `IsoMapTP2Val_i_tt.jbl`
  - `pathFinalTP2Val.jbl`
  - `IsoMapETP2Iso_i_tt.jbl`
  - `IsoMapEIso2TP_i_tt.jbl`
  - `pathFinalMapETP2Iso.jbl`
  - （以及一些中间产物 `IsoMapP_i_tt.jbl`、`pathFinalP.jbl`，是否保留由你决定）
  见 `main/mainObtainIsoMap0901.py`。
- 敌方轨迹 profile 目前写入 `map/<TimeMap>/<slot>/`（slot 为时间戳），并包含：
  - `PathE2Val_true.jbl`
  - `pathFinalE2ValIn.jbl`
  - `IsoMapE2ValIn_i_tt.jbl`
  - （可选 `PathMidPts.json`、`PathPreview.png`、`WebPickerBG.png`）
  见 `main/mainObtainPathETrue.py`。

## 需要同步修改的环境读取方式（marl）
- 修改 `TODCMARLEnv._discover_evader_profile_dirs(map_base)`（在 `marl/envs/todc_env.py`）使其**只**发现 `map/<TimeMap>/<TimeEpath>/` 形式的 profile 目录。\n+  - 约定：`map_base = map/<TimeMap>/`，其下每个子目录都被视为一个 profile slot（TimeEpath）。\n+  - 每个 slot 目录必须包含：`PathE2Val_true.jbl`、`pathFinalE2ValIn.jbl`。\n+  - 不做其它结构兜底（缺失则直接报错）。\n+- 可选（如需过滤无关子目录）：忽略以 `.` 开头、或名为 `__pycache__`、或不含 required 文件的目录。

## 三个脚本的重写方案（CLI + YAML）
### 1) 《环境初始化》批量脚本
- **新脚本**：`main/batch_env_init.py`
- **输入**：`configs/env_init.yaml`
- **职责**：循环生成 `n_maps` 个 TimeMap 目录，每个目录写入 `Map.jbl`（以及可选的 `Map.png`）。
- **实现方式**：把 `main/mainObtainMap.py` 的 Map 参数段抽成“可由 YAML 覆盖”的 dict，然后调用 `obtainMap`。
- **输出**：`map/<TimeMap>/Map.jbl`（与现一致）。

### 2) 《环境构建》批量脚本
- **新脚本**：`main/batch_env_build.py`
- **输入**：`configs/env_build.yaml`
- **职责**：对每个 `map/<TimeMap>/Map.jbl` 运行 IsoMap/Ref 构建逻辑（复用 `main/mainObtainIsoMap0901.py` 的计算），把所有静态资产写回 `map/<TimeMap>/`。
- **失败策略**：
  - 若某个 TimeMap 构建报错（任何 Exception），则**删除整个 `map/<TimeMap>/`**（你要求的回滚），并把失败原因记录到一个统一日志（例如 `map/_build_failures.jsonl` 或 `output/build_failures.log`）。
- **输出**：上面列出的 IsoMap 与 pathFinal 资产（与 `TODCMARLEnv._load_assets` 的 required 列表对齐）。

### 3) 《敌方无人机路径构建》批量脚本
- **新脚本**：`main/batch_evader_paths.py`
- **输入**：`configs/evader_paths.yaml`
- **职责**：对每个 TimeMap 生成 `m_profiles` 个 profile，保存到：
  - `map/<TimeMap>/<TimeEpath>/`（TimeEpath 为时间戳）
- **交互支持（你选择保留）**：提供两种运行模式：
  - **交互式**：逐条 profile 启动一次网页选点（复用 `main/mainObtainPathETrue.py` 的 HTTP picker 逻辑），用户保存后继续下一个。
  - **回放式**：YAML 中指定 `points_file` 列表或模板路径，直接加载中间点 JSON 生成轨迹（适合批量不想每次点选）。
- **输出**：`PathE2Val_true.jbl`、`pathFinalE2ValIn.jbl`、`IsoMapE2ValIn_i_tt.jbl`（与现一致，只是 slot 目录直接挂在 `map/<TimeMap>/` 下）。

## 配置文件（YAML）建议最小字段
- `configs/env_init.yaml`
  - `n_maps`
  - `map_params`: mapsize、sure、Stepsize、障碍物数量/半径范围、设施点数/分布、追捕者数/分布、resolution_map_pos 等
  - `seed`（可选）
- `configs/env_build.yaml`
  - `time_maps`: 可选显式列表；为空则自动扫描 `map/*/Map.jbl`
  - `v_P`, `v_E`（写入 Map 并用于 WH_* 生成）
  - `cleanup_on_fail: true`
- `configs/evader_paths.yaml`
  - `m_profiles`
  - `evader_init`: Evader 初始位置/朝向（支持固定数组或随机采样规则）
  - `value_id_list` 或 `pairing_rules`（决定 Evader 对应设施点）
  - `mode: interactive|replay`
  - `web_picker`: host/port/points_file_template

## 执行方式（严格三阶段串行）
- 阶段 1：
  - `python -m main.batch_env_init --config configs/env_init.yaml`
- 阶段 2：
  - `python -m main.batch_env_build --config configs/env_build.yaml`
- 阶段 3：
  - `python -m main.batch_evader_paths --config configs/evader_paths.yaml`

## 验证点（不改训练逻辑前提下）
- 对任意生成的 `TimeMap`：`TODCMARLEnv({'time_map': TimeMap, 'evader_profile_mode': 'cycle'})` 能在 `reset()` 时发现并轮换多个 `TimeEpath` profile。
- `TODCMARLEnv._load_assets` 能在 `map/<TimeMap>/` 找到 required 的静态资产文件列表，否则阶段 2 必须判定为失败并删除该 TimeMap。