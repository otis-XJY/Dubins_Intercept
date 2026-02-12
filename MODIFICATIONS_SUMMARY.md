# 代码修改总结

## 目标
使PosP、PosE和相关路径数据保持原始顺序（与初始代理ID对应），被拦截的代理用None占位符代替。

## 主要修改点

### 1. 初始化部分 (第259-290行)
**修改内容：**
- 从创建固定大小的列表而不是numpy数组
- PosE: `list of 3D arrays | None` (保持原始顺序)
- PosP: `list of 3D arrays | None` (保持原始顺序)
- PathPtrue: `list of 2D arrays` (列向量，保持原始顺序)
- PathE: `list of 2D arrays | None` (保持原始顺序)
- PathP: `list of 2D arrays | None` (保持原始顺序)

**原因：** 
这样可以通过ID直接索引，拦截后标记为None而不是删除

### 2. 循环中的位置更新 (第369-409行)
**修改内容：**
- 不再对PosP和PosE进行排序
- 直接在原始ID对应的索引位置更新值
- 拦截的代理标记为None

**核心逻辑：**
```python
for pid in UnCapPid:
    PosP[pid] = PathP[pid][:, curr_idx_p]  # 更新未拦截
    
for pid in range(num_P):
    if pid not in UnCapPid:
        PosP[pid] = None  # 拦截标记
```

### 3. 路径数据处理 (第459-473行)
**修改内容：**
- PathE2TP、PathP2TP也保持固定大小列表格式
- 初始化为 `[None] * num_E/num_P`
- 只填充未拦截代理的索引位置

### 4. 绘图部分 (第671-733行)
**修改内容：**
- 遍历时通过UnCapEid和UnCapPid获取未拦截代理
- 在访问PosE[eid]和PosP[pid]前检查是否为None
- 路径绘制也检查None值

**检查方式：**
```python
if PosE[eid] is None or PosP[pid] is None:
    continue
```

### 5. 距离计算 (第742-756行)
**修改内容：**
- 改为循环计算而不是向量化
- 已拦截的配对距离设为无穷大(np.inf)

**原因：** 
确保while循环条件在所有配对中都满足

## 数据结构变化

### 原始设计
```
PosP: (num_P, 3) numpy array - 所有追捕者
PosE: (num_E, 3) numpy array - 所有逃避者
PathP: list of (2, N) - 所有追捕者的规划路径
```

### 新设计
```
PosP: [3D array | None] * num_P - 原始顺序，拦截为None
PosE: [3D array | None] * num_E - 原始顺序，拦截为None
PathP: [2D array | None] * num_P - 原始顺序，拦截为None
PathE: [2D array | None] * num_E - 原始顺序，拦截为None
```

## 索引映射说明

```
ID对应关系保持不变：
- Evader ID: 0 ~ (num_E-1)
- Pursuer ID: 0 ~ (num_P-1)
- 任何时刻，PosE[i]、PosP[i]、PathE[i]、PathP[i] 对应原始代理i
```

## 可能的问题和注意事项

1. **函数兼容性**
   - obtainPTP2TP_IsoPos_timeShift2 等函数可能期望列表/数组格式，需要查看实际定义
   - 部分函数可能需要调整以处理None值

2. **性能影响**
   - 循环计算距离代替向量化运算可能有性能损失
   - 但保持了代码清晰性和拦截状态追踪

3. **绘图准确性**
   - 需要确保绘图时的代理ID映射与原始设计一致
   - 当有代理拦截时，图表会自动跳过None值

## 测试建议

1. 验证拦截后的None标记是否正确
2. 确保最终绘图中拦截的代理不会显示
3. 检查多次重规划时的ID对应关系
4. 监控与涉及PosP/PosE的函数调用的兼容性
