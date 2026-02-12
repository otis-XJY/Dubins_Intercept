import numpy as np
from A_dubins.coreCode.Dubins_obs_nocircle_oneside import Dubins_obs_nocircle_oneside

def Dubins_obs_nocircle_oneside_first(Start_Point, End_Point, outline_all_positive, outline, 
                                      r_s, r_e, R, Stepsize, pos_id, parent_id, 
                                      obs_no_circle, End_Point_now, param_safe):
    """
    Dubins 避障逻辑的首次尝试：根据方向筛选边缘点并进行路径拼接。
    """
    param_best = []
    param_final = []
    
    # 提取点坐标 (N, 2)
    points = outline_all_positive[:, :2]
    
    # 定义直线参数 (ax + by + c = 0)
    x0, y0 = Start_Point[0], Start_Point[1]
    dx = (End_Point_now[0] - Start_Point[0])
    # 处理斜率无穷大的情况（防止除以0）
    if abs(dx) < 1e-9:
        # 垂直线方程: 1*x + 0*y - x0 = 0
        a, b, c = 1.0, 0.0, -x0
    else:
        m = (End_Point_now[1] - Start_Point[1]) / dx
        a = m
        b = -1.0
        c = y0 - m * x0
    
    denom = np.sqrt(a**2 + b**2)
    
    # 计算 End_Point 到直线的带符号距离
    singal = (a * End_Point[0] + b * End_Point[1] + c) / denom
    # 计算所有 outline 点到直线的带符号距离
    distances = (a * points[:, 0] + b * points[:, 1] + c) / denom
    
    farthest_point = None
    
    # 根据 End_Point 的象限/侧向筛选点
    if singal < 0:
        max_idx = np.argmax(distances)
        farthest_point = points[max_idx, :]
        # 过滤距离大于等于 0 的点 (MATLAB: distances >= 0)
        outline_all_positive_filtered = outline_all_positive[distances >= 0, :]
    else:
        min_idx = np.argmin(distances)
        farthest_point = points[min_idx, :]
        # 过滤距离小于 0 的点 (MATLAB: distances < 0)
        outline_all_positive_filtered = outline_all_positive[distances < 0, :]

    if farthest_point is not None:
        # 调用 oneside 路径规划函数
        param_all, param_best = Dubins_obs_nocircle_oneside(
            Start_Point, End_Point_now, farthest_point, 
            outline_all_positive_filtered, outline, 
            r_s, R, R, Stepsize, pos_id, parent_id, obs_no_circle
        )
        
        # 拼接旧的 param_safe 和新的 param_all
        # MATLAB 中的嵌套循环 j+i 索引意味着生成所有可能的组合
        for i in range(len(param_safe)):
            for j in range(len(param_all)):
                # 检查是否存在 pos_id 字段且非空
                if param_safe[i].get('pos_id') is not None and param_all[j].get('pos_id') is not None:
                    
                    # 构造拼接后的节点字典
                    node = {}
                    # 索引转换说明: 
                    # MATLAB point(:, 4:6) -> Python point[:, 3:6]
                    node['point'] = np.hstack([param_all[j]['point'], param_safe[i]['point'][:, 3:6]])
                    
                    # MATLAB length(3:5) -> Python length[2:5]
                    node['length'] = np.concatenate([param_all[j]['length'], param_safe[i]['length'][2:5]])
                    
                    # MATLAB phy(2:3) -> Python phy[1:3]
                    node['phy'] = np.concatenate([param_all[j]['phy'], param_safe[i]['phy'][1:3]])
                    
                    # MATLAB theta(3:5) -> Python theta[2:5]
                    node['theta'] = np.concatenate([param_all[j]['length'], param_safe[i]['theta'][2:5]])
                    
                    # MATLAB center(:, 2:3) -> Python center[:, 1:3]
                    node['center'] = np.hstack([param_all[j]['center'], param_safe[i]['center'][:, 1:3]])
                    
                    node['r'] = [r_s, r_e]
                    node['R'] = R
                    node['type'] = -1
                    
                    # MATLAB vtheta(2:4) -> Python vtheta[1:4]
                    node['vtheta'] = np.concatenate([param_all[j]['vtheta'], param_safe[i]['vtheta'][1:4]])
                    
                    node['vtheta_plot'] = param_all[j]['vtheta_plot'] + param_safe[i]['vtheta_plot']
                    node['Length'] = np.sum(node['length'])
                    
                    # MATLAB path(3:5) -> Python path[2:5]
                    node['path'] = param_all[j]['path'] + param_safe[i]['path'][2:5]
                    
                    # MATLAB vtheta_all(3:5) -> Python vtheta_all[2:5]
                    node['vtheta_all'] = param_all[j]['vtheta_all'] + param_safe[i]['vtheta_all'][2:5]
                    
                    node['pos_id'] = pos_id
                    node['parent_id'] = parent_id
                    
                    param_final.append(node)
        
        # 将结果赋值回 param_safe
        param_safe = param_final

    # 如果没有生成任何有效路径，返回空字典/列表（模拟 MATLAB exist 和 struct 逻辑）
    if not param_safe:
        param_safe = []

    return param_safe, param_best