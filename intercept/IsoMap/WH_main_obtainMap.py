import numpy as np
from A_dubins.dubinspath.obtain_path import obtain_path  # 假设导入路径规划算法
from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm  # 假设导入路径规划算法
from intercept.IsoMap.obtainPath import obtainPath

from dataclasses import dataclass
import numpy as np

@dataclass
class IsoMapData:
    """存储等值线数据的结构类"""
    IsoPos: np.ndarray      # 等值线位置坐标
    pathid: np.ndarray      # 路径ID
    IsoVtheta: np.ndarray   # 等值线速度角度

def WH_main_obtainMapP(Map, draw=True, draw_interactive=False):
    """主流程：计算所有 UAV->Target 的路径并生成 IsoMap 数据。
    draw: 是否调用外部 DrawIso 程序进行绘图（默认 True）
    draw_interactive: 如果绘图，是否启用交互式逐条绘制（模拟原脚本的实时绘图）
    """

    # 提取Map字典中的相关数据
    PStart_Point = Map['PStart_Point']
    Trans_Point = Map['Trans_Point']
    obs = Map['obs']
    sure = Map['sure']
    r = Map['r']
    resolution = Map['resolution']
    Stepsize = Map['Stepsize']
    obs_no_circle = Map['obs_no_circle']
    obs_no_circle_in = Map['obs_no_circle_in']
    ValuePos = Map['ValuePos']
    outline_all = Map['outline_all']

    final_pathUAV = {}
    vthetaAllP = {}
    vthetaAll_ = []
    pathFinalP = [[None for _ in range(len(Trans_Point[:, 1]))] for _ in range(len(PStart_Point[:, 1]))]
    lengthPathP = []

    # 计算每个无人机的路径
    for i in range(len(PStart_Point[:, 1])):
        for j in range(len(Trans_Point[:, 1])):
            print(f'################################UAV {i}, Target {j}')
            # 使用 dubins 路径规划算法计算路径
            final_pathUAV[i, j], open_list, close_list,_ = A_dubins_nocircle_swarm(
                PStart_Point[i, :], Trans_Point[j, :], obs, sure, r,
                obs_no_circle, outline_all, Stepsize,
                1,
                resolution,
                depth=0,max_depth=1
            )

            # 提取路径中的速度信息（vtheta_all 表示路径中每个点的速度角度）
            for tt in range(len(final_pathUAV[i, j]) - 1):
                for k in range(len(final_pathUAV[i, j][tt + 1]['vtheta_all'])):
                    vthetaAll_.extend(final_pathUAV[i, j][tt + 1]['vtheta_all'][k])
            vthetaAllP[i, j] = vthetaAll_

            pathFinalP[i][j] = obtainPath(final_pathUAV[i, j], vthetaAllP[i, j], 0)  # 假设这是一个函数，返回路径
            lengthPathP.append(len(pathFinalP[i][j][1, :]))

            vthetaAll_ = []

    # 生成每个无人机在不同时间点的轨迹
    Maxlength = max(lengthPathP) if len(lengthPathP) > 0 else 0
    numTime =max(Maxlength* Stepsize//250 if Maxlength > 0 else Map['numTime'], Map['numTime'])
    Map['numTime']=numTime
    IsoMapP_i_tt = [[{} for _ in range(numTime)] for _ in range(len(PStart_Point[:, 1]))]
    v_P = Map['v_P']


    timePlot = np.linspace(1, Maxlength / v_P, numTime) if Maxlength > 0 else np.linspace(1, 1, numTime)
    time = timePlot * Stepsize
    Map['timePlot'] = timePlot
    Map['time'] = time

    timeIsoRes_ = np.diff(time)  # numpy 里 diff 是一维差分
    timeIsoRes = timeIsoRes_[0] if len(timeIsoRes_) > 0 else 0  # 第一个元素
    Map["timeIsoRes"] = timeIsoRes

    for tt in range(numTime):
        for i in range(len(PStart_Point[:, 1])):
            IsoPos = []
            IsoVtheta = []
            pathid = []

            for j in range(len(Trans_Point[:, 1])):
                if  len(pathFinalP[i][j][0, :]) >= round(v_P * timePlot[tt]):
                    IsoPos.append(pathFinalP[i][j][:, round(v_P * timePlot[tt])-1])
                    IsoVtheta.append(vthetaAllP[i, j][round(v_P * timePlot[tt])-1])
                    pathid.append([i, j, round(v_P * timePlot[tt])-1])

            IsoMapP_i_tt[i][tt] = IsoMapData(
                IsoPos=np.round(np.array(IsoPos, dtype=np.float32), 2).T if len(IsoPos) > 0 else np.array([[]]),
                pathid=np.array(pathid, dtype=np.int32),
                IsoVtheta=np.round(np.array(IsoVtheta, dtype=np.float32), 4)
            )            
    # 如果需要绘图，调用 DrawIso 模块进行绘制
    if draw:
        try:
            from intercept.IsoMap.DrawIso import draw_iso

            draw_iso(Map, final_pathUAV, pathFinalP, vthetaAllP, interactive=draw_interactive)
        except Exception as e:
            print('调用 DrawIso 绘图失败：', e)

    return Map, IsoMapP_i_tt, pathFinalP




import numpy as np
import time
from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm # 确保此路径正确

def WH_main_obtainMapRef(Map, PointFrom, PointTo, flagAll):
    """
    计算 PointFrom 到 PointTo 之间的路径及速度偏角集合。
    
    参数:
    Map: 包含环境信息的字典 (obs, sure, r, obs_no_circle, outline_all, Stepsize, resolution)
    PointFrom: 起点坐标矩阵 (N x 2 或 N x 3)
    PointTo: 终点坐标矩阵 (M x 2 或 M x 3)
    flagAll: 1 表示计算所有组合；其他值表示仅在 i != j 时计算
    
    返回:
    Map: 更新后的 Map 字典
    final_pathE: 存储路径的字典，键为 (i, j)
    vthetaAllE: 存储速度角度集合的字典，键为 (i, j)
    """
    
    start_time = time.time()
    
    final_pathE = {}
    vthetaAllE = {}
    
    # 提取 Map 中的参数以提高读取速度
    obs = Map.get('obs')
    sure = Map.get('sure')
    r = Map.get('r')
    obs_no_circle = Map.get('obs_no_circle')
    outline_all = Map.get('outline_all')
    step_size = Map.get('Stepsize')
    resolution = Map.get('resolution')

    num_from = PointFrom.shape[0]
    num_to = PointTo.shape[0]

    for i in range(num_from):
        for j in range(num_to):
            # 判断逻辑：如果 flagAll 为 1，或者 i 不等于 j，则执行计算
            if flagAll == 1 or i != j:
                print(f'#################################PointFrom {i}, PointTo {j}')
                
                # 调用 Dubins 路径规划
                # 注意：MATLAB 中函数返回 [path, open, close]，Python 中我们也解构它们
                path_segments, _, _,_ = A_dubins_nocircle_swarm(
                    PointFrom[i, :], 
                    PointTo[j, :], 
                    obs, 
                    sure, 
                    r, 
                    obs_no_circle, 
                    outline_all, 
                    step_size, 
                    1, 
                    resolution,
                    depth=0,max_depth=1
                )
                
                final_pathE[i, j] = path_segments
                
                # 提取 vtheta_all
                # MATLAB 逻辑：从第二个节点开始 (tt_=1:length-1, 获取 tt_+1)
                # 对应 Python：从索引 1 开始到结束
                vtheta_list = []
                if path_segments is not None:
                    # 遍历路径段（跳过第一个点，模拟 MATLAB 的 tt_+1 逻辑）
                    for tt in range(1, len(path_segments)):
                        segment_vtheta = path_segments[tt].get('vtheta_all', [])
                        # segment_vtheta 假设是一个列表的列表，需要展平
                        for k_theta in segment_vtheta:
                            # 这里的 extend 相当于 MATLAB 的 [vthetaAll_, val]
                            if isinstance(k_theta, (list, np.ndarray)):
                                vtheta_list.extend(k_theta)
                            else:
                                vtheta_list.append(k_theta)
                
                vthetaAllE[i, j] = np.round(np.array(vtheta_list, dtype=np.float32), 4)

    end_time = time.time()
    print(f"路径计算完成，耗时: {end_time - start_time:.4f} 秒")

    return Map, final_pathE, vthetaAllE


import numpy as np
from intercept.IsoMap.obtainPath import obtainPath

def WH_main_obtainIso(Map, v, final_pathE, vthetaAllE, verse):
    """
    根据给定的路径和速度，生成等时线数据。

    参数:
        Map: 包含时间信息的字典（Map.numTime, Map.timePlot)
        v: 速度 (m/s)
        final_pathE: 存储最终路径的字典 (由 WH_main_obtainMapRef 生成)
        vthetaAllE: 存储所有速度角度信息的字典
        verse: 用于 obtainPath 函数的参数

    返回:
        pathFinal: 存储处理后路径的字典
        IsoMapTP2Iso_i_tt: 存储等时线数据的字典
    """
    # 画图
    lengtehPathE = []
    grid_size = np.max(list(final_pathE.keys()), axis=0) + 1
    ii, jj = grid_size[0], grid_size[1]

    pathFinal = [[None for _ in range(jj)] for _ in range(ii)]
    #  确保获取行数和列数

    for i in range(ii):
        for j in range(jj):
            # 确保不修改变量名，即便在Pythonic的风格中，可以使用get的方式避免报错
            if (i, j) in final_pathE and final_pathE[i, j] is not None:
                pathFinal[i][j] = obtainPath(final_pathE[i, j], vthetaAllE.get((i, j), np.array([])), 0)
                if pathFinal[i][j] is not None:
                    lengtehPathE.append(len(pathFinal[i][j][0, :])) # 添加路径长度

    # 计算等时线数据
    IsoMapTP2Iso_i_tt = [[{} for _ in range(Map['numTime'])] for _ in range(ii)]
    v_E = v  # m/s
    PointNumIsoUav = {}

    for tt in range(Map['numTime']):  # 假设 Map 是一个字典
        for i in range(ii):
            IsoPos = []
            IsoVtheta = []
            pathid = []

            for j in range(jj):
                # 检查 pathFinal[i, j] 是否存在以及是否非空
                if pathFinal[i][j] is not None and len(pathFinal[i][j][0, :]) >= round(v_E * Map['timePlot'][tt]):
                        IsoPos.append(pathFinal[i][j][:, round(v_E * Map['timePlot'][tt])-1])
                        IsoVtheta.append(vthetaAllE.get((i, j), np.array([]))[round(v_E * Map['timePlot'][tt])-1])
                        pathid.append([i, j, round(v_E * Map['timePlot'][tt])-1])
            
            # 创建字典，存储数据
            IsoMapTP2Iso_i_tt[i][tt] = IsoMapData(
                IsoPos=np.round(np.array(IsoPos, dtype=np.float32), 2).T if len(IsoPos) > 0 else np.array([[]]),
                pathid=np.array(pathid, dtype=np.int32),
                IsoVtheta=np.round(np.array(IsoVtheta, dtype=np.float32), 4)
            )
    
    IsoMapIso2TP_i_tt = [[{} for _ in range(Map['numTime'])] for _ in range(ii)]

    if verse == 1:
            # 计算等时线数据

        v_E = v  # m/s

        for tt in range(Map['numTime']):  # 假设 Map 是一个字典
            for i in range(ii):
                IsoPos = []
                IsoVtheta = []
                pathid = []

                for j in range(jj):
                    # 检查 pathFinal[i, j] 是否存在以及是否非空
                    if pathFinal[j][i] is not None and len(pathFinal[j][i][0, :]) >= round(v_E * Map['timePlot'][tt]):
                            IsoPos.append(pathFinal[j][i][:, -round(v_E * Map['timePlot'][tt])])
                            IsoVtheta.append(vthetaAllE.get((j, i), np.array([]))[round(v_E * Map['timePlot'][tt])-1])
                            pathid.append([j, i, round(v_E * Map['timePlot'][tt])-1])
                
                # 创建字典，存储数据
                IsoMapIso2TP_i_tt[i][tt] = IsoMapData(
                    IsoPos=np.round(np.array(IsoPos, dtype=np.float32), 2).T if len(IsoPos) > 0 else np.array([[]]),
                    pathid=np.array(pathid, dtype=np.int32),
                    IsoVtheta=np.round(np.array(IsoVtheta, dtype=np.float32), 4)
                )

    return pathFinal, IsoMapTP2Iso_i_tt,IsoMapIso2TP_i_tt