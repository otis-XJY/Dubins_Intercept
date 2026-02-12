import numpy as np
import math
from A_dubins.coreCode.obtain_vtheta_path import obtain_vtheta_path
from A_dubins.dubinspath.dubins_path import dubins_path


def Dubins_obs(Start_Point, End_Point, Avoid_Center, r_s, r_e, R, Stepsize, pos_id, parent_id):
    param_best = {
        'point': [],  # 切点集合
        'length': [],  # 每个路段长度
        'phy': [],  # 每个弧段的角度
        'center': [],  # 圆心
        'theta': [],
        'r': [r_s, r_e],
        'R': R,
        'type': -1,  # 1-->RSR;2-->RSL;3-->LSL;4-->LSR 
        'pos_id': pos_id
    }
    
    param5 = {}
    param6 = {}
    param7 = {}
    param8 = {}
    LengthRSR = float('inf')
    LengthRSL = float('inf')
    LengthLSL = float('inf')
    LengthLSR = float('inf')
    Theta_SRG = Theta_SLG = x_SRG = y_SRG = x_SRF = y_SRF = None
    x_SLG = y_SLG = x_SLF = y_SLF = None
    
    Start_X = Start_Point[0]
    Start_Y = Start_Point[1]
    Start_Theta = Start_Point[2]
    
    End_X = End_Point[0]
    End_Y = End_Point[1]
    End_Theta = End_Point[2]

    # 检查 center1 是否是一维数组
    if Avoid_Center.ndim == 1:  # 一维数组，直接访问
        Dx, Dy = Avoid_Center[0], Avoid_Center[1]
    else:  # 多行的情况 (n x 2)
        Dx, Dy = Avoid_Center[0, 0], Avoid_Center[0, 1]


    # Dx = Avoid_Center[0]
    # Dy = Avoid_Center[0]
    
    # 计算初始圆心
    CenterRs_x = Start_X + r_s * np.cos(Start_Theta - np.pi/2)
    CenterRs_y = Start_Y + r_s * np.sin(Start_Theta - np.pi/2)  # 右转圆
    CenterLs_x = Start_X + r_s * np.cos(Start_Theta + np.pi/2)
    CenterLs_y = Start_Y + r_s * np.sin(Start_Theta + np.pi/2)  # 左转圆
    
    # 计算末位置圆心
    CenterRf_x = End_X + r_e * np.cos(End_Theta - np.pi/2)
    CenterRf_y = End_Y + r_e * np.sin(End_Theta - np.pi/2)  # 右转圆
    CenterLf_x = End_X + r_e * np.cos(End_Theta + np.pi/2)
    CenterLf_y = End_Y + r_e * np.sin(End_Theta + np.pi/2)  # 左转圆
    
    # 终点圆与目标区域的切点（右转圆）
    cR = np.sqrt((CenterRf_x - Dx)**2 + (CenterRf_y - Dy)**2)
    alphaSR_out = np.arcsin((r_e - R) / cR)
    if not np.isnan(alphaSR_out):
        beltSR = np.arctan2((CenterRf_y - Dy), (CenterRf_x - Dx))
        
        Theta_SRG0 = beltSR + alphaSR_out + np.pi/2
        Theta_SRF0 = beltSR + alphaSR_out + np.pi/2
        Theta_SRG = np.mod(Theta_SRG0, 2 * np.pi)
        Theta_SRF = np.mod(Theta_SRF0, 2 * np.pi)
        
        x_SRG = Dx + R * np.cos(Theta_SRF)
        y_SRG = Dy + R * np.sin(Theta_SRF)  # 障碍圆上的点
        x_SRF = CenterRf_x + r_e * np.cos(Theta_SRG)
        y_SRF = CenterRf_y + r_e * np.sin(Theta_SRG)  # 终点圆上的点

    # 终点圆与目标区域的切点（左转圆）
    cL = np.sqrt((CenterLf_x - Dx) ** 2 + (CenterLf_y - Dy) ** 2)
    alphaSL_out = np.arcsin((r_e - R) / cL)
    if not np.isnan(alphaSL_out):
        beltSL = np.arctan2((CenterLf_y - Dy), (CenterLf_x - Dx))

        Theta_SLG0 = beltSL - alphaSL_out + 3 * np.pi / 2
        Theta_SLF0 = beltSL - alphaSL_out + 3 * np.pi / 2
        Theta_SLG = np.mod(Theta_SLG0, 2 * np.pi)
        Theta_SLF = np.mod(Theta_SLF0, 2 * np.pi)

        x_SLG = Dx + R * np.cos(Theta_SLF)
        y_SLG = Dy + R * np.sin(Theta_SLF)
        x_SLF = CenterLf_x + r_e * np.cos(Theta_SLG)
        y_SLF = CenterLf_y + r_e * np.sin(Theta_SLG)

    # 有障碍物情况：
    # 外公切 RSR_SR
    c1 = np.sqrt((CenterRs_x - Dx)**2 + (CenterRs_y - Dy)**2)
    alpha1_out = np.arcsin((R - r_s) / c1)
    if not np.isnan(alpha1_out) and Theta_SRG is not None:
    # 这样只有在终点切点成功计算的情况下，才会计算这一条路径
        belt1 = np.arctan2((Dy - CenterRs_y), (Dx - CenterRs_x))
        
        Theta_RSRG0 = belt1 + alpha1_out + np.pi/2
        Theta_RSRF0 = belt1 + alpha1_out + np.pi/2
        Theta_RSRG = np.mod(Theta_RSRG0, 2 * np.pi)
        Theta_RSRF = np.mod(Theta_RSRF0, 2 * np.pi)
        
        x_RSRG = CenterRs_x + r_s * np.cos(Theta_RSRG)
        y_RSRG = CenterRs_y + r_s * np.sin(Theta_RSRG)  # 起始点的右转圆切点
        x_RSRF = Dx + R * np.cos(Theta_RSRF)
        y_RSRF = Dy + R * np.sin(Theta_RSRF)  # 障碍圆的右
        
        phy_RSRs = np.mod((Start_Theta - Theta_RSRG + np.pi/2) , 2*np.pi)
        phy_RSRf = np.mod((Theta_RSRG - Theta_SRG) , 2*np.pi)
        phy_SRf = np.mod((Theta_SRF - End_Theta + 3*np.pi/2) , 2*np.pi)
        
        Length_RSR = np.zeros(5)
        Length_RSR[0] = phy_RSRs * r_s
        Length_RSR[1] = np.sqrt((x_RSRF - x_RSRG)**2 + (y_RSRF - y_RSRG)**2)
        Length_RSR[2] = phy_RSRf * R
        Length_RSR[3] = np.sqrt((x_SRF - x_SRG)**2 + (y_SRF - y_SRG)**2)
        Length_RSR[4] = phy_SRf * r_e
        LengthRSR = np.sum(Length_RSR)
        
        Point_RSR = np.array([
            [Start_X, x_RSRG, x_RSRF, x_SRG, x_SRF, End_X],
            [Start_Y, y_RSRG, y_RSRF, y_SRG, y_SRF, End_Y]
        ])
        
        vtheta_RSR = np.zeros(2)
        vtheta_RSR[0] = np.arctan2((Point_RSR[1, 2] - Point_RSR[1, 1]), (Point_RSR[0, 2] - Point_RSR[0, 1]))
        vtheta_RSR[1] = np.arctan2((Point_RSR[1, 4] - Point_RSR[1, 3]), (Point_RSR[0, 4] - Point_RSR[0, 3]))
        
        Phy_RSR = [phy_RSRs, phy_RSRf, phy_SRf]
        Theta_RSR = [Start_Theta + np.pi/2, Theta_RSRG, Theta_RSRF, Theta_SRG, Theta_SRF]
        Center_RSR = np.array([
            [CenterRs_x, Dx, CenterRf_x],
            [CenterRs_y, Dy, CenterRf_y]
        ])
        
        param5 = {
            'point': Point_RSR,
            'length': Length_RSR,
            'phy': np.array(Phy_RSR),
            'theta': np.array(Theta_RSR),
            'center': Center_RSR,
            'r': np.array([r_s, r_e]),
            'R': R,
            'type': 5,
            'vtheta': np.array([Start_Theta, vtheta_RSR[0], vtheta_RSR[1], End_Theta]),
            'vtheta_plot': np.array([Start_Theta, vtheta_RSR[0], vtheta_RSR[1], End_Theta]),
            'Length': np.sum(Length_RSR),
            'pos_id': pos_id,
            'parent_id': parent_id
        }
        
        # 需要dubins_path函数
        path5 = dubins_path(param5, Stepsize)
        param5['path'] = path5

    # 内公切 RSL_SL
    alpha1_in = np.arcsin((R + r_s) / c1)
    if not np.isnan(alpha1_in) and Theta_SLG is not None:
        belt1 = np.arctan2((Dy - CenterRs_y), (Dx - CenterRs_x))
        Theta_RSLG0 = belt1 - alpha1_in + np.pi / 2
        Theta_RSLF0 = belt1 - alpha1_in + 3 * np.pi / 2
        Theta_RSLG = np.mod(Theta_RSLG0, 2 * np.pi)
        Theta_RSLF = np.mod(Theta_RSLF0, 2 * np.pi)

        x_RSLG = CenterRs_x + r_s * np.cos(Theta_RSLG)
        y_RSLG = CenterRs_y + r_s * np.sin(Theta_RSLG)
        x_RSLF = Dx + R * np.cos(Theta_RSLF)
        y_RSLF = Dy + R * np.sin(Theta_RSLF)

        phy_RSLs = np.mod(Start_Theta - Theta_RSLG + np.pi / 2, 2 * np.pi)
        phy_RSLf = np.mod(Theta_SLG - Theta_RSLF, 2 * np.pi)
        phy_SLf = np.mod(End_Theta - Theta_SLF + 3 * np.pi / 2, 2 * np.pi)

        Length_RSL = np.zeros(5)
        Length_RSL[0] = phy_RSLs * r_s
        Length_RSL[1] = np.sqrt((x_RSLF - x_RSLG) ** 2 + (y_RSLF - y_RSLG) ** 2)
        Length_RSL[2] = phy_RSLf * R
        Length_RSL[3] = np.sqrt((x_SLF - x_SLG) ** 2 + (y_SLF - y_SLG) ** 2)
        Length_RSL[4] = phy_SLf * r_e
        LengthRSL = np.sum(Length_RSL)

        Point_RSL = np.array([
            [Start_X, x_RSLG, x_RSLF, x_SLG, x_SLF, End_X],
            [Start_Y, y_RSLG, y_RSLF, y_SLG, y_SLF, End_Y]
        ])

        vtheta_RSL = np.zeros(2)
        vtheta_RSL[0] = np.arctan2(Point_RSL[1, 2] - Point_RSL[1, 1], Point_RSL[0, 2] - Point_RSL[0, 1])
        vtheta_RSL[1] = np.arctan2(Point_RSL[1, 4] - Point_RSL[1, 3], Point_RSL[0, 4] - Point_RSL[0, 3])

        Phy_RSL = [phy_RSLs, phy_RSLf, phy_SLf]
        Theta_RSL = [Start_Theta + np.pi / 2, Theta_RSLG, Theta_RSLF, Theta_SLG, Theta_SLF]
        Center_RSL = np.array([
            [CenterRs_x, Dx, CenterLf_x],
            [CenterRs_y, Dy, CenterLf_y]
        ])

        param6 = {
            'point': Point_RSL,
            'length': Length_RSL,
            'phy': np.array(Phy_RSL),
            'theta': np.array(Theta_RSL),
            'center': Center_RSL,
            'r': np.array([r_s, r_e]),
            'R': R,
            'type': 6,
            'vtheta': np.array([Start_Theta, vtheta_RSL[0], vtheta_RSL[1], End_Theta]),
            'vtheta_plot': np.array([Start_Theta, vtheta_RSL[0], vtheta_RSL[1], End_Theta]),
            'Length': np.sum(Length_RSL),
            'pos_id': pos_id,
            'parent_id': parent_id
        }

        path6 = dubins_path(param6, Stepsize)
        param6['path'] = path6

    # 外公切 LSL_SL
    c2 = np.sqrt((CenterLs_x - Dx) ** 2 + (CenterLs_y - Dy) ** 2)
    alpha2_out = np.arcsin((R - r_s) / c2)
    if not np.isnan(alpha2_out) and Theta_SLG is not None:
        belt2 = np.arctan2((Dy - CenterLs_y), (Dx - CenterLs_x))

        Theta_LSLG0 = belt2 - alpha2_out + 3 * np.pi / 2
        Theta_LSLF0 = belt2 - alpha2_out + 3 * np.pi / 2
        Theta_LSLG = np.mod(Theta_LSLG0, 2 * np.pi)
        Theta_LSLF = np.mod(Theta_LSLF0, 2 * np.pi)

        x_LSLG = CenterLs_x + r_s * np.cos(Theta_LSLG)
        y_LSLG = CenterLs_y + r_s * np.sin(Theta_LSLG)
        x_LSLF = Dx + R * np.cos(Theta_LSLF)
        y_LSLF = Dy + R * np.sin(Theta_LSLF)

        phy_LSLs = np.mod(Theta_LSLG - Start_Theta + np.pi / 2, 2 * np.pi)
        phy_LSLf = np.mod(Theta_SLG - Theta_LSLF, 2 * np.pi)
        phy_SLf = np.mod(End_Theta - Theta_SLF + 3 * np.pi / 2, 2 * np.pi)

        Length_LSL = np.zeros(5)
        Length_LSL[0] = phy_LSLs * r_s
        Length_LSL[1] = np.sqrt((x_LSLF - x_LSLG) ** 2 + (y_LSLF - y_LSLG) ** 2)
        Length_LSL[2] = phy_LSLf * R
        Length_LSL[3] = np.sqrt((x_SLF - x_SLG) ** 2 + (y_SLF - y_SLG) ** 2)
        Length_LSL[4] = phy_SLf * r_e
        LengthLSL = np.sum(Length_LSL)

        Point_LSL = np.array([
            [Start_X, x_LSLG, x_LSLF, x_SLG, x_SLF, End_X],
            [Start_Y, y_LSLG, y_LSLF, y_SLG, y_SLF, End_Y]
        ])

        vtheta_LSL = np.zeros(2)
        vtheta_LSL[0] = np.arctan2(Point_LSL[1, 2] - Point_LSL[1, 1], Point_LSL[0, 2] - Point_LSL[0, 1])
        vtheta_LSL[1] = np.arctan2(Point_LSL[1, 4] - Point_LSL[1, 3], Point_LSL[0, 4] - Point_LSL[0, 3])

        Phy_LSL = [phy_LSLs, phy_LSLf, phy_SLf]
        Theta_LSL = [Start_Theta - np.pi / 2, Theta_LSLG, Theta_LSLF, Theta_SLG, Theta_SLF]
        Center_LSL = np.array([
            [CenterLs_x, Dx, CenterLf_x],
            [CenterLs_y, Dy, CenterLf_y]
        ])

        param7 = {
            'point': Point_LSL,
            'length': Length_LSL,
            'theta': np.array(Theta_LSL),
            'phy': np.array(Phy_LSL),
            'center': Center_LSL,
            'r': np.array([r_s, r_e]),
            'R': R,
            'type': 7,
            'vtheta': np.array([Start_Theta, vtheta_LSL[0], vtheta_LSL[1], End_Theta]),
            'vtheta_plot': np.array([Start_Theta, vtheta_LSL[0], vtheta_LSL[1], End_Theta]),
            'Length': np.sum(Length_LSL),
            'pos_id': pos_id,
            'parent_id': parent_id
        }

        path7 = dubins_path(param7, Stepsize)
        param7['path'] = path7

    # 内公切 LSR_SR
    alpha2_in = np.arcsin((R + r_s) / c2)
    if not np.isnan(alpha2_in) and Theta_SRG is not None:
        belt2 = np.arctan2((Dy - CenterLs_y), (Dx - CenterLs_x))
        Theta_LSRG0 = belt2 + alpha2_in + 3 * np.pi / 2
        Theta_LSRF0 = belt2 + alpha2_in + np.pi / 2
        Theta_LSRG = np.mod(Theta_LSRG0, 2 * np.pi)
        Theta_LSRF = np.mod(Theta_LSRF0, 2 * np.pi)

        x_LSRG = CenterLs_x + r_s * np.cos(Theta_LSRG)
        y_LSRG = CenterLs_y + r_s * np.sin(Theta_LSRG)
        x_LSRF = Dx + R * np.cos(Theta_LSRF)
        y_LSRF = Dy + R * np.sin(Theta_LSRF)

        phy_LSRs = np.mod(Theta_LSRG - Start_Theta + np.pi / 2, 2 * np.pi)
        phy_LSRf = np.mod(Theta_LSRF - Theta_SRG, 2 * np.pi)
        phy_SRf = np.mod(Theta_SRF - End_Theta + 3 * np.pi / 2, 2 * np.pi)

        Length_LSR = np.zeros(5)
        Length_LSR[0] = phy_LSRs * r_s
        Length_LSR[1] = np.sqrt((x_LSRF - x_LSRG) ** 2 + (y_LSRF - y_LSRG) ** 2)
        Length_LSR[2] = phy_LSRf * R
        Length_LSR[3] = np.sqrt((x_SRF - x_SRG) ** 2 + (y_SRF - y_SRG) ** 2)
        Length_LSR[4] = phy_SRf * r_e
        LengthLSR = np.sum(Length_LSR)

        Point_LSR = np.array([
            [Start_X, x_LSRG, x_LSRF, x_SRG, x_SRF, End_X],
            [Start_Y, y_LSRG, y_LSRF, y_SRG, y_SRF, End_Y]
        ])

        vtheta_LSR = np.zeros(2)
        vtheta_LSR[0] = np.arctan2(Point_LSR[1, 2] - Point_LSR[1, 1], Point_LSR[0, 2] - Point_LSR[0, 1])
        vtheta_LSR[1] = np.arctan2(Point_LSR[1, 4] - Point_LSR[1, 3], Point_LSR[0, 4] - Point_LSR[0, 3])

        Phy_LSR = [phy_LSRs, phy_LSRf, phy_SRf]
        Theta_LSR = [Start_Theta - np.pi / 2, Theta_LSRG, Theta_LSRF, Theta_SRG, Theta_SRF]
        Center_LSR = np.array([
            [CenterLs_x, Dx, CenterRf_x],
            [CenterLs_y, Dy, CenterRf_y]
        ])

        param8 = {
            'point': Point_LSR,
            'length': Length_LSR,
            'theta': np.array(Theta_LSR),
            'phy': np.array(Phy_LSR),
            'center': Center_LSR,
            'r': np.array([r_s, r_e]),
            'R': R,
            'type': 8,
            'vtheta': np.array([Start_Theta, vtheta_LSR[0], vtheta_LSR[1], End_Theta]),
            'vtheta_plot': np.array([Start_Theta, vtheta_LSR[0], vtheta_LSR[1], End_Theta]),
            'Length': np.sum(Length_LSR),
            'pos_id': pos_id,
            'parent_id': parent_id
        }

        path8 = dubins_path(param8, Stepsize)
        param8['path'] = path8

    # 选择最优路径
    paramcell = [param5, param6, param7, param8]
    Dubins_Length2 = [LengthRSR, LengthRSL, LengthLSL, LengthLSR]
    
    index2 = np.argmin(Dubins_Length2)  # 找到最短的路径索引

    param_best = paramcell[index2]
    
    param_all = []
    if param5:
        param_all.append(param5)
    if param6:
        param_all.append(param6)
    if param7:
        param_all.append(param7)
    if param8:
        param_all.append(param8)
    
    # 需要obtain_vtheta_path函数
    if param_all:
        param_all = obtain_vtheta_path(param_all)
    else:
        param_all = []
    
    return param_all, param_best