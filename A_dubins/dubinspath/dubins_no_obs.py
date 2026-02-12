import numpy as np
from A_dubins.dubinspath.dubins_path import dubins_path
from A_dubins.coreCode.obtain_vtheta_path import obtain_vtheta_path

def Dubins_no_obs(Start_Point, End_Point, r_s, r_e, Stepsize, pos_id, parent_id):
    param_best = {
        'point': np.array([]),  # 切点集合
        'length': np.array([]),  # 每个路段长度
        'phy': np.array([]),  # 每个弧段的角度
        'center': np.array([]),  # 圆心
        'theta': np.array([]),
        'r': np.array([r_s, r_e]),
        'type': -1,  # 1-->RSR; 2-->RSL; 3-->LSL; 4-->LSR
        'pos_id': pos_id,
        'parent_id': parent_id
    }

    # 初始化 param1, param2, param3, param4
    param1 = {}
    param2 = {}
    param3 = {}
    param4 = {}

    LengthfsRSR = float('inf')
    LengthfsRSL = float('inf')
    LengthfsLSL = float('inf')
    LengthfsLSR = float('inf')

    Start_X, Start_Y, Start_Theta = Start_Point
    End_X, End_Y, End_Theta = End_Point

    # 计算初始圆心
    CenterRs_x = Start_X + r_s * np.cos(Start_Theta - np.pi / 2)
    CenterRs_y = Start_Y + r_s * np.sin(Start_Theta - np.pi / 2)
    CenterLs_x = Start_X + r_s * np.cos(Start_Theta + np.pi / 2)
    CenterLs_y = Start_Y + r_s * np.sin(Start_Theta + np.pi / 2)

    # 计算末位置圆心
    CenterRf_x = End_X + r_e * np.cos(End_Theta - np.pi / 2)
    CenterRf_y = End_Y + r_e * np.sin(End_Theta - np.pi / 2)
    CenterLf_x = End_X + r_e * np.cos(End_Theta + np.pi / 2)
    CenterLf_y = End_Y + r_e * np.sin(End_Theta + np.pi / 2)

    # ------------------------无障碍物RSR情况------------------------------------------
    Centerfs1_dis = np.sqrt((CenterRs_x - CenterRf_x) ** 2 + (CenterRs_y - CenterRf_y) ** 2)
    alphafs1_out = np.arcsin((r_e - r_s) / Centerfs1_dis)
    beltfs1 = np.arctan2((CenterRf_y - CenterRs_y), (CenterRf_x - CenterRs_x))

    if not np.isnan(alphafs1_out):
        Thetafs_RSRG0 = beltfs1 + alphafs1_out + np.pi / 2
        Thetafs_RSRF0 = beltfs1 + alphafs1_out + np.pi / 2
        Thetafs_RSRG = np.mod(Thetafs_RSRG0, 2 * np.pi)
        Thetafs_RSRF = np.mod(Thetafs_RSRF0, 2 * np.pi)

        xfs_RSRG = CenterRs_x + r_s * np.cos(Thetafs_RSRG)
        yfs_RSRG = CenterRs_y + r_s * np.sin(Thetafs_RSRG)
        xfs_RSRF = CenterRf_x + r_e * np.cos(Thetafs_RSRF)
        yfs_RSRF = CenterRf_y + r_e * np.sin(Thetafs_RSRF)

        phyfs_RSRs = np.mod(Start_Theta - Thetafs_RSRG + np.pi / 2, 2 * np.pi)
        phyfs_RSRf = np.mod(Thetafs_RSRF - End_Theta + 3 * np.pi / 2, 2 * np.pi)

        Lengthfs_RSR = [phyfs_RSRs * r_s,
                        np.sqrt((xfs_RSRF - xfs_RSRG) ** 2 + (yfs_RSRF - yfs_RSRG) ** 2),
                        phyfs_RSRf * r_e]
        LengthfsRSR = sum(Lengthfs_RSR)

        Pointfs_RSR = [[Start_X, xfs_RSRG, xfs_RSRF, End_X],
                       [Start_Y, yfs_RSRG, yfs_RSRF, End_Y]]

        vthetafs_RSR = np.arctan2((Pointfs_RSR[1][2] - Pointfs_RSR[1][1]),
                                  (Pointfs_RSR[0][2] - Pointfs_RSR[0][1]))

        Phyfs_RSR = [phyfs_RSRs, phyfs_RSRf]
        Thetafs_RSR = [Start_Theta + np.pi / 2, Thetafs_RSRG, Thetafs_RSRF]
        Centerfs_RSR = [[CenterRs_x, CenterRf_x], [CenterRs_y, CenterRf_y]]

        param1 = {
            'point': np.array(Pointfs_RSR),
            'length': np.array(Lengthfs_RSR),
            'phy': np.array(Phyfs_RSR),
            'theta': np.array(Thetafs_RSR),
            'center': np.array(Centerfs_RSR),
            'r': np.array([r_s, r_e]),
            'type': 1,
            'vtheta': np.array([Start_Theta, vthetafs_RSR, End_Theta]),
            'vtheta_plot': np.array([Start_Theta, vthetafs_RSR, End_Theta]),
            'Length': sum(Lengthfs_RSR)
        }

        path1 = dubins_path(param1, Stepsize)
        param1['path'] = path1
        param1['pos_id'] = pos_id
        param1['parent_id'] = parent_id

    # ------------------------无障碍物RSL情况------------------------------------------
    # 右转-左转 (RSL)
    Centerfs2_dis = np.sqrt((CenterRs_x - CenterLf_x) ** 2 + (CenterRs_y - CenterLf_y) ** 2)
    alphafs1_in = np.arcsin((r_e + r_s) / Centerfs2_dis)
    beltfs2 = np.arctan2((CenterLf_y - CenterRs_y), (CenterLf_x - CenterRs_x))

    if not np.isnan(alphafs1_in):
        Thetafs_RSLG0 = beltfs2 - alphafs1_in + np.pi / 2
        Thetafs_RSLF0 = beltfs2 - alphafs1_in + 3 * np.pi / 2
        Thetafs_RSLG = np.mod(Thetafs_RSLG0, 2 * np.pi)
        Thetafs_RSLF = np.mod(Thetafs_RSLF0, 2 * np.pi)

        xfs_RSLG = CenterRs_x + r_s * np.cos(Thetafs_RSLG)
        yfs_RSLG = CenterRs_y + r_s * np.sin(Thetafs_RSLG)
        xfs_RSLF = CenterLf_x + r_e * np.cos(Thetafs_RSLF)
        yfs_RSLF = CenterLf_y + r_e * np.sin(Thetafs_RSLF)

        phyfs_RSLs = np.mod(Start_Theta - Thetafs_RSLG + np.pi / 2, 2 * np.pi)
        phyfs_RSLf = np.mod(End_Theta - Thetafs_RSLF + 3 * np.pi / 2, 2 * np.pi)

        Lengthfs_RSL = [phyfs_RSLs * r_s,
                        np.sqrt((xfs_RSLF - xfs_RSLG) ** 2 + (yfs_RSLF - yfs_RSLG) ** 2),
                        phyfs_RSLf * r_e]
        LengthfsRSL = sum(Lengthfs_RSL)

        Pointfs_RSL = [[Start_X, xfs_RSLG, xfs_RSLF, End_X],
                       [Start_Y, yfs_RSLG, yfs_RSLF, End_Y]]

        vthetafs_RSL = np.arctan2((Pointfs_RSL[1][2] - Pointfs_RSL[1][1]),
                                  (Pointfs_RSL[0][2] - Pointfs_RSL[0][1]))

        Phyfs_RSL = [phyfs_RSLs, phyfs_RSLf]
        Thetafs_RSL = [Start_Theta + np.pi / 2, Thetafs_RSLG, Thetafs_RSLF]
        Centerfs_RSL = [[CenterRs_x, CenterLf_x], [CenterRs_y, CenterLf_y]]

        param2 = {
            'point': np.array(Pointfs_RSL),
            'length': np.array(Lengthfs_RSL),
            'phy': np.array(Phyfs_RSL),
            'theta': np.array(Thetafs_RSL),
            'center': np.array(Centerfs_RSL),
            'r': np.array([r_s, r_e]),
            'type': 2,
            'vtheta': np.array([Start_Theta, vthetafs_RSL, End_Theta]),
            'vtheta_plot': np.array([Start_Theta, vthetafs_RSL, End_Theta]),
            'Length': sum(Lengthfs_RSL)
        }

        path2 = dubins_path(param2, Stepsize)
        param2['path'] = path2
        param2['pos_id'] = pos_id
        param2['parent_id'] = parent_id

    # ------------------------无障碍物LSL情况------------------------------------------
    # 左转-左转 (LSL)
    Centerfs3_dis = np.sqrt((CenterLs_x - CenterLf_x) ** 2 + (CenterLs_y - CenterLf_y) ** 2)
    alphafs2_out = np.arcsin((r_e - r_s) / Centerfs3_dis)
    beltfs3 = np.arctan2((CenterLf_y - CenterLs_y), (CenterLf_x - CenterLs_x))

    if not np.isnan(alphafs2_out):
        Thetafs_LSLG0 = beltfs3 - alphafs2_out + 3 * np.pi / 2
        Thetafs_LSLF0 = beltfs3 - alphafs2_out + 3 * np.pi / 2
        Thetafs_LSLG = np.mod(Thetafs_LSLG0, 2 * np.pi)
        Thetafs_LSLF = np.mod(Thetafs_LSLF0, 2 * np.pi)

        xfs_LSLG = CenterLs_x + r_s * np.cos(Thetafs_LSLG)
        yfs_LSLG = CenterLs_y + r_s * np.sin(Thetafs_LSLG)
        xfs_LSLF = CenterLf_x + r_e * np.cos(Thetafs_LSLF)
        yfs_LSLF = CenterLf_y + r_e * np.sin(Thetafs_LSLF)

        phyfs_LSLs = np.mod(Thetafs_LSLG - Start_Theta + np.pi / 2, 2 * np.pi)
        phyfs_LSLf = np.mod(End_Theta - Thetafs_LSLF + 3 * np.pi / 2, 2 * np.pi)

        Lengthfs_LSL = [phyfs_LSLs * r_s,
                        np.sqrt((xfs_LSLF - xfs_LSLG) ** 2 + (yfs_LSLF - yfs_LSLG) ** 2),
                        phyfs_LSLf * r_e]
        LengthfsLSL = sum(Lengthfs_LSL)

        Pointfs_LSL = [[Start_X, xfs_LSLG, xfs_LSLF, End_X],
                       [Start_Y, yfs_LSLG, yfs_LSLF, End_Y]]

        vthetafs_LSL = np.arctan2((Pointfs_LSL[1][2] - Pointfs_LSL[1][1]),
                                  (Pointfs_LSL[0][2] - Pointfs_LSL[0][1]))

        Phyfs_LSL = [phyfs_LSLs, phyfs_LSLf]
        Thetafs_LSL = [Start_Theta - np.pi / 2, Thetafs_LSLG, Thetafs_LSLF]
        Centerfs_LSL = [[CenterLs_x, CenterLf_x], [CenterLs_y, CenterLf_y]]

        param3 = {
            'point': np.array(Pointfs_LSL),
            'length': np.array(Lengthfs_LSL),
            'phy': np.array(Phyfs_LSL),
            'theta': np.array(Thetafs_LSL),
            'center': np.array(Centerfs_LSL),
            'r': np.array([r_s, r_e]),
            'type': 3,
            'vtheta': np.array([Start_Theta, vthetafs_LSL, End_Theta]),
            'vtheta_plot': np.array([Start_Theta, vthetafs_LSL, End_Theta]),
            'Length': sum(Lengthfs_LSL)
        }

        path3 = dubins_path(param3, Stepsize)
        param3['path'] = path3
        param3['pos_id'] = pos_id
        param3['parent_id'] = parent_id

    # ------------------------无障碍物LSR情况------------------------------------------
    # 左转-右转 (LSR)
    Centerfs4_dis = np.sqrt((CenterLs_x - CenterRf_x) ** 2 + (CenterLs_y - CenterRf_y) ** 2)
    alphafs2_in = np.arcsin((r_e + r_s) / Centerfs4_dis)
    beltfs4 = np.arctan2((CenterRf_y - CenterLs_y), (CenterRf_x - CenterLs_x))

    if not np.isnan(alphafs2_in):
        Thetafs_LSRG0 = beltfs4 + alphafs2_in + 3 * np.pi / 2
        Thetafs_LSRF0 = beltfs4 + alphafs2_in + np.pi / 2
        Thetafs_LSRG = np.mod(Thetafs_LSRG0, 2 * np.pi)
        Thetafs_LSRF = np.mod(Thetafs_LSRF0, 2 * np.pi)

        xfs_LSRG = CenterLs_x + r_s * np.cos(Thetafs_LSRG)
        yfs_LSRG = CenterLs_y + r_s * np.sin(Thetafs_LSRG)
        xfs_LSRF = CenterRf_x + r_e * np.cos(Thetafs_LSRF)
        yfs_LSRF = CenterRf_y + r_e * np.sin(Thetafs_LSRF)

        phyfs_LSRs = np.mod(Thetafs_LSRG - Start_Theta + np.pi / 2, 2 * np.pi)
        phyfs_LSRf = np.mod(Thetafs_LSRF - End_Theta + 3 * np.pi / 2, 2 * np.pi)

        Lengthfs_LSR = [phyfs_LSRs * r_s,
                        np.sqrt((xfs_LSRF - xfs_LSRG) ** 2 + (yfs_LSRF - yfs_LSRG) ** 2),
                        phyfs_LSRf * r_e]
        LengthfsLSR = sum(Lengthfs_LSR)

        Pointfs_LSR = [[Start_X, xfs_LSRG, xfs_LSRF, End_X],
                       [Start_Y, yfs_LSRG, yfs_LSRF, End_Y]]

        vthetafs_LSR = np.arctan2((Pointfs_LSR[1][2] - Pointfs_LSR[1][1]),
                                  (Pointfs_LSR[0][2] - Pointfs_LSR[0][1]))

        Phyfs_LSR = [phyfs_LSRs, phyfs_LSRf]
        Thetafs_LSR = [Start_Theta - np.pi / 2, Thetafs_LSRG, Thetafs_LSRF]
        Centerfs_LSR = [[CenterLs_x, CenterRf_x], [CenterLs_y, CenterRf_y]]

        param4 = {
            'point': np.array(Pointfs_LSR),
            'length': np.array(Lengthfs_LSR),
            'phy': np.array(Phyfs_LSR),
            'theta': np.array(Thetafs_LSR),
            'center': np.array(Centerfs_LSR),
            'r': np.array([r_s, r_e]),
            'type': 4,
            'vtheta': np.array([Start_Theta, vthetafs_LSR, End_Theta]),
            'vtheta_plot': np.array([Start_Theta, vthetafs_LSR, End_Theta]),
            'Length': sum(Lengthfs_LSR)
        }

        path4 = dubins_path(param4, Stepsize)
        param4['path'] = path4
        param4['pos_id'] = pos_id
        param4['parent_id'] = parent_id

    # 最短路径选择逻辑
    paramcell = [param1, param2, param3, param4]

    # 计算路径长度
    Dubins_Length1 = [LengthfsRSR, LengthfsRSL, LengthfsLSL, LengthfsLSR]

    index1 = np.argmin(Dubins_Length1)  # 找到最短的路径索引

    param_best = paramcell[index1]

    # 将结果返回给param_all
    param_all = []
    if param1:
        param_all.append(param1)
    if param2:
        param_all.append(param2)
    if param3:
        param_all.append(param3)
    if param4:
        param_all.append(param4)

    # print(11111)

    param_all = obtain_vtheta_path(param_all)

    # print(22222)

    return param_all, param_best