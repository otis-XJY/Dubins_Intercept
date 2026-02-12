import numpy as np

def dubins_path(param, Stepsize):
    """
    根据路径类型调用不同的Dubins路径生成函数
    """
    t = param['type']
    if t == 1:
        path = path_fs_RSR(param, Stepsize)
    elif t == 2:
        path = path_fs_RSL(param, Stepsize)
    elif t == 3:
        path = path_fs_LSL(param, Stepsize)
    elif t == 4:
        path = path_fs_LSR(param, Stepsize)
    elif t == 5:
        path = path_RSR(param, Stepsize)
    elif t == 6:
        path = path_RSL(param, Stepsize)
    elif t == 7:
        path = path_LSL(param, Stepsize)
    elif t == 8:
        path = path_LSR(param, Stepsize)
    else:
        raise ValueError("未知路径类型")
    return path

# --------------------------- 无障碍物路径函数 ---------------------------
def path_fs_RSR(param, Stepsize):
    # 根据角度和线段长度计算步数
    step = [int(param['length'][i]/Stepsize) for i in range(3)]
    # 直线路段
    Line1x = np.linspace(param['point'][0][1], param['point'][0][2], step[1])
    Line1y = np.linspace(param['point'][1][1], param['point'][1][2], step[1])
    # 圆弧段
    C1 = np.linspace(0, param['phy'][0], step[0])
    C2 = np.linspace(0, param['phy'][1], step[2])
    Round1x = param['center'][0][0] + param['r'][0]*np.cos(param['theta'][0] - C1)
    Round1y = param['center'][1][0] + param['r'][0]*np.sin(param['theta'][0] - C1)
    Round2x = param['center'][0][1] + param['r'][1]*np.cos(param['theta'][2] - C2)
    Round2y = param['center'][1][1] + param['r'][1]*np.sin(param['theta'][2] - C2)
    return [np.vstack((Round1x, Round1y)).tolist(), np.vstack((Line1x, Line1y)).tolist(), np.vstack((Round2x, Round2y)).tolist()]

def path_fs_RSL(param, Stepsize):
    step = [int(param['length'][i]/Stepsize) for i in range(3)]
    Line1x = np.linspace(param['point'][0][1], param['point'][0][2], step[1])
    Line1y = np.linspace(param['point'][1][1], param['point'][1][2], step[1])
    C1 = np.linspace(0, param['phy'][0], step[0])
    C2 = np.linspace(0, param['phy'][1], step[2])
    Round1x = param['center'][0][0] + param['r'][0]*np.cos(param['theta'][0] - C1)
    Round1y = param['center'][1][0] + param['r'][0]*np.sin(param['theta'][0] - C1)
    Round2x = param['center'][0][1] + param['r'][1]*np.cos(param['theta'][2] + C2)
    Round2y = param['center'][1][1] + param['r'][1]*np.sin(param['theta'][2] + C2)
    return [np.vstack((Round1x, Round1y)).tolist(), np.vstack((Line1x, Line1y)).tolist(), np.vstack((Round2x, Round2y)).tolist()]

def path_fs_LSL(param, Stepsize):
    step = [int(param['length'][i]/Stepsize) for i in range(3)]
    Line1x = np.linspace(param['point'][0][1], param['point'][0][2], step[1])
    Line1y = np.linspace(param['point'][1][1], param['point'][1][2], step[1])
    C1 = np.linspace(0, param['phy'][0], step[0])
    C2 = np.linspace(0, param['phy'][1], step[2])
    Round1x = param['center'][0][0] + param['r'][0]*np.cos(param['theta'][0] + C1)
    Round1y = param['center'][1][0] + param['r'][0]*np.sin(param['theta'][0] + C1)
    Round2x = param['center'][0][1] + param['r'][1]*np.cos(param['theta'][2] + C2)
    Round2y = param['center'][1][1] + param['r'][1]*np.sin(param['theta'][2] + C2)
    return [np.vstack((Round1x, Round1y)).tolist(), np.vstack((Line1x, Line1y)).tolist(), np.vstack((Round2x, Round2y)).tolist()]

def path_fs_LSR(param, Stepsize):
    step = [int(param['length'][i]/Stepsize) for i in range(3)]
    Line1x = np.linspace(param['point'][0][1], param['point'][0][2], step[1])
    Line1y = np.linspace(param['point'][1][1], param['point'][1][2], step[1])
    C1 = np.linspace(0, param['phy'][0], step[0])
    C2 = np.linspace(0, param['phy'][1], step[2])
    Round1x = param['center'][0][0] + param['r'][0]*np.cos(param['theta'][0] + C1)
    Round1y = param['center'][1][0] + param['r'][0]*np.sin(param['theta'][0] + C1)
    Round2x = param['center'][0][1] + param['r'][1]*np.cos(param['theta'][2] - C2)
    Round2y = param['center'][1][1] + param['r'][1]*np.sin(param['theta'][2] - C2)
    return [np.vstack((Round1x, Round1y)).tolist(), np.vstack((Line1x, Line1y)).tolist(), np.vstack((Round2x, Round2y)).tolist()]

# --------------------------- 有障碍物或多段路径函数 ---------------------------
def path_RSR(param, Stepsize):
    step = [int(param['length'][i]/Stepsize) for i in range(5)]
    Line1x = np.linspace(param['point'][0][1], param['point'][0][2], step[1])
    Line1y = np.linspace(param['point'][1][1], param['point'][1][2], step[1])
    Line2x = np.linspace(param['point'][0][3], param['point'][0][4], step[3])
    Line2y = np.linspace(param['point'][1][3], param['point'][1][4], step[3])
    C1 = np.linspace(0, param['phy'][0], step[0])
    C2 = np.linspace(0, param['phy'][1], step[2])
    C3 = np.linspace(0, param['phy'][2], step[4])
    Round1x = param['center'][0][0] + param['r'][0]*np.cos(param['theta'][0] - C1)
    Round1y = param['center'][1][0] + param['r'][0]*np.sin(param['theta'][0] - C1)
    Round2x = param['center'][0][1] + param['R']*np.cos(param['theta'][2] - C2)
    Round2y = param['center'][1][1] + param['R']*np.sin(param['theta'][2] - C2)
    Round3x = param['center'][0][2] + param['r'][1]*np.cos(param['theta'][4] - C3)
    Round3y = param['center'][1][2] + param['r'][1]*np.sin(param['theta'][4] - C3)
    return [np.vstack((Round1x, Round1y)).tolist(), np.vstack((Line1x, Line1y)).tolist(), 
            np.vstack((Round2x, Round2y)).tolist(), np.vstack((Line2x, Line2y)).tolist(), np.vstack((Round3x, Round3y)).tolist()]

def path_RSL(param, Stepsize):
    step = [int(param['length'][i]/Stepsize) for i in range(5)]
    Line1x = np.linspace(param['point'][0][1], param['point'][0][2], step[1])
    Line1y = np.linspace(param['point'][1][1], param['point'][1][2], step[1])
    Line2x = np.linspace(param['point'][0][3], param['point'][0][4], step[3])
    Line2y = np.linspace(param['point'][1][3], param['point'][1][4], step[3])
    C1 = np.linspace(0, param['phy'][0], step[0])
    C2 = np.linspace(0, param['phy'][1], step[2])
    C3 = np.linspace(0, param['phy'][2], step[4])
    Round1x = param['center'][0][0] + param['r'][0]*np.cos(param['theta'][0] - C1)
    Round1y = param['center'][1][0] + param['r'][0]*np.sin(param['theta'][0] - C1)
    Round2x = param['center'][0][1] + param['R']*np.cos(param['theta'][2] + C2)
    Round2y = param['center'][1][1] + param['R']*np.sin(param['theta'][2] + C2)
    Round3x = param['center'][0][2] + param['r'][1]*np.cos(param['theta'][4] + C3)
    Round3y = param['center'][1][2] + param['r'][1]*np.sin(param['theta'][4] + C3)
    return [np.vstack((Round1x, Round1y)).tolist(), np.vstack((Line1x, Line1y)).tolist(), 
            np.vstack((Round2x, Round2y)).tolist(), np.vstack((Line2x, Line2y)).tolist(), np.vstack((Round3x, Round3y)).tolist()]

def path_LSL(param, Stepsize):
    step = [int(param['length'][i]/Stepsize) for i in range(5)]
    Line1x = np.linspace(param['point'][0][1], param['point'][0][2], step[1])
    Line1y = np.linspace(param['point'][1][1], param['point'][1][2], step[1])
    Line2x = np.linspace(param['point'][0][3], param['point'][0][4], step[3])
    Line2y = np.linspace(param['point'][1][3], param['point'][1][4], step[3])
    C1 = np.linspace(0, param['phy'][0], step[0])
    C2 = np.linspace(0, param['phy'][1], step[2])
    C3 = np.linspace(0, param['phy'][2], step[4])
    Round1x = param['center'][0][0] + param['r'][0]*np.cos(param['theta'][0] + C1)
    Round1y = param['center'][1][0] + param['r'][0]*np.sin(param['theta'][0] + C1)
    Round2x = param['center'][0][1] + param['R']*np.cos(param['theta'][2] + C2)
    Round2y = param['center'][1][1] + param['R']*np.sin(param['theta'][2] + C2)
    Round3x = param['center'][0][2] + param['r'][1]*np.cos(param['theta'][4] + C3)
    Round3y = param['center'][1][2] + param['r'][1]*np.sin(param['theta'][4] + C3)
    return [np.vstack((Round1x, Round1y)).tolist(), np.vstack((Line1x, Line1y)).tolist(), 
            np.vstack((Round2x, Round2y)).tolist(), np.vstack((Line2x, Line2y)).tolist(), np.vstack((Round3x, Round3y)).tolist()]

def path_LSR(param, Stepsize):
    step = [int(param['length'][i]/Stepsize) for i in range(5)]
    Line1x = np.linspace(param['point'][0][1], param['point'][0][2], step[1])
    Line1y = np.linspace(param['point'][1][1], param['point'][1][2], step[1])
    Line2x = np.linspace(param['point'][0][3], param['point'][0][4], step[3])
    Line2y = np.linspace(param['point'][1][3], param['point'][1][4], step[3])
    C1 = np.linspace(0, param['phy'][0], step[0])
    C2 = np.linspace(0, param['phy'][1], step[2])
    C3 = np.linspace(0, param['phy'][2], step[4])
    Round1x = param['center'][0][0] + param['r'][0]*np.cos(param['theta'][0] + C1)
    Round1y = param['center'][1][0] + param['r'][0]*np.sin(param['theta'][0] + C1)
    Round2x = param['center'][0][1] + param['R']*np.cos(param['theta'][2] - C2)
    Round2y = param['center'][1][1] + param['R']*np.sin(param['theta'][2] - C2)
    Round3x = param['center'][0][2] + param['r'][1]*np.cos(param['theta'][4] - C3)
    Round3y = param['center'][1][2] + param['r'][1]*np.sin(param['theta'][4] - C3)
    return [np.vstack((Round1x, Round1y)).tolist(), np.vstack((Line1x, Line1y)).tolist(), 
            np.vstack((Round2x, Round2y)).tolist(), np.vstack((Line2x, Line2y)).tolist(), np.vstack((Round3x, Round3y)).tolist()]