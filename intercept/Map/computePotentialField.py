
import numpy as np
from scipy.interpolate import griddata

def computePotentialField(obs_no_circle, E_TranPoint, PPStart_Point, Trans_Point, obs_mapsize, downsample_factor, r):
    """
    计算潜势场
    obs_no_circle: 不规则多边形障碍物列表，每个元素为Nx2数组
    E_TranPoint: 终点集合 Nx2
    PPStart_Point: 起点集合 Nx2
    Trans_Point: 中间点集合 Nx2
    obs_mapsize: 地图大小 [x_max, y_max]
    downsample_factor: 降采样因子
    r: 半径参数，用于高斯势场
    """

    sigma = r / np.sqrt(2)
    x_max, y_max = obs_mapsize[:,1]

    # 降采样后的网格
    x_max_ds = int(np.ceil(x_max / downsample_factor))
    y_max_ds = int(np.ceil(y_max / downsample_factor))
    PotentialField_ds = np.zeros((x_max_ds, y_max_ds))

    # 网格坐标
    X_grid, Y_grid = np.meshgrid(np.linspace(1, x_max, x_max_ds),
                                 np.linspace(1, y_max, y_max_ds))
    grid_points = np.column_stack((X_grid.ravel(), Y_grid.ravel()))

    # 调整起点和终点
    PPStart_Point[:,1] = PPStart_Point[:,1] + 2*r
    E_TranPoint[:,1] = E_TranPoint[:,1] - 2*r

    # 合并所有目标点
    all_points = np.vstack([E_TranPoint[:, :2], PPStart_Point[:, :2], Trans_Point[:, :2]])
    for obs in obs_no_circle:
        if obs is not None and len(obs) > 0:
            all_points = np.vstack([all_points, obs[:, :2]])

    num_points = all_points.shape[0]
    point_weights = np.ones(num_points)
    point_weights[len(E_TranPoint)+len(PPStart_Point):
                  len(E_TranPoint)+len(PPStart_Point)+len(Trans_Point)] = 2

    # 计算距离矩阵
    distances = np.sqrt(np.sum(grid_points**2, axis=1)[:,None] +
                        np.sum(all_points**2, axis=1)[None,:] -
                        2 * grid_points.dot(all_points.T))

    # 高斯势场
    PotentialField_ds = np.exp(-distances**2 / (2 * sigma**2))
    PotentialField_ds *= point_weights
    PotentialField_ds = PotentialField_ds.sum(axis=1)
    PotentialField_ds = PotentialField_ds.reshape(x_max_ds, y_max_ds)

    # 归一化到 [0,1]
    PotentialField_ds = (PotentialField_ds - np.min(PotentialField_ds)) / \
                       (np.max(PotentialField_ds) - np.min(PotentialField_ds))

    # 插值到原始分辨率
    X_full, Y_full = np.meshgrid(np.arange(1, x_max+1), np.arange(1, y_max+1))
    PotentialField = griddata((X_grid.ravel(), Y_grid.ravel()),
                              PotentialField_ds.ravel(),
                              (X_full, Y_full),
                              method='linear')

    return X_full, Y_full, PotentialField, X_grid, Y_grid, PotentialField_ds
