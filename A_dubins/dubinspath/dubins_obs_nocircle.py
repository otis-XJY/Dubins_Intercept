import numpy as np

from A_dubins.dubinspath.dubins_obs_nocircle_oneside_step1 import Dubins_obs_nocircle_oneside_step1
from A_dubins.coreCode.obtain_new_center import obtain_new_center
from A_dubins.coreCode.obtain_min_or_onlypath_from_safe import obtain_min_or_onlypath_from_safe
from A_dubins.coreCode.obtain_safeparam import obtain_safeparam
from A_dubins.coreCode.SafeFlag import if_safe_point
from A_dubins.search.update_first import update_first

def dubins_obs_nocircle(Start_Point, End_Point, outline_all, r_s, r_e, R, Stepsize, pos_id, parent_id, obs_no_circle,
                        total_field, resolution,depth,max_depth):
    print(f"from {parent_id} avoid {pos_id}")

    resolution=0.1*R
    # protect against empty outline input
    if outline_all is None or getattr(outline_all, 'size', 0) == 0:
        # no contour info - return no columns (caller will handle as empty)
        return [], []

    # 解析轮廓
    outline = outline_all[np.where(outline_all[:, 2] == pos_id)]

    points = outline[:, :2]
    x0, y0 = Start_Point[0], Start_Point[1]  # 起点坐标
    m = (End_Point[1] - Start_Point[1]) / (End_Point[0] - Start_Point[0])  # 计算直线斜率

    a = m
    b = -1
    c = y0 - m * x0

    if abs(a) != float('inf'):
        distances = (a * points[:, 0] + b * points[:, 1] + c) / np.sqrt(a ** 2 + b ** 2)
    else:
        distances = points[:, 0] - Start_Point[0]

    max_distance, max_index = np.max(distances), np.argmax(distances)
    min_distance, min_index = np.min(distances), np.argmin(distances)

    outline_all_positive = outline[distances >= 0]
    outline_all_negative = outline[distances < 0]

    farthest_point_max = points[max_index, :]
    farthest_point_min = points[min_index, :]

    flag_safe_max = if_safe_point(obs_no_circle, farthest_point_max, 0)
    flag_safe_min = if_safe_point(obs_no_circle, farthest_point_min, 0)

    # 穿过检查
    if outline_all_positive.size > 0 and outline_all_negative.size > 0:
        print('1')
        if flag_safe_max == 1:
            print('1,1')
            param_safe1, param_best1,Insert1 = Dubins_obs_nocircle_oneside_step1(Start_Point, End_Point, farthest_point_max,
                                                                         outline_all, outline, r_s, r_e, R,
                                                                         Stepsize, pos_id, parent_id, obs_no_circle,resolution)

        if flag_safe_min == 1:
            print('1,2')
            param_safe2, param_best2,Insert2 = Dubins_obs_nocircle_oneside_step1(Start_Point, End_Point, farthest_point_min,
                                                                         outline_all, outline, r_s, r_e, R,
                                                                         Stepsize, pos_id, parent_id, obs_no_circle,resolution)
        

        if Insert1['flagSucced'] and Insert2['flagSucced']:
            
            from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm
            if Insert1['End_Point_Cost'] < Insert2['End_Point_Cost']:     
                Insertfinal=Insert1           

            else:
                Insertfinal=Insert2 

            if Insertfinal['End_Point_now'] is not None:

                print('需要插入点'+str(Insertfinal['End_Point_Cost']))
                _,_,_,param_all_2 = A_dubins_nocircle_swarm(
                    Start_Point, Insertfinal['End_Point_now'], 0, 0, r_s, obs_no_circle, outline_all, 
                    Stepsize, 0,
                    resolution,
                    depth=depth+1,
                    max_depth=max_depth
                )
                param_safe1=update_first(Insertfinal['param_all_1'], param_all_2)
        



    # 一侧=全负
    elif outline_all_positive.size == 0:
        print('2')
        center_new, poit_num = obtain_new_center(a, b, c, Start_Point, R, farthest_point_max, 1, total_field, 1,
                                                 resolution)
        flag_safe = if_safe_point(obs_no_circle, center_new[0], r_e)
        pose = 1
        while flag_safe == 0 and pose < poit_num:

            flag_safe = if_safe_point(obs_no_circle, center_new[pose], r_e)
            pose += 1

        if flag_safe == 1:
            print('2,1')
            param_safe1, param_best1,Insert1 = Dubins_obs_nocircle_oneside_step1(Start_Point, End_Point, center_new,
                                                                         outline_all, outline, r_s, r_e, R,
                                                                         Stepsize, pos_id, parent_id, obs_no_circle,resolution)
        # 第二次尝试：基于 negative 点集合找最小（最远的负方向）
        distance_all_negative = distances[distances < 0]
        points_negative = points[distances < 0]
        # if distance_all_negative.size > 0:
        _, idx_local = np.min(distance_all_negative), np.argmin(distance_all_negative)
        farthest_point_min = points_negative[idx_local, :]

        center_new2, poit_num2 = obtain_new_center(a, b, c, Start_Point, R, farthest_point_min, 2, total_field, 1, resolution)
        flag_safe2 = if_safe_point(obs_no_circle, center_new2[0], r_e)
        pose2 = 1
        while not flag_safe2 and pose2 < poit_num2:
            flag_safe2 = if_safe_point(obs_no_circle, center_new2[pose2], r_e)
            pose2 += 1

        if flag_safe2:
            print('2,2')
            param_safe2, param_best2,Insert2 = Dubins_obs_nocircle_oneside_step1(
                Start_Point, End_Point, center_new2, outline_all, outline,
                r_s, r_e, R, Stepsize, pos_id, parent_id, obs_no_circle,resolution
            )
        
        if Insert1['flagSucced'] and Insert2['flagSucced']:
            
            from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm
            if Insert1['End_Point_Cost'] < Insert2['End_Point_Cost']:     
                Insertfinal=Insert1           

            else:
                Insertfinal=Insert2 

            if Insertfinal['End_Point_now'] is not None:

                print('需要插入点'+str(Insertfinal['End_Point_Cost']))
                _,_,_,param_all_2 = A_dubins_nocircle_swarm(
                    Start_Point, Insertfinal['End_Point_now'], 0, 0, r_s, obs_no_circle, outline_all, 
                    Stepsize, 0,
                    resolution,
                    depth=depth+1,
                    max_depth=max_depth
                )
                param_safe1=update_first(Insertfinal['param_all_1'], param_all_2)


    # 另一侧=全正
    elif outline_all_negative.size == 0:
        print('3')
        center_new, poit_num = obtain_new_center(a, b, c, Start_Point, R, farthest_point_min, 1, total_field, 1,
                                                 resolution)
        flag_safe = if_safe_point(obs_no_circle, center_new[0], r_e)
        pose = 1
        while flag_safe == 0 and pose < poit_num:

            flag_safe = if_safe_point(obs_no_circle, center_new[pose], r_e)
            pose += 1

        if flag_safe == 1:
            print('3,1')
            param_safe1, param_best1,Insert1 = Dubins_obs_nocircle_oneside_step1(Start_Point, End_Point, center_new,
                                                                         outline_all, outline, r_s, r_e, R,
                                                                         Stepsize, pos_id, parent_id, obs_no_circle,resolution)

        # 第二次尝试：基于 positive 点集合找最小（最远的正方向）
        distance_all_positive  = distances[distances > 0]
        points_positive  = points[distances > 0]
        # if distance_all_negative.size > 0:
        _, idx_local = np.min(distance_all_positive), np.argmin(distance_all_positive )
        farthest_point_max  = points_positive [idx_local, :]

        center_new2, poit_num2 = obtain_new_center(a, b, c, Start_Point, R, farthest_point_max, 2, total_field, 1, resolution)
        flag_safe2 = if_safe_point(obs_no_circle, center_new2[0], r_e)
        pose2 = 1
        while not flag_safe2 and pose2 < poit_num2:
            flag_safe2 = if_safe_point(obs_no_circle, center_new2[pose2], r_e)
            pose2 += 1

        if flag_safe2:
            print('3,2')
            param_safe2, param_best2,Insert2 = Dubins_obs_nocircle_oneside_step1(
                Start_Point, End_Point, center_new2, outline_all, outline,
                r_s, r_e, R, Stepsize, pos_id, parent_id, obs_no_circle,resolution
            )

        if Insert1['flagSucced'] and Insert2['flagSucced']:
            
            from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm
            if Insert1['End_Point_Cost'] < Insert2['End_Point_Cost']:     
                Insertfinal=Insert1           

            else:
                Insertfinal=Insert2 

            if Insertfinal['End_Point_now'] is not None:

                print('需要插入点'+str(Insertfinal['End_Point_Cost']))
                _,_,_,param_all_2 = A_dubins_nocircle_swarm(
                    Start_Point, Insertfinal['End_Point_now'], 0, 0, r_s, obs_no_circle, outline_all, 
                    Stepsize, 0,
                    resolution,
                    depth=depth+1,
                    max_depth=max_depth
                )
                param_safe1=update_first(Insertfinal['param_all_1'], param_all_2)
    # 返回结果
    param_all = []
    if 'param_safe1' in locals():
        param_all.append(param_safe1)
    if 'param_safe2' in locals():
        param_all.append(param_safe2)
    
    # 如果没有任何安全路径，返回与列数约定一致的空列（list-of-lists）
    if not param_all:
        # use an empty column to indicate no candidates for this obstacle
        param_all = [[]]

    param_best = []

    return param_all, param_best
