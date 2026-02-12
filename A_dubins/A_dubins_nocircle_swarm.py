import numpy as np
import math
from collections import deque
import heapq

from A_dubins.dubinspath.dubins_obs_nocircle import dubins_obs_nocircle
from A_dubins.coreCode.obtain_short_param import obtain_short_param
from A_dubins.search.update_open_mayday import update_open_mayday
from A_dubins.coreCode.obtain_h import obtain_h
from A_dubins.search.check_in_open_new import check_in_open_new
from A_dubins.dubinspath.obtain_path import obtain_path
from A_dubins.dubinspath.dubins_no_obs import Dubins_no_obs
from A_dubins.coreCode.obtain_flag_safe import obtain_flag_safe
from A_dubins.coreCode.obtain_min_or_onlypath_from_safe import obtain_min_or_onlypath_from_safe
from A_dubins.coreCode.obtain_safeparam import obtain_safeparam
from A_dubins.search.update_close import update_close
from A_dubins.coreCode.exHelp import effective_len

def A_dubins_nocircle_swarm(Start_Point, End_Point, pos_idInsert, parent_idInsert, r, obs_no_circle, outline_all, Stepsize, total_field,
                            resolution,depth=0,max_depth=3):
    if depth>max_depth:
        return None,[],[],[]
    # A*算法开始
    open_ = np.array([])
    close = np.array([])
    open_f = []
    pos_id_mayday = []

    # 初始节点
    node = {
        'point': np.transpose([Start_Point[0:2]]),
        'vtheta': Start_Point[2],
        'g': 0,
        'f': 0 + obtain_h(Start_Point[0:2], End_Point[0:2]),
        'pos_id': -1,  # 起点为-1,终点为-2
        'parent_id': -1
    }
    close = np.append(close, node)

    # 产生初始路径（统一为 list-of-lists：每个元素为对应障碍/列的候选参数列表）
    param_all_start = []
    param_best_start = []
    param_all_start_, param_best_start_ = Dubins_no_obs(Start_Point, End_Point, r, r, Stepsize, -2, -1)
    # single column of candidates (list-of-lists: each element is a list of candidate params)
    param_all_start.append(param_all_start_)
    param_best_start.append(param_best_start_)

    flag_all, obs_id_all = obtain_flag_safe(1, param_all_start, obs_no_circle, outline_all)
    min_path, path_safe_flag, flag_min = obtain_min_or_onlypath_from_safe(1, param_all_start, flag_all)

    if path_safe_flag is not None and len(path_safe_flag) > 0:
        print('Find Goal!!')
        close = update_close(path_safe_flag[flag_min], param_all_start, End_Point, close)
        final_path,final_param = obtain_path(close, pos_idInsert, parent_idInsert)
        return final_path, open_, close,final_param

    else:
        obs_to_avoid = np.unique(obs_id_all)

        # 生成避障路线
        param_all = []
        param_best = []

        for i in range(len(obs_to_avoid)):
            param_all_, param_best_ = dubins_obs_nocircle(
                Start_Point, End_Point, outline_all, r, r, r, Stepsize,
                obs_to_avoid[i], -1, obs_no_circle, total_field, resolution,depth,max_depth
            )
            # dubins_obs_nocircle returns a list of columns; extend to flatten into top-level columns list
            if isinstance(param_all_, list):
                param_all.extend(param_all_)
            else:
                param_all.append(param_all_)
            if isinstance(param_best_, list):
                param_best.extend(param_best_)
            else:
                param_best.append(param_best_)

        # 检查避障路线的安全性
        flag_all, obs_id_all = obtain_flag_safe(2, param_all, obs_no_circle, outline_all)
        min_path, path_safe_flag, flag_min = obtain_min_or_onlypath_from_safe(1, param_all, flag_all)

        # param_safe is a list-of-lists: param_safe[obs_id] -> list of candidate params
        param_safe = obtain_safeparam(path_safe_flag, param_all)
        safe_obs_num = effective_len(param_safe)

        for i in range(safe_obs_num):
            open_f, open_ = check_in_open_new(param_safe, End_Point, open_f, open_, close, i)

        while len(open_f) == 0 or all(np.isnan(open_f)):
            print('open is empty! Mayday! even in begin!!')

            open_f, open_, pos_id_mayday = update_open_mayday(
                    Start_Point, End_Point, outline_all, r, Stepsize,
                    obs_no_circle, total_field, close, open_, open_f,
                    pos_id_mayday, resolution,depth,max_depth
                )
            
            times = sum(1 for x in pos_id_mayday if x == close[-1]['pos_id'])

            if times > 1:
                print(f'add open to close more {times - 1}')

                sorted_indices = np.argsort(open_f)

                if len(open_f) >= times - 1:
                    min_indices = sorted_indices[0:times - 1]


                    nodes_to_add = [open_[i] for i in min_indices]
                    close = np.append(close,nodes_to_add)
                    open_ = np.delete(open_, min_indices)
                    open_f = np.delete(open_f, min_indices)

                    break
                else:
                    print('sorry, no path in begin')
                    break

        # 选择最短节点
        flag_min = np.argmin(open_f)

        # 移除open，加入close
        close = np.append(close, [open_[flag_min]])
        open_ = np.delete(open_, flag_min)
        open_f = np.delete(open_f, flag_min)


        # 循环开始
        while True:
            param_all = []
            param_best=[]
            # 1. 提取 point 并确保其为 2D 数组，然后取最后一列
            p_last = np.atleast_2d(close[-1]['point'])[:, -1]

            # 2. 提取 vtheta 并确保其为 1D 数组，然后取最后一个值
            # 即使 vtheta 只是一个数字，np.atleast_1d 也会把它变成 [val]
            v_last = np.atleast_1d(close[-1]['vtheta'])[-1]
            Start_Point_=np.hstack([p_last,v_last])
            param_all_, param_best_ = Dubins_no_obs(
                Start_Point_,
                End_Point, r, r, Stepsize, -2, close[-1]['pos_id']
            )

            param_all.append(param_all_)
            param_best.append(param_best_)

            # shorten candidate lists, obtain_short_param expects list-of-groups
            param_all = obtain_short_param(param_all)
            flag_all, obs_id_all = obtain_flag_safe(1, param_all, obs_no_circle, outline_all)
            min_path, path_safe_flag, flag_min = obtain_min_or_onlypath_from_safe(1, param_all, flag_all)

            if path_safe_flag is not None and len(path_safe_flag) > 0:
                print('Find Goal!!')
                close = update_close(path_safe_flag[flag_min], param_all, End_Point, close)
                final_path,final_param = obtain_path(close, pos_idInsert, parent_idInsert)
                break

            obs_num = obs_id_all
            obs_to_avoid = np.unique(obs_num)

            rmove_id = []
            for i in range(len(obs_to_avoid)):
                if obs_to_avoid[i] == close[-1]['pos_id']:
                    rmove_id.append(i)

            # obs_to_avoid = np.delete(obs_to_avoid, rmove_id)

            # 生成避障路线
            param_all = []
            param_best = []

            for i in range(len(obs_to_avoid)):
                SStart_Point = np.hstack([close[-1]['point'][:,-1],close[-1]['vtheta']])
                param_all_, param_best_ = dubins_obs_nocircle(
                    SStart_Point,
                    End_Point, outline_all, r, r, r, Stepsize,
                    obs_to_avoid[i], close[-1]['pos_id'], obs_no_circle, total_field, resolution,depth,max_depth
                )
                if isinstance(param_all_, list):
                    param_all.extend(param_all_)
                else:
                    param_all.append(param_all_)
                if isinstance(param_best_, list):
                    param_best.extend(param_best_)
                else:
                    param_best.append(param_best_)

            # 检查安全性
            flag_all, obs_id_all = obtain_flag_safe(2, param_all, obs_no_circle, outline_all)
            min_path, path_safe_flag, flag_min = obtain_min_or_onlypath_from_safe(2, param_all, flag_all)

            param_safe = obtain_safeparam(path_safe_flag, param_all)
            safe_obs_num = effective_len(param_safe)

            for i in range(safe_obs_num):
                open_f, open_ = check_in_open_new(param_safe, End_Point, open_f, open_, close, i)

            while len(open_f) == 0 or all(np.isnan(open_f)):
                print('open is empty! Mayday!')
                close = close[:-1]

                if len(close) == 0:
                    print('sorry, no path')
                    break

                times = sum(1 for x in pos_id_mayday if x == close[-1]['pos_id'])

                while True:
                    if times >= 1:
                        print(f'add open to close more {times}')
                        open_f, open_, pos_id_mayday = update_open_mayday(
                            Start_Point, End_Point, outline_all, r, Stepsize,
                            obs_no_circle, total_field, close, open_, open_f,
                            pos_id_mayday, resolution,depth,max_depth
                        )

                        sorted_indices = np.argsort(open_f)

                        if len(open_f) >= times:
                            min_indices = sorted_indices[0:times]

                            nodes_to_add = [open_[i] for i in min_indices]
                            close = np.append(close,nodes_to_add)
                            open_ = np.delete(open_, min_indices)
                            open_f = np.delete(open_f, min_indices)

                            break
                        else:
                            print('再回退一个')
                            close = close[:-1]

                            if len(close) == 0:
                                print('sorry, no path')
                                break

                            times = sum(1 for x in pos_id_mayday if x == close[-1]['pos_id'])
                    else:
                        print('第一次加入mayday')
                        open_f, open_, pos_id_mayday = update_open_mayday(
                            Start_Point, End_Point, outline_all, r, Stepsize,
                            obs_no_circle, total_field, close, open_, open_f,
                            pos_id_mayday, resolution,depth,max_depth
                        )
                        break

            # 选择最短节点
            flag_min = np.argmin(open_f)

            # 移除open，加入close

            close = np.append(close, [open_[flag_min]])
            open_ = np.delete(open_, flag_min)
            open_f = np.delete(open_f, flag_min)


    return final_path, open_, close,final_param


