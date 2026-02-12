from A_dubins.dubinspath.dubins_obs_nocircle import dubins_obs_nocircle
from A_dubins.coreCode.obtain_flag_safe import obtain_flag_safe
from A_dubins.coreCode.obtain_min_or_onlypath_from_safe import obtain_min_or_onlypath_from_safe
from A_dubins.coreCode.obtain_safeparam import obtain_safeparam
from A_dubins.coreCode.obtain_obs_to_avoid import obtain_obs_to_avoid
from A_dubins.coreCode.exHelp import effective_len
from A_dubins.search.check_in_open_new import check_in_open_new

import numpy as np
def update_open_mayday(Start_Point, End_Point, outline_all, r, Stepsize, obs_no_circle, total_field, close, open,
                       open_f, pos_id_mayday, resolution,depth,max_depth):
    param_all = []
    param_best = []

    if len(close) == 1:
        print("length(close)==1")

        for i in range(len(obs_no_circle)):
            if obs_no_circle[i] is not None and len(obs_no_circle[i]) > 0:
                param_all_, param_best_ = dubins_obs_nocircle(Start_Point, End_Point, outline_all, r, r, r, Stepsize, i, -1, obs_no_circle,
                                        total_field, resolution,depth,max_depth)
                # param_all_ may itself be a list of columns; flatten into top-level list-of-columns
                if isinstance(param_all_, list):
                    param_all.extend(param_all_)
                else:
                    param_all.append(param_all_)
                if isinstance(param_best_, list):
                    param_best.extend(param_best_)
                else:
                    param_best.append(param_best_)

        # 检查1-2的安全性
        flag_all, obs_id_all = obtain_flag_safe(2, param_all, obs_no_circle, outline_all)

        # 如果有安全的路径
        min_path, path_safe_flag, flag_min = obtain_min_or_onlypath_from_safe(2, param_all, flag_all)

        # param_safe is a list-of-lists (per-obstacle candidate lists)
        param_safe = obtain_safeparam(path_safe_flag, param_all)
        safe_obs_num = effective_len(param_safe)
        pos_id_mayday.append(-1)

    else:
        obs_to_avoid = obtain_obs_to_avoid(close[-1]['point'][:,-1], End_Point, outline_all)

        for i in range(len(obs_to_avoid)):
            if obs_no_circle[int(obs_to_avoid[i])] is not None and len(obs_no_circle[int(obs_to_avoid[i])]) > 0 and obs_to_avoid[i] != close[-1]['pos_id']:
                SStart_Point = np.hstack([close[-1]['point'][:, -1], close[-1]['vtheta']])
                param_all_, param_best_ = dubins_obs_nocircle(SStart_Point, End_Point, outline_all, r, r, r,
                                        Stepsize, obs_to_avoid[i], close[-1]['pos_id'], obs_no_circle, total_field,
                                        resolution,depth,max_depth)
                if isinstance(param_all_, list):
                    param_all.extend(param_all_)
                else:
                    param_all.append(param_all_)
                if isinstance(param_best_, list):
                    param_best.extend(param_best_)
                else:
                    param_best.append(param_best_)

        # 检查1-2的安全性
        flag_all, obs_id_all = obtain_flag_safe(2, param_all, obs_no_circle, outline_all)

        # 如果有安全的路径
        min_path, path_safe_flag, flag_min = obtain_min_or_onlypath_from_safe(2, param_all, flag_all)

        param_safe = obtain_safeparam(path_safe_flag, param_all)
        safe_obs_num = effective_len(param_safe)
        pos_id_mayday.append(close[-1]['pos_id'])

    # 更新open和open_f
    for i in range(safe_obs_num):
        open_f, open = check_in_open_new(param_safe, End_Point, open_f, open, close, i)

    return open_f, open, pos_id_mayday
