import numpy as np
from concurrent.futures import ThreadPoolExecutor
from A_dubins.coreCode.SafeFlag import if_safe_no_circle
import numpy as np


def obtain_flag_safe(type, param_all, obs_no_circle, outline_all):
    # 根据输入的type值，调用不同的检查函数
    if type == 1:
        flag_all, obs_id_all = obtain_all_or_part(param_all, obs_no_circle, outline_all, 1, 3)
    elif type == 2:
        flag_all, obs_id_all = obtain_all_or_part(param_all, obs_no_circle, outline_all, 1, 2)
    elif type == 3:
        flag_all, obs_id_all = obtain_all_or_part(param_all, obs_no_circle, outline_all, 3, 5)
    elif type == 4:
        flag_all, obs_id_all = obtain_all_or_part(param_all, obs_no_circle, outline_all, 2, 3)
    elif type == 5:
        flag_all, obs_id_all = obtain_all_or_part(param_all, obs_no_circle, outline_all, 1, 5)
    else:
        raise Warning('No type')

    return flag_all, obs_id_all


def obtain_all_or_part(param_all, obs_no_circle, outline_all, from_path, to_path):
    flag_all = []
    obs_id_all = np.array([])

    # param_all is expected to be a list of "columns" (each column is a list of path parameter dicts)
    colCount = len(param_all)

    # Iterate columns and each candidate param in that column
    for j in range(colCount):
        for i in range(len(param_all[j])):
            p_ = param_all[j][i]
            if type(p_) is list:
                for k in range(len(p_)):
                    p=p_[k]
                    if p.get('path') and not np.isnan(p.get('Length', np.nan)):
                        if outline_all.size > 0:
                            flag_safe, obs_id = if_safe_no_circle(p['r'][0], p['path'], obs_no_circle,
                                                                outline_all, from_path, to_path, p['point'])
                            flag_all.append(flag_safe)
                            obs_id_all = np.append(obs_id_all, obs_id)
                        else:
                            flag_all.append(1)
                    else:
                        flag_all.append(0)                   
            else:
                p=p_

                if p.get('path') and not np.isnan(p.get('Length', np.nan)):
                    if outline_all.size > 0:
                        flag_safe, obs_id = if_safe_no_circle(p['r'][0], p['path'], obs_no_circle,
                                                            outline_all, from_path, to_path, p['point'])
                        flag_all.append(flag_safe)
                        obs_id_all = np.append(obs_id_all, obs_id)
                    else:
                        flag_all.append(1)
                else:
                    flag_all.append(0)

    return flag_all, obs_id_all

