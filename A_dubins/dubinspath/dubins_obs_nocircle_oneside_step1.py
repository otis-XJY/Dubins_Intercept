# 导入必要的库
import numpy as np
from A_dubins.dubinspath.dubins_obs import Dubins_obs
from A_dubins.coreCode.obtain_flag_safe import obtain_flag_safe
from A_dubins.coreCode.obtain_min_or_onlypath_from_safe import obtain_min_or_onlypath_from_safe
from A_dubins.coreCode.obtain_safeparam import obtain_safeparam
from A_dubins.coreCode.SafeFlag import if_safe_point,obtain_safe_point
from A_dubins.search.update_first import update_first

# 假设这些函数已经定义：Dubins_obs, obtain_flag_safe, obtain_min_or_onlypath_from_safe, obtain_safeparam, if_safe_point

def Dubins_obs_nocircle_oneside_step1(Start_Point, End_Point, center1, outline_all, outline, r_s, r_e, R,
                                      Stepsize, pos_id, parent_id, obs_no_circle,resolution):
    param_all = []
    param_best = []
    param_safe =[]

    Insert={}

    # 调用Dubins_obs函数（需要提供该函数）
    param_all_1, param_best_1 = Dubins_obs(Start_Point, End_Point, center1, r_s, r_e, R, Stepsize, pos_id, parent_id)
    # 直接把候选列表作为一列添加（符合 list-of-lists 约定）
    param_all.append(param_all_1)
    param_best.append(param_best_1)

    # 获取路径的安全标志（需要提供该函数）
    flag_all, obs_id_all = obtain_flag_safe(2, param_all, obs_no_circle, outline)  # 检查1-2段
    min_path, path_safe_flag, flag_min = obtain_min_or_onlypath_from_safe(2, param_all, flag_all)  # 获取最小路径
    Insert['flagSucced']=True
    # 如果找到了安全路径
    if path_safe_flag:
        print('直接找到1-2安全路径')
        param_safe = obtain_safeparam(path_safe_flag, param_all)  # 获取安全参数
        # 只选择第一个安全路径（该函数返回 list-of-lists）
        param_safe = param_safe[0]  # 只选择第一个安全路径
        Insert['flagSucced']=False
    else:
        End_Point_nowLsit=[]
        End_Point_CostList=[]
        flag_all_point=[]
        for i, param in enumerate(param_all_1):
            if param['point'] is not None:
                flag = if_safe_point(obs_no_circle, param['point'][:, 2], 0)  # 检查点是否安全
                if flag == 1:
                    flag_all_point.append(i)
                    End_Point_now = np.append(param_all_1[i]['point'][:, 2], param_all_1[i]['vtheta'][1])
                    End_Point_nowCost=obtain_safe_point(End_Point_now, outline, r_s)
                    End_Point_nowLsit.append(End_Point_now)
                    End_Point_CostList.append(End_Point_nowCost)

        if len(flag_all_point)==0:
            print('没有找到安全点')
            End_Point_nowFinal=None
            costFial=float('inf')
            param_allFinal=None
        else:
            # 转换为 numpy 数组进行快速索引
            costs = np.array(End_Point_CostList)
            min_idx = np.argmin(costs)
            End_Point_nowFinal = End_Point_nowLsit[min_idx]
            costFial=costs[min_idx] 
            param_allFinal=param_all_1[flag_all_point[min_idx]]
        
        Insert['End_Point_now'] = End_Point_nowFinal
        Insert['End_Point_Cost'] = costFial         
        Insert['param_all_1']=param_allFinal
        



        # # 如果找到了安全的初始点，则插入新的路径点
        # if flag_all_point != 0:
        #     print('需要插入点')
        #     End_Point_now = np.append(param_all_1[flag_all_point]['point'][:, 2], param_all_1[flag_all_point]['vtheta'][1])
            
    #         # 延迟导入避免循环依赖
    #         from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm
            
    #         # 需要调用Dubins_obs_nocircle_oneside_first函数（你可以提供该函数）
    #         _,_,_,param_all_2 = A_dubins_nocircle_swarm(
    #             Start_Point, End_Point_now, 0, 0, r_s, obs_no_circle, outline_all, 
    #             Stepsize, 0,
    #             resolution
    #         )

    #         param_safe=update_first(param_all_1[flag_all_point], param_all_2)


    # 如果没有找到安全路径，则返回空的路径
    if 'param_safe' not in locals():
        param_safe = []
    
    return param_safe, param_best ,Insert
