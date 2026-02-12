import numpy as np
import copy
import numpy as np

import numpy as np

def optimize_node_linking(close):
    # 1. 使用浅拷贝或直接操作（除非你必须保留原有的 close 列表不变）
    # 如果 close 里的字典以后会变，且不希望影响 choose，才用 copy()
    # 对于父节点引用，通常不需要 deepcopy，直接引用对象即可
    choose = [node.copy() for node in close] 

    # 2. 建立“末端点”到“节点对象”的映射表
    # key: 坐标转换成的 tuple（因为 numpy 数组不可哈希）
    # value: 节点对象的引用
    last_point_map = {}

    for node in choose:
        # 获取当前节点的起点和终点（假设是 1D 数组或列向量）
        start_pt = tuple(node['point'][:, 0].flatten())
        end_pt = tuple(node['point'][:, -1].flatten())

        # 3. 直接通过哈希表查找父节点 (替代了内层的 j 循环)
        # 如果当前节点的起点 是 之前某个节点的终点
        if start_pt in last_point_map:
            # 直接建立引用，不建议使用 deepcopy
            node['father'] = last_point_map[start_pt]
        
        # 将当前节点的终点记录到 map 中，供后续节点匹配
        # 注意：如果有多个节点终点相同，后面的会覆盖前面的，
        # 如果逻辑允许这种情况，map 足够；若需全部匹配，value 可设为 list
        last_point_map[end_pt] = node

    return choose
def obtain_path(close,pos_id,parent_id):
    """
    从close节点列表中获取最终的路径。

    Parameters:
        close : list of dict
            节点列表，每个节点应包含'point'和'father'属性。

    Returns:
        final_path : list of dict
            从终点到起点的路径节点列表。
    """
    # # 使用深拷贝
    # choose = copy.deepcopy(close)

    # # 确定每个节点的父节点
    # for i in range(len(choose)):
    #     for j in range(i + 1, len(choose)):
    #         if np.array_equal(choose[i]['point'][:, -1], choose[j]['point'][:, 0]):
    #             choose[j]['father'] = copy.deepcopy(choose[i])


    choose=optimize_node_linking(close)

    # 获取最终路径
    check_cell = choose[-1]
    final_path = [choose[-1]]
    final_param=choose[-1].copy()
    final_param['length'] = choose[-1]['g']
    while 'father' in check_cell:
        final_path = [check_cell['father']] + final_path
        # final_path.pop('father')
        if check_cell['father'].get('center') is not None and check_cell['father']['center'].size > 0:


            final_param['point'] = np.hstack([check_cell['father']['point'][:,:-1], final_param['point']])
            
            # MATLAB length(3:5) -> Python length[2:5]
            # final_param['length'] = check_cell['father']['g']
            
            # MATLAB phy(2:3) -> Python phy[1:3]
            # final_param['phy'] = np.concatenate([check_cell['father']['phy'], final_param['phy']])
            
            # MATLAB theta(3:5) -> Python theta[2:5]
            # final_param['theta'] = np.concatenate([check_cell['father']['theta'], final_param['theta']])
            
            # MATLAB center(:, 2:3) -> Python center[:, 1:3]
            final_param['center'] = np.hstack([check_cell['father']['center'], final_param['center']])
            
            final_param['r'] = check_cell['father']['r']
            # final_param['type'] = -1
            
            # MATLAB vtheta(2:4) -> Python vtheta[1:4]
            final_param['vtheta'] = np.hstack([check_cell['father']['vtheta'], final_param['vtheta']])
            
            final_param['vtheta_plot'] = check_cell['father']['vtheta_plot']
            final_param['Length'] = np.sum(final_param['length'])
            
            # MATLAB path(3:5) -> Python path[2:5]
            final_param['path'] = check_cell['father']['path'] + final_param['path']
            
            # MATLAB vtheta_all(3:5) -> Python vtheta_all[2:5]
            final_param['vtheta_all'] = check_cell['father']['vtheta_all'] + final_param['vtheta_all']
            
            final_param['pos_id'] = pos_id
            final_param['parent_id'] = parent_id

        check_cell = check_cell['father']
        
        

    return final_path,final_param
