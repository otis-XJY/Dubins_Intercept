import numpy as np

def update_first(param_old, param_new):

    param_all = []
    # 拼接旧的 param_safe 和新的 param_all
            # MATLAB 中的嵌套循环 j+i 索引意味着生成所有可能的组合
    # 检查是否存在 pos_id 字段且非空
    if len(param_new)!=0:
        if param_old.get('pos_id') is not None and param_new.get('pos_id') is not None:
            
            # 构造拼接后的节点字典
            node = {}
            # 索引转换说明: 
            # MATLAB point(:, 4:6) -> Python point[:, 3:6]
            node['point'] = np.hstack([param_new['point'], param_old['point'][:, 3:6]])
            
            # MATLAB length(3:5) -> Python length[2:5]
            node['length'] = np.hstack([param_new['length'], param_old['length'][2:5]])
            
            # MATLAB phy(2:3) -> Python phy[1:3]
            # node['phy'] = np.hstack([param_new['phy'], param_old['phy'][1:3]])
            
            # MATLAB theta(3:5) -> Python theta[2:5]
            # node['theta'] = np.hstack([param_new['theta'], param_old['theta'][2:5]])
            
            # MATLAB center(:, 2:3) -> Python center[:, 1:3]
            node['center'] = np.hstack([param_new['center'], param_old['center'][:, 1:3]])
            
            node['r'] = param_old['r']
            # node['R'] = param_old['R']
            # node['type'] = -1
            
            # MATLAB vtheta(2:4) -> Python vtheta[1:4]
            node['vtheta'] = np.hstack([param_new['vtheta'], param_old['vtheta'][1:4]])
            
            node['vtheta_plot'] = np.hstack([param_new['vtheta_plot'], param_old['vtheta_plot']])
            node['Length'] = np.sum(node['length'])
            
            # MATLAB path(3:5) -> Python path[2:5]
            node['path'] = param_new['path'] + param_old['path'][2:5]
            
            # MATLAB vtheta_all(3:5) -> Python vtheta_all[2:5]
            node['vtheta_all'] = param_new['vtheta_all'] + param_old['vtheta_all'][2:5]
            
            node['pos_id'] = param_old['pos_id']
            node['parent_id'] = param_old['parent_id']
                
            param_all.append(node)


    return param_all