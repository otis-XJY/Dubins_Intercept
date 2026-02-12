from A_dubins.coreCode.obtain_h import obtain_h
from A_dubins.coreCode.exHelp import effective_len
import numpy as np
def check_in_open_new(param_safe, End_Point, open_f, open, close, obs_num_):
    """
    检查是否需要更新open列表中的路径节点。

    Parameters:
        param_safe : dict
            包含路径参数的结构体（包括R和r）。
        End_Point : list
            目标点坐标。
        open_f : list
            open列表中的f值。
        open : list
            open列表，存储路径节点。
        close : list
            close列表，存储已访问的路径节点。
        obs_num_ : int
            当前障碍物编号。

    Returns:
        open_f : list
            更新后的open列表中的f值。
        open : list
            更新后的open列表，包含新的路径节点。
    """
    pos_id_all =  np.array([node['pos_id'] for node in open])
    parent_id_all =  np.array([node['parent_id'] for node in open])


    # 处理 param_safe（list-of-lists），使用零基索引并防止越界
    if obs_num_ < effective_len(param_safe):
        if param_safe[obs_num_]:
            for param in param_safe[obs_num_]:
                if param.get('pos_id'):
                    pos_in_open = np.where((pos_id_all == param['pos_id']) & (parent_id_all == param['parent_id']))[0]
                    num = len(param['path'])
                    if pos_in_open is not None and len(pos_in_open) > 0:
                        for pos in pos_in_open:
                            Length = sum(param['length'][:-3])
                            new_g = close[-1]['g'] + Length
                            new_f = new_g + obtain_h(param['point'][:, -1].T, End_Point)

                            if new_f < open[pos]['f']:
                                print(
                                    f'change: {new_f}, {open[pos]["f"]}, {open[pos]["pos_id"]}, {open[pos]["parent_id"]}')

                                # 更新路径信息
                                open[pos]['point'] = param['point'][:, :-3]
                                open[pos]['vtheta'] = param['vtheta'][-3:-2]
                                open[pos]['vtheta_plot'] = param['vtheta_plot']
                                open[pos]['vtheta_all'] = param['vtheta_all'][:-3]
                                open[pos]['g'] = new_g
                                open[pos]['f'] = new_f
                                open[pos]['path'] = param['path'][:- 3]
                                open[pos]['r'] = param.get('r', [])
                                open[pos]['center'] = param['center']
                                open[pos]['pos_id'] = param['pos_id']
                                open[pos]['parent_id'] = param['parent_id']
                                open_f[pos] = open[pos]['f']
                            else:
                                print(f'ignore:{open[pos]["f"]}, {open[pos]["pos_id"]}, {open[pos]["parent_id"]}')
                    else:
                        # 如果该路径不在open列表中
                        node = {
                            'point': param['point'][:, :-3],
                            'vtheta': param['vtheta'][-3:-2],
                            'vtheta_plot': param['vtheta_plot'],
                            'vtheta_all': param['vtheta_all'][:-3],
                            'g': close[-1]['g'] + sum(param['length'][:- 3]),
                            'f': close[-1]['g'] + sum(param['length'][:- 3]) + obtain_h(param['point'][:, -1].T,
                                                                                            End_Point),
                            'path': param['path'][:- 3],
                            'r': param.get('r', []),
                            'center': param['center'],
                            'pos_id': param['pos_id'],
                            'parent_id': param['parent_id']
                        }
                        open_f = np.append(open_f, node['f'])
                        open = np.append(open, node)
    return open_f, open
