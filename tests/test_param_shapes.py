import numpy as np
from A_dubins.dubinspath.dubins_no_obs import Dubins_no_obs
from A_dubins.coreCode.obtain_flag_safe import obtain_flag_safe
from A_dubins.coreCode.obtain_min_or_onlypath_from_safe import obtain_min_or_onlypath_from_safe
from A_dubins.coreCode.obtain_safeparam import obtain_safeparam
from A_dubins.A_dubins_nocircle_swarm import A_dubins_nocircle_swarm


def test_dubins_no_obs_and_flag():
    p_all, p_best = Dubins_no_obs([0,0,0], [10,0,0], 1, 1, 0.5, -2, -1)
    # param_all should be a list of candidate dicts
    assert isinstance(p_all, list)
    assert len(p_all) > 0
    assert isinstance(p_all[0], dict)

    flag_all, obs_id_all = obtain_flag_safe(1, [p_all], [], np.array([]))
    # flag_all should be a list of ints and same length as total candidates
    assert isinstance(flag_all, list)
    assert len(flag_all) == len(p_all)


def test_obtain_min_or_onlypath_from_safe():
    # construct param_all with two columns
    c1 = [{'Length': 5.0, 'path':[1], 'r':[1], 'point':np.zeros((2,2))}, {'Length': 8.0, 'path':[1], 'r':[1], 'point':np.zeros((2,2))}]
    c2 = [{'Length': 3.0, 'path':[1], 'r':[1], 'point':np.zeros((2,2))}]
    param_all = [c1, c2]

    # type 3: absolute shortest among all
    minp, path_safe_flag, flagmin = obtain_min_or_onlypath_from_safe(3, param_all, [1,1,1])
    assert minp == 3.0

    # simulate flags where only first column candidate 0 and column 1 candidate 0 are safe
    flag_all = [1, 0, 1]
    minp2, path_safe_flag2, flagmin2 = obtain_min_or_onlypath_from_safe(1, param_all, flag_all)
    # path_safe_flag2 should be flattened safe indices
    assert path_safe_flag2 == [0,2]


def test_obtain_safeparam_and_swarm_basic():
    from A_dubins.dubinspath.dubins_no_obs import Dubins_no_obs
    p_all, p_best = Dubins_no_obs([0,0,0], [10,0,0], 1, 1, 0.5, -2, -1)
    flag_all, obs_id_all = obtain_flag_safe(1, [p_all], [], np.array([]))
    _, path_safe_flag, _ = obtain_min_or_onlypath_from_safe(1, [p_all], flag_all)
    param_safe = obtain_safeparam(path_safe_flag, [p_all])
    # param_safe should be a list-of-lists
    assert isinstance(param_safe, list)

    # A_dubins_nocircle_swarm basic call should find direct goal for no obstacles
    final_path, open_, close,_ = A_dubins_nocircle_swarm([0,0,0], [10,0,0], [], [], 1, [], np.array([]), 0.5, 0, 0.1)
    assert len(final_path) == 2


def test_empty_column_and_mayday_behavior():
    # empty column should be represented as a list-of-lists with an empty list
    param_all = [[]]
    flag_all, obs_id_all = obtain_flag_safe(2, param_all, [], np.array([]))
    assert flag_all == []

    minp, path_safe_flag, flag_min = obtain_min_or_onlypath_from_safe(1, param_all, flag_all)
    param_safe = obtain_safeparam(path_safe_flag, param_all)
    assert isinstance(param_safe, list) and len(param_safe) == 1 and isinstance(param_safe[0], list)

    # check check_in_open_new can accept param_safe with empty column without crashing
    from A_dubins.search.check_in_open_new import check_in_open_new
    close = [{'point': np.transpose([[0,0]]), 'vtheta': 0, 'g': 0, 'f': 0, 'pos_id': -1, 'parent_id': -1}]
    open_f = []
    open_ = []
    End_Point = [10,0]
    open_f2, open2 = check_in_open_new(param_safe, End_Point, open_f, open_, close, 0)
    assert open_f2 == open_f and open2 == open_


def test_dubins_obs_nocircle_flattening():
    from A_dubins.dubinspath.dubins_obs_nocircle import dubins_obs_nocircle
    from A_dubins.coreCode.obtain_min_or_onlypath_from_safe import obtain_min_or_onlypath_from_safe

    # generate a param_all_ (list of columns) from a sample call
    param_all_, param_best_ = dubins_obs_nocircle([0,0,0], [10,0,0], np.array([]), 1, 1, 1, 0.5, 0, -1, [], 0, 0.1)
    # simulate outer assembly that incorrectly did append
    param_all_bad = []
    param_all_bad.append(param_all_)

    # Good assembly should extend
    param_all_good = []
    if isinstance(param_all_, list):
        param_all_good.extend(param_all_)
    else:
        param_all_good.append(param_all_)

    # ensure the good assembly does not raise
    flag_all, obs_id_all = obtain_flag_safe(2, param_all_good, [], np.array([]))
    # and min selection works
    minp, path_safe_flag, flag_min = obtain_min_or_onlypath_from_safe(1, param_all_good, flag_all)
    # If path_safe_flag is empty, still returns structures (no exception)
    assert isinstance(path_safe_flag, list)

