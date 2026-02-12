import numpy as np
from A_dubins.search.update_open_mayday import update_open_mayday
from A_dubins.dubinspath.dubins_obs_nocircle import dubins_obs_nocircle


def test_update_open_mayday_flattening_basic():
    Start = [0,0,0]
    End = [10,0,0]
    # minimal fake obs_no_circle: two obstacles with simple outlines
    obs_no_circle = [np.array([[0,0]]), np.array([[1,1]])]
    # minimal outline_all with (x,y,pos_id) rows
    outline_all = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    r = 1
    Stepsize = 0.5
    total_field = 0
    resolution = 0.1

    # close with single root node (len==1) triggers first branch
    close = [{'point': np.transpose([[0,0]]), 'vtheta': 0, 'g': 0, 'f': 0, 'pos_id': -1, 'parent_id': -1}]
    open_ = []
    open_f = []
    pos_id_mayday = []

    # monkeypatch dubins_obs_nocircle to return deterministic columns (avoid geometric edge cases)
    def fake_dubins_obs_nocircle(*args, **kwargs):
        num = 6
        c1 = {
            'Length': 5.0,
            'path': [np.zeros((2,1)) for _ in range(num)],
            'r': [1, 1],
            'point': np.zeros((2, num)),
            'length': np.ones(num),
            'vtheta': np.zeros(num),
            'vtheta_plot': np.zeros(num),
            'vtheta_all': [[], [] , []],
            'center': np.zeros((2, 2)),
            'pos_id': -1,
            'parent_id': -1
        }
        return [[c1]], []

    import A_dubins.search.update_open_mayday as uom
    uom.dubins_obs_nocircle = fake_dubins_obs_nocircle
    # monkeypatch obtain_flag_safe to avoid geometry code
    import A_dubins.coreCode.obtain_flag_safe as of
    def fake_obtain_flag_safe(*args, **kwargs):
        # one column with one candidate -> flattened flags [1]
        return [1], np.array([-1.0])
    of.obtain_flag_safe = fake_obtain_flag_safe

    # Run update_open_mayday: should not raise and should return structures
    open_f2, open2, pos_list = update_open_mayday(Start, End, outline_all, r, Stepsize, obs_no_circle, total_field,
                                                   close, open_, open_f, pos_id_mayday, resolution)

    assert isinstance(open_f2, list)
    assert isinstance(open2, list)
    # pos_id_mayday should have appended -1
    assert pos_list[-1] == -1


def test_update_open_mayday_followup_branch():
    Start = [0,0,0]
    End = [10,0,0]
    obs_no_circle = [np.array([[0,0]]), np.array([[1,1]])]
    # minimal outline_all with (x,y,pos_id) rows
    outline_all = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    r = 1
    Stepsize = 0.5
    total_field = 0
    resolution = 0.1

    # close with two nodes triggers else branch
    close = [
        {'point': np.transpose([[0,0]]), 'vtheta': 0, 'g': 0, 'f': 0, 'pos_id': -1, 'parent_id': -1},
        {'point': np.transpose([[1,1]]), 'vtheta': 0, 'g': 0, 'f': 0, 'pos_id': 5, 'parent_id': -1}
    ]
    open_ = []
    open_f = []
    pos_id_mayday = []

    # monkeypatch dubins_obs_nocircle for deterministic return
    def fake_dubins_obs_nocircle(*args, **kwargs):
        num = 6
        c1 = {
            'Length': 3.0,
            'path': [np.zeros((2,1)) for _ in range(num)],
            'r': [1, 1],
            'point': np.zeros((2, num)),
            'length': np.ones(num),
            'vtheta': np.zeros(num),
            'vtheta_plot': np.zeros(num),
            'vtheta_all': [[], [] , []],
            'center': np.zeros((2, 2)),
            'pos_id': 5,
            'parent_id': -1
        }
        return [[c1]], []

    import A_dubins.search.update_open_mayday as uom
    uom.dubins_obs_nocircle = fake_dubins_obs_nocircle
    # monkeypatch obtain_flag_safe for deterministic behavior
    import A_dubins.coreCode.obtain_flag_safe as of
    def fake_obtain_flag_safe(*args, **kwargs):
        # one column with one candidate -> flattened flags [1]
        return [1], np.array([5.0])
    of.obtain_flag_safe = fake_obtain_flag_safe

    open_f2, open2, pos_list = update_open_mayday(Start, End, outline_all, r, Stepsize, obs_no_circle, total_field,
                                                   close, open_, open_f, pos_id_mayday, resolution)
    assert isinstance(open_f2, list)
    assert isinstance(open2, list)
    assert pos_list[-1] == 5
