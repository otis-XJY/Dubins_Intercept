import numpy as np

from intercept.IsoPair.obtainRLOutput import obtainNeighbour


def test_obtain_neighbour_list_mode_compatible():
    pos_p = np.array([[0.0, 0.0], [10.0, 0.0]])
    value_pos = np.array([[1.0, 0.0], [20.0, 0.0]])

    out = obtainNeighbour(pos_p, value_pos, distance=2.0)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape[0] == 1
    assert out[1].shape[0] == 0


def test_obtain_neighbour_pid_dict_mode():
    pos_p = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    pos_e = np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

    query_ids = np.array([5, 9])
    target_ids = np.array([100, 101, 102])

    out = obtainNeighbour(
        pos_p,
        pos_e,
        distance=2.0,
        query_ids=query_ids,
        target_ids=target_ids,
        return_mode="dict",
    )

    assert set(out.keys()) == {5, 9}
    assert np.array_equal(out[5]["target_ids"], np.array([100]))
    assert np.array_equal(out[9]["target_ids"], np.array([101]))
    assert out[5]["states"].shape[1] == 3
