"""观测：pairs_realE2P 存活集、零填充与设施常驻。"""

import numpy as np
import pytest

from marl.obs_generator import TODCObservationGenerator, alive_pid_eid_sets


def test_alive_pid_eid_sets():
    pairs = np.array([[0, 1], [2, 3]], dtype=np.int64)
    ap, ae = alive_pid_eid_sets(pairs, num_p=5, num_e=5)
    assert ap == {1, 3}
    assert ae == {0, 2}
    assert alive_pid_eid_sets(None, 3, 3) == (set(), set())
    assert alive_pid_eid_sets(np.empty((0, 2)), 3, 3) == (set(), set())


def test_zero_dead_pursuer_and_evader_columns():
    gen = TODCObservationGenerator()
    num_p, num_e, k = 3, 2, 2
    # Only pid 0 paired with eid 0
    pairs = np.array([[0, 0]], dtype=np.int64)
    v_p = np.zeros((num_p, 6), dtype=np.float32)
    v_p[:, 0] = [1.0, 2.0, 3.0]
    v_p[:, 1] = [0.0, 0.0, 0.0]
    v_p[:, 4] = 1.0
    v_p[:, 5] = 0.0
    pos_p = np.zeros((num_p, 3), dtype=np.float64)
    pos_p[:, 0] = v_p[:, 0]
    pos_p[:, 1] = v_p[:, 1]
    pos_p[:, 2] = 0.0
    pos_e = np.zeros((num_e, 3), dtype=np.float64)
    pos_e[:, 0] = [10.0, 20.0]
    value_pos = np.array([[0.0, 0.0], [100.0, 100.0]], dtype=np.float32)
    inferred = np.array([[5.0, 5.0], [6.0, 6.0]], dtype=np.float32)

    ic = np.zeros((4, 25), dtype=np.float64)
    ic[:, 11] = [0, 0, 1, 1]
    ic[:, 12] = [0, 0, 1, 1]
    ic[:, gen.cols.tp] = 0
    ic[:, gen.cols.te] = 0
    ic[:, gen.cols.Delta_t] = 0.1
    ic[:, gen.cols.Delta_d] = 0.1
    ic[:, gen.cols.Delta_theta] = 0.1
    ic[:, gen.cols.path_L] = 1.0
    ic[:, gen.cols.Delta_V] = 0.1
    ic[:, gen.cols.cost_t] = 0.0
    ic[:, gen.cols.cost_d] = 0.0
    ic[:, gen.cols.cost_theta] = 0.0
    ic[:, gen.cols.cost_L] = 0.0
    ic[:, gen.cols.cost_V] = 0.0

    def cand_fn(row, pid):
        return float(pos_p[pid, 0]), float(pos_p[pid, 1]), 0.0

    out = gen.generate(
        pos_p=pos_p,
        pos_e=pos_e,
        v_p=1.0,
        v_e=1.0,
        value_pos=value_pos,
        num_p=num_p,
        num_e=num_e,
        ic_candidates=ic,
        candidate_pos_fn=cand_fn,
        inferred_targets=inferred,
        pairs_realE2P=pairs,
    )

    assert np.allclose(out["self_uav"][0], [[[1.0, 0.0, 0.0]]], atol=1e-5)
    assert np.all(out["self_uav"][1:] == 0)
    assert not np.any(out["self_pts_mask"][1])
    assert int(out["pursuer_active"][0]) == 1 and int(np.sum(out["pursuer_active"])) == 1
    assert np.all(out["enemies"][:, 1, :] == 0)
    assert not np.any(out["enemy_mask"][:, 1])
    # 设施全保留
    assert out["assets"].shape == (num_p, value_pos.shape[0], 2)
    assert np.all(out["asset_mask"])
    # 死 pid 无友机候选、无友机槽
    assert np.all(out["ally_pts"][1:] == 0)
    assert not np.any(out["ally_pts_mask"][1:])
    assert not np.any(out["ally_mask"][1:])


def test_ally_channels_only_alive_neighbors():
    """存活机 ally_pts 仅拼接其它存活机候选，不包含死 pid 占位块。"""
    gen = TODCObservationGenerator(ally_perception_radius=float("inf"))
    num_p, num_e, k = 3, 2, 2
    pairs = np.array([[0, 0], [1, 1]], dtype=np.int64)
    v_p = np.zeros((num_p, 6), dtype=np.float32)
    v_p[:, 0] = [0.0, 1.0, 99.0]
    v_p[:, 1] = 0.0
    v_p[:, 4] = 1.0
    v_p[:, 5] = 0.0
    pos_p = np.zeros((num_p, 3), dtype=np.float64)
    pos_p[:, 0] = v_p[:, 0]
    pos_p[:, 2] = 0.0
    pos_e = np.zeros((num_e, 3), dtype=np.float64)
    pos_e[:, 0] = [10.0, 20.0]
    value_pos = np.array([[0.0, 0.0], [50.0, 50.0]], dtype=np.float32)
    inferred = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)

    ic = np.zeros((4, 25), dtype=np.float64)
    ic[:, 11] = [0, 0, 1, 1]
    ic[:, 12] = [0, 0, 1, 1]
    ic[:, gen.cols.tp] = 0
    ic[:, gen.cols.te] = 0
    ic[:, gen.cols.Delta_t] = 0.1
    ic[:, gen.cols.Delta_d] = 0.1
    ic[:, gen.cols.Delta_theta] = 0.1
    ic[:, gen.cols.path_L] = 1.0
    ic[:, gen.cols.Delta_V] = 0.1
    ic[:, gen.cols.cost_t] = 0.0
    ic[:, gen.cols.cost_d] = 0.0
    ic[:, gen.cols.cost_theta] = 0.0
    ic[:, gen.cols.cost_L] = 0.0
    ic[:, gen.cols.cost_V] = 0.0

    def cand_fn(row, pid):
        return float(pos_p[pid, 0]), float(pos_p[pid, 1]), 0.0

    out = gen.generate(
        pos_p=pos_p,
        pos_e=pos_e,
        v_p=1.0,
        v_e=1.0,
        value_pos=value_pos,
        num_p=num_p,
        num_e=num_e,
        ic_candidates=ic,
        candidate_pos_fn=cand_fn,
        inferred_targets=inferred,
        pairs_realE2P=pairs,
    )

    # pid 2 无配对：友机通道全空
    assert np.all(out["ally_pts"][2] == 0)
    assert not np.any(out["ally_pts_mask"][2])
    assert not np.any(out["ally_mask"][2])
    # pid 0、1 各仅 1 个存活友机 -> 有效 ally_pts 长度 k，而非 (num_p-1)*k
    assert int(out["ally_pts_mask"][0].sum()) == k
    assert int(out["ally_pts_mask"][1].sum()) == k
