import numpy as np

from marl.rewards import TODCRewardFunction


def test_reward_module_dist_progress_and_sim_time():
    reward_fn = TODCRewardFunction.from_env_config(
        {
            "reward": {
                "dist_progress_scale": 1.0,
            }
        }
    )

    obs = {
        "self_pts": np.zeros((2, 4, 8), dtype=np.float32),
        "self_pts_mask": np.ones((2, 4), dtype=bool),
        "reward_nodes": np.zeros((2, 4, 8), dtype=np.float32),
        "self_uav": np.zeros((2, 1, 3), dtype=np.float32),
        "enemies": np.zeros((2, 1, 3), dtype=np.float32),
    }
    actions = np.zeros((2, 4), dtype=np.float32)
    actions[:, 0] = 1.0

    _, details = reward_fn.compute_step_rewards(
        actions=actions,
        obs=obs,
        curr_min_dist=np.array([8.0, 7.0], dtype=np.float32),
        last_min_dist=np.array([10.0, 10.0], dtype=np.float32),
        num_p=2,
        sim_time_elapsed=0.0,
        sim_dt=1.0,
        return_details=True,
    )
    assert details["p_0"]["r_progress"] == 2.0
    assert details["p_1"]["r_progress"] == 3.0
    assert details["p_0"]["r_time"] == 0.0

    _, d2 = reward_fn.compute_step_rewards(
        actions=actions,
        obs=obs,
        curr_min_dist=np.array([8.0, 7.0], dtype=np.float32),
        last_min_dist=np.array([10.0, 10.0], dtype=np.float32),
        num_p=2,
        sim_time_elapsed=2.0,
        sim_dt=1.0,
        return_details=True,
    )
    sc = float(reward_fn.config.step_cost)
    assert abs(d2["p_0"]["r_time"] - 2.0 * sc) < 1e-9
