import numpy as np

from marl.rewards import TODCRewardFunction


def test_reward_module_dist_progress_and_capture_bonus():
    reward_fn = TODCRewardFunction.from_env_config(
        {
            "reward": {
                "dist_progress_scale": 1.0,
                "entropy_scale": 0.0,
                "capture_bonus": 10.0,
            }
        }
    )

    obs = {
        "self_pts": np.zeros((2, 4, 8), dtype=np.float32),
        "self_pts_mask": np.ones((2, 4), dtype=bool),
    }
    actions = np.zeros((2, 4), dtype=np.float32)
    actions[:, 0] = 1.0

    rewards = reward_fn.compute_step_rewards(
        actions=actions,
        obs=obs,
        curr_min_dist=np.array([8.0, 7.0], dtype=np.float32),
        last_min_dist=np.array([10.0, 10.0], dtype=np.float32),
        num_p=2,
    )

    # progress reward = [2, 3]
    assert rewards["p_0"] == 2.0
    assert rewards["p_1"] == 3.0

    reward_fn.apply_capture_bonus(rewards, captured_delta=1, num_e=2)
    # bonus = 10/2 = 5
    assert rewards["p_0"] == 7.0
    assert rewards["p_1"] == 8.0
