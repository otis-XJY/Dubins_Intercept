import numpy as np

from marl.MARL_env import TODCMARLEnv


def test_marl_env_reset_and_step_smoke():
    env = TODCMARLEnv(
        {
            "render_mode": "none",
            "max_episode_steps": 20,
            "step_mode": "time",
        }
    )

    obs, info = env.reset()
    assert "self_uav" in obs
    assert "enemies" in obs
    assert "self_pts" in obs
    assert "self_pts_mask" in obs
    assert obs["self_pts"].shape[1] >= 1
    assert isinstance(info["real_mode"], bool)

    done = False
    trunc = False
    steps = 0
    while not done and not trunc and steps < 10:
        action = np.zeros((env.num_P,), dtype=np.int64)
        _, rewards, terminations, truncations, infos = env.step(action)
        assert len(rewards) == env.num_P
        assert "__all__" in terminations
        assert "__all__" in truncations
        assert all(k in infos for k in [f"p_{i}" for i in range(env.num_P)])
        done = terminations["__all__"]
        trunc = truncations["__all__"]
        steps += 1

    assert steps > 0