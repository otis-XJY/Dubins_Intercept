import numpy as np

from marl.MARL_env import TODCMARLEnv


def test_marl_env_decision_step_mode_online_fields():
    env = TODCMARLEnv(
        {
            "allow_dummy_if_missing": True,
            "render_mode": "none",
            "max_episode_steps": 10,
            "k_max": 8,
            "step_mode": "decision",
            "max_inner_ticks": 5,
        }
    )

    obs, info = env.reset()
    assert "candidates" in obs
    assert isinstance(info["real_mode"], bool)

    action = np.zeros((env.num_P,), dtype=np.int64)
    _, rewards, terminations, truncations, infos = env.step(action)

    assert len(rewards) == env.num_P
    assert "__all__" in terminations
    assert "__all__" in truncations

    for i in range(env.num_P):
        ii = infos[f"p_{i}"]
        assert ii["step_mode"] == "decision"
        assert ii["inner_ticks"] >= 1
        assert "forced_decision" in ii
        assert "decision_step" in ii
