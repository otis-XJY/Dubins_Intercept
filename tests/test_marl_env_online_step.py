import numpy as np

from marl.envs import TODCMARLEnv


def test_marl_env_online_step_info_fields():
    env = TODCMARLEnv(
        {
            "render_mode": "none",
            "max_episode_steps": 10,
        }
    )

    obs, info = env.reset()
    assert "self_pts" in obs
    assert isinstance(info["real_mode"], bool)
    assert "t_all" in info

    action = np.zeros((env.num_P,), dtype=np.int64)
    _, rewards, terminations, truncations, infos = env.step(action)

    assert len(rewards) == env.num_P
    assert "__all__" in terminations
    assert "__all__" in truncations
    assert "global_t_all" in infos
    assert "path_exec_t" in infos

    for i in range(env.num_P):
        ii = infos[f"p_{i}"]
        assert ii["delta_t_all"] >= 0.0
        assert ii["sim_dt"] > 0.0
        assert ii["t_all"] >= 0.0
        assert "decision_step" in ii
