import numpy as np

from marl.MARL_env import TODCMARLEnv


def test_phase2_obs_shapes_and_mask_consistency():
    env = TODCMARLEnv(
        {
            "allow_dummy_if_missing": True,
            "render_mode": "none",
            "k_max": 12,
            "max_episode_steps": 5,
        }
    )

    obs, _ = env.reset()

    assert "V_P" in obs
    assert "V_E" in obs
    assert "V_C" in obs
    assert "V_C_mask" in obs

    assert obs["V_P"].shape == (env.num_P, 6)
    assert obs["V_E"].shape == (env.num_E, 8)
    assert obs["V_C"].shape == (env.num_P, env.k_max, 6)
    assert obs["V_C_mask"].shape == (env.num_P, env.k_max)

    # mask 只能是 0/1
    mask_vals = np.unique(obs["V_C_mask"])
    assert np.all(np.isin(mask_vals, [0.0, 1.0]))

    # 老字段与新字段保持一致，避免后续 Phase3 迁移成本
    assert np.allclose(obs["V_P"], obs["pursuers"])
    assert np.allclose(obs["V_E"], obs["evaders"])
    assert np.allclose(obs["V_C"], obs["candidates"])
    assert np.allclose(obs["V_C_mask"], obs["candidate_mask"])


def test_phase2_action_masking_fallback_distribution():
    env = TODCMARLEnv(
        {
            "allow_dummy_if_missing": True,
            "render_mode": "none",
            "k_max": 10,
            "max_episode_steps": 5,
        }
    )
    obs, _ = env.reset()

    # 全 0 动作应被归一化到有效 mask 上
    zero_action = np.zeros((env.num_P, env.k_max), dtype=np.float32)
    normalized = env._normalize_action(zero_action)

    mask = obs["V_C_mask"]
    for i in range(env.num_P):
        if np.sum(mask[i]) > 0:
            assert np.isclose(np.sum(normalized[i]), 1.0)
            assert np.all(normalized[i][mask[i] == 0] == 0)