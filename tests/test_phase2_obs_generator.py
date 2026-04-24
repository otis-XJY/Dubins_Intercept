import numpy as np

from marl.envs import TODCMARLEnv


def test_phase2_obs_shapes_and_mask_consistency():
    env = TODCMARLEnv(
        {
            "render_mode": "none",
            "max_episode_steps": 5,
        }
    )

    obs, _ = env.reset()

    assert "self_uav" in obs
    assert "allies_local" in obs
    assert "self_pts" in obs
    assert "self_pts_mask" in obs
    assert "pursuer_active" in obs
    assert obs["pursuer_active"].shape == (env.num_P,)

    assert obs["self_uav"].shape == (env.num_P, 1, 3)
    assert obs["enemies"].shape == (env.num_P, env.num_E, 3)
    k_curr = obs["self_pts"].shape[1]
    assert k_curr >= 1
    assert obs["self_pts"].shape == (env.num_P, k_curr, 8)
    assert obs["self_pts_mask"].shape == (env.num_P, k_curr)

    mask_vals = np.unique(obs["self_pts_mask"].astype(np.int32))
    assert np.all(np.isin(mask_vals, [0, 1]))


def test_phase2_obs_direct_model_alignment_keys_exist_and_shapes():
    env = TODCMARLEnv(
        {
            "render_mode": "none",
            "max_episode_steps": 5,
        }
    )

    obs, _ = env.reset()
    ally_slots = max(1, env.num_P - 1)
    k_curr = obs["self_pts"].shape[1]
    ally_pts_slots = max(1, (env.num_P - 1) * k_curr)

    assert obs["self_uav"].shape == (env.num_P, 1, 3)
    assert obs["allies_local"].shape == (env.num_P, ally_slots, 3)
    assert obs["enemy_assigned_self"].shape == (env.num_P, 1, 3)
    assert obs["enemy_assigned_per_ally"].shape == (env.num_P, ally_slots, 3)
    assert obs["self_pts"].shape == (env.num_P, k_curr, 8)
    assert obs["ally_pts"].shape == (env.num_P, ally_pts_slots, 8)
    assert obs["enemies"].shape == (env.num_P, env.num_E, 3)
    assert obs["targets"].shape == (env.num_P, env.num_E, 2)
    assert obs["asset_target_self"].shape == (env.num_P, 1, 2)
    assert obs["asset_target_per_ally"].shape == (env.num_P, ally_slots, 2)

    assert obs["ally_mask"].shape == (env.num_P, ally_slots)
    assert obs["ally_enemy_mask"].shape == (env.num_P, ally_slots)
    assert obs["enemy_self_mask"].shape == (env.num_P, 1)
    assert obs["self_pts_mask"].shape == (env.num_P, k_curr)
    assert obs["ally_pts_mask"].shape == (env.num_P, ally_pts_slots)
    assert obs["enemy_mask"].shape == (env.num_P, env.num_E)
    assert obs["target_mask"].shape == (env.num_P, env.num_E)

    assert obs["ally_mask"].dtype == np.bool_
    assert obs["self_pts_mask"].dtype == np.bool_
    assert obs["ally_pts_mask"].dtype == np.bool_
    assert obs["enemy_mask"].dtype == np.bool_
    assert obs["target_mask"].dtype == np.bool_


def test_phase2_action_masking_fallback_distribution():
    env = TODCMARLEnv(
        {
            "render_mode": "none",
            "max_episode_steps": 5,
        }
    )
    obs, _ = env.reset()

    zero_action = np.zeros((env.num_P, obs["self_pts"].shape[1]), dtype=np.float32)
    normalized, _idx, _obs_snap = env._normalize_action(zero_action)

    mask = obs["self_pts_mask"].astype(np.float32)
    for i in range(env.num_P):
        if np.sum(mask[i]) > 0:
            assert np.isclose(np.sum(normalized[i]), 1.0)
            assert np.all(normalized[i][mask[i] == 0] == 0)
