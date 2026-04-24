import numpy as np
import pytest

from marl.envs import TODCMARLEnv


def _first_valid_actions(obs: dict, num_p: int) -> np.ndarray:
    """Pick one valid candidate index per pursuer from ``self_pts_mask`` (fail-fast if a row is empty)."""
    mask = np.asarray(obs["self_pts_mask"])
    out = np.zeros((num_p,), dtype=np.int64)
    for i in range(num_p):
        valid = np.where(mask[i] > 0)[0]
        assert valid.size > 0, f"no valid candidates for pursuer {i}"
        out[i] = int(valid[0])
    return out


def test_marl_env_reset_and_step_smoke():
    env = TODCMARLEnv(
        {
            "render_mode": "none",
            "max_episode_steps": 20,
        }
    )

    obs, info = env.reset()
    assert "self_uav" in obs
    assert "enemies" in obs
    assert "self_pts" in obs
    assert "self_pts_mask" in obs
    assert obs["self_pts"].shape[1] >= 1
    assert isinstance(info["real_mode"], bool)

    # ``_normalize_action`` requires every pursuer row in ``self_pts_mask`` to have a valid candidate.
    mask = np.asarray(obs["self_pts_mask"])
    if not bool(np.all(np.sum(mask, axis=1) > 0)):
        pytest.skip("Reset state has pursuer(s) with no candidates; step smoke needs full mask rows")

    done = False
    trunc = False
    steps = 0
    while not done and not trunc and steps < 10:
        mask = np.asarray(obs["self_pts_mask"])
        if not bool(np.all(np.sum(mask, axis=1) > 0)):
            break
        action = _first_valid_actions(obs, env.num_P)
        obs, rewards, terminations, truncations, infos = env.step(action)
        assert len(rewards) == env.num_P
        assert "__all__" in terminations
        assert "__all__" in truncations
        assert all(k in infos for k in [f"p_{i}" for i in range(env.num_P)])
        done = terminations["__all__"]
        trunc = truncations["__all__"]
        steps += 1

    assert steps > 0