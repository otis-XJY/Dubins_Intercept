import pytest

torch = pytest.importorskip("torch")

from marl.nn.models import (
    SCHEME_KEY_TO_NAME,
    UAVInterceptionNetwork,
    build_actor_critic_schemes,
    design_mode_to_name,
)


def _fake_obs(bsz, a, m, ma):
    return {
        "self_uav": torch.randn(bsz, 1, 3),
        "allies_local": torch.randn(bsz, a, 3),
        "enemy_assigned_self": torch.randn(bsz, 1, 3),
        "enemy_assigned_per_ally": torch.randn(bsz, a, 3),
        "asset_target_self": torch.randn(bsz, 1, 2),
        "asset_target_per_ally": torch.randn(bsz, a, 2),
        "self_pts": torch.randn(bsz, m, 8),
        "ally_pts": torch.randn(bsz, ma, 8),
        "ally_mask": torch.ones(bsz, a, dtype=torch.bool),
        "enemy_self_mask": torch.ones(bsz, 1, dtype=torch.bool),
        "ally_enemy_mask": torch.ones(bsz, a, dtype=torch.bool),
        "self_pts_mask": torch.ones(bsz, m, dtype=torch.bool),
        "ally_pts_mask": torch.ones(bsz, ma, dtype=torch.bool),
    }


def test_phase3_model_forward_and_masking():
    bsz, a, m, ma = 2, 3, 8, 6
    model = UAVInterceptionNetwork(hidden_dim=64, num_heads=4, use_soft_gating=False)

    obs = _fake_obs(bsz, a, m, ma)

    obs["self_pts_mask"][0, 4:] = False
    obs["self_pts_mask"][1, :] = False

    out = model(obs)
    logits = out["action_logits"]
    probs = out["action_probs"]
    idx = out["best_candidate_idx"]
    value = out["value"]

    assert logits.shape == (bsz, m)
    assert probs.shape == (bsz, m)
    assert idx.shape == (bsz,)
    assert value.shape == (bsz,)

    assert torch.allclose(probs[0].sum(), torch.tensor(1.0), atol=1e-5)

    assert torch.isfinite(probs[1]).all()
    assert idx[1].item() == -1


@pytest.mark.parametrize(
    "alias,expected_name",
    [
        ("A", "Concatenative Query Network"),
        ("Concatenative Query Network", "Concatenative Query Network"),
        ("gqn", "Gated Query Network"),
        ("Point-Wise Scoring Network", "Point-Wise Scoring Network"),
    ],
)
def test_design_mode_aliases(alias, expected_name):
    model = UAVInterceptionNetwork(hidden_dim=32, num_heads=4, design_mode=alias)
    assert model.design_name == expected_name
    assert design_mode_to_name(alias) == expected_name


def test_build_actor_critic_schemes_single_and_multi():
    single = build_actor_critic_schemes("Concatenative Query Network", hidden_dim=32)
    assert list(single.keys()) == ["Concatenative Query Network"]

    multi = build_actor_critic_schemes(["A", "Gated Query Network", "C"], hidden_dim=32)
    assert list(multi.keys()) == [
        SCHEME_KEY_TO_NAME["A"],
        SCHEME_KEY_TO_NAME["B"],
        SCHEME_KEY_TO_NAME["C"],
    ]


def test_legacy_use_soft_gating_compatibility():
    model_a = UAVInterceptionNetwork(hidden_dim=32, num_heads=4, use_soft_gating=False)
    model_b = UAVInterceptionNetwork(hidden_dim=32, num_heads=4, use_soft_gating=True)
    assert model_a.design_name == "Concatenative Query Network"
    assert model_b.design_name == "Gated Query Network"
