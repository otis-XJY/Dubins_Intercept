import pytest

torch = pytest.importorskip("torch")

from marl.models import (
    SCHEME_KEY_TO_NAME,
    UAVInterceptionNetwork,
    build_actor_critic_schemes,
    design_mode_to_name,
)


def test_phase3_model_forward_and_masking():
    bsz, a, m, ma, n, v = 2, 3, 8, 6, 4, 5
    model = UAVInterceptionNetwork(hidden_dim=64, num_heads=4, use_soft_gating=False)

    obs = {
        "self_uav": torch.randn(bsz, 1, 3),
        "ally_uavs": torch.randn(bsz, a, 3),
        "self_pts": torch.randn(bsz, m, 8),
        "ally_pts": torch.randn(bsz, ma, 8),
        "enemies": torch.randn(bsz, n, 3),
        "targets": torch.randn(bsz, v, 2),
        "ally_mask": torch.ones(bsz, a, dtype=torch.bool),
        "self_pts_mask": torch.ones(bsz, m, dtype=torch.bool),
        "ally_pts_mask": torch.ones(bsz, ma, dtype=torch.bool),
        "enemy_mask": torch.ones(bsz, n, dtype=torch.bool),
        "target_mask": torch.ones(bsz, v, dtype=torch.bool),
    }

    # Mask-out half of self points for one sample and all self points for another sample.
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

    # Valid-mask rows should still form a probability simplex.
    assert torch.allclose(probs[0].sum(), torch.tensor(1.0), atol=1e-5)

    # Fully masked row should still produce finite outputs and a valid index.
    assert torch.isfinite(probs[1]).all()
    assert 0 <= idx[1].item() < m


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