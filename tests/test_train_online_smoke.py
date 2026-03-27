import pytest
from pathlib import Path

torch = pytest.importorskip("torch")

from marl.train_online import TrainConfig, _load_config_file, _parse_gpu_ids, train_online


def test_train_online_single_scheme_smoke():
    cfg = TrainConfig(
        episodes=1,
        max_episode_steps=3,
        gamma=0.95,
        value_coef=0.5,
        entropy_coef=0.0,
        lr=1e-3,
        hidden_dim=32,
        num_heads=4,
        seed=7,
        device="cpu",
        schemes=["Concatenative Query Network"],
        step_mode="time",
        allow_dummy_if_missing=True,
        enable_dwa_replan=False,
        wandb_mode="disabled",
        log_interval=1,
    )

    history = train_online(cfg)
    assert "Concatenative Query Network" in history
    assert len(history["Concatenative Query Network"]) == 1
    assert "return" in history["Concatenative Query Network"][0]


def test_train_online_eval_best_and_replay(tmp_path: Path):
    save_dir = tmp_path / "ckpt"
    replay_dir = tmp_path / "replay"
    cfg = TrainConfig(
        episodes=1,
        max_episode_steps=2,
        hidden_dim=32,
        num_heads=4,
        device="cpu",
        schemes=["Gated Query Network"],
        step_mode="time",
        allow_dummy_if_missing=True,
        enable_dwa_replan=False,
        wandb_mode="disabled",
        eval_interval=1,
        eval_episodes=1,
        save_dir=str(save_dir),
        save_best=True,
        save_replay=True,
        replay_dir=str(replay_dir),
    )

    train_online(cfg)

    assert (save_dir / "Gated_Query_Network_last.pt").exists()
    assert (save_dir / "Gated_Query_Network_best.pt").exists()
    replay_files = list((replay_dir / "Gated_Query_Network").glob("episode_*.npz"))
    assert len(replay_files) >= 1


def test_load_config_file_yaml_and_json(tmp_path: Path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text(
        "episodes: 3\n"
        "step_mode: decision\n"
        "multi_gpu: true\n"
        "gpu_ids: [0, 1]\n"
        "reward:\n"
        "  dist_progress_scale: 0.08\n",
        encoding="utf-8",
    )
    cfg_yaml = _load_config_file(str(yaml_path))
    assert cfg_yaml["episodes"] == 3
    assert cfg_yaml["step_mode"] == "decision"
    assert cfg_yaml["multi_gpu"] is True
    assert cfg_yaml["gpu_ids"] == [0, 1]
    assert cfg_yaml["reward"]["dist_progress_scale"] == 0.08

    json_path = tmp_path / "cfg.json"
    json_path.write_text('{"episodes": 4, "time_res": 0.5}', encoding="utf-8")
    cfg_json = _load_config_file(str(json_path))
    assert cfg_json["episodes"] == 4
    assert cfg_json["time_res"] == 0.5


def test_parse_gpu_ids():
    assert _parse_gpu_ids(None) is None
    assert _parse_gpu_ids("") is None
    assert _parse_gpu_ids("0") == [0]
    assert _parse_gpu_ids("0,1,3") == [0, 1, 3]
