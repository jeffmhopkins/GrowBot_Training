from __future__ import annotations

from pathlib import Path

import pytest

from growbot_train import config as config_mod

SHIPPED = Path(__file__).parent.parent / "configs" / "heading_walk.toml"


def test_shipped_config_matches_the_code_defaults() -> None:
    # The TOML is documentation as much as config: it must say what the code does.
    assert config_mod.load(SHIPPED).to_dict() == config_mod.Config().to_dict()


def test_override_parses_toml_values() -> None:
    cfg = config_mod.load(
        SHIPPED,
        [
            "reward.speed_cap_m_s=0.05",
            "task.retarget_every_s=[1, 2]",
            "randomization.enabled=false",
            "train.n_envs=3",
        ],
    )
    assert cfg.reward.speed_cap_m_s == 0.05
    assert cfg.task.retarget_every_s == (1.0, 2.0)
    assert cfg.randomization.enabled is False
    assert cfg.train.n_envs == 3


@pytest.mark.parametrize("bad", ["reward.nope=1", "nosection.x=1", "reward=1"])
def test_unknown_settings_fail_loudly(bad: str) -> None:
    with pytest.raises((KeyError, ValueError)):
        config_mod.load(None, [bad])


def test_round_trip_through_saved_json(tmp_path: Path) -> None:
    import json

    cfg = config_mod.load(SHIPPED, ['task.body="olie"'])
    cfg.save(tmp_path / "c.json")
    again = config_mod.from_dict(json.loads((tmp_path / "c.json").read_text()))
    assert again.to_dict() == cfg.to_dict()


def test_load_reads_a_saved_run_config_with_overrides(tmp_path: Path) -> None:
    cfg = config_mod.load(SHIPPED, ["reward.speed_cap_m_s=0.04"])
    cfg.save(tmp_path / "config.json")
    again = config_mod.load(tmp_path / "config.json", ["train.n_envs=2"])
    assert again.reward.speed_cap_m_s == 0.04
    assert again.train.n_envs == 2


def test_camera_walk_differs_only_where_it_says() -> None:
    cam = config_mod.load(SHIPPED.with_name("camera_walk.toml")).to_dict()
    base = config_mod.Config().to_dict()
    changed = {
        (section, key)
        for section, values in cam.items()
        if isinstance(values, dict)
        for key, value in values.items()
        if base[section][key] != value
    }
    assert changed == {
        ("reward", "stance"),
        ("reward", "turn"),
        ("task", "facing"),
        ("task", "stand_lift_deg"),
        ("reward", "back_lean"),
        ("reward", "speed_cap_m_s"),
        ("reward", "overspeed"),
        ("reward", "stop_drift_free"),
        ("task", "back_fall_pitch_deg"),
        ("task", "gait_hz"),
        ("task", "speed"),
        ("task", "tilt_command"),
        ("task", "episode_seconds"),
        ("task", "retarget_every_s"),
        ("train", "total_steps"),
        # Still on upstream's sim body until the tail-first foot work settles.
        ("task", "body"),
        ("task", "surfaces"),
        ("task", "start_pitch_deg"),
    }
    assert cam["name"] == "camera_walk"
    marked = SHIPPED.with_name("camera_walk.toml").read_text().count("# CHANGED:")
    assert marked == len(changed)


def test_look_differs_from_the_camera_walk_only_where_it_says() -> None:
    look = config_mod.load(SHIPPED.with_name("look.toml")).to_dict()
    cam = config_mod.load(SHIPPED.with_name("camera_walk.toml")).to_dict()
    changed = {
        (section, key)
        for section, values in look.items()
        if isinstance(values, dict)
        for key, value in values.items()
        if cam[section][key] != value
    }
    assert changed == {
        ("reward", "heading"),
        ("reward", "wobble"),
        ("reward", "turn"),
        ("reward", "speed_cap_m_s"),
        ("reward", "overspeed"),
        ("reward", "stop_drift_free"),
        ("task", "gait_hz"),
        ("task", "retarget_every_s"),
        ("task", "speed"),
        ("train", "total_steps"),
        ("task", "body"),
        ("task", "surfaces"),
        ("task", "start_pitch_deg"),
    }
    assert look["name"] == "look"
    assert SHIPPED.with_name("look.toml").read_text().count("# CHANGED:") == len(changed)


def test_tall_waddle_differs_from_the_camera_walk_only_where_it_says() -> None:
    from growbot_train.gait import Keyframes

    path = SHIPPED.with_name("tall_waddle.toml")
    waddle = config_mod.load(path)
    cam = config_mod.load(SHIPPED.with_name("camera_walk.toml")).to_dict()
    changed = {
        (section, key)
        for section, values in waddle.to_dict().items()
        if isinstance(values, dict)
        for key, value in values.items()
        if cam[section][key] != value
    }
    assert path.read_text().count("# CHANGED:") == len(changed)
    # The cloned rhythm must tick with the policy's clock and fit inside its action range.
    gait = Keyframes.load(SHIPPED.parents[1] / waddle.train.warm_gait)
    assert gait.hz == waddle.task.gait_hz
    assert abs(gait.keys).max() < 0.9 * waddle.task.action_to_rad
