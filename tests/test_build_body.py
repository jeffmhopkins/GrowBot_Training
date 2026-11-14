"""The printed-build body, per-surface materials and floors, and cloning a gait into a
policy before PPO."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from growbot_train.config import Config
from growbot_train.env import CONTROL_DT, GrowBotHeadingEnv
from growbot_train.gait import Keyframes, clone_gait
from growbot_train.surfaces import FLOORS, SURFACES
from growbot_train.train import CLIP_OBS


def _build_cfg(surfaces=("pla", "pla", "pla", "rubber")) -> Config:
    cfg = Config()
    cfg.task.body = "build"
    cfg.task.surfaces = surfaces
    cfg.task.start_pitch_deg = 25.0
    cfg.task.servo_torque_nm = 0.25
    cfg.task.action_to_rad = 1.5
    return cfg


def test_build_body_sits_on_its_tail_and_materials_land_on_their_surfaces() -> None:
    env = GrowBotHeadingEnv(_build_cfg())
    env.reset(seed=0, options={"nominal": True})
    assert 30 < env._rest_pitch < 45
    m = env.model
    hard = FLOORS["hard"].materials
    for surf, mat in zip(SURFACES, ("pla", "pla", "pla", "rubber"), strict=True):
        for g in env._surface_geoms[surf]:
            assert m.geom_friction[g, 0] == pytest.approx(hard[mat])
    assert m.geom_friction[m.geom("floor").id, 0] == 0.0
    assert m.actuator_forcerange[0, 1] == pytest.approx(0.25)


def test_floors_and_surface_friction_are_drawn_per_episode() -> None:
    env = GrowBotHeadingEnv(_build_cfg())
    seen = set()
    for seed in range(12):
        env.reset(seed=seed)
        p = env._body_params
        seen.add(p["floor"])
        lo, hi = env.cfg.randomization.surface_friction_scale
        assert all(lo <= v <= hi for v in p["surface_scale"].values())
        toe = env._surface_geoms["toe_nose"][0]
        want = FLOORS[p["floor"]].materials["rubber"]
        want *= p["surface_scale"]["toe_nose"] * p["floor_scale"]
        assert env.model.geom_friction[toe, 0] == pytest.approx(want)
        if p["floor"] == "carpet":
            assert env.model.geom_solref[toe, 0] == pytest.approx(FLOORS["carpet"].solref[0])
    assert seen == {"hard", "carpet"}


def test_longer_legs_move_their_contact_halves_too() -> None:
    env = GrowBotHeadingEnv(_build_cfg())
    env.reset(seed=0, options={"overrides": {"leg_scale": [1.1, 1.1]}})
    m = env.model
    for g in env._leg_all[0]:
        assert m.geom_size[g, 2] == pytest.approx(env._leg_half[0])


def test_bad_material_names_fail_loudly() -> None:
    with pytest.raises(ValueError):
        GrowBotHeadingEnv(_build_cfg(("pla", "pla", "skin", "pla")))  # bench-only ratchet
    with pytest.raises(ValueError):
        GrowBotHeadingEnv(_build_cfg(("pla", "pla")))


def test_legacy_setup_has_one_friction_everywhere() -> None:
    # With no surfaces (camera_walk.toml): no split, one friction, no extra draws.
    cfg = Config()
    cfg.task.body, cfg.task.surfaces = "rollout", ()
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=3)
    assert "floor" not in env._body_params
    assert np.unique(env.model.geom_friction[:, 0]).size == 1


def _gait_json(tmp_path, hz: float) -> str:
    k = 8
    ph = np.arange(k) / k
    rock = (0.6 * np.sin(2 * math.pi * ph)).tolist()
    path = tmp_path / "gait.json"
    path.write_text(json.dumps({"gait": {"hz": hz, "keys_rad": [rock, rock]}}))
    return str(path)


def test_keyframes_interpolate_and_wrap(tmp_path) -> None:
    g = Keyframes.load(_gait_json(tmp_path, 2.0))
    assert g(0.0) == pytest.approx([0.0, 0.0])
    assert g(0.125)[0] == pytest.approx(0.6)  # a quarter cycle at 2 Hz
    assert g(0.5) == pytest.approx(g(0.0))


def test_cloning_a_gait_makes_the_policy_play_it(tmp_path) -> None:
    cfg = _build_cfg()
    cfg.task.gait_hz = 2.0
    cfg.task.episode_seconds = 2.0
    gait = Keyframes.load(_gait_json(tmp_path, 2.0))
    vec = VecNormalize(DummyVecEnv([lambda: GrowBotHeadingEnv(cfg)]), clip_obs=CLIP_OBS)
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=64,
        batch_size=64,
        seed=0,
        policy_kwargs={"net_arch": {"pi": [32, 32], "vf": [16, 16]}},
        device="cpu",
    )
    mse = clone_gait(model, vec, cfg, gait, samples=1500, epochs=60, log=lambda *_: None)
    assert mse < 0.05
    obs = vec.reset()
    errs = []
    for _ in range(50):
        step = vec.get_attr("step_index")[0]
        action, _ = model.predict(obs, deterministic=True)
        want = gait(step * CONTROL_DT) / cfg.task.action_to_rad
        errs.append(abs(np.tanh(action[0]) - want).max())
        obs, *_ = vec.step(action)
    assert np.mean(errs) < 0.1


def test_cloning_refuses_a_clock_that_does_not_match(tmp_path) -> None:
    cfg = _build_cfg()
    cfg.task.gait_hz = 1.0
    vec = VecNormalize(DummyVecEnv([lambda: GrowBotHeadingEnv(cfg)]), clip_obs=CLIP_OBS)
    model = PPO("MlpPolicy", vec, n_steps=64, batch_size=64, device="cpu")
    with pytest.raises(ValueError, match="gait_hz"):
        clone_gait(model, vec, cfg, Keyframes.load(_gait_json(tmp_path, 2.0)), 64, 1)
