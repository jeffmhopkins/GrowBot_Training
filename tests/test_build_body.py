"""The printed-build body, per-surface materials and floors, and cloning a gait into a
policy before PPO."""

from __future__ import annotations

import json
import math

import mujoco
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
    cfg.task.start_pitch_deg = 35.0
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


def test_random_robots_all_start_resting_on_their_tails() -> None:
    # Two ways to rest: on the tail's bottom edge and the feet (about 40 degrees), or
    # flat on the tail's cut face (about 60, the cut is 30 degrees off upright).
    env = GrowBotHeadingEnv(_build_cfg(("pla",) * 4))
    for seed in range(60):
        env.reset(seed=seed)
        assert 25 < env._rest_pitch < 65, (seed, env._body_params["leg_scale"])
    for scale in (0.85, 1.15):
        env.reset(seed=0, options={"overrides": {"leg_scale": [scale, scale]}})
        assert 25 < env._rest_pitch < 65, scale


def test_the_tail_is_the_cut_back_end_and_the_legs_draw_once() -> None:
    env = GrowBotHeadingEnv(_build_cfg(("pla",) * 4))
    env.reset(seed=0, options={"nominal": True})
    m, d = env.model, env.data
    tail = set(env._surface_geoms["tail"])
    assert {m.geom(g).name for g in tail} >= {"tail_0", "tail_1"}
    floor = m.geom("floor").id
    hits = {c.geom1 if c.geom2 == floor else c.geom2 for c in d.contact[: d.ncon]}
    assert hits & tail and not hits & set(env._surface_geoms["belly"])
    # Each leg's mass box is hidden: its two contact halves draw it.
    for leg in env._leg_all:
        assert sum(m.geom_rgba[g, 3] > 0 for g in leg) == 2


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


@pytest.mark.parametrize("scale", [1.04, 1.1, 1.15])
def test_longer_legs_stand_on_the_floor_not_in_it(scale) -> None:
    env = GrowBotHeadingEnv(_build_cfg(("pla",) * 4))
    env.reset(seed=0, options={"overrides": {"leg_scale": [scale, scale]}})
    m, d = env.model, env.data
    for g in env._leg_all[0]:
        assert m.geom_size[g, 2] == pytest.approx(env._leg_half[0])
    d.ctrl[:] = 0.5  # swing the feet down onto the floor
    for _ in range(200):
        mujoco.mj_step(m, d)
    floor = m.geom("floor").id
    touching = set()
    for c in d.contact[: d.ncon]:
        if floor in (c.geom1, c.geom2):
            touching.add(c.geom2 if c.geom1 == floor else c.geom1)
            assert c.dist > -0.001  # resting on it, not sunk in
    toes = {g for s in ("toe_nose", "toe_tail") for g in env._surface_geoms[s]}
    assert touching & toes


def test_floor_edges_in_the_sweep_match_the_training_range() -> None:
    from growbot_train.evaluate import sweep_cases

    cfg = _build_cfg()
    cases = sweep_cases(cfg)
    assert cases["slippery"] == {"floor_scale": cfg.randomization.floor_friction_scale[0]}
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"overrides": cases["grippy"]})
    assert env._body_params["floor_scale"] == cfg.randomization.floor_friction_scale[1]


def test_bad_material_names_fail_loudly() -> None:
    with pytest.raises(ValueError):
        GrowBotHeadingEnv(_build_cfg(("pla", "pla", "skin", "pla")))  # bench-only ratchet
    with pytest.raises(ValueError):
        GrowBotHeadingEnv(_build_cfg(("pla", "pla")))
    cfg = _build_cfg()
    cfg.task.floors = ()
    with pytest.raises(ValueError, match="floor"):
        GrowBotHeadingEnv(cfg)


def test_a_run_saved_before_surfaces_keeps_its_old_physics(tmp_path) -> None:
    from dataclasses import asdict

    from growbot_train import config as config_mod

    old = asdict(Config())
    old["task"]["body"] = "rollout"
    for k in ("surfaces", "floors", "start_pitch_deg", "servo_torque_nm"):
        del old["task"][k]
    cfg = config_mod.from_dict(old, saved=True)
    assert (cfg.task.surfaces, cfg.task.start_pitch_deg, cfg.task.servo_torque_nm) == ((), 0, 0)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(old))
    assert config_mod.load(path).task.surfaces == ()
    # A hand-written config without the key gets today's defaults.
    assert config_mod.from_dict({"task": {"gait_hz": 2.0}}).task.surfaces == Config().task.surfaces


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


def test_the_replay_draws_each_leg_once() -> None:
    from growbot_train.replay import _geoms

    env = GrowBotHeadingEnv(_build_cfg(("pla",) * 4))
    drawn = set(_geoms(env))
    for leg in env._leg_all:
        assert len(drawn & set(leg)) == 2
