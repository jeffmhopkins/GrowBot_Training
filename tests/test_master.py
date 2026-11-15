"""The master distillation: each teacher must see exactly what it trained on, the mode
rule must follow the speed's sign, and the exported file must be one the app runner
accepts."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from growbot_train import config as config_mod
from growbot_train import master
from growbot_train.env import GrowBotHeadingEnv, channels_for
from growbot_train.evaluate import env_for
from growbot_train.export import _residual_spec
from growbot_train.policy import JsonPolicy, runner_forward


@pytest.mark.parametrize("mode", master.MODES)
def test_a_teacher_sees_the_frames_its_own_env_builds(mode: str) -> None:
    path = master.DEFAULT_TEACHERS[mode]
    pol = JsonPolicy(path)
    env = env_for(pol)
    teacher = master.Teacher.load(mode, path)
    fwd = 0.0 if mode == "look" else 1.0
    obs, _ = env.reset(
        seed=3,
        options={"retarget": False, "speed_cmd": fwd, "tilt_cmd": 1.0, "target_error_deg": 40.0},
    )
    for _ in range(25):
        f = env.hist[0]
        s = master.Senses(
            acc=f[0:3],
            gyro=f[3:6],
            prev_rad=env.prev_action * env.cfg.task.action_to_rad,
            err_lens=env._err_obs,
            err_body=env._err_obs,
            # The student asks the nose walk with a negative speed; it sees the size.
            forward=-fwd if mode == "back" else fwd,
            tilt=env.tilt_cmd,
            step_index=env.step_index,
        )
        assert np.allclose(master.teacher_frame(teacher, s), f, atol=1e-12)
        obs, _, term, _, _ = env.step(pol.env_action(obs))
        assert not term


def test_only_the_speeds_sign_picks_the_waddle() -> None:
    for err in (0.0, 90.0, -170.0, 180.0):
        assert master.choose_mode(0.0, err) == "look"
        assert master.choose_mode(1.0, err) != "back"
    for err in (0.0, 44.0, -45.0):
        assert master.choose_mode(-0.2, err) == "back"
    assert master.choose_mode(1.0, 29.0) == "walk"
    assert master.choose_mode(1.0, -31.0) == "look"
    assert master.choose_mode(1.0, 170.0) == "look"


def test_the_walk_lets_go_only_past_its_hysteresis() -> None:
    assert master.choose_mode(1.0, 35.0, "walk") == "walk"
    assert master.choose_mode(1.0, 35.0, "look") == "look"
    assert master.choose_mode(1.0, 45.0, "walk") == "look"


def test_backing_wide_of_the_target_spins_with_look_first() -> None:
    for err in (46.0, -90.0, 180.0):
        assert master.choose_mode(-0.2, err) == "look"
    assert master.choose_mode(-0.2, 50.0, "back") == "back"
    assert master.choose_mode(-0.2, 50.0, "look") == "look"
    assert master.choose_mode(-0.2, -56.0, "back") == "look"


def test_teacher_actions_are_matched_in_radians() -> None:
    rad = np.array([0.5, -0.25])
    assert np.allclose(master.to_student(rad) * master.ACTION_TO_RAD, rad)
    assert np.all(np.abs(master.to_student(np.array([1.57, -1.57]))) <= master.SQUASH)


def test_a_residual_teacher_sees_its_frames_and_sends_its_legs(tmp_path) -> None:
    # A stand-in for the residual nose walk: its config, small random weights.
    cfg = config_mod.load("configs/nose_walk.toml")
    assert cfg.task.residual_gait and cfg.task.residual_steer_deg
    env = GrowBotHeadingEnv(cfg)
    channels = channels_for(cfg)
    n = cfg.task.history_frames * len(channels)
    rng = np.random.default_rng(0)
    policy = {
        "obs_spec": {
            "segments": [{"channels": list(channels), "frames": cfg.task.history_frames}],
            "cycles": {"gait_sin": {"type": "sin", "frequency": cfg.task.gait_hz}},
            "action_to_rad": cfg.task.action_to_rad,
            **_residual_spec(cfg),
        },
        "mean": [0.0] * n,
        "std": [1.0] * n,
        "layers": [{"W": (rng.normal(0, 0.3, (n, 2))).tolist(), "b": [0.0, 0.0]}],
        "training": {"config": cfg.to_dict()},
    }
    path = tmp_path / "residual.json"
    path.write_text(json.dumps(policy))
    teacher = master.Teacher.load("back", path)
    assert teacher.gait is not None
    env.reset(
        seed=4,
        options={"retarget": False, "speed_cmd": 1.0, "target_error_deg": 35.0},
    )
    for _ in range(40):
        s = master.Senses(
            acc=env.hist[0][0:3],
            gyro=env.hist[0][3:6],
            prev_rad=env.prev_action * cfg.task.residual_rad + teacher.base,
            err_lens=env._err_obs,
            err_body=env._err_obs,
            forward=-1.0,
            tilt=1.0,
            step_index=env.step_index,
        )
        teacher.observe(s)
        assert np.allclose(teacher.hist[0], env.hist[0], atol=1e-12)
        out = runner_forward(teacher.data, teacher.hist.reshape(-1))
        want = env.gait_rad() + env.steer_rad() + out * cfg.task.residual_rad
        assert np.allclose(teacher.act_rad(), np.clip(want, -1.57, 1.57), atol=1e-12)
        _, _, term, _, _ = env.step(np.arctanh(out))
        assert not term


def test_the_rollout_judges_falls_by_the_active_teachers_rule() -> None:
    r = master.Rollout(master.DEFAULT_TEACHERS)
    assert r.env.dr_scale == min(t.ramp_to for t in r.teachers.values())
    for mode, t in r.teachers.items():
        r.judge_as(mode)
        assert r.env.cfg.task.back_fall_pitch_deg == t.back_fall_deg


def test_distill_writes_a_policy_the_app_runner_accepts(tmp_path) -> None:
    path = master.distill(
        tmp_path,
        master.DEFAULT_TEACHERS,
        rounds=(1.0, 0.0),
        episodes=2,
        seconds=1.0,
        epochs=1,
        workers=1,
        log=lambda _row: None,
    )
    policy = json.loads(path.read_text())
    spec = policy["obs_spec"]
    seg = spec["segments"][0]
    assert policy["obs_size"] == seg["frames"] * len(seg["channels"]) == len(policy["mean"])
    # growbot_runner.js refuses a cycle channel without a declared cycle and control_hz.
    assert spec["control_hz"] > 0
    for c in seg["channels"]:
        if c.startswith("cycle:"):
            decl = spec["cycles"][c[len("cycle:") :]]
            assert decl["type"] in ("sin", "cos") and decl["frequency"] > 0
    assert spec["cycles"]["walk_sin"]["frequency"] == pytest.approx(2.347)
    out = runner_forward(policy, np.zeros(policy["obs_size"]))
    assert out.shape == (2,) and np.all(np.isfinite(out))
    rounds = json.loads((tmp_path / "distill.json").read_text())
    assert [r["beta"] for r in rounds] == [1.0, 0.0]


def test_each_clock_runs_at_its_teachers_rate() -> None:
    teachers = {m: master.Teacher.load(m, p) for m, p in master.DEFAULT_TEACHERS.items()}
    clocks = master.student_clocks(teachers)
    s = master.Senses(
        acc=np.zeros(3),
        gyro=np.zeros(3),
        prev_rad=np.zeros(2),
        err_lens=0.0,
        err_body=0.0,
        forward=0.0,
        tilt=1.0,
        step_index=7,
    )
    frame = master.build_frame(master.STUDENT_CHANNELS, s, master.ACTION_TO_RAD, True, clocks)
    i = master.STUDENT_CHANNELS.index("cycle:walk_sin")
    assert frame[i] == pytest.approx(math.sin(2 * math.pi * 2.347 * 7 * 0.04))


def test_the_switch_test_runs_teacher_and_master_on_the_same_scripts() -> None:
    r = master.Rollout(master.DEFAULT_TEACHERS)
    policy = json.loads((master.Path("policies/master-1/policy_master.json")).read_text())
    runner = master.MasterRunner(policy, r.teachers)
    for who in (None, runner):
        out = master.switch_test(r, who, [1], 2.0)
        assert out["episodes"] == 1 and out["falls"] in (0, 1)


def test_teacher_and_master_get_the_same_script() -> None:
    r = master.Rollout(master.DEFAULT_TEACHERS)
    policy = json.loads((master.Path("policies/master-1/policy_master.json")).read_text())
    targets = []
    for student in (None, policy):
        seen = []
        orig = master.Script.tick

        def spy(self, env, t, seen=seen, orig=orig):
            orig(self, env, t)
            seen.append((round(env.target_deg, 6), self.forward, self.tilt))

        master.Script.tick = spy
        try:
            r.episode(5, 20.0, student, beta=0.0)
        finally:
            master.Script.tick = orig
        targets.append(seen)
    n = min(len(targets[0]), len(targets[1]))
    assert n > 0 and targets[0][:n] == targets[1][:n]


def _stand_in_residual(tmp_path, scale: float = 0.3):
    cfg = config_mod.load("configs/nose_walk.toml")
    channels = channels_for(cfg)
    n = cfg.task.history_frames * len(channels)
    rng = np.random.default_rng(0)
    policy = {
        "obs_spec": {
            "segments": [{"channels": list(channels), "frames": cfg.task.history_frames}],
            "cycles": {"gait_sin": {"type": "sin", "frequency": cfg.task.gait_hz}},
            "action_to_rad": cfg.task.action_to_rad,
            **_residual_spec(cfg),
        },
        "mean": [0.0] * n,
        "std": [1.0] * n,
        "layers": [{"W": (rng.normal(0, scale, (n, 2))).tolist(), "b": [0.0, 0.0]}],
        "training": {"config": cfg.to_dict()},
    }
    path = tmp_path / "residual.json"
    path.write_text(json.dumps(policy))
    return cfg, path, channels


def test_a_residual_teacher_reads_back_its_own_correction_in_the_student_env(
    tmp_path,
) -> None:
    # The real path: senses() from the student env, legs sent through to_student. The
    # teacher must see its own last correction (none on the first tick).
    cfg, path, channels = _stand_in_residual(tmp_path, scale=0.05)
    teachers = dict(master.DEFAULT_TEACHERS)
    teachers["back"] = str(path)
    r = master.Rollout(teachers)
    env, teacher = r.env, r.teachers["back"]
    env.reset(seed=2, options={"retarget": False, "speed_cmd": 0.0, "target_error_deg": 20.0})
    teacher.reset()
    i0 = channels.index("prev_action_0")
    last = np.zeros(2)
    for _ in range(30):
        teacher.observe(master.senses(env, -1.0, 1.0))
        assert np.allclose(teacher.hist[0][i0 : i0 + 2], last, atol=1e-6)
        last = runner_forward(teacher.data, teacher.hist.reshape(-1))
        env.step(np.arctanh(master.to_student(teacher.act_rad())))


def _senses(err: float, forward: float = 0.0, tilt: float = 1.0) -> master.Senses:
    return master.Senses(
        acc=np.zeros(3),
        gyro=np.zeros(3),
        prev_rad=np.zeros(2),
        err_lens=err,
        err_body=err,
        forward=forward,
        tilt=tilt,
        step_index=0,
    )


def _settle(hold: master.Hold, errs) -> list[np.ndarray]:
    return [hold.legs_rad(_senses(e), "look", np.array([0.1 * k, -0.1 * k])) for k, e in errs]


def test_the_hold_freezes_the_last_seconds_mean_stance_once_settled() -> None:
    hold = master.Hold()
    w = master.HOLD_WINDOW
    out = _settle(hold, [(k, 2.0) for k in range(w)])
    assert hold.holding
    # The tick that completes the window still sends look's own legs; the next is held.
    assert out[-1] == pytest.approx([0.1 * (w - 1), -0.1 * (w - 1)])
    mean = 0.1 * (w - 1) / 2
    held = hold.legs_rad(_senses(3.0), "look", np.array([9.0, 9.0]))
    assert held == pytest.approx([mean, -mean])


def test_the_hold_waits_out_a_sway_that_reaches_too_far() -> None:
    hold = master.Hold()
    # Averages near zero, but swings past HOLD_SPREAD_DEG within the second.
    big = master.HOLD_SPREAD_DEG + 1.0
    _settle(hold, [(k, big if k % 2 else -big) for k in range(master.HOLD_WINDOW)])
    assert not hold.holding
    hold.reset()
    _settle(hold, [(k, master.HOLD_ENTER_DEG + 1.0) for k in range(master.HOLD_WINDOW)])
    assert not hold.holding


@pytest.mark.parametrize(
    ("mode", "senses"),
    [
        ("look", _senses(master.HOLD_EXIT_DEG + 1.0)),
        ("look", _senses(0.0, tilt=1.0 + 2 * master.HOLD_TILT_BAND)),
        ("look", _senses(0.0, forward=1.0)),
        ("walk", _senses(0.0, forward=1.0)),
    ],
)
def test_the_hold_lets_go_when_aim_tilt_or_a_move_asks(mode: str, senses) -> None:
    hold = master.Hold()
    _settle(hold, [(k, 0.0) for k in range(master.HOLD_WINDOW)])
    assert hold.holding
    own = np.array([0.5, -0.5])
    assert hold.legs_rad(senses, mode, own) == pytest.approx(own)
    assert not hold.holding


def test_teacher_driven_look_stands_still_once_held() -> None:
    r = master.Rollout(master.DEFAULT_TEACHERS)
    d = master.drive(r, None, 0, 0.0, 0.0, 6.0, nominal=True)
    assert not d["fell"] and d["drift_cm"] < 1.0


def test_the_master_job_distils_scores_and_reports_progress(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "master.toml"
    teachers = "\n".join(f'{m} = "{p}"' for m, p in master.DEFAULT_TEACHERS.items())
    cfg.write_text(f"[master]\n{teachers}\nepisodes = 2\nrounds = [1.0, 0.0]\nepochs = 1\n")
    assert master.is_job(cfg) and not master.is_job(master.Path("configs/look.toml"))
    row = {"toward": 0.0, "speed": 0.02, "err": 4.0, "turn_s": 1.0, "frame_pct": 99.0}
    row |= {"img_x": 2.0, "falls": 0, "runs": 5}
    report = {m: {"+0": {"teacher": row, "master": {**row, "speed": 0.019}}} for m in master.MODES}
    sw = {"falls": 1, "switches": 90, "episodes": 20}
    report["switch"] = {"teacher": sw, "master": sw}
    monkeypatch.setattr(master, "evaluate", lambda _path: report)
    monkeypatch.setattr(master, "distill", _tiny_distill(master.distill))
    path = master.run_job(tmp_path / "run", cfg)
    assert path.name == "policy_master.json"
    # JBrain's Launcher fails a GrowBot run whose export_verify.log doesn't end so.
    verify = (tmp_path / "run" / "export_verify.log").read_text().splitlines()
    assert verify[-1] == "EXPORT OK"
    broken = json.loads(path.read_text())
    broken["obs_size"] += 1
    path.write_text(json.dumps(broken))
    assert not master.verify_export(path, tmp_path / "v.log")
    progress = json.loads((tmp_path / "run" / "live" / "progress.json").read_text())
    assert progress["state"] == "finishing" and progress["summary"][0]["value"] == "2 of 2"
    tail = (tmp_path / "run" / "eval.txt").read_text().splitlines()[-1]
    assert tail == (
        "eval summary: speed within 5% of the teachers, aim within 0.0 deg, "
        "falls 0/15, 1 in 90 switches"
    )


def _tiny_distill(real):
    def run(out, paths, rounds, episodes, **kw):
        return real(out, paths, rounds, episodes, seconds=1.0, **{**kw, "workers": 1})

    return run


def test_a_master_config_must_name_a_teacher_per_mode(tmp_path) -> None:
    cfg = tmp_path / "master.toml"
    cfg.write_text('[master]\nlook = "policies/look-v9/policy_look.json"\n')
    with pytest.raises(ValueError, match="teacher per mode"):
        master.load_job(cfg)


def test_a_camera_height_ask_restarts_the_settling_second() -> None:
    hold = master.Hold()
    w = master.HOLD_WINDOW
    own = np.array([0.1, 0.1])
    for _ in range(w - 1):
        hold.legs_rad(_senses(0.0), "look", own)
    # Look is still moving to the new stance: the hold waits a full second past the ask.
    for _ in range(w - 1):
        hold.legs_rad(_senses(0.0, tilt=1.05), "look", own)
    assert not hold.holding
    hold.legs_rad(_senses(0.0, tilt=1.05), "look", own)
    assert hold.holding


def test_teacher_driven_look_settles_after_a_turn_and_the_episode_counts_it() -> None:
    r = master.Rollout(master.DEFAULT_TEACHERS)
    d = master.drive(r, None, 0, 0.0, 90.0, 20.0, nominal=True)
    assert not d["fell"] and d["drift_cm"] < 1.0
    ep = r.episode(0, 60.0, None, beta=1.0)
    assert 0 < ep["held"] < len(ep["obs"])
    assert r.episode(0, 60.0, None, beta=1.0)["held"] == ep["held"]


def test_a_master_config_can_pin_the_bodies_and_the_policy_keeps_them(tmp_path) -> None:
    job = master.load_job(master.Path("configs/master.toml"))
    assert job["body"]["leg_length_scale"] == [1.0, 1.0]
    r = master.Rollout(master.DEFAULT_TEACHERS, job["body"])
    assert r.env.cfg.randomization.leg_length_scale == (1.0, 1.0)
    assert master.Rollout(master.DEFAULT_TEACHERS).env.cfg.randomization.leg_length_scale != (
        1.0,
        1.0,
    )
    path = master.distill(
        tmp_path,
        master.DEFAULT_TEACHERS,
        rounds=(1.0,),
        episodes=1,
        seconds=1.0,
        epochs=1,
        workers=1,
        log=lambda _row: None,
        body=job["body"],
    )
    saved = json.loads(path.read_text())["training"]["master"]["body"]
    assert saved == job["body"]


@pytest.mark.parametrize("bad", ["ramp_to = [0, 1]", "nope = [0, 1]", "mass_scale = [2, 1]"])
def test_a_master_body_takes_only_randomization_ranges(tmp_path, bad: str) -> None:
    cfg = tmp_path / "master.toml"
    teachers = "\n".join(f'{m} = "{p}"' for m, p in master.DEFAULT_TEACHERS.items())
    cfg.write_text(f"[master]\n{teachers}\n[master.body]\n{bad}\n")
    with pytest.raises(ValueError, match="randomization range"):
        master.load_job(cfg)


def test_the_box_job_and_the_command_line_distil_the_same_master() -> None:
    job = master.load_job(master.Path("configs/master.toml"))
    assert {m: job[m] for m in master.MODES} == master.DEFAULT_TEACHERS
    assert job["body"] == master.DEFAULT_BODY
