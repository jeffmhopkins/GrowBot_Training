"""The master distillation: each teacher must see exactly what it trained on, the mode
rule must follow the speed's sign, and the exported file must be one the app runner
accepts."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from growbot_train import master
from growbot_train.evaluate import env_for
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
        assert master.choose_mode(-0.2, err) == "back"
        assert master.choose_mode(0.0, err) == "look"
    assert master.choose_mode(1.0, 59.0) == "walk"
    assert master.choose_mode(1.0, -61.0) == "look"
    assert master.choose_mode(1.0, 170.0) == "look"


def test_the_walk_lets_go_only_past_its_hysteresis() -> None:
    assert master.choose_mode(1.0, 65.0, "walk") == "walk"
    assert master.choose_mode(1.0, 65.0, "look") == "look"
    assert master.choose_mode(1.0, 75.0, "walk") == "look"


def test_teacher_actions_are_matched_in_radians() -> None:
    a = np.array([0.5, -0.25])
    out = master.to_student(a, 1.0)
    assert np.allclose(out * master.ACTION_TO_RAD, a * 1.0)
    assert np.all(np.abs(master.to_student(np.array([1.0, -1.0]), 1.57)) <= master.SQUASH)


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
