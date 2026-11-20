import math

import numpy as np
import pytest

from growbot_train import sim_match as sm


def test_the_series_reads_as_the_robot_page_plays_it() -> None:
    # The same points JBrain2's test_growbot_site reads off growbot-box.js's simMatchAt; at 9
    # a step's two keys share a time and the later one holds.
    got = [sm.at(t) for t in (0, 1.5, 2, 9, 14.5, 28.5, 1e3)]
    assert got == [(0, 0), (0.25, 0.25), (0.5, 0.5), (0.4, 0.4), (0.5, 0), (0.9, 0.9), (0, 0)]
    assert sm.TOTAL_S == pytest.approx(70.45)
    assert sm.ladder_starts()[0] == 25 and len(sm.ladder_starts()) == len(sm.LADDER)


def _run(tilt: list[float], v: str = "sm1") -> dict:
    t0 = 1_000_000
    log = [{"k": "e", "e": "simmatch_start", "t": t0, "v": v}]
    for i, b in enumerate(tilt):
        st = i / sm.HZ
        log.append({"k": "s", "t": t0 + i * 40, "mode": "simmatch", "st": st,
                    "act": list(sm.at(st)), "ob": b})  # fmt: skip
    return {"log": log}


def test_a_recorder_run_gives_the_robots_trace() -> None:
    r = sm.robot_trace(_run([45, 46, 47]))
    assert r["st"].tolist() == [0, 0.04, 0.08] and r["tilt"].tolist() == [45, 46, 47]
    with pytest.raises(ValueError, match="no Sim match"):
        sm.robot_trace({"log": []})
    with pytest.raises(ValueError, match="sm9"):
        sm.robot_trace(_run([45], v="sm9"))


def test_a_sim_that_tilts_as_he_did_scores_zero() -> None:
    tilt = [45 + 10 * math.sin(i / 5) for i in range(100)]
    robot = sm.robot_trace(_run(tilt))
    # Its own frame: a different starting pitch, the same changes.
    sim = {"st": robot["st"], "pitch": np.array(tilt) - 3, "z": np.zeros(100)}
    got = sm.compare(robot, sim)
    assert got["rms_deg"] == pytest.approx(0, abs=1e-9)
    assert math.isnan(got["robot_tip_s"]) and math.isnan(got["sim_tip_s"])
    assert got["act_err"] == pytest.approx(0, abs=1e-9)


def test_the_trace_ends_at_the_sim_matchs_own_end_and_skips_other_moves() -> None:
    run = _run([45, 46, 47])
    log = run["log"]
    log.insert(0, {"k": "s", "t": log[0]["t"] - 40, "mode": "simmatch", "st": 9, "ob": 1})
    log.append({"k": "e", "e": "simmatch_end", "t": log[-1]["t"]})
    log.append({"k": "s", "t": log[-1]["t"] + 40, "mode": "simmatch", "st": 0.12, "ob": 9})
    log.append({"k": "s", "t": log[-1]["t"] + 40, "mode": "walk", "st": 0.16, "ob": 9})
    assert sm.robot_trace(run)["tilt"].tolist() == [45, 46, 47]


def test_a_wrong_logged_action_shows_but_a_tick_beside_a_step_does_not() -> None:
    robot = sm.robot_trace(_run([45] * 300))
    assert sm._act_err(robot) == 0
    robot["st"][-1], robot["act"][-1] = 9.0005, [0, 0]  # read just before the step at 9
    assert sm._act_err(robot) == 0
    robot["act"][10] = [0.1, -0.1]
    assert sm._act_err(robot) == pytest.approx(0.1)


def test_a_sim_that_never_overlaps_his_run_is_refused() -> None:
    robot = sm.robot_trace(_run([45, 46, 47]))
    sim = {"st": np.array([0.0]), "pitch": np.array([0.0]), "z": np.zeros(1)}
    with pytest.raises(ValueError, match="don't overlap"):
        sm.compare(robot, sim)


def test_the_series_plays_on_a_real_sim_body_with_his_pitch_sign() -> None:
    pytest.importorskip("mujoco")
    from growbot_train.evaluate import env_for
    from growbot_train.policy import JsonPolicy

    env = env_for(JsonPolicy("policies/lift-fall-3/policy_lift_fall.json"))
    got = sm.sim_trace(env, {}, until_s=3.0)
    assert len(got["st"]) == 3 * sm.HZ + 1
    # It settles nose-up on the stance, and the both-legs sweep to +0.5 moves it.
    assert got["pitch"][0] > 20
    assert np.ptp(got["pitch"]) > 1
