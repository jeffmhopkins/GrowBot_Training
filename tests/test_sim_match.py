import math

import numpy as np
import pytest

from growbot_train import sim_match as sm


def test_the_series_reads_as_the_robot_page_plays_it() -> None:
    # The same points growbot-box.js's own test reads off simMatchAt (JBrain2 test_growbot_site).
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
