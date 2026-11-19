"""Every kept walk, run the way the robot runs it, turns the way it is asked and walks the
end it should lead with. The JBrain2 page's own runner (growbot_runner.js) drives the sim
body wired as the owner's robot is; around it, this file restates the page's side of
growbot-box.js start() (walkerFor's leg swap, bodyAccel/bodyGyro, the stance levelling and
the gyro heading), so a change there must be copied here: only the runner is the page's
own code. simbody's phone readings are bodyAccel/bodyGyro's inverse by construction, so the
axes rest on the facts below, not on this test. Needs node and a JBrain2 checkout: set
GROWBOT_SITE to its growbot/site.

The robot's facts this rests on (owner, 2026-11-19, and body test #40):
- the wire's `l` is Dave's own left leg, seen from the camera end he faces out of: the
  model's `right_leg` (-y), which the sim names from the nose;
- the stock calibration (L_SIGN -1, R_SIGN +1) sends a positive joint, the foot toward the
  camera end, as the Body tab's "Sweep forward" checks;
- the phone lies screen up, its top toward the nose: the camera's picture is upright for a
  face in front of the camera end, which the face tracker finds.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest

from growbot_train import simbody as sb
from growbot_train.env import CONTROL_DT

SITE = os.environ.get("GROWBOT_SITE")
pytestmark = pytest.mark.skipif(
    not SITE or not shutil.which("node"), reason="set GROWBOT_SITE and install node"
)
ROOT = Path(__file__).resolve().parents[1]
DRIVER = Path(__file__).parent / "js" / "page_walk.mjs"
ROBOT = sb.Wiring(swap=True, l_sign=1, r_sign=-1)
STOCK_CAL = {"L_SIGN": -1, "R_SIGN": 1, "L_OFF": 0, "R_OFF": 0}
SECS = 12.0

# What each job is asked, and the command a skill set gives it (growbot-drive.js inputs,
# growbot-box.js setCommand): a turn by degrees stands at camera height 0 (TURN_PITCH), and a
# nose walk's +1 backs away.
LEFT = ("left", {"target": 90.0, "fwd": 0.0, "pitch": 0.0})
RIGHT = ("right", {"target": -90.0, "fwd": 0.0, "pitch": 0.0})
FORWARD = ("forward", {"target": 0.0, "fwd": 1.0, "pitch": 1.0})
NOSE_BACK = ("back", {"target": 0.0, "fwd": 1.0, "pitch": 1.0})
BACK = ("back", {"target": 0.0, "fwd": -1.0, "pitch": 1.0})
MOVES = {
    "growbot_look": (LEFT, RIGHT),
    "growbot_lift_fall": (FORWARD,),
    "growbot_lift_fall_turns": (FORWARD,),
    "growbot_nose_walk": (NOSE_BACK,),
    "growbot_nose_walk_smooth": (NOSE_BACK,),
    "growbot_master": (LEFT, RIGHT, FORWARD, BACK),
}


def _kept() -> list[tuple[str, str, str, dict[str, float]]]:
    index = tomllib.loads((ROOT / "policies" / "index.toml").read_text())
    # A job this file doesn't know yet fails its own case rather than the whole collection.
    unknown = (("unknown", {}),)
    return [
        (p["slug"], p["file"], name, cmd)
        for p in index["policy"]
        for name, cmd in MOVES.get(p["job"], unknown)
    ]


def _levelling(u: np.ndarray) -> np.ndarray:
    """growbot-box.js levelling: the smallest rotation taking u onto +z."""
    a, b, c = u / np.linalg.norm(u)
    v = np.array([[0.0, 0.0, -a], [0.0, 0.0, -b], [a, b, 0.0]])
    return np.eye(3) + v + v @ v / (1.0 + c)


def _body(m: dict) -> tuple[np.ndarray, np.ndarray]:
    """growbot-box.js bodyAccel and bodyGyro (rad/s)."""
    a, r = m["accel"], m["rate"]
    accel = np.array([a["y"], -a["x"], a["z"]])
    gyro = np.radians([r["beta"], -r["alpha"], r["gamma"]])
    return accel, gyro


def _xyz(v: np.ndarray) -> dict[str, float]:
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _wrap(d: float) -> float:
    return (d + 180.0) % 360.0 - 180.0


def _walk(policy: Path, cmd: dict[str, float], swap: bool = True) -> dict[str, float]:
    """One trained walk as start() runs it: legs neutral 1 s, level on the stance's
    gravity, then tick the Walker. Returns the turn (degrees, + counter-clockwise) and the
    travel toward the camera end (cm), by the sim's own pose."""
    sim = sb.SimBody(wiring=ROBOT)
    sim.person.present = False
    sim.reset("standing")
    js = subprocess.Popen(
        ["node", str(DRIVER), str(Path(SITE) / "growbot_runner.js"), str(policy)]
        + [json.dumps(STOCK_CAL)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert js.stdin and js.stdout
    if not swap:
        js.stdin.write(json.dumps({"swap": False}) + "\n")
    grav = []
    for i in range(round(1.0 / CONTROL_DT)):
        sim.servos(90, 90)
        sim.tick()
        if i * CONTROL_DT >= 0.6:
            grav.append(_body(sim.motion())[0])
    rot = _levelling(np.mean(grav, axis=0))
    s0 = sim.state()
    heading = 0.0
    try:
        for _ in range(round(SECS / CONTROL_DT)):
            accel, gyro = (rot @ v for v in _body(sim.motion()))
            heading += math.degrees(gyro[2]) * CONTROL_DT
            yaw = _wrap(cmd["target"] - heading) / 180.0
            fed = {"fwd": cmd["fwd"], "yaw": yaw, "pitch": cmd["pitch"]}
            line = {"A": _xyz(accel), "G": _xyz(gyro), "cmd": fed}
            js.stdin.write(json.dumps(line) + "\n")
            js.stdin.flush()
            reply = js.stdout.readline()
            if not reply:
                js.kill()
                raise AssertionError(f"the runner stopped: {js.stderr.read() if js.stderr else ''}")
            out = json.loads(reply)
            sim.servos(out["l"], out["r"])
            sim.tick()
    finally:
        if js.poll() is None:
            js.stdin.close()
        js.wait()
    s = sim.state()
    nose = math.radians(s0["nose_deg"])
    ahead = (s["x"] - s0["x"]) * math.cos(nose) + (s["y"] - s0["y"]) * math.sin(nose)
    return {"turned": _wrap(s["nose_deg"] - s0["nose_deg"]), "camera_cm": -100.0 * ahead}


@pytest.mark.parametrize(("slug", "file", "move", "cmd"), _kept(), ids=lambda v: str(v))
def test_kept_walk_goes_the_way_it_is_asked(
    slug: str, file: str, move: str, cmd: dict[str, float]
) -> None:
    assert move != "unknown", f"{slug}: add its job to MOVES"
    got = _walk(ROOT / file, cmd)
    if move == "left":
        assert got["turned"] > 30.0, got  # counter-clockwise from above: Dave's own left
    elif move == "right":
        assert got["turned"] < -30.0, got
    elif move == "forward":
        assert got["camera_cm"] > 30.0 and abs(got["turned"]) < 30.0, got
    else:
        # Away from what the camera sees, and clearly more than the same file standing
        # still creeps: a stand can drift a few centimetres either way.
        still = _walk(ROOT / file, {**cmd, "fwd": 0.0})
        assert got["camera_cm"] < still["camera_cm"] - 5.0, (got, still)


def test_without_the_page_swap_a_look_left_turns_right() -> None:
    """The same run with the runner's own a0=right mapping, unswapped: the leg mapping is
    what decides the turn's direction, so the checks above tell the two apart."""
    got = _walk(ROOT / "policies" / "look-v9" / "policy_look.json", LEFT[1], swap=False)
    assert got["turned"] < -30.0, got
