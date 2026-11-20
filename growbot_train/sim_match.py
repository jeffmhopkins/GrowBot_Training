"""Fit the sim's body to the robot's: replay the Body tab's Sim match (JBrain2
growbot-box.js gbSimMatch) and compare the sim's tilt with the robot's.

The owner's walks on 2026-11-19 tipped where the same leg commands, replayed here, never
tip the sim: it hops about 16 mm and lands back on its tail edge. A walk is too chaotic to fit
a body to, so the robot plays a fixed series of leg moves instead, from the walk's stance, in
the policy's action space at 25 Hz: slow sweeps and steps of both legs and of one, small
oscillations, then a ladder of full-stroke lifts, faster each time, meant to end in a tip. The
flight recorder keeps each tick's action and series time beside the phone's tilt; this replays
the same series on a sim body and scores how far its tilt is from his.

    python -m growbot_train.sim_match RUN.json                # nominal body
    python -m growbot_train.sim_match RUN.json --set floor=carpet --set foot_width=0.6

RUN.json is the debug API's GET /growbot/bodytests/{n} of the recorder run that holds it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

# SIM_MATCH_V "sm1" in growbot-box.js: [seconds from the start, a0, a1], linear between, two
# keys at one time a step. Change both together, under a new version.
VERSION = "sm1"
HZ = 25
_KEYS: list[tuple[float, float, float]] = [
    (0, 0, 0), (1, 0, 0),
    (2, 0.5, 0.5), (3, 0.5, 0.5), (4, 0, 0), (5, 0, 0),
    (6, -0.5, -0.5), (7, -0.5, -0.5), (8, 0, 0), (9, 0, 0),
    (9, 0.4, 0.4), (10, 0.4, 0.4), (10, 0, 0), (11.5, 0, 0),
    (11.5, -0.4, -0.4), (12.5, -0.4, -0.4), (12.5, 0, 0), (14, 0, 0),
    (14, 0.5, 0), (15, 0.5, 0), (15, 0, 0), (16, 0, 0),
    (16, 0, 0.5), (17, 0, 0.5), (17, 0, 0), (18, 0, 0),
    (18.25, 0.3, 0.3), (18.75, -0.3, -0.3), (19.25, 0.3, 0.3), (19.75, -0.3, -0.3),
    (20.25, 0.3, 0.3), (20.75, -0.3, -0.3), (21, 0, 0), (22, 0, 0),
    (22.1, 0.3, 0.3), (22.3, -0.3, -0.3), (22.5, 0.3, 0.3), (22.7, -0.3, -0.3),
    (22.9, 0.3, 0.3), (23.1, -0.3, -0.3), (23.3, 0.3, 0.3), (23.5, -0.3, -0.3), (23.6, 0, 0),
    (25, 0, 0),
]  # fmt: skip
LADDER = (3, 2, 1.5, 1, 0.7, 0.5, 0.35, 0.25, 0.15)


def keys() -> list[tuple[float, float, float]]:
    out = list(_KEYS)
    t = out[-1][0]
    for up in LADDER:
        out += [
            (t + up, 0.9, 0.9),
            (t + up + 0.5, 0.9, 0.9),
            (t + up + 2.5, 0, 0),
            (t + up + 4, 0, 0),
        ]
        t += up + 4
    return out


KEYS = keys()
TOTAL_S = KEYS[-1][0]


def at(t: float) -> tuple[float, float]:
    """The series' action at `t` seconds, as growbot-box.js simMatchAt reads it."""
    if t <= KEYS[0][0]:
        return KEYS[0][1], KEYS[0][2]
    i = 0
    while i < len(KEYS) - 1 and KEYS[i + 1][0] <= t:
        i += 1
    if i >= len(KEYS) - 1:
        return KEYS[i][1], KEYS[i][2]
    a, b = KEYS[i], KEYS[i + 1]
    f = (t - a[0]) / (b[0] - a[0])
    return a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f


def ladder_starts() -> list[float]:
    """When each of the ladder's lifts begins, seconds into the series."""
    t, out = _KEYS[-1][0], []
    for up in LADDER:
        out.append(t)
        t += up + 4
    return out


def robot_trace(run: dict[str, Any]) -> dict[str, np.ndarray]:
    """The robot's side of a recorder run: each sample during the series, by series time,
    with its action and tilt (the phone's beta, degrees). Raises if the run holds none, or
    one of another version."""
    log = run.get("log") or []
    starts = [r for r in log if r.get("k") == "e" and r.get("e") == "simmatch_start"]
    if not starts:
        raise ValueError("this run holds no Sim match")
    v = starts[-1].get("v")
    if v not in (None, VERSION):
        raise ValueError(f"the run played series {v!r}, this replays {VERSION!r}")
    t0 = starts[-1]["t"]
    rows = [
        r for r in log
        if r.get("k") == "s" and r["t"] >= t0 and r.get("mode") == "simmatch"
        and isinstance(r.get("st"), (int, float)) and r.get("ob") is not None
    ]  # fmt: skip
    if not rows:
        raise ValueError("the Sim match in this run has no samples")
    st = np.array([r["st"] for r in rows], float)
    return {
        "st": st,
        "act": np.array([r.get("act") or [math.nan, math.nan] for r in rows], float),
        "tilt": np.array([r["ob"] for r in rows], float),
        "up": np.array([r.get("up") is not False for r in rows]),
    }


def _pitch(env: Any) -> float:
    w, x, y, z = env.data.qpos[3:7]
    return -math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))


def sim_trace(
    env: Any, overrides: dict[str, Any], until_s: float = TOTAL_S
) -> dict[str, np.ndarray]:
    """The series on a sim body: settled at the stance for a second, as the robot's prep
    settles, then each 25 Hz tick's action. Pitch in degrees nose-up, height in metres."""
    opts = {"nominal": True, "overrides": overrides, "retarget": False, "speed_cmd": 1.0}
    env.reset(seed=0, options={**opts, "tilt_cmd": 1.0, "target_error_deg": 0.0})
    for _ in range(HZ):
        env.step(np.zeros(2))
    st, pitch, z = [], [], []
    for k in range(int(until_s * HZ) + 1):
        t = k / HZ
        a = np.clip(np.array(at(t)), -0.999, 0.999)
        _, _, term, _, _ = env.step(np.arctanh(a))
        st.append(t)
        pitch.append(_pitch(env))
        z.append(float(env.data.qpos[2]))
        if term:
            break
    return {"st": np.array(st), "pitch": np.array(pitch), "z": np.array(z)}


def compare(robot: dict[str, np.ndarray], sim: dict[str, np.ndarray]) -> dict[str, float]:
    """How far the sim's tilt is from his, each as a change from where it stood when the
    series began (so the phone's mount and the sim's frame drop out), over the time both
    ran; and when each first tipped 45 degrees past that."""
    end = min(robot["st"][-1], sim["st"][-1])
    keep = robot["st"] <= end
    real = robot["tilt"][keep] - robot["tilt"][0]
    mine = np.interp(robot["st"][keep], sim["st"], sim["pitch"] - sim["pitch"][0])

    def tipped(t: np.ndarray, d: np.ndarray) -> float:
        hit = np.nonzero(np.abs(d) > 45)[0]
        return float(t[hit[0]]) if len(hit) else math.nan

    return {
        "rms_deg": float(np.sqrt(np.mean((mine - real) ** 2))),
        "robot_tip_s": tipped(robot["st"], robot["tilt"] - robot["tilt"][0]),
        "sim_tip_s": tipped(sim["st"], sim["pitch"] - sim["pitch"][0]),
        "act_err": float(np.nanmax(np.abs(robot["act"] - np.array([at(t) for t in robot["st"]])))),
        "until_s": float(end),
    }


def _value(v: str) -> Any:
    try:
        return float(v)
    except ValueError:
        return v


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("run", type=Path, help="GET /growbot/bodytests/{n} of the recorder run")
    ap.add_argument(
        "--policy",
        default="policies/lift-fall-3/policy_lift_fall.json",
        help="a policy whose training config builds the sim (the body and floors)",
    )
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="a body override (env._draw's keys: floor, foot_width, mass_scale, ...)",
    )
    args = ap.parse_args(argv)
    from growbot_train.evaluate import env_for
    from growbot_train.policy import JsonPolicy

    robot = robot_trace(json.loads(args.run.read_text()))
    over = {}
    for kv in args.set:
        k, _, v = kv.partition("=")
        over[k] = _value(v)
    env = env_for(JsonPolicy(args.policy))
    sim = sim_trace(env, over, until_s=float(robot["st"][-1]))
    got = compare(robot, sim)
    print(json.dumps({"overrides": over, **{k: round(v, 3) for k, v in got.items()}}))
    for i, t in enumerate(ladder_starts()):
        print(f"ladder lift {i + 1} ({LADDER[i]} s) at {t:.2f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
