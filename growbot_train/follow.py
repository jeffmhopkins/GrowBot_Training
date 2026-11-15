"""The follow-me harness (docs/follow_me.md, roadmap phase 2): a scripted person walks
around the robot, a phone-like camera sees them, a tracker and follower turn that into
the policy's commands, and the whole loop runs on one trained skill.

Nothing reaches the policy except what the camera would see: the person's bearing,
height in frame and an eye-spacing distance, with jitter, a slowly drifting distance
bias, dropouts, frame latency and a field of view. The person's true position is only
used to score the run.

This first version runs one skill at a time, with the follower limited to that skill's
mode (the single-skill baseline the master policy has to beat): look turns and tilts
only, the nose walk backs away, and a walk with no speed input always walks.

    python -m growbot_train.follow policies/look-v6/policy_look.json --seconds 120
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from growbot_train.env import CONTROL_DT, GrowBotHeadingEnv, wrap_deg
from growbot_train.evaluate import env_for
from growbot_train.policy import JsonPolicy

# Where the phone's camera sits above the torso origin, metres along the lens axis.
LENS_OFFSET_M = 0.03


@dataclass
class CameraSpec:
    """A phone's front camera as the tracker sees through it. Field of view is the
    full angle; the phone lies screen up, so "horizontal" is across the robot."""

    hfov_deg: float = 60.0
    vfov_deg: float = 75.0
    latency_s: tuple[float, float] = (0.10, 0.20)
    jitter_deg: float = 1.0
    distance_bias: tuple[float, float] = (-0.15, 0.15)
    distance_noise: float = 0.05
    # Chance per frame of missing a face that is in view, at 1 m and growing with
    # distance (a smaller face), so far faces drop out more.
    dropout_at_1m: float = 0.05
    dropout_per_m: float = 0.05


@dataclass
class Person:
    """A person who walks to random spots around the robot, pauses, and sits or stands.

    Waypoints are drawn in a ring around the robot's start so they stay in a room-sized
    space; walking speed and pauses are drawn per leg."""

    seed: int
    ring_m: tuple[float, float] = (0.7, 2.5)
    speed_m_s: tuple[float, float] = (0.3, 0.7)
    pause_s: tuple[float, float] = (1.0, 6.0)
    face_height_m: tuple[float, float] = (1.1, 1.65)
    pos: np.ndarray = field(init=False)
    face_z: float = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.pos = self._spot()
        self.face_z = self.face_height_m[1]
        self._goal = self.pos.copy()
        self._speed = 0.0
        self._wait = float(self.rng.uniform(*self.pause_s))

    def _spot(self) -> np.ndarray:
        r = self.rng.uniform(*self.ring_m)
        a = self.rng.uniform(-math.pi, math.pi)
        return np.array([r * math.cos(a), r * math.sin(a)])

    def step(self, dt: float) -> None:
        if self._wait > 0:
            self._wait -= dt
            if self._wait <= 0:
                self._goal = self._spot()
                self._speed = float(self.rng.uniform(*self.speed_m_s))
                # Sit about one leg in four: the face drops and the camera has to follow.
                lo, hi = self.face_height_m
                self.face_z = lo if self.rng.random() < 0.25 else hi
            return
        to_goal = self._goal - self.pos
        dist = float(np.linalg.norm(to_goal))
        stride = self._speed * dt
        if dist <= stride:
            self.pos = self._goal.copy()
            self._wait = float(self.rng.uniform(*self.pause_s))
        else:
            self.pos = self.pos + to_goal / dist * stride


@dataclass
class Sighting:
    """One camera frame's reading of the person: what the phone's tracker reports."""

    bearing_deg: float  # left of the lens axis, CCW seen from above
    pitch_deg: float  # above the lens axis
    distance_m: float


def lens_pose(env: GrowBotHeadingEnv) -> tuple[np.ndarray, np.ndarray]:
    """The camera's position and unit viewing direction in the world."""
    axis = env.lens_dir()
    pos = env.data.xpos[env._torso].copy() + LENS_OFFSET_M * axis
    return pos, axis


def true_sighting(env: GrowBotHeadingEnv, person: Person) -> Sighting:
    """Where the person's face really is from the lens, before any camera error.

    The bearing is taken against the heading the policy is scored on (env.yaw_deg(), the
    lens's compass direction), so a bearing of 0 is the policy's target; the pitch is the
    face's elevation against the lens axis's elevation."""
    cam, axis = lens_pose(env)
    face = np.array([person.pos[0], person.pos[1], person.face_z])
    v = face - cam
    flat = math.hypot(v[0], v[1])
    bearing = wrap_deg(math.degrees(math.atan2(v[1], v[0])) - env.yaw_deg())
    face_el = math.degrees(math.atan2(v[2], flat))
    axis_el = math.degrees(math.asin(max(-1.0, min(1.0, float(axis[2])))))
    return Sighting(bearing, face_el - axis_el, float(np.linalg.norm(v)))


class Camera:
    """The phone camera between the world and the tracker: field of view, dropouts,
    jitter, a distance bias that drifts, and frame latency."""

    def __init__(self, spec: CameraSpec, rng: np.random.Generator) -> None:
        self.spec = spec
        self.rng = rng
        self.bias = float(rng.uniform(*spec.distance_bias))
        lag = float(rng.uniform(*spec.latency_s))
        self._queue: deque[Sighting | None] = deque(
            [None] * max(1, round(lag / CONTROL_DT)), maxlen=max(1, round(lag / CONTROL_DT))
        )

    def in_view(self, s: Sighting) -> bool:
        return abs(s.bearing_deg) <= self.spec.hfov_deg / 2 and abs(s.pitch_deg) <= (
            self.spec.vfov_deg / 2
        )

    def frame(self, truth: Sighting) -> Sighting | None:
        """This tick's reading, which the tracker gets latency_s later (None: nothing)."""
        sp = self.spec
        # The eye-spacing assumption is off by a bias that wanders slowly.
        lo, hi = sp.distance_bias
        self.bias = float(np.clip(self.bias + self.rng.normal(0, 0.002), lo, hi))
        seen: Sighting | None = None
        if self.in_view(truth):
            miss = sp.dropout_at_1m + sp.dropout_per_m * max(0.0, truth.distance_m - 1.0)
            if self.rng.random() >= miss:
                seen = Sighting(
                    truth.bearing_deg + self.rng.normal(0, sp.jitter_deg),
                    truth.pitch_deg + self.rng.normal(0, sp.jitter_deg),
                    truth.distance_m * (1 + self.bias + self.rng.normal(0, sp.distance_noise)),
                )
        late = self._queue[0]
        self._queue.append(seen)
        return late


@dataclass
class Tracker:
    """Smooths the camera's readings and remembers where the person was last seen."""

    smooth: float = 0.5
    lost_after_s: float = 0.5
    last: Sighting | None = None
    age_s: float = math.inf

    def update(self, seen: Sighting | None, turned_deg: float) -> Sighting | None:
        """Fold in this tick's reading. turned_deg is how far the robot turned since the
        last tick (from its gyro), so a remembered bearing stays pointed at the same spot."""
        self.age_s += CONTROL_DT
        if self.last is not None:
            self.last = Sighting(
                wrap_deg(self.last.bearing_deg - turned_deg),
                self.last.pitch_deg,
                self.last.distance_m,
            )
        if seen is not None:
            if self.last is None or self.age_s > self.lost_after_s:
                self.last = seen
            else:
                k = self.smooth
                self.last = Sighting(
                    wrap_deg(
                        self.last.bearing_deg
                        + k * wrap_deg(seen.bearing_deg - self.last.bearing_deg)
                    ),
                    self.last.pitch_deg + k * (seen.pitch_deg - self.last.pitch_deg),
                    self.last.distance_m + k * (seen.distance_m - self.last.distance_m),
                )
            self.age_s = 0.0
        return self.last

    @property
    def lost(self) -> bool:
        return self.age_s > self.lost_after_s


@dataclass
class Follower:
    """Turns the tracked person into the three commands (docs/follow_me.md, Follower)."""

    set_distance_m: float = 1.0
    dead_band_m: float = 0.15
    back_ramp_m: float = 0.5
    walk_within_deg: float = 30.0
    # Camera-height command per degree of pitch error per tick. Taller (+1) tips the
    # lens toward the horizon, so a face above the lens axis needs it lower.
    pitch_gain: float = 0.02
    search_deg: float = 90.0  # the heading error sent while searching: a steady spin
    near_m: float = 0.5
    far_m: float = 2.5
    step_m: float = 0.25
    stay: bool = False
    tilt: float = 1.0

    def voice(self, said: str) -> None:
        if said == "follow me":
            self.stay = False
        elif said == "come":
            self.stay = False
            self.set_distance_m = self.near_m
        elif said == "come closer":
            self.set_distance_m = max(self.near_m, self.set_distance_m - self.step_m)
        elif said == "go further back":
            self.set_distance_m = min(self.far_m, self.set_distance_m + self.step_m)
        elif said == "stay":
            self.stay = True
        else:
            raise ValueError(f"unknown voice command {said!r}")

    def command(self, tracked: Sighting | None, lost: bool) -> tuple[float, float, float]:
        """(heading error deg, signed speed -1..1, camera height -1..1). Lost or never
        seen: speed 0, and spin toward where they were last seen."""
        if tracked is None:
            # Never seen: spin on the spot until they come into view.
            return self.search_deg, 0.0, self.tilt
        bearing = tracked.bearing_deg
        if lost:
            # Turn toward where they were last seen, at least a search spin's worth, so
            # a person who left the frame on the left is looked for on the left.
            return math.copysign(max(abs(bearing), self.search_deg), bearing), 0.0, self.tilt
        self.tilt = float(np.clip(self.tilt - self.pitch_gain * tracked.pitch_deg, -1.0, 1.0))
        speed = 0.0
        err = tracked.distance_m - self.set_distance_m
        if not self.stay:
            if err > self.dead_band_m and abs(bearing) <= self.walk_within_deg:
                speed = 1.0
            elif err < -self.dead_band_m:
                speed = -min(1.0, (-err - self.dead_band_m) / self.back_ramp_m)
        return bearing, speed, self.tilt


DEFAULT_SCRIPT: tuple[tuple[float, str], ...] = (
    (0.0, "follow me"),
    (30.0, "come closer"),
    (50.0, "go further back"),
    (70.0, "stay"),
    (85.0, "follow me"),
    (100.0, "come"),
)


def _mode_speed(policy_speed: str, speed: float) -> tuple[float | None, bool]:
    """The speed the skill can take, and whether it can obey this command at all: look
    only stands, the nose walk only backs (its input runs 0..1 nose first), a walk with a
    speed input only goes forward, and a walk at its cap ignores speed altogether."""
    if policy_speed == "still":
        return None, speed == 0.0
    if policy_speed == "cap":
        return None, speed > 0.0
    return speed, True


def run(
    policy: JsonPolicy,
    seed: int,
    seconds: float = 120.0,
    script: tuple[tuple[float, str], ...] = DEFAULT_SCRIPT,
    camera: CameraSpec | None = None,
    nominal: bool = False,
) -> dict[str, Any]:
    """One follow run on one skill. Returns its summary and a per-tick trace."""
    env = env_for(policy)
    nose_first = env.cfg.task.facing == "nose"
    backs = nose_first and env.cfg.task.speed == "command"
    opts: dict[str, Any] = {"retarget": False, "speed_cmd": 0.0, "tilt_cmd": 1.0}
    if nominal:
        opts["nominal"] = True
    obs, _ = env.reset(seed=seed, options=opts)
    env.max_steps = round(seconds / CONTROL_DT) + 1
    rng = np.random.default_rng(seed + 7919)
    person = Person(seed)
    cam = Camera(camera or CameraSpec(), rng)
    tracker = Tracker()
    follower = Follower()
    pending = sorted(script)
    trace: list[dict[str, float]] = []
    yaw_prev = env.yaw_deg()
    fell = False
    for i in range(round(seconds / CONTROL_DT)):
        t = i * CONTROL_DT
        while pending and pending[0][0] <= t:
            follower.voice(pending.pop(0)[1])
        person.step(CONTROL_DT)
        truth = true_sighting(env, person)
        yaw = env.yaw_deg()
        tracked = tracker.update(cam.frame(truth), wrap_deg(yaw - yaw_prev))
        yaw_prev = yaw
        bearing, speed, tilt = follower.command(tracked, tracker.lost)
        # The single-skill baseline: a command this skill can't carry out becomes the
        # nearest one it can (the nose walk stands for anything but backing, look for
        # anything but standing).
        if backs:
            fwd = max(0.0, -speed)
        else:
            fwd, _ = _mode_speed(env.cfg.task.speed, speed)
        env.target_deg = wrap_deg(env.yaw_deg() + bearing)
        env.heading_bias = 0.0
        if env.cfg.task.speed == "command":
            env.speed_cmd = env._fixed_speed = 0.0 if fwd is None else float(fwd)
        if env.cfg.task.tilt_command:
            env.tilt_cmd = env._fixed_tilt = tilt
            env._set_stance()
        obs, _, term, _, info = env.step(policy.env_action(obs))
        trace.append(
            {
                "t": t,
                "bearing_true": truth.bearing_deg,
                "pitch_true": truth.pitch_deg,
                "distance": truth.distance_m,
                "set_distance": follower.set_distance_m,
                "in_view": float(cam.in_view(truth)),
                "lost": float(tracker.lost),
                "cmd_yaw": bearing / 180.0,
                "cmd_speed": speed,
                "cmd_tilt": tilt,
                "stay": float(follower.stay),
                "robot_speed": info["speed"],
            }
        )
        if term:
            fell = True
            break
    return {"summary": summarise(trace, fell, seconds), "trace": trace}


def summarise(trace: list[dict[str, float]], fell: bool, seconds: float) -> dict[str, float]:
    if not trace:
        return {"fell": float(fell)}
    arr = {k: np.array([r[k] for r in trace]) for k in trace[0]}
    stay = arr["stay"] > 0
    return {
        "seconds": len(trace) * CONTROL_DT,
        "fell": float(fell),
        "in_view_pct": 100.0 * float(arr["in_view"].mean()),
        "lost_pct": 100.0 * float(arr["lost"].mean()),
        "aim_err_deg": float(np.abs(arr["bearing_true"]).mean()),
        "pitch_err_deg": float(np.abs(arr["pitch_true"]).mean()),
        "distance_err_m": float(np.abs(arr["distance"] - arr["set_distance"]).mean()),
        "stay_speed_cm_s": 100.0 * float(arr["robot_speed"][stay].mean()) if stay.any() else 0.0,
        "completed": float(len(trace) * CONTROL_DT >= seconds - CONTROL_DT),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the follow-me harness on one skill")
    ap.add_argument("policy", help="policy JSON (export.py output)")
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--seeds", type=int, default=4, help="runs, each a new person and body")
    ap.add_argument("--nominal", action="store_true", help="the nominal body, no noise")
    ap.add_argument("--json", help="write the summaries (and traces) here")
    args = ap.parse_args(argv)
    policy = JsonPolicy(args.policy)
    runs = [run(policy, s, args.seconds, nominal=args.nominal) for s in range(args.seeds)]
    keys = [k for k in runs[0]["summary"] if k != "completed"]
    print("seed  " + "  ".join(f"{k:>14}" for k in keys))
    for s, r in enumerate(runs):
        print(f"{s:>4}  " + "  ".join(f"{r['summary'].get(k, float('nan')):>14.2f}" for k in keys))
    mean = {k: float(np.mean([r["summary"][k] for r in runs])) for k in keys}
    print("mean  " + "  ".join(f"{mean[k]:>14.2f}" for k in keys))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"mean": mean, "runs": runs}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
