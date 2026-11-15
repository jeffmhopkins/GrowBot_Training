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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from growbot_train.env import (
    ACTION_LIMIT,
    CONTROL_DT,
    LENS_OFFSET_M,
    GrowBotHeadingEnv,
    wrap_deg,
)
from growbot_train.evaluate import env_for
from growbot_train.policy import JsonPolicy


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
    # Where the face falls in the image, degrees left of and above its centre; only the
    # field-of-view test uses these. The tracker reports compass bearing and pitch, as the
    # app will after turning image angles into them with the phone's own attitude.
    image_x_deg: float = 0.0
    image_y_deg: float = 0.0
    in_front: bool = True


def lens_pose(env: GrowBotHeadingEnv) -> tuple[np.ndarray, np.ndarray]:
    """The camera's position and unit viewing direction in the world."""
    axis = env.lens_dir()
    pos = env.data.xpos[env._torso].copy() + LENS_OFFSET_M * axis
    return pos, axis


def camera_heading_deg(env: GrowBotHeadingEnv) -> float:
    """The compass direction the camera looks: the policy's heading for a camera-first
    skill, and half a turn from it for a nose-first one, whose heading is its nose's."""
    yaw = env.yaw_deg()
    return yaw if env.cfg.task.facing == "camera" else wrap_deg(yaw + 180.0)


def lens_elevation_deg(env: GrowBotHeadingEnv) -> float:
    axis = env.lens_dir()
    return math.degrees(math.asin(max(-1.0, min(1.0, float(axis[2])))))


def true_sighting(env: GrowBotHeadingEnv, person: Person) -> Sighting:
    """Where the person's face really is from the lens, before any camera error.

    The bearing is taken against the camera's compass heading, so 0 is dead ahead in the
    image; the pitch is the face's elevation against the lens axis's; the distance is the
    straight line from lens to face, which is what eye spacing in the image measures."""
    cam, axis = lens_pose(env)
    face = np.array([person.pos[0], person.pos[1], person.face_z])
    v = face - cam
    flat = math.hypot(v[0], v[1])
    bearing = wrap_deg(math.degrees(math.atan2(v[1], v[0])) - camera_heading_deg(env))
    face_el = math.degrees(math.atan2(v[2], flat))
    # The image axes: left is level and square to the lens axis, up completes the frame.
    # With the lens tipped up, a compass offset shrinks in the image by about cos(el).
    left = np.array([-axis[1], axis[0], 0.0])
    left /= max(float(np.linalg.norm(left)), 1e-9)
    up = np.cross(axis, left)
    ahead = float(v @ axis)
    return Sighting(
        bearing,
        face_el - lens_elevation_deg(env),
        float(np.linalg.norm(v)),
        math.degrees(math.atan2(float(v @ left), ahead)),
        math.degrees(math.atan2(float(v @ up), ahead)),
        ahead > 0.0,
    )


def floor_from_camera(env: GrowBotHeadingEnv, s: Sighting) -> Sighting:
    """The tracker's reading with its distance put on the floor. Eye spacing gives the
    straight line to the face; the phone knows its own pitch, so the floor distance
    follows from the face's elevation."""
    el = math.radians(lens_elevation_deg(env) + s.pitch_deg)
    return Sighting(s.bearing_deg, s.pitch_deg, s.distance_m * math.cos(el))


def skill_target_deg(env: GrowBotHeadingEnv, bearing_deg: float) -> float:
    """The heading to give the skill for a person bearing_deg off the camera's axis. For
    a nose-first skill yaw_deg() is the nose's, so this is the person's direction turned
    half a turn: the nose points away from them and the camera at them."""
    return wrap_deg(env.yaw_deg() + bearing_deg)


def floor_distance_m(env: GrowBotHeadingEnv, person: Person) -> float:
    """How far the person stands from the robot along the floor: what the set distance
    means, and what the run is scored on."""
    cam, _ = lens_pose(env)
    return float(np.hypot(*(person.pos - cam[:2])))


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
        return (
            s.in_front
            and abs(s.image_x_deg) <= self.spec.hfov_deg / 2
            and abs(s.image_y_deg) <= self.spec.vfov_deg / 2
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
    # Start walking up past this, then keep walking until within walk_stop_m: the spin's
    # own drift (10-25 cm from a half turn) is taken back in one walk, not left in the band.
    dead_band_m: float = 0.10
    walk_stop_m: float = 0.05
    # The camera's distance swings about 0.3 m with each waddle's pitch, so the follower
    # acts on its average over about this long; acting on each frame chattered between
    # walking and standing and never closed the gap (Jeff, 2026-11-15: "zero creep").
    distance_avg_s: float = 1.5
    # Too close backs away past this, as before the walk got its own bands.
    back_band_m: float = 0.15
    distance: float | None = None
    frames: int = 0  # fresh frames in the average; a walk waits for a second's worth
    walking: bool = False
    back_ramp_m: float = 0.5
    walk_within_deg: float = 30.0
    # Camera-height command per degree of pitch error per tick. Taller (+1) tips the
    # lens toward the horizon, so a face above the lens axis needs it lower.
    # About 0.12 a second per degree: slow enough not to ring against the camera's
    # latency and the stance's own lag.
    pitch_gain: float = 0.005
    # Look turns barely at all with the camera low (Master v2 and look-v9: 11-45 degrees of a
    # 90-degree turn in 10 s at -1, 80-88 at 0), so a turn or a search is sent with the
    # camera no lower than this; the height asked for comes back once it faces them.
    turn_tilt_min: float = 0.0
    turn_tilt_deg: float = 20.0
    search_deg: float = 90.0  # the heading error sent while searching: a steady spin
    search_sign: float = 1.0
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

    def command(
        self, tracked: Sighting | None, lost: bool, fresh: bool = True
    ) -> tuple[float, float, float]:
        """(heading error deg, signed speed -1..1, camera height -1..1). Lost or never
        seen: speed 0, and spin toward the side they were last seen on. tracked.distance_m
        is along the floor (see floor_from_camera); fresh says this tick brought a new
        frame, the only time the camera height moves, so a held pitch isn't counted twice."""
        if tracked is None or lost:
            # A distance from before they were lost is stale by the time they're found.
            self.distance, self.frames, self.walking = None, 0, False
        turning_tilt = max(self.tilt, self.turn_tilt_min)
        if tracked is None:
            # Never seen: spin on the spot until they come into view.
            return self.search_sign * self.search_deg, 0.0, turning_tilt
        bearing = tracked.bearing_deg
        if lost:
            # Spin toward the side of the image they were last on, and keep spinning that
            # way: chasing the remembered bearing would stall facing the spot they left.
            return self.search_sign * self.search_deg, 0.0, turning_tilt
        self.search_sign = 1.0 if bearing >= 0 else -1.0
        if fresh:
            self.frames += 1
            # A plain mean until it holds distance_avg_s of frames, then a moving one: an
            # early long frame weighs no more than any other.
            a = max(1.0 / self.frames, CONTROL_DT / self.distance_avg_s)
            d = tracked.distance_m
            self.distance = d if self.distance is None else self.distance + a * (d - self.distance)
        speed = 0.0
        err = (tracked.distance_m if self.distance is None else self.distance) - self.set_distance_m
        # As the master's walk does, a walk lets go of the heading only 10 degrees wider.
        within = self.walk_within_deg + (10.0 if self.walking else 0.0)
        # One frame can read 0.3 m long mid-waddle, so a walk waits until the average
        # holds about a second of frames.
        settled = self.frames * CONTROL_DT >= 1.0
        self.walking = (
            not self.stay
            and (self.walking or settled)
            and err > (self.walk_stop_m if self.walking else self.dead_band_m)
            and abs(bearing) <= within
        )
        if not self.stay:
            if self.walking:
                speed = 1.0
            elif err < -self.back_band_m:
                speed = -min(1.0, (-err - self.back_band_m) / self.back_ramp_m)
        # Only look takes a camera height, so it is steered only while standing and roughly
        # facing them: a walk's own stance pitches the lens 20-30 degrees, which read as the
        # face far above and drove the height to its limit in a second.
        if fresh and speed == 0.0 and abs(bearing) <= self.turn_tilt_deg:
            self.tilt = float(np.clip(self.tilt - self.pitch_gain * tracked.pitch_deg, -1.0, 1.0))
        if abs(bearing) > self.turn_tilt_deg:
            return bearing, speed, turning_tilt
        return bearing, speed, self.tilt


DEFAULT_SCRIPT: tuple[tuple[float, str], ...] = (
    (0.0, "follow me"),
    (30.0, "come closer"),
    (50.0, "go further back"),
    (70.0, "stay"),
    (85.0, "follow me"),
    (100.0, "come"),
)


@dataclass
class Hold:
    """Stops calling the policy once the camera is on the person and the robot is asked to
    stand: the servos hold the average of the last second of commands, so the legs stop
    fiddling and the robot stops creeping (Jeff, 2026-11-15). Averaging over about one
    gait cycle lands on the stance the policy was holding, where the last single command
    would freeze a leg mid-swing. It lets go when the person moves past exit_deg, when the
    robot is asked to walk, when the camera height is asked to move, or when the person
    is lost (the search spin needs the policy). The gap between enter and exit keeps it
    from flicking in and out."""

    enter_deg: float = 6.0
    exit_deg: float = 15.0
    settle_s: float = 1.0
    tilt_band: float = 0.15
    window: int = round(1.0 / CONTROL_DT)
    held: np.ndarray | None = None
    _tilt: float = 0.0
    _on_s: float = 0.0
    _recent: deque[np.ndarray] = field(default_factory=deque)

    def action(
        self,
        policy_action: Callable[[], np.ndarray],
        bearing_deg: float,
        standing: bool,
        lost: bool,
        tilt: float,
    ) -> np.ndarray:
        """This tick's action: the held pose, or the policy's."""
        if self.held is not None and (
            lost
            or not standing
            or abs(bearing_deg) > self.exit_deg
            or abs(tilt - self._tilt) > self.tilt_band
        ):
            self.held = None
            self._on_s = 0.0
            self._recent.clear()
        if self.held is not None:
            return self.held
        a = policy_action()
        self._recent.append(a)
        while len(self._recent) > self.window:
            self._recent.popleft()
        on = standing and not lost and abs(bearing_deg) < self.enter_deg
        self._on_s = self._on_s + CONTROL_DT if on else 0.0
        if self._on_s >= self.settle_s and len(self._recent) == self.window:
            # Averaged as joint targets (after the env's clip and tanh), then turned back
            # into the env's input, so a briefly saturated command can't skew the pose.
            servo = np.tanh(np.clip(np.asarray(self._recent), -ACTION_LIMIT, ACTION_LIMIT))
            self.held = np.arctanh(np.clip(servo.mean(axis=0), -0.999999, 0.999999))
            self._tilt = tilt
        return a


def run(
    policy: JsonPolicy,
    seed: int,
    seconds: float = 120.0,
    script: tuple[tuple[float, str], ...] = DEFAULT_SCRIPT,
    camera: CameraSpec | None = None,
    nominal: bool = False,
    hold: bool = True,
) -> dict[str, Any]:
    """One follow run on one skill. Returns its summary and a per-tick trace. hold puts
    the robot in Hold while it stands on target; False leaves the policy running."""
    env = env_for(policy)
    nose_first = env.cfg.task.facing == "nose"
    backs = nose_first and env.cfg.task.speed == "command"
    opts: dict[str, Any] = {
        "retarget": False,
        "speed_cmd": 0.0,
        "tilt_cmd": 1.0,
        "target_error_deg": 0.0,
    }
    if nominal:
        opts["nominal"] = True
    obs, _ = env.reset(seed=seed, options=opts)
    env.max_steps = round(seconds / CONTROL_DT) + 1
    rng = np.random.default_rng(seed + 7919)
    person = Person(seed)
    cam = Camera(camera or CameraSpec(), rng)
    tracker = Tracker()
    follower = Follower()
    holder = Hold() if hold else None
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
        seen = cam.frame(truth)
        tracked = tracker.update(seen, wrap_deg(yaw - yaw_prev))
        yaw_prev = yaw
        if tracked is not None:
            tracked = floor_from_camera(env, tracked)
        bearing, speed, tilt = follower.command(tracked, tracker.lost, seen is not None)
        if not env.cfg.task.tilt_command:
            follower.tilt = tilt = 1.0
        # The single-skill baseline: a command this skill can't carry out becomes the
        # nearest one it can (the nose walk stands for anything but backing, look for
        # anything but standing).
        fwd = max(0.0, -speed) if backs else max(0.0, speed)
        env.target_deg = skill_target_deg(env, bearing)
        env.heading_bias = 0.0
        if env.cfg.task.speed == "command":
            env.speed_cmd = env._fixed_speed = float(fwd)
        if env.cfg.task.tilt_command:
            env.tilt_cmd = env._fixed_tilt = tilt
            env._set_stance()
        if holder is None:
            act = policy.env_action(obs)
        else:
            act = holder.action(
                lambda o=obs: policy.env_action(o),
                bearing,
                speed == 0.0 and fwd == 0.0,
                tracker.lost,
                tilt,
            )
        obs, _, term, _, info = env.step(act)
        trace.append(
            {
                "t": t,
                "bearing_true": truth.bearing_deg,
                "pitch_true": truth.pitch_deg,
                "distance": floor_distance_m(env, person),
                "set_distance": follower.set_distance_m,
                "in_view": float(cam.in_view(truth)),
                "lost": float(tracker.lost),
                "cmd_yaw": bearing / 180.0,
                "cmd_speed": speed,
                "cmd_tilt": tilt,
                "stay": float(follower.stay),
                "robot_speed": info["speed"],
                "held": float(holder is not None and holder.held is not None),
            }
        )
        if term:
            fell = True
            break
    return {"summary": summarise(trace, fell, seconds), "trace": trace}


def hold_check(
    policy: JsonPolicy, seed: int, hold: bool, nominal: bool = False, hold_s: float = 20.0
) -> dict[str, float] | None:
    """Creep on target, with and without Hold: turn 60 degrees to a fixed target, settle,
    then stand on it for hold_s, then turn 90 more (that the robot still turns from a held
    pose). None when it never settled within 6 degrees."""
    env = env_for(policy)
    opts: dict[str, Any] = {"retarget": False, "tilt_cmd": 1.0, "target_error_deg": 60.0}
    if nominal:
        opts["nominal"] = True
    obs, _ = env.reset(seed=seed, options=opts)

    def err() -> float:
        return abs(wrap_deg(env.target_deg - env.yaw_deg()))

    def step(a: np.ndarray) -> tuple[np.ndarray, bool]:
        # A perfect compass, as in run(): with the gyro's drift left in, the policy
        # chases the drift and that reads as creep and turn error.
        env.heading_bias = 0.0
        o, _, term, _, _ = env.step(a)
        return o, term

    holder = Hold()
    for _ in range(round(40.0 / CONTROL_DT)):
        obs, term = step(holder.action(lambda o=obs: policy.env_action(o), err(), True, False, 1.0))
        if term:
            return {"fell": 1.0}
        if holder.held is not None:
            break
    else:
        return None
    pose = holder.held
    p0, yaw0 = env.data.qpos[0:2].copy(), env.yaw_deg()
    for _ in range(round(hold_s / CONTROL_DT)):
        obs, term = step(pose if hold else policy.env_action(obs))
        if term:
            return {"fell": 1.0}
    out = {
        "fell": 0.0,
        "creep_cm_s": 100.0 * float(np.linalg.norm(env.data.qpos[0:2] - p0)) / hold_s,
        "yaw_drift_deg": abs(wrap_deg(env.yaw_deg() - yaw0)),
    }
    env.target_deg = wrap_deg(env.target_deg + 90.0)
    for _ in range(round(25.0 / CONTROL_DT)):
        obs, term = step(policy.env_action(obs))
        if term:
            return {**out, "fell": 1.0}
    return {**out, "turn_err_deg": err()}


def summarise(trace: list[dict[str, float]], fell: bool, seconds: float) -> dict[str, float]:
    if not trace:
        return {"fell": float(fell)}
    arr = {k: np.array([r[k] for r in trace]) for k in trace[0]}
    stay = arr["stay"] > 0
    # Distance counts only while it is being held: not on "stay", not while searching.
    held = ~stay & (arr["lost"] == 0)
    gap = np.abs(arr["distance"] - arr["set_distance"])
    return {
        "seconds": len(trace) * CONTROL_DT,
        "fell": float(fell),
        "in_view_pct": 100.0 * float(arr["in_view"].mean()),
        "lost_pct": 100.0 * float(arr["lost"].mean()),
        "aim_err_deg": float(np.abs(arr["bearing_true"]).mean()),
        "pitch_err_deg": float(np.abs(arr["pitch_true"]).mean()),
        "distance_err_m": float(gap[held].mean()) if held.any() else float("nan"),
        "held_pct": 100.0 * float(arr["held"].mean()),
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
    ap.add_argument("--no-hold", action="store_true", help="keep the policy running on target")
    ap.add_argument(
        "--hold-check", action="store_true", help="creep on target with and without Hold"
    )
    args = ap.parse_args(argv)
    policy = JsonPolicy(args.policy)
    if args.hold_check:
        for hold in (False, True):
            rs = [hold_check(policy, s, hold, args.nominal) for s in range(args.seeds)]
            kept = [r for r in rs if r and "creep_cm_s" in r]
            falls = sum(1 for r in rs if r and r["fell"])
            m = {
                k: float(np.nanmean([r.get(k, np.nan) for r in kept])) if kept else math.nan
                for k in ("creep_cm_s", "yaw_drift_deg", "turn_err_deg")
            }
            print(
                f"{'hold' if hold else 'policy'}: settled and held {len(kept)}/{len(rs)}, "
                f"falls {falls}, creep {m['creep_cm_s']:.2f} cm/s, "
                f"yaw drift {m['yaw_drift_deg']:.1f} deg, then a 90 degree turn ends "
                f"{m['turn_err_deg']:.1f} deg off"
            )
        return 0
    runs = [
        run(policy, s, args.seconds, nominal=args.nominal, hold=not args.no_hold)
        for s in range(args.seeds)
    ]
    widest = max((r["summary"] for r in runs), key=len)
    keys = [k for k in widest if k != "completed"]
    print("seed  " + "  ".join(f"{k:>14}" for k in keys))
    for s, r in enumerate(runs):
        print(f"{s:>4}  " + "  ".join(f"{r['summary'].get(k, float('nan')):>14.2f}" for k in keys))
    mean = {k: float(np.nanmean([r["summary"].get(k, float("nan")) for r in runs])) for k in keys}
    print("mean  " + "  ".join(f"{mean[k]:>14.2f}" for k in keys))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"mean": mean, "runs": runs}, f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
