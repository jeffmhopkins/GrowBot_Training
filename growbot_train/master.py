"""The master policy: one student distilled from the single-skill teachers, so the app
runs one file whatever it is asked (docs/follow_me.md, "One master policy").

The student takes the follower's three commands and sees one gait clock per teacher, so
each teacher's timing survives the distillation; the app runner already steps any number
of declared `cycle:` channels. Which teacher labels a tick (`choose_mode`):

- speed below 0, the heading error within `BACK_DEG`: the nose walk, backing away nose
  first;
- speed above 0, the heading error within `WALK_DEG`: lift and fall;
- otherwise (standing, or asked to move with the target well off to the side): look's
  spin, until the heading comes inside the limit. Jeff, 2026-11-15: the heading never
  picks the waddle, only the speed's sign does; a wide heading only delays it.

DAgger: the first round lets the teachers drive, later rounds let the student drive more
and more often while the teachers keep labelling, so the student also learns to recover
from its own mistakes and the hand-overs between modes. Each teacher keeps its own frame
history every tick, active or not, so it takes over with a full history.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from growbot_train import config as config_mod
from growbot_train.env import CONTROL_DT, CTRL_LIMIT, GrowBotHeadingEnv, wrap_deg
from growbot_train.export import FORMAT, LENS_BLEND
from growbot_train.gait import Keyframes
from growbot_train.policy import runner_forward

MODES = ("look", "walk", "back")
WALK_DEG = 30.0  # Jeff, 2026-11-15: wider than this, stop, spin with look, walk again
WALK_HYST_DEG = 10.0  # leave the walk only past 40 degrees, so the switch doesn't chatter
# The nose walk steers weakly to its left: past about 45 degrees it ends 15-20 off
# (locomotion thread's last-10 s test at 22.5M), so wider than this it spins with look
# first, as the walk does past WALK_DEG.
BACK_DEG = 45.0
BACK_HYST_DEG = 10.0
ACTION_TO_RAD = CTRL_LIMIT  # the student may reach every teacher's swing
FRAMES = 5
SQUASH = 0.9993  # tanh(ACTION_LIMIT): the furthest a squashed action reaches

DEFAULT_TEACHERS = {
    "look": "policies/look-v7/policy_look.json",
    "walk": "policies/lift-fall-2/policy_lift_fall.json",
    "back": "policies/nose-walk-1/policy_nose_walk.json",
}

STUDENT_CHANNELS = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "prev_action_0",
    "prev_action_1",
    "cmd_yaw",
    "cmd_forward",
    "cmd_pitch",
    *(f"cycle:{m}_{fn}" for m in MODES for fn in ("sin", "cos")),
)


@dataclass
class Teacher:
    mode: str
    data: dict[str, Any]
    channels: tuple[str, ...]
    frames: int
    hz: float
    a2r: float
    lens: bool
    back_fall_deg: float
    ramp_to: float
    # A residual-gait teacher (the nose walk since #59) outputs a correction on top of a
    # gait and a steering rule; its leg angles are base + output x residual_rad.
    gait: Keyframes | None = None
    residual_rad: float = 0.0
    steer_deg: float = 0.0
    steer_rad: float = 0.0
    hist: np.ndarray = field(default_factory=lambda: np.zeros(0))
    base: np.ndarray = field(default_factory=lambda: np.zeros(2))

    @classmethod
    def load(cls, mode: str, path: str | Path) -> Teacher:
        data = json.loads(Path(path).read_text())
        spec = data["obs_spec"]
        seg = spec["segments"][0]
        cfg = data.get("training", {}).get("config", {})
        task = cfg.get("task", {})
        ramp_to = cfg.get("randomization", {}).get("ramp_to")
        t = cls(
            mode=mode,
            data=data,
            channels=tuple(seg["channels"]),
            frames=int(seg["frames"]),
            hz=float(spec["cycles"]["gait_sin"]["frequency"]),
            a2r=float(spec["action_to_rad"]),
            lens=task.get("heading_from") == "lens",
            back_fall_deg=float(task.get("back_fall_pitch_deg", 0.0)),
            ramp_to=1.0 if ramp_to is None else float(ramp_to),
        )
        res = spec.get("residual_gait")
        if res:
            t.gait = Keyframes(res["hz"], res["keys_rad"])
            t.residual_rad = float(res["residual_rad"])
            t.steer_deg = float(res.get("steer_deg", 0.0))
            t.steer_rad = float(res.get("steer_rad", 0.0))
        t.reset()
        return t

    def reset(self) -> None:
        self.hist = np.zeros((self.frames, len(self.channels)))
        self.base = np.zeros(2)

    def base_rad(self, s: Senses) -> np.ndarray:
        """A residual teacher's gait and steering this tick, as its env builds them. Its
        clock is the shared step count: backing is only ever asked at full speed, where
        its own clock runs at exactly that rate (a slower ask needs a speed-scaled clock
        in the app runner, which it doesn't have)."""
        if self.gait is None:
            return np.zeros(2)
        ease = min(1.0, s.step_index * CONTROL_DT)
        base = ease * self.gait(s.step_index * CONTROL_DT)
        if self.steer_deg and self.mode != "look" and s.forward != 0.0:
            err = s.err_lens if self.lens else s.err_body
            u = float(np.clip(err / self.steer_deg, -1.0, 1.0))
            base = base + self.steer_rad * np.array([max(u, 0.0), max(-u, 0.0)])
        return base

    def observe(self, s: Senses) -> None:
        """Push this tick's frame (its last action read against last tick's base, as
        its env does) and set this tick's base."""
        self.hist = np.concatenate([teacher_frame(self, s)[None], self.hist[:-1]])
        self.base = self.base_rad(s)

    def act_rad(self) -> np.ndarray:
        """Its leg commands in radians."""
        out = runner_forward(self.data, self.hist.reshape(-1))
        if self.gait is None:
            return out * self.a2r
        return np.clip(self.base + out * self.residual_rad, -CTRL_LIMIT, CTRL_LIMIT)


@dataclass
class Senses:
    """One tick's readings, shared by the student and every teacher."""

    acc: np.ndarray
    gyro: np.ndarray
    prev_rad: np.ndarray
    err_lens: float
    err_body: float
    forward: float
    tilt: float
    step_index: int


def build_frame(
    channels: tuple[str, ...], s: Senses, a2r: float, lens: bool, clocks: dict[str, float]
) -> np.ndarray:
    """One frame in a policy's channel order: the same arithmetic as the env's own
    frame, so a teacher sees what it trained on."""
    out = []
    for c in channels:
        if c.startswith("accel_"):
            out.append(s.acc["xyz".index(c[-1])])
        elif c.startswith("gyro_"):
            out.append(s.gyro["xyz".index(c[-1])])
        elif c.startswith("prev_action_"):
            out.append(float(np.clip(s.prev_rad[int(c[-1])] / a2r, -1.0, 1.0)))
        elif c == "cmd_yaw":
            out.append((s.err_lens if lens else s.err_body) / 180.0)
        elif c == "cmd_forward":
            out.append(s.forward)
        elif c == "cmd_pitch":
            out.append(s.tilt)
        elif c.startswith("cycle:"):
            name = c[len("cycle:") :]
            fn = math.sin if name.endswith("sin") else math.cos
            out.append(fn(2 * math.pi * clocks[name] * s.step_index * CONTROL_DT))
        else:
            raise ValueError(f"unknown channel {c}")
    return np.asarray(out, dtype=np.float64)


def teacher_frame(t: Teacher, s: Senses) -> np.ndarray:
    """The teacher's own inputs: its speed is the size of the student's in its mode (the
    nose walk's +1 is backing at full speed), and its clock its own rate."""
    fwd = abs(s.forward) if t.mode != "look" else 0.0
    shown = Senses(**{**s.__dict__, "forward": fwd})
    clocks = {"gait_sin": t.hz, "gait_cos": t.hz}
    if t.gait is None:
        return build_frame(t.channels, shown, t.a2r, t.lens, clocks)
    # Its last action is the correction that, on last tick's base, gives the legs sent;
    # none on the first tick, as its env starts (the legs are the settled stance then).
    shown.prev_rad = np.zeros(2) if s.step_index == 0 else s.prev_rad - t.base
    return build_frame(t.channels, shown, t.residual_rad, t.lens, clocks)


def student_clocks(teachers: dict[str, Teacher]) -> dict[str, float]:
    return {f"{m}_{fn}": teachers[m].hz for m in MODES for fn in ("sin", "cos")}


def choose_mode(forward: float, err_body: float, current: str | None = None) -> str:
    if forward < 0:
        limit = BACK_DEG + (BACK_HYST_DEG if current == "back" else 0.0)
        if abs(err_body) <= limit:
            return "back"
    elif forward > 0:
        limit = WALK_DEG + (WALK_HYST_DEG if current == "walk" else 0.0)
        if abs(err_body) <= limit:
            return "walk"
    return "look"


def to_student(rad: np.ndarray) -> np.ndarray:
    """A teacher's leg angles as the student's squashed action: matched in radians."""
    return np.clip(np.asarray(rad) / ACTION_TO_RAD, -SQUASH, SQUASH)


def student_env(look: Teacher) -> GrowBotHeadingEnv:
    """The robot the student drives: look's printed build and randomisation (lens
    heading, so the env's heading input is what the tracker reports), with the
    student's full servo swing. The script below sets every command."""
    cfg = config_mod.from_dict(look.data["training"]["config"], saved=True)
    cfg.task.action_to_rad = ACTION_TO_RAD
    cfg.task.history_frames = FRAMES
    return GrowBotHeadingEnv(cfg)


def senses(env: GrowBotHeadingEnv, forward: float, tilt: float) -> Senses:
    frame = env.hist[0]
    lens_yaw = env.yaw_deg()
    return Senses(
        acc=frame[0:3].copy(),
        gyro=frame[3:6].copy(),
        prev_rad=env.prev_action * env.cfg.task.action_to_rad,
        err_lens=float(env._err_obs),
        err_body=wrap_deg(env._err_obs + lens_yaw - env.body_yaw_deg()),
        forward=forward,
        tilt=tilt,
        step_index=env.step_index,
    )


@dataclass
class Script:
    """Random commands for one episode: a mode held for a few seconds, then another,
    with the target and camera height moving on their own clocks."""

    rng: np.random.Generator
    walk_speeds: bool
    back_speeds: bool
    forward: float = 0.0
    tilt: float = 1.0
    next_mode: float = 0.0
    next_target: float = 0.0
    next_tilt: float = 0.0

    def tick(self, env: GrowBotHeadingEnv, t: float) -> None:
        rng = self.rng
        moved = False
        if t >= self.next_mode:
            r = rng.random()
            if r < 0.35:
                self.forward = 0.0
            elif r < 0.7:
                self.forward = float(rng.uniform(0.3, 1.0)) if self.walk_speeds else 1.0
            else:
                self.forward = -(float(rng.uniform(0.3, 1.0)) if self.back_speeds else 1.0)
            self.next_mode = t + rng.uniform(3.0, 12.0)
            if rng.random() < 0.6:
                env.target_deg = wrap_deg(env.target_deg + rng.uniform(-180, 180))
                moved = True
        if t >= self.next_target:
            if t > 0:
                env.target_deg = wrap_deg(env.target_deg + rng.uniform(-45, 45))
                moved = True
            self.next_target = t + rng.uniform(5.0, 15.0)
        if t >= self.next_tilt:
            self.tilt = 1.0 if rng.random() < 0.5 else float(rng.uniform(-1.0, 1.0))
            env.tilt_cmd = env._fixed_tilt = self.tilt
            env._set_stance()
            self.next_tilt = t + rng.uniform(8.0, 20.0)
        if moved:
            env._observe_heading(force=True)


class Rollout:
    """One worker's simulator and teachers, reused across episodes."""

    def __init__(self, teacher_paths: dict[str, str]) -> None:
        self.teachers = {m: Teacher.load(m, p) for m, p in teacher_paths.items()}
        self.env = student_env(self.teachers["look"])
        # Bodies only as far from the as-built one as every teacher trained on: past
        # that a teacher's labels are guesses (lift-fall-2 saw 35% of the ranges).
        self.env.dr_scale = min(t.ramp_to for t in self.teachers.values())
        self.clocks = student_clocks(self.teachers)

    def judge_as(self, mode: str) -> None:
        """Fall by the active teacher's rule: lift and fall's gait rears up past look's
        back-fall line by design."""
        self.env.cfg.task.back_fall_pitch_deg = self.teachers[mode].back_fall_deg

    def episode(
        self,
        seed: int,
        seconds: float,
        student: dict[str, Any] | None,
        beta: float,
    ) -> dict[str, Any]:
        env, teachers = self.env, self.teachers
        rng = np.random.default_rng(seed + 104729)
        # Who drives draws from its own stream, so the script is the same whoever drives.
        coin = np.random.default_rng(seed + 7919)
        env.reset(
            seed=seed,
            options={
                "retarget": False,
                "speed_cmd": 0.0,
                "tilt_cmd": 1.0,
                "target_error_deg": float(rng.uniform(-180, 180)),
            },
        )
        n = round(seconds / CONTROL_DT)
        env.max_steps = n + 1
        script = Script(
            rng,
            walk_speeds="cmd_forward" in teachers["walk"].channels
            and teachers["walk"].gait is None,
            back_speeds="cmd_forward" in teachers["back"].channels
            and teachers["back"].gait is None,
        )
        for t in teachers.values():
            t.reset()
        hist = np.zeros((FRAMES, len(STUDENT_CHANNELS)))
        obs, labels, modes = [], [], []
        mode = None
        switches = 0
        fell = False
        for i in range(n):
            script.tick(env, i * CONTROL_DT)
            s = senses(env, script.forward, script.tilt)
            for t in teachers.values():
                t.observe(s)
            frame = build_frame(STUDENT_CHANNELS, s, ACTION_TO_RAD, True, self.clocks)
            hist = np.concatenate([frame[None], hist[:-1]])
            new = choose_mode(script.forward, s.err_body, mode)
            switches += mode is not None and new != mode
            mode = new
            self.judge_as(mode)
            teacher = teachers[mode]
            label = to_student(teacher.act_rad())
            x = hist.reshape(-1).copy()
            obs.append(x)
            labels.append(label)
            modes.append(MODES.index(mode))
            if student is None or coin.random() < beta:
                drive = label
            else:
                drive = np.clip(runner_forward(student, x), -SQUASH, SQUASH)
            _, _, term, _, _ = env.step(np.arctanh(drive))
            if term:
                fell = True
                break
        return {
            "obs": np.asarray(obs, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.float32),
            "modes": np.asarray(modes, dtype=np.int8),
            "fell": fell,
            "switches": switches,
        }


_WORKER: Rollout | None = None


def _init(teacher_paths: dict[str, str]) -> None:
    global _WORKER
    _WORKER = Rollout(teacher_paths)


def _run(args: tuple[int, float, dict[str, Any] | None, float]) -> dict[str, Any]:
    assert _WORKER is not None
    return _WORKER.episode(*args)


def collect(
    teacher_paths: dict[str, str],
    seeds: list[int],
    seconds: float,
    student: dict[str, Any] | None,
    beta: float,
    workers: int,
) -> list[dict[str, Any]]:
    jobs = [(s, seconds, student, beta) for s in seeds]
    if workers <= 1:
        _init(teacher_paths)
        return [_run(j) for j in jobs]
    with get_context("fork").Pool(workers, initializer=_init, initargs=(teacher_paths,)) as pool:
        return pool.map(_run, jobs)


def fit(
    obs: np.ndarray,
    labels: np.ndarray,
    norm: tuple[np.ndarray, np.ndarray],
    init: dict[str, Any] | None,
    epochs: int,
    seed: int,
    hidden: int = 128,
) -> tuple[dict[str, Any], float]:
    """Behaviour cloning on the squashed actions, in the student's units. Returns the
    student as runner layers (mean, std, layers) and the last epoch's loss."""
    import torch

    torch.manual_seed(seed)
    mean, std = norm
    net = torch.nn.Sequential(
        torch.nn.Linear(obs.shape[1], hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.Tanh(),
        torch.nn.Linear(hidden, 2),
    ).double()
    if init is not None:
        linears = [m for m in net if isinstance(m, torch.nn.Linear)]
        with torch.no_grad():
            for lin, layer in zip(linears, init["layers"], strict=True):
                lin.weight.copy_(torch.tensor(layer["W"]).T)
                lin.bias.copy_(torch.tensor(layer["b"]))
    x = torch.tensor((obs - mean) / std)
    y = torch.tensor(labels, dtype=torch.float64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    batch = 4096
    loss_v = float("nan")
    for _ in range(epochs):
        perm = torch.randperm(len(x))
        total = 0.0
        for k in range(0, len(x), batch):
            idx = perm[k : k + batch]
            loss = torch.mean((torch.tanh(net(x[idx])) - y[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        loss_v = total / len(x)
    linears = [m for m in net if isinstance(m, torch.nn.Linear)]
    layers = [
        {
            "W": lin.weight.detach().T.tolist(),
            "b": lin.bias.detach().tolist(),
            "act": "tanh" if i < len(linears) - 1 else "linear",
        }
        for i, lin in enumerate(linears)
    ]
    return {"mean": mean.tolist(), "std": std.tolist(), "layers": layers}, loss_v


def normaliser(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = obs.mean(axis=0).astype(np.float64)
    std = np.sqrt(obs.var(axis=0).astype(np.float64) + 1e-8)
    return mean, np.maximum(std, 1e-3)


def export_policy(
    student: dict[str, Any], teachers: dict[str, Teacher], teacher_paths: dict[str, str]
) -> dict[str, Any]:
    look_cfg = teachers["look"].data["training"]["config"]["task"]
    lo, hi = look_cfg["tilt_lift_deg"]
    caps = {m: teachers[m].data["training"]["speed_cap_m_s"] for m in ("walk", "back")}
    n = FRAMES * len(STUDENT_CHANNELS)
    return {
        "format": FORMAT,
        "version": 1,
        "name": "master",
        "obs_size": n,
        "act_size": 2,
        "activation": "tanh",
        "mean": student["mean"],
        "std": student["std"],
        "layers": student["layers"],
        "obs_spec": {
            "control_hz": round(1 / CONTROL_DT),
            "action_to_rad": ACTION_TO_RAD,
            "action_map": {"0": "right_leg", "1": "left_leg"},
            "imu_frame": teachers["look"].data["obs_spec"]["imu_frame"],
            "segments": [
                {
                    "channels": list(STUDENT_CHANNELS),
                    "frames": FRAMES,
                    "order": "newest_first",
                    "layout": "frame_major",
                }
            ],
            # One clock per teacher, each at its own gait's rate, all from the start.
            "cycles": {
                f"{m}_{fn}": {"type": fn, "frequency": teachers[m].hz}
                for m in MODES
                for fn in ("sin", "cos")
            },
            "cmd_yaw": {
                "meaning": "heading_error",
                "scale": "degrees / 180",
                "range": [-1, 1],
                "sign": (
                    "positive when the target is counter-clockwise of where the camera "
                    "points seen from above, i.e. turn left"
                ),
                "facing": "camera",
                "heading_from": "lens",
                "lens_fade": LENS_BLEND,
            },
            "cmd_forward": {
                "meaning": "speed_command",
                "range": [-1, 1],
                "scale": (
                    f"+1 walks camera first toward the target at {caps['walk'] * 100:g} "
                    f"cm/s, -1 backs away nose first at {caps['back'] * 100:g} cm/s, 0 "
                    "stands and turns to face it; the sign picks the gait, and a wide heading "
                    "turns first"
                ),
            },
            "cmd_pitch": {
                "meaning": "camera_height",
                "range": [-1, 1],
                "scale": (
                    f"-1 holds the body {-lo:g} degrees flatter than its resting stance, "
                    f"+1 {hi:g} degrees taller; only while standing (speed 0)"
                ),
            },
        },
        "training": {
            "master": {
                "teachers": {m: str(p) for m, p in teacher_paths.items()},
                "walk_deg": WALK_DEG,
                "walk_hysteresis_deg": WALK_HYST_DEG,
                "back_deg": BACK_DEG,
                "back_hysteresis_deg": BACK_HYST_DEG,
            },
            "speed_cap_m_s": caps["walk"],
            "speed": "command",
        },
    }


def distill(
    out: Path,
    teacher_paths: dict[str, str],
    rounds: tuple[float, ...] = (1.0, 0.5, 0.25, 0.0, 0.0),
    episodes: int = 64,
    seconds: float = 60.0,
    epochs: int = 40,
    workers: int = 4,
    seed: int = 0,
    log: Any = print,
) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    teachers = {m: Teacher.load(m, p) for m, p in teacher_paths.items()}
    obs_all: list[np.ndarray] = []
    lab_all: list[np.ndarray] = []
    student: dict[str, Any] | None = None
    norm = None
    history = []
    for r, beta in enumerate(rounds):
        seeds = [seed * 100003 + r * 10007 + k for k in range(episodes)]
        eps = collect(teacher_paths, seeds, seconds, student, beta, workers)
        obs_all += [e["obs"] for e in eps if len(e["obs"])]
        lab_all += [e["labels"] for e in eps if len(e["labels"])]
        falls = sum(e["fell"] for e in eps)
        switches = sum(e["switches"] for e in eps)
        obs = np.concatenate(obs_all)
        labels = np.concatenate(lab_all)
        if norm is None:
            norm = normaliser(obs)
        student, loss = fit(obs, labels, norm, student, epochs, seed + r)
        row = {
            "round": r,
            "beta": beta,
            "samples": int(len(obs)),
            "falls": int(falls),
            "episodes": episodes,
            "switches": int(switches),
            "loss": loss,
        }
        history.append(row)
        log(json.dumps(row))
    assert student is not None
    policy = export_policy(student, teachers, teacher_paths)
    path = out / "policy_master.json"
    path.write_text(json.dumps(policy))
    (out / "distill.json").write_text(json.dumps(history, indent=1))
    return path


# ------------------------------------------------------------------ evaluation

SCENARIOS = {
    # mode: (forward command, start errors)
    "look": (0.0, (90.0, -90.0, 180.0)),
    "walk": (1.0, (0.0, 25.0, -25.0)),  # inside WALK_DEG, so it measures the walk alone
    "back": (
        -1.0,
        (0.0, 35.0, -35.0),
    ),  # inside BACK_DEG with room for noise, so it measures the nose walk alone
}


class MasterRunner:
    def __init__(self, policy: dict[str, Any], teachers: dict[str, Teacher]) -> None:
        self.policy = policy
        self.clocks = student_clocks(teachers)
        self.hist = np.zeros((FRAMES, len(STUDENT_CHANNELS)))

    def reset(self) -> None:
        self.hist[:] = 0

    def act(self, s: Senses) -> np.ndarray:
        frame = build_frame(STUDENT_CHANNELS, s, ACTION_TO_RAD, True, self.clocks)
        self.hist = np.concatenate([frame[None], self.hist[:-1]])
        return np.clip(runner_forward(self.policy, self.hist.reshape(-1)), -SQUASH, SQUASH)


def drive(
    rollout: Rollout,
    master: MasterRunner | None,
    seed: int,
    forward: float,
    start_err: float,
    seconds: float,
    nominal: bool,
) -> dict[str, float]:
    """One fixed command held for the whole run, driven by the master or (None) by the
    teachers through choose_mode: the same robot, so the two compare directly."""
    env, teachers = rollout.env, rollout.teachers
    opts: dict[str, Any] = {
        "retarget": False,
        "speed_cmd": 0.0,
        "tilt_cmd": 1.0,
        "target_error_deg": start_err,
    }
    if nominal:
        opts["nominal"] = True
    env.reset(seed=seed, options=opts)
    n = round(seconds / CONTROL_DT)
    env.max_steps = n + 1
    env.tilt_cmd = env._fixed_tilt = 1.0
    env._set_stance()
    for t in teachers.values():
        t.reset()
    if master is not None:
        master.reset()
    mode = None
    along, errs, speed = [], [], []
    framed, img_x = [], []
    turn_s = float("nan")
    fell = False
    for _ in range(n):
        s = senses(env, forward, 1.0)
        for t in teachers.values():
            t.observe(s)
        mode = choose_mode(forward, s.err_body, mode)
        rollout.judge_as(mode)
        if master is None:
            a = to_student(teachers[mode].act_rad())
        else:
            a = master.act(s)
        _, _, term, _, info = env.step(np.arctanh(a))
        # Scored as the follower sees it: where the camera points, against the person.
        err = abs(wrap_deg(env.target_deg - env.yaw_deg()))
        along.append(info["v_along"])
        speed.append(info["speed"])
        errs.append(err)
        framed.append(info["in_frame"])
        img_x.append(abs(info["image_x_avg_deg"]) if info["in_frame"] else math.nan)
        if math.isnan(turn_s) and err < 15:
            turn_s = env.step_index * CONTROL_DT
        if term:
            fell = True
            break
    half = len(errs) // 2
    return {
        "toward": float(np.mean(along)),
        "speed": float(np.mean(speed[half:])),
        "err": float(np.mean(errs[half:])),
        "turn_s": turn_s,
        "frame_pct": 100.0 * float(np.mean(framed)),
        "img_x": float(np.nanmean(img_x[half:])) if any(framed[half:]) else math.nan,
        "fell": float(fell),
    }


def switch_test(
    rollout: Rollout, master: MasterRunner | None, seeds: list[int], seconds: float
) -> dict[str, float]:
    """Episodes of random commands, falls against mode switches: driven by the master,
    or (None) by the teachers on the same scripts, so a fall rate the teachers share
    shows as theirs."""
    falls = switches = 0
    for seed in seeds:
        policy = None if master is None else master.policy
        ep = rollout.episode(seed, seconds, policy, beta=0.0)
        falls += ep["fell"]
        switches += ep["switches"]
    return {"falls": falls, "switches": switches, "episodes": len(seeds)}


def evaluate(
    policy_path: Path, episodes: int = 4, seconds: float = 30.0, switch_episodes: int = 20
) -> dict[str, Any]:
    policy = json.loads(Path(policy_path).read_text())
    teacher_paths = policy["training"]["master"]["teachers"]
    rollout = Rollout(teacher_paths)
    master = MasterRunner(policy, rollout.teachers)
    report: dict[str, Any] = {}
    for mode, (fwd, starts) in SCENARIOS.items():
        for start in starts:
            for who, m in (("teacher", None), ("master", master)):
                runs = [
                    drive(rollout, m, 9000 + k, fwd, start, seconds, nominal=False)
                    for k in range(episodes)
                ]
                runs.append(drive(rollout, m, 0, fwd, start, seconds, nominal=True))
                agg = {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0] if k != "fell"}
                agg["falls"] = sum(r["fell"] for r in runs)
                agg["runs"] = len(runs)
                report.setdefault(mode, {}).setdefault(f"{start:+.0f}", {})[who] = agg
    seeds = [7000 + k for k in range(switch_episodes)]
    report["switch"] = {
        "teacher": switch_test(rollout, None, seeds, 60),
        "master": switch_test(rollout, master, seeds, 60),
    }
    return report


def format_report(report: dict[str, Any]) -> str:
    lines = []
    for mode in MODES:
        lines.append(f"-- {mode}")
        lines.append(" start  who       toward   speed    err  turn_s  frame%  img_x  falls")
        for start, row in report[mode].items():
            for who in ("teacher", "master"):
                r = row[who]
                lines.append(
                    f"{start:>6}  {who:<8} {r['toward']:7.4f} {r['speed']:7.4f} "
                    f"{r['err']:6.1f} {r['turn_s']:7.2f}  {r['frame_pct']:6.1f} "
                    f"{r['img_x']:6.1f}  {r['falls']}/{r['runs']}"
                )
    for who in ("teacher", "master"):
        sw = report["switch"][who]
        lines.append(
            f"-- switches, {who}-driven: {sw['falls']} falls in {sw['switches']} mode "
            f"switches over {sw['episodes']} minute-long runs"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("distill", help="distil the teachers into one student")
    d.add_argument("--out", type=Path, required=True)
    for m in MODES:
        d.add_argument(f"--{m}", default=DEFAULT_TEACHERS[m], help=f"{m} teacher JSON")
    d.add_argument("--episodes", type=int, default=64, help="minute-long episodes per round")
    d.add_argument("--rounds", default="1,0.5,0.25,0,0", help="teacher-driving share per round")
    d.add_argument("--epochs", type=int, default=40)
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("evaluate", help="score the master against its teachers")
    e.add_argument("policy", type=Path)
    e.add_argument("--episodes", type=int, default=4)
    e.add_argument("--switch-episodes", type=int, default=20)
    e.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    if args.cmd == "distill":
        paths = {m: getattr(args, m) for m in MODES}
        rounds = tuple(float(b) for b in args.rounds.split(","))
        path = distill(
            args.out,
            paths,
            rounds,
            args.episodes,
            epochs=args.epochs,
            workers=args.workers,
            seed=args.seed,
        )
        print(path)
        return 0
    report = evaluate(args.policy, args.episodes, switch_episodes=args.switch_episodes)
    print(format_report(report))
    if args.json:
        args.json.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
