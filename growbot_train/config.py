"""Every tunable number in one place, loaded from a TOML file.

The defaults here are the ones the shipped config uses; ``configs/heading_walk.toml``
repeats them with plain-English comments so a run can be changed without reading code.
A run saves the exact config it used next to its weights, so a result can always be
traced back to the numbers that made it.
"""

from __future__ import annotations

import dataclasses
import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RewardConfig:
    # Walking toward the target, per step, at the speed cap. Below the cap it scales
    # linearly; above the cap it earns nothing extra (and overspeed is penalised). Both
    # use the speed averaged over ``speed_window_s``, so a paddling gait's lurches within
    # one step don't count as speeding.
    progress: float = 2.0
    # Paid for pointing the right way: weight * (1 + cos(error)) / 2, or with
    # heading_width_deg > 0, weight * exp(-(error / width)^2): pays only near the target,
    # so facing it is worth far more than sitting 60 degrees off (the cosine pays 75% there).
    heading: float = 0.5
    heading_width_deg: float = 0.0
    # With progress_width_deg > 0, progress (paid or charged) is scaled by
    # exp(-(error / width)^2), so it counts only while pointing near the target. Speed
    # along the target alone still pays 77% walking 40 degrees off, too little to make a
    # body that steers slowly finish its turns.
    progress_width_deg: float = 0.0
    # A one-off bonus each target that starts at least arrive_within_deg off, paid once
    # the error has stayed under it for arrive_hold_s: weight * (1 - time since the target
    # moved / arrive_by_s), the hold included, never below 0. The turn term pays the same
    # for a slow turn as a quick one; this pays for quick. 0 = off.
    arrive: float = 0.0
    arrive_within_deg: float = 15.0
    arrive_hold_s: float = 1.0
    arrive_by_s: float = 20.0
    speed_cap_m_s: float = 0.03
    speed_window_s: float = 1.0
    overspeed: float = 4.0
    tilt: float = 1.0
    fall: float = 10.0
    action_rate: float = 0.05
    effort: float = 0.01
    # Paid for standing taller than the body's resting stance: weight * exp(-(miss/tol)^2),
    # where "miss" is how far the body's pitch is from its resting pitch plus
    # task.stand_lift_deg. 0 turns it off (the first policy just keeps its resting stance).
    stance: float = 0.0
    stance_tol_deg: float = 6.0
    # Leaning back past the stance it should hold (its resting pitch plus
    # task.stand_lift_deg plus this margin) costs weight * (excess / 10 degrees)^2, a
    # warning that grows long before the point of no return. 0 turns it off.
    back_lean: float = 0.0
    back_lean_margin_deg: float = 12.0
    # With task.speed "command" (unless task.speed_reward is "progress") or "still",
    # progress pays for matching the wanted velocity instead: weight * (|wanted| - miss)
    # / cap, floored at -weight, where miss is how far the averaged velocity is from the
    # wanted one (direction included), in m/s.
    # Standing still always scores exactly zero, as it does in "cap" mode: an earlier
    # form paid a third of the weight for standing still when told to walk.
    # During a stop (the command is exactly 0, and always in "still" mode) drift up to
    # stop_drift_free * cap is free. Turning on the spot drifts a little, and charging
    # for it outbid the heading reward: the first camera walk and look runs froze rather
    # than turn to face the target.
    stop_drift_free: float = 0.5
    # When 0 or more, the free band narrows from stop_drift_free to this as the
    # curriculum ramps (randomization.ramp_steps): a policy that has already learned to
    # turn can be charged for the slow creep that would have frozen it at the start.
    # -1 = no narrowing.
    stop_drift_free_final: float = -1.0
    # While stopped (speed command 0, or always in "still" mode), wandering from where the
    # stop began (or the target last moved) costs weight * (distance - anchor_free_m) / 15 cm
    # per tick, at most weight. The speed band above only sees drift faster than
    # stop_drift_free; slower drift adds up unseen.
    # The policy isn't told the offset (the phone can't measure it), so it has to learn a
    # turn that doesn't wander. 0 = off.
    anchor: float = 0.0
    anchor_free_m: float = 0.02
    # Rocking costs weight * (roll rate^2 + pitch rate^2), rad/s: a steadier camera.
    wobble: float = 0.0
    # Keeping the person's face in the phone camera's picture, paid per tick: weight while
    # a virtual face (drawn per target, out in the person's direction) is inside the
    # frame, nothing outside it. Standing (speed 0), it also has to be centred across the
    # picture: weight * (1 - (x / half the width)^2). Walking sways the camera and that's
    # fine (Jeff, 2026-11-15) as long as the face stays in the picture. 0 = off.
    frame: float = 0.0
    frame_hfov_deg: float = 60.0  # follow.py's CameraSpec: the phone lies screen up
    frame_vfov_deg: float = 75.0
    frame_distance_m: tuple[float, float] = (0.7, 2.5)
    frame_face_height_m: tuple[float, float] = (1.1, 1.65)
    # Pay per degree the heading error shrank this tick (negative when it grew), measured
    # against the current target so a retarget's jump is never paid. It adds up to
    # weight * (error at start - error at end), so it can't be farmed. Turning on the spot
    # manages about 8 degrees a second, which moves the (1 + cos) heading term only
    # ~0.004 a tick: too faint for PPO to find, and the first look run never turned.
    turn: float = 0.0


@dataclass
class TaskConfig:
    body: str = "build"
    episode_seconds: float = 20.0
    # How far off the first target may be, in degrees either side of where the robot
    # starts out facing. 180 means anywhere.
    initial_error_deg: float = 180.0
    # Each episode the target moves this often (seconds between moves, drawn at random
    # from the range), so the policy learns to change course mid-walk.
    retarget_every_s: tuple[float, float] = (6.0, 14.0)
    retarget_max_deg: float = 120.0
    gait_hz: float = 1.0
    # A gait the robot always walks (a Keyframes JSON, repo-relative), the policy adding
    # only a correction of up to residual_rad per leg: a robust open-loop gait stays in
    # charge, so PPO can't unlearn it (the cloned nose walk drifted to half the gait's
    # own speed). Its hz must equal gait_hz. The speed command sets the step rate (the
    # clock runs at |speed_cmd| x gait_hz; a stop freezes it), the way the bench showed
    # this gait's speed scales. Empty = off: the policy drives the legs on its own.
    residual_gait: str = ""
    residual_rad: float = 0.5
    # A steering rule built into the residual gait's base, so PPO refines turning rather
    # than having to find it (from scratch the nose walk was still 47 degrees off at 7.5M;
    # this rule alone holds 15-23 on random robots): one leg pushed out by up to
    # residual_steer_rad, in proportion to the observed heading error over
    # residual_steer_deg, the right leg to turn left. Negative rad flips it, for a gait
    # that turns the other way. 0 = off.
    residual_steer_deg: float = 0.0
    residual_steer_rad: float = 0.5
    history_frames: int = 5
    # action [-1, 1] -> servo angle in radians; 1.0 rad is about 57 degrees each way.
    action_to_rad: float = 1.0
    fall_tilt_deg: float = 55.0
    # Which end walks first. "nose" is the end the legs lean toward; "camera" is the
    # other end, where the phone's front (screen-side) camera looks when it lies screen
    # up on top of a nose-up body. The heading error is measured from this end, so for
    # "camera" a heading error of 0 means "the face the camera sees is straight ahead".
    facing: str = "nose"
    # With facing "camera", where the heading is read from. "body" is the tail end's
    # direction, which leaning and rolling don't move. "lens" is where the phone's rear
    # camera actually points (its optical axis, the body's up axis, projected flat): the
    # body's tail end at rest, but rolling sideways swings it. With "lens", facing is
    # scored on the worse of the lens and body errors, so a roll that swings the lens
    # toward the target while the body stays put earns nothing: only turning does.
    heading_from: str = "body"
    # How much higher than its resting stance to hold the body (nose-up pitch, degrees).
    # Only rewarded when reward.stance > 0; the tilt penalty then measures from this
    # taller stance instead of the resting one.
    stand_lift_deg: float = 0.0
    # Share of episodes that start folded (both legs tucked to one angle drawn from
    # lying_leg_rad, inside what the policy itself can command) instead of
    # standing on neutral legs, so the policy learns to get up before it walks.
    lying_start_prob: float = 0.0
    lying_leg_rad: tuple[float, float] = (-0.95, -0.75)
    # Going over backwards: past about 95 degrees nose-up this body lands on its back
    # with its legs in the air and can't get up. An episode ends as a fall once the
    # pitch passes this many degrees, well before that point. 0 = only fall_tilt_deg.
    back_fall_pitch_deg: float = 0.0
    # How fast to go. "cap": always walk toward the target at the speed cap (the first
    # policies). "command": a second input, cmd_forward in [-1, 1], says how fast: +1 the
    # cap toward the target, 0 stand still (still turning to face it), below 0 back away.
    # "still": never walk, only turn on the spot to face the target.
    speed: str = "cap"
    # With "command", each episode redraws the command this often (seconds): a stop with
    # speed_stop_prob, otherwise anything from speed_back_min up to +1.
    speed_command_every_s: tuple[float, float] = (3.0, 8.0)
    speed_stop_prob: float = 0.3
    speed_back_min: float = -1.0
    # Curriculum for the speed command: every command starts at full speed (no stops) and
    # the spread widens to the above over this many steps. Scored on matching the command,
    # a policy that can't walk yet is paid most for keeping still, so asked for slow
    # walks and stops from the start it never learns to walk. 0 = full spread at once.
    speed_ramp_steps: int = 0
    # How a speed command is scored. "match": by how far the velocity is from the
    # commanded one (backing away included). "progress": like the cap walk, paid for speed
    # along the target up to the command and charged past it, so early flailing costs no
    # more than standing still; forward commands only (speed_back_min >= 0).
    speed_reward: str = "match"
    # A third input, cmd_pitch in [-1, 1], says how high to hold the camera end: -1 is
    # tilt_lift_deg[0] from the resting pitch (low, nearly flat: the camera looks almost
    # straight up), +1 is tilt_lift_deg[1] (standing tall). The stance bonus then pays for
    # matching it instead of stand_lift_deg. Redrawn like the speed command; standing tall
    # with tilt_tall_prob.
    tilt_command: bool = False
    tilt_lift_deg: tuple[float, float] = (-30.0, 6.0)
    tilt_command_every_s: tuple[float, float] = (3.0, 8.0)
    tilt_tall_prob: float = 0.5
    # The robot as built: "build" is the printed body with a phone on top (from upstream's
    # STLs, research/locomotion/make_body.py), "rollout" and "olie" upstream's sim bodies.
    # What each floor-touching surface is made of, one material per surface in
    # growbot_train/surfaces.py order (tail, belly, toe_tail, toe_nose), e.g. "pla" bare
    # print or "rubber" liquid dip. Empty = the original setup, one friction on every geom.
    surfaces: tuple[str, ...] = ("pla", "pla", "pla", "pla")
    # With surfaces: the floors an episode may be on, one drawn each episode.
    floors: tuple[str, ...] = ("hard", "carpet")
    # Set down tail first at this nose-up angle (degrees) and let it settle, instead of
    # level. The printed build needs it: level, it balances on its soles. Just under
    # where it rests (about 40 degrees), so it settles without rocking: from 25 it swung
    # past 55 and a body with long legs went over onto its back.
    start_pitch_deg: float = 35.0
    # Servo torque limit (N m); 0 keeps the body file's. An MG90S stalls near 0.2 N m,
    # upstream's model allows 1.5.
    servo_torque_nm: float = 0.25


@dataclass
class RandomizationConfig:
    enabled: bool = True
    mass_scale: tuple[float, float] = (0.80, 1.25)
    com_x_m: tuple[float, float] = (-0.030, 0.030)
    com_y_m: tuple[float, float] = (-0.015, 0.015)
    com_z_m: tuple[float, float] = (-0.010, 0.015)
    leg_length_scale: tuple[float, float] = (0.85, 1.15)
    leg_asymmetry: tuple[float, float] = (0.96, 1.04)
    servo_gain_scale: tuple[float, float] = (0.75, 1.25)
    # Upstream's shipped walks train on 0.2 to 2.2 (policies/forward-0930's dr).
    friction: tuple[float, float] = (0.2, 2.2)
    # With task.surfaces, friction is drawn per surface and per floor instead.
    surface_friction_scale: tuple[float, float] = (0.7, 1.3)
    floor_friction_scale: tuple[float, float] = (0.5, 1.6)
    servo_slew_rad_s: tuple[float, float] = (8.7, 10.5)
    servo_lag_s: tuple[float, float] = (0.015, 0.035)
    relay_delay_s: tuple[float, float] = (0.020, 0.160)
    # Scales the servo torque limit: upstream draws 0.13 to 0.3 N m, MG90Ss vary that much.
    servo_torque_scale: tuple[float, float] = (0.5, 1.2)
    # Each leg's zero is off by up to this much (rad): a horn is seated a spline tooth
    # out, or the trim is set by eye. Upstream draws +-4 to 5 degrees.
    trim_err_rad: float = 0.087
    # The phone's sensors arrive this late (s), separate from the relay's delay.
    sensor_delay_s: tuple[float, float] = (0.0, 0.12)
    # The foot's width along the body, as a share of the printed 21 mm: a worn or
    # rounded edge stands on less. Upstream draws a 6 to 10.5 mm half-width.
    foot_width_scale: tuple[float, float] = (0.57, 1.0)
    relay_drop_prob: float = 0.08
    imu_tilt_rad: float = 0.15
    accel_noise_m_s2: float = 0.3
    gyro_noise_rad_s: float = 0.1
    gyro_bias_rad_s: float = 0.02
    # Heading-error input: noise, slow drift (a gyro-integrated heading wanders), and a
    # refresh period in control ticks (a camera target updates slower than the IMU).
    heading_noise_deg: float = 3.0
    heading_drift_deg_per_s: float = 0.5
    heading_refresh_ticks: tuple[int, int] = (1, 6)
    # Shoves: every so often (seconds, drawn from the range) the body gets a sudden
    # nose-up or nose-down spin of up to push_pitch_rad_s, like a bump or a hand, so the
    # policy learns to catch itself. (0, 0) = no shoves.
    push_every_s: tuple[float, float] = (0.0, 0.0)
    push_pitch_rad_s: float = 0.0
    # Curriculum: start every body, servo, friction and sensor-offset range (and how far
    # off the target starts or jumps; not the sensor noise or refresh rate) at
    # ramp_from of its full width around the as-built robot, widening to the full range
    # over ramp_steps of training. A fragile gait cloned before PPO only pays on robots
    # near the one it was found on; on the full spread from step 0 PPO unlearns it.
    # 0 = full ranges from the start.
    ramp_steps: int = 0
    ramp_from: float = 0.15
    # Where the body, servo, friction and sensor-offset ranges stop widening, as a share
    # of their full width, in training and evaluation alike; the target ranges still
    # widen to full. A gait that collapses on the wider robots holds the spread it
    # learned on. 1 = the full ranges.
    ramp_to: float = 1.0


@dataclass
class TrainConfig:
    total_steps: int = 30_000_000
    # 0 = one environment per CPU thread, times envs_per_worker.
    n_envs: int = 0
    # Simulators each worker process steps per round trip (growbot_train/vec.py). 1 is
    # SB3's one process per simulator, which on the 32-thread box left the cores 86%
    # idle waiting on pipes. More per worker means more simulators in all, so a rollout
    # holds n_steps * n_envs samples: lower n_steps to keep it the same size.
    envs_per_worker: int = 1
    n_steps: int = 256
    batch_size: int = 4096
    n_epochs: int = 5
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    log_std_init: float = -0.5
    policy_layers: tuple[int, ...] = (128, 128)
    value_layers: tuple[int, ...] = (256, 256)
    checkpoint_every_steps: int = 2_000_000
    live_replay_every_steps: int = 500_000
    seed: int = 0
    # Before PPO, clone this open-loop gait (a search result or bare gait JSON,
    # see growbot_train/gait.py) into the policy so training starts from a moving robot.
    # Its hz must equal task.gait_hz. Empty = start from scratch.
    warm_gait: str = ""
    warm_gait_samples: int = 200_000
    warm_gait_epochs: int = 40
    # Clone warm_gait as a steered spin instead of a walk: the gait must turn the robot
    # counter-clockwise; with the target clockwise of the heading it is cloned with its
    # legs swapped (which spins it the other way), and within warm_gait_deadband_deg of
    # the target as standing still (zero action), so the clone already turns toward the
    # target and stops on it.
    warm_gait_steer: bool = False
    warm_gait_deadband_deg: float = 10.0
    # Legs offset (both the same, in action units) added to the cloned gait at camera
    # height command -1 and +1, linear between: without it the clone holds one posture
    # whatever the command, and the policy learned to ignore the input. Negative lowers
    # the body and aims the camera higher. Empty = no offset.
    warm_gait_tilt_offset: tuple[float, ...] = ()


@dataclass
class Config:
    name: str = "heading_walk"
    reward: RewardConfig = field(default_factory=RewardConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    randomization: RandomizationConfig = field(default_factory=RandomizationConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")


# Settings that have been removed but still sit in older runs' saved config.json (and in
# the config each exported policy carries), skipped so those runs still load.
_RETIRED = {("reward", "speed_track_tol")}

_SECTIONS = {
    "reward": RewardConfig,
    "task": TaskConfig,
    "randomization": RandomizationConfig,
    "train": TrainConfig,
}


def _coerce(cls: type, key: str, value: Any) -> Any:
    fields = {f.name: f for f in dataclasses.fields(cls)}
    if key not in fields:
        raise KeyError(f"unknown setting [{cls.__name__}] {key}")
    default = getattr(cls(), key)
    # TOML has arrays, not tuples; keep the dataclass's tuple shape so ranges stay ranges.
    if isinstance(default, tuple):
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{key} must be a list")
        return tuple(value)
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise TypeError(f"{key} must be true or false")
        return value
    if isinstance(default, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(default, float):
        return float(value)
    return value


# A run saved before task.surfaces existed ran the old physics: one friction on every
# geom, set down level, no torque cap. Its config.json must not pick up today's defaults.
_LEGACY_TASK = {"surfaces": (), "start_pitch_deg": 0.0, "servo_torque_nm": 0.0}
# Likewise a run saved before the servo-torque, trim, sensor-delay and foot-width draws.
_LEGACY_RANDOMIZATION = {
    "servo_torque_scale": (1.0, 1.0),
    "trim_err_rad": 0.0,
    "sensor_delay_s": (0.0, 0.0),
    "foot_width_scale": (1.0, 1.0),
}


def from_dict(data: dict[str, Any], saved: bool = False) -> Config:
    """``saved``: ``data`` is a run's or policy's own config, not a hand-written one."""
    cfg = Config()
    if saved and "surfaces" not in data.get("task", {}):
        for k, v in _LEGACY_TASK.items():
            setattr(cfg.task, k, v)
    if saved and "trim_err_rad" not in data.get("randomization", {}):
        for k, v in _LEGACY_RANDOMIZATION.items():
            setattr(cfg.randomization, k, v)
    for key, value in data.items():
        if key == "name":
            cfg.name = str(value)
            continue
        if key not in _SECTIONS:
            raise KeyError(f"unknown section [{key}]")
        section = getattr(cfg, key)
        for k, v in value.items():
            if (key, k) in _RETIRED:
                continue
            setattr(section, k, _coerce(_SECTIONS[key], k, v))
    return cfg


def load(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    """A TOML config file, or a run's saved ``config.json``, plus ``--set`` overrides."""
    data: dict[str, Any] = {}
    saved = path is not None and str(path).endswith(".json")
    if saved:
        data = json.loads(Path(path).read_text())
    elif path is not None:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    for item in overrides or []:
        # "reward.speed_cap_m_s=0.05": parsed as a TOML value so lists and bools work.
        dotted, _, raw = item.partition("=")
        section, _, key = dotted.strip().partition(".")
        if not key:
            raise ValueError(f"override {item!r} must look like section.key=value")
        value = tomllib.loads(f"v = {raw.strip()}")["v"]
        data.setdefault(section, {})[key] = value
    return from_dict(data, saved=saved)
