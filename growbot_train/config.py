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
    # Paid for pointing the right way: weight * (1 + cos(error)) / 2.
    heading: float = 0.5
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


@dataclass
class TaskConfig:
    body: str = "rollout"
    episode_seconds: float = 20.0
    # How far off the first target may be, in degrees either side of where the robot
    # starts out facing. 180 means anywhere.
    initial_error_deg: float = 180.0
    # Each episode the target moves this often (seconds between moves, drawn at random
    # from the range), so the policy learns to change course mid-walk.
    retarget_every_s: tuple[float, float] = (6.0, 14.0)
    retarget_max_deg: float = 120.0
    gait_hz: float = 1.0
    history_frames: int = 5
    # action [-1, 1] -> servo angle in radians; 1.0 rad is about 57 degrees each way.
    action_to_rad: float = 1.0
    fall_tilt_deg: float = 55.0
    # Which end walks first. "nose" is the end the legs lean toward; "camera" is the
    # other end, where the phone's rear camera looks when it lies on top of a nose-up
    # body. The heading error is measured from this end, so for "camera" a heading error
    # of 0 means "the face the camera sees is straight ahead".
    facing: str = "nose"
    # How much higher than its resting stance to hold the body (nose-up pitch, degrees).
    # Only rewarded when reward.stance > 0; the tilt penalty then measures from this
    # taller stance instead of the resting one.
    stand_lift_deg: float = 0.0
    # Share of episodes that start folded flat (legs tucked to lying_leg_rad) instead of
    # standing on neutral legs, so the policy learns to get up before it walks.
    lying_start_prob: float = 0.0
    lying_leg_rad: tuple[float, float] = (-1.2, -0.8)


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
    friction: tuple[float, float] = (0.6, 1.4)
    servo_slew_rad_s: tuple[float, float] = (8.7, 10.5)
    servo_lag_s: tuple[float, float] = (0.015, 0.035)
    relay_delay_s: tuple[float, float] = (0.020, 0.090)
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


@dataclass
class TrainConfig:
    total_steps: int = 30_000_000
    # 0 = one environment per CPU thread.
    n_envs: int = 0
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


def from_dict(data: dict[str, Any]) -> Config:
    cfg = Config()
    for key, value in data.items():
        if key == "name":
            cfg.name = str(value)
            continue
        if key not in _SECTIONS:
            raise KeyError(f"unknown section [{key}]")
        section = getattr(cfg, key)
        for k, v in value.items():
            setattr(section, k, _coerce(_SECTIONS[key], k, v))
    return cfg


def load(path: str | Path | None, overrides: list[str] | None = None) -> Config:
    """A TOML config file, or a run's saved ``config.json``, plus ``--set`` overrides."""
    data: dict[str, Any] = {}
    if path is not None and str(path).endswith(".json"):
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
    return from_dict(data)
