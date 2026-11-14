"""The heading-following walk task: a MuJoCo GrowBot that learns to walk slowly toward a
target direction it is told about as one number, the heading error.

What the policy sees is exactly what the phone on the real robot can supply, frame by
frame, newest first (the JBrain2 app's ``growbot_runner.js`` builds the same vector):

    accel_x, accel_y, accel_z      accelerometer incl. gravity, m/s^2, body frame, z up
    gyro_x, gyro_y, gyro_z         rad/s, body frame
    prev_action_0, prev_action_1   the action sent last tick (right leg, left leg)
    cmd_yaw                        heading error / 180, in [-1, 1]
    cycle:gait_sin, cycle:gait_cos a fixed-rate clock, counted in ticks from the start

Heading error is ``target - current`` in degrees, wrapped to [-180, 180]: positive means
the target is counter-clockwise of the robot's nose, seen from above (turn left).
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from growbot_train.config import Config

BODIES_DIR = Path(__file__).parent / "bodies"
CONTROL_DT = 0.04  # 25 Hz, the rate the app's runner ticks at
CHANNELS: tuple[str, ...] = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "prev_action_0",
    "prev_action_1",
    "cmd_yaw",
    "cycle:gait_sin",
    "cycle:gait_cos",
)
ACTION_LIMIT = 4.0  # pre-squash bound; tanh(4) = 0.9993, so the legs can reach full swing
CTRL_LIMIT = 1.57
DELAY_SLOTS = 6


def wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def _pitch_deg(up_in_body: np.ndarray) -> float:
    """Nose-up pitch of the body, from world up expressed in the body frame."""
    return math.degrees(math.atan2(up_in_body[0], up_in_body[2]))


def body_xml(name: str) -> str:
    """The body's MJCF with an IMU site and its two sensors added to the torso."""
    path = BODIES_DIR / f"{name}.xml"
    if not path.exists():
        known = sorted(p.stem for p in BODIES_DIR.glob("*.xml"))
        raise FileNotFoundError(f"no body {name!r}; known: {', '.join(known)}")
    root = ET.fromstring(path.read_text())
    torso = root.find(".//body[@name='base_body']")
    if torso is None:
        raise ValueError(f"{path} has no base_body")
    # The phone lies on the torso's top face; its exact spot barely changes the readings
    # of a body this small, and the mount tilt is randomised anyway.
    ET.SubElement(torso, "site", name="imu", pos="0 0 0.01", size="0.005")
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "accelerometer", name="accel", site="imu")
    ET.SubElement(sensor, "gyro", name="gyro", site="imu")
    return ET.tostring(root, encoding="unicode")


def _rot(axis_angle: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        return np.eye(3)
    k = axis_angle / angle
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def _rot_onto_z(v: np.ndarray) -> np.ndarray:
    """The smallest rotation taking direction ``v`` onto +z (the app runner's rotZto,
    inverted): it tilts, never spins about the vertical, so x stays toward the nose."""
    u = v / np.linalg.norm(v)
    axis = np.cross(u, [0.0, 0.0, 1.0])
    s = float(np.linalg.norm(axis))
    if s < 1e-9:
        return np.eye(3) if u[2] > 0 else np.diag([1.0, -1.0, -1.0])
    return _rot(axis / s * math.atan2(s, float(u[2])))


@dataclass
class Episode:
    """Per-episode draws: the randomised body and the sensor/relay imperfections."""

    slew: np.ndarray
    lag: np.ndarray
    delay_s: float
    imu_rot: np.ndarray
    gyro_bias: np.ndarray
    heading_drift_deg_s: float
    heading_refresh: int


class GrowBotHeadingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, cfg: Config, render_mode: str | None = None) -> None:
        self.cfg = cfg
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_string(body_xml(cfg.task.body))
        self.data = mujoco.MjData(self.model)
        self.n_substeps = max(1, round(CONTROL_DT / self.model.opt.timestep))
        self.max_steps = round(cfg.task.episode_seconds / CONTROL_DT)
        m = self.model
        if cfg.task.facing not in ("nose", "camera"):
            raise ValueError(f'task.facing must be "nose" or "camera", not {cfg.task.facing!r}')
        self._camera_first = cfg.task.facing == "camera"
        self._torso = m.body("base_body").id
        self._legs = [m.body("right_leg").id, m.body("left_leg").id]
        self._leg_geoms = [m.geom("lower_leg_1").id, m.geom("lower_leg_2").id]
        self._accel_adr = m.sensor_adr[m.sensor("accel").id]
        self._gyro_adr = m.sensor_adr[m.sensor("gyro").id]
        self._nominal = {
            k: getattr(m, k).copy()
            for k in (
                "body_mass",
                "body_inertia",
                "body_ipos",
                "geom_size",
                "geom_pos",
                "geom_rbound",
                "geom_friction",
                "actuator_gainprm",
                "actuator_biasprm",
            )
        }
        self.frames = cfg.task.history_frames
        n_obs = len(CHANNELS) * self.frames
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-ACTION_LIMIT, ACTION_LIMIT, (2,), np.float32)
        self._renderer: mujoco.Renderer | None = None
        self.target_deg = 0.0

    # ---------------------------------------------------------------- randomisation
    def _u(self, lo_hi: tuple[float, float]) -> float:
        return float(self.np_random.uniform(lo_hi[0], lo_hi[1]))

    def _apply_body(self, params: dict[str, Any]) -> None:
        m, nom = self.model, self._nominal
        for k, v in nom.items():
            getattr(m, k)[:] = v
        ms = params["mass_scale"]
        m.body_mass[:] = nom["body_mass"] * ms
        m.body_inertia[:] = nom["body_inertia"] * ms
        m.body_ipos[self._torso] = nom["body_ipos"][self._torso] + params["com"]
        self._leg_half = []
        for leg, geom, s in zip(self._legs, self._leg_geoms, params["leg_scale"], strict=True):
            half0 = nom["geom_size"][geom, 2]
            half = half0 * s
            hx, hy = nom["geom_size"][geom, 0], nom["geom_size"][geom, 1]
            mass = nom["body_mass"][leg] * s * ms
            m.geom_size[geom, 2] = half
            m.geom_pos[geom, 2] = -half
            m.body_mass[leg] = mass
            m.body_ipos[leg, 2] = -half
            a, b, c = 2 * hx, 2 * hy, 2 * half
            m.body_inertia[leg] = [
                mass / 12 * (b * b + c * c),
                mass / 12 * (a * a + c * c),
                mass / 12 * (a * a + b * b),
            ]
            m.geom_rbound[geom] = math.sqrt(hx * hx + hy * hy + half * half)
            self._leg_half.append(half)
        m.actuator_gainprm[:] = nom["actuator_gainprm"] * params["gain"]
        m.actuator_biasprm[:] = nom["actuator_biasprm"] * params["gain"]
        m.geom_friction[:, 0] = params["friction"]

    def _draw(self, overrides: dict[str, Any], randomize: bool) -> tuple[dict[str, Any], Episode]:
        r = self.cfg.randomization
        rng = self.np_random
        ng = self.model.ngeom
        if randomize:
            shared = self._u(r.leg_length_scale)
            body = {
                "mass_scale": self._u(r.mass_scale),
                "com": np.array([self._u(r.com_x_m), self._u(r.com_y_m), self._u(r.com_z_m)]),
                "leg_scale": [shared * self._u(r.leg_asymmetry) for _ in range(2)],
                "gain": self._u(r.servo_gain_scale),
                # One value for every geom: MuJoCo takes the larger of two touching
                # geoms' friction, so per-geom draws would skew the floor grippy.
                "friction": np.full(ng, self._u(r.friction)),
            }
            ep = Episode(
                slew=rng.uniform(*r.servo_slew_rad_s, 2),
                lag=rng.uniform(*r.servo_lag_s, 2),
                delay_s=self._u(r.relay_delay_s),
                imu_rot=_rot(rng.uniform(-r.imu_tilt_rad, r.imu_tilt_rad, 3)),
                gyro_bias=rng.uniform(-r.gyro_bias_rad_s, r.gyro_bias_rad_s, 3),
                heading_drift_deg_s=self._u(
                    (-r.heading_drift_deg_per_s, r.heading_drift_deg_per_s)
                ),
                heading_refresh=int(
                    rng.integers(r.heading_refresh_ticks[0], r.heading_refresh_ticks[1] + 1)
                ),
            )
        else:
            body = {
                "mass_scale": 1.0,
                "com": np.zeros(3),
                "leg_scale": [1.0, 1.0],
                "gain": 1.0,
                "friction": self._nominal["geom_friction"][:, 0].copy(),
            }
            ep = Episode(
                slew=np.full(2, 9.6),
                lag=np.full(2, 0.025),
                delay_s=0.04,
                imu_rot=np.eye(3),
                gyro_bias=np.zeros(3),
                heading_drift_deg_s=0.0,
                heading_refresh=1,
            )
        # A robustness sweep pins single parameters to the edge of their range.
        for k, v in overrides.items():
            if k == "friction":
                body["friction"] = np.full(ng, float(v))
            elif k in body:
                body[k] = v
            elif hasattr(ep, k):
                setattr(ep, k, v)
            else:
                raise KeyError(f"unknown override {k!r}")
        return body, ep

    # ---------------------------------------------------------------------- gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        # "nominal": the as-designed body with perfect sensors, for a clean look.
        randomize = self.cfg.randomization.enabled and not options.get("nominal", False)
        self._body_params, self.ep = self._draw(options.get("overrides", {}), randomize)
        self._apply_body(self._body_params)
        self._noisy = randomize

        m, d = self.model, self.data
        mujoco.mj_setConst(m, d)  # mass-derived constants follow the randomised body
        mujoco.mj_resetData(m, d)
        yaw0 = float(self.np_random.uniform(-math.pi, math.pi))
        tilt = self.np_random.uniform(-0.03, 0.03, 2)
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array([tilt[0], tilt[1], yaw0]), "xyz")
        leg_top = m.body_pos[self._legs[0], 2]
        d.qpos[0:3] = [0.0, 0.0, 2 * max(self._leg_half) - leg_top + 0.003]
        d.qpos[3:7] = q
        d.qpos[7:9] = self.np_random.uniform(-0.05, 0.05, 2)
        d.ctrl[:] = 0.0
        for _ in range(round(1.0 / m.opt.timestep)):  # settle onto the feet
            mujoco.mj_step(m, d)
        d.time = 0.0
        # This body rests nose-up (about 34 degrees on the nominal build), so "upright"
        # means "as it stood at the start", the same reference the app's tip check uses.
        self._up0 = self._up_in_body()
        # Level the IMU the way the app does after its stance settle: rotate the phone's
        # readings so the gravity it measures right now reads straight +z. The policy
        # then sees the same "level" whatever the body's resting pitch or the mount.
        self._level = np.eye(3)
        self._level = _rot_onto_z(self._imu_raw()[0])
        self._rest_pitch = _pitch_deg(self._up0)
        if self.cfg.task.stand_lift_deg:
            lift = math.radians(self._rest_pitch + self.cfg.task.stand_lift_deg)
            self._stance_up = np.array([math.sin(lift), 0.0, math.cos(lift)])
        else:
            self._stance_up = self._up0
        # A folded start: legs tucked, body lying nearly flat. The levelling above was
        # already done on neutral legs, as the app does, so this only changes the pose.
        prob = self.cfg.task.lying_start_prob
        lying = prob > 0 and self.np_random.random() < prob
        legs = np.zeros(2)
        if lying:
            legs[:] = self._u(self.cfg.task.lying_leg_rad)
            d.ctrl[:] = legs
            for _ in range(round(0.6 / m.opt.timestep)):
                mujoco.mj_step(m, d)
            d.time = 0.0

        self.step_index = 0
        self.motor = legs.copy()
        self.queue = np.tile(legs, (DELAY_SLOTS, 1))
        self.prev_action = np.clip(legs / self.cfg.task.action_to_rad, -1.0, 1.0)
        self.vel_avg = np.zeros(2)
        self.heading_bias = 0.0
        if "target_error_deg" in options:
            first_err = float(options["target_error_deg"])
        else:
            lim = self.cfg.task.initial_error_deg
            first_err = float(self.np_random.uniform(-lim, lim))
        self.target_deg = wrap_deg(self.yaw_deg() + first_err)
        self.retarget = bool(options.get("retarget", True))
        self._next_retarget = self._u(self.cfg.task.retarget_every_s)
        pushes = self.cfg.randomization.push_every_s
        self._next_push = self._u(pushes) if pushes[1] > 0 else 0.0
        self._err_obs = 0.0
        self._observe_heading(force=True)
        self.hist = np.zeros((self.frames, len(CHANNELS)))
        self._push_frame()
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg, m, d = self.cfg, self.model, self.data
        a = np.tanh(np.clip(np.asarray(action, dtype=np.float64), -ACTION_LIMIT, ACTION_LIMIT))
        self._actuate(a * cfg.task.action_to_rad)
        p0 = d.qpos[0:2].copy()
        for _ in range(self.n_substeps):
            mujoco.mj_step(m, d)
        self.step_index += 1
        t = self.step_index * CONTROL_DT
        self.heading_bias += self.ep.heading_drift_deg_s * CONTROL_DT

        if self._noisy and self._next_push and t >= self._next_push:
            # Free-joint angular velocity is in the body frame; nose-up is about -y.
            d.qvel[4] += self._u((-1.0, 1.0)) * cfg.randomization.push_pitch_rad_s
            self._next_push = t + self._u(cfg.randomization.push_every_s)

        if self.retarget and t >= self._next_retarget:
            lim = cfg.task.retarget_max_deg
            self.target_deg = wrap_deg(self.target_deg + self._u((-lim, lim)))
            self._next_retarget = t + self._u(cfg.task.retarget_every_s)

        vel = (d.qpos[0:2] - p0) / CONTROL_DT
        k = min(1.0, CONTROL_DT / max(cfg.reward.speed_window_s, CONTROL_DT))
        self.vel_avg = (1 - k) * self.vel_avg + k * vel
        err_true = wrap_deg(self.target_deg - self.yaw_deg())
        up = self._up_in_body()
        stance_cos = float(np.dot(up, self._up0))
        tall_cos = float(np.dot(up, self._stance_up))
        r = cfg.reward
        cap = r.speed_cap_m_s
        tr = math.radians(self.target_deg)
        v_along = float(self.vel_avg[0] * math.cos(tr) + self.vel_avg[1] * math.sin(tr))
        speed = float(np.linalg.norm(self.vel_avg))
        terms = {
            "progress": r.progress * min(v_along, cap) / cap,
            "heading": r.heading * (1 + math.cos(math.radians(err_true))) / 2,
            "overspeed": -r.overspeed * (max(0.0, speed - cap) / cap) ** 2,
            "tilt": -r.tilt * (1 - tall_cos),
            "action_rate": -r.action_rate * float(np.sum((a - self.prev_action) ** 2)),
            "effort": -r.effort * float(np.sum(a**2)),
        }
        if r.stance:
            miss = math.degrees(math.acos(max(-1.0, min(1.0, tall_cos))))
            terms["stance"] = r.stance * math.exp(-((miss / r.stance_tol_deg) ** 2))
        pitch = _pitch_deg(up)
        if r.back_lean:
            limit = self._rest_pitch + cfg.task.stand_lift_deg + r.back_lean_margin_deg
            terms["back_lean"] = -r.back_lean * (max(0.0, pitch - limit) / 10.0) ** 2
        fell = stance_cos < math.cos(math.radians(cfg.task.fall_tilt_deg))
        if cfg.task.back_fall_pitch_deg and pitch > cfg.task.back_fall_pitch_deg:
            fell = True
        if fell:
            terms["fall"] = -r.fall
        reward = float(sum(terms.values()))

        self.prev_action = a
        self._observe_heading()
        self._push_frame()
        truncated = self.step_index >= self.max_steps
        info = {
            "terms": terms,
            "v_along": v_along,
            "speed": speed,
            "abs_err_deg": abs(err_true),
            "fell": fell,
            "pitch_deg": pitch,
            "lift_deg": pitch - self._rest_pitch,
        }
        return self._obs(), reward, fell, truncated, info

    # ------------------------------------------------------------------- internals
    def _actuate(self, cmd_rad: np.ndarray) -> None:
        """The relay and servo between the phone's command and the joint: dropped
        packets, a fractional delay, a slew-rate limit and a first-order lag."""
        ep = self.ep
        if self._noisy and self.np_random.random() < self.cfg.randomization.relay_drop_prob:
            cmd_rad = self.queue[0]
        self.queue = np.concatenate([cmd_rad[None], self.queue[:-1]])
        idx = min(ep.delay_s / CONTROL_DT, DELAY_SLOTS - 1.001)
        i0 = int(idx)
        frac = idx - i0
        delayed = (1 - frac) * self.queue[i0] + frac * self.queue[i0 + 1]
        step = ep.slew * CONTROL_DT
        slewed = np.clip(delayed, self.motor - step, self.motor + step)
        alpha = CONTROL_DT / (ep.lag + CONTROL_DT)
        self.motor = (1 - alpha) * self.motor + alpha * slewed
        self.data.ctrl[:] = np.clip(self.motor, -CTRL_LIMIT, CTRL_LIMIT)

    def _up_in_body(self) -> np.ndarray:
        # World "up" expressed in the torso frame: the bottom row of the rotation matrix.
        return self.data.xmat[self._torso][6:9].copy()

    def yaw_deg(self) -> float:
        """The heading of the end that walks first (task.facing), degrees CCW from +x."""
        xm = self.data.xmat[self._torso]
        nose = math.degrees(math.atan2(xm[3], xm[0]))
        return wrap_deg(nose + 180.0) if self._camera_first else nose

    def _observe_heading(self, force: bool = False) -> None:
        if not force and self.step_index % self.ep.heading_refresh:
            return
        err = wrap_deg(self.target_deg - self.yaw_deg() - self.heading_bias)
        if self._noisy:
            err = wrap_deg(err + self.np_random.normal(0, self.cfg.randomization.heading_noise_deg))
        self._err_obs = err

    def _imu_raw(self) -> tuple[np.ndarray, np.ndarray]:
        """Accel and gyro as the (possibly crooked) phone reads them, then levelled."""
        d = self.data
        rot = self._level @ self.ep.imu_rot
        acc = rot @ d.sensordata[self._accel_adr : self._accel_adr + 3]
        gyro = rot @ d.sensordata[self._gyro_adr : self._gyro_adr + 3]
        return acc, gyro

    def _frame(self) -> np.ndarray:
        r = self.cfg.randomization
        acc, gyro = self._imu_raw()
        if self._noisy:
            acc = acc + self.np_random.normal(0, r.accel_noise_m_s2, 3)
            gyro = gyro + self.ep.gyro_bias + self.np_random.normal(0, r.gyro_noise_rad_s, 3)
        phase = 2 * math.pi * self.cfg.task.gait_hz * self.step_index * CONTROL_DT
        return np.concatenate(
            [
                acc,
                gyro,
                self.prev_action,
                [self._err_obs / 180.0, math.sin(phase), math.cos(phase)],
            ]
        )

    def _push_frame(self) -> None:
        self.hist = np.concatenate([self._frame()[None], self.hist[:-1]])

    def _obs(self) -> np.ndarray:
        return self.hist.reshape(-1).astype(np.float32)

    def render(self) -> np.ndarray | None:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 640)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.data.qpos[0:3]
        cam.distance, cam.azimuth, cam.elevation = 0.6, 135.0, -25.0
        self._renderer.update_scene(self.data, cam)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
