"""The heading-following walk task: a MuJoCo GrowBot that learns to walk slowly toward a
target direction it is told about as one number, the heading error.

What the policy sees is exactly what the phone on the real robot can supply, frame by
frame, newest first (the JBrain2 app's ``growbot_runner.js`` builds the same vector):

    accel_x, accel_y, accel_z      accelerometer incl. gravity, m/s^2, body frame, z up
    gyro_x, gyro_y, gyro_z         rad/s, body frame
    prev_action_0, prev_action_1   the action sent last tick (right leg, left leg)
    cmd_yaw                        heading error / 180, in [-1, 1]
    cmd_forward                    speed command in [-1, 1] (task.speed = "command" only)
    cmd_pitch                      camera height command in [-1, 1] (task.tilt_command)
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
from growbot_train.surfaces import FLOORS, SURFACES, split_contacts, surface_geoms

BODIES_DIR = Path(__file__).parent / "bodies"
# How far (flat) the lens axis must reach out over the tail end before the heading is
# the lens's alone; below it the heading fades into the tail end's. The lowest look
# stance reaches about 0.38.
LENS_BLEND = 0.35
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
SPEED_MODES = ("cap", "command", "still")


def channels_for(cfg: Config) -> tuple[str, ...]:
    """The input layout for this config: the speed command rides the app runner's
    cmd_forward channel and the camera height its cmd_pitch, right after cmd_yaw, only
    when the policy takes them."""
    extra = []
    if cfg.task.speed == "command":
        extra.append("cmd_forward")
    if cfg.task.tilt_command:
        extra.append("cmd_pitch")
    i = CHANNELS.index("cmd_yaw") + 1
    return (*CHANNELS[:i], *extra, *CHANNELS[i:])


ACTION_LIMIT = 4.0  # pre-squash bound; tanh(4) = 0.9993, so the legs can reach full swing
CTRL_LIMIT = 1.57
DELAY_SLOTS = 6


def wrap_deg(x: float) -> float:
    return (x + 180.0) % 360.0 - 180.0


def _pitch_deg(up_in_body: np.ndarray) -> float:
    """Nose-up pitch of the body, from world up expressed in the body frame."""
    return math.degrees(math.atan2(up_in_body[0], up_in_body[2]))


def body_xml(name: str, split: bool = False, torque_nm: float = 0.0) -> str:
    """The body's MJCF with an IMU site and its two sensors added to the torso; with
    `split`, contact surfaces split out for per-surface materials (surfaces.py), and with
    `torque_nm`, the servos' torque limited to it."""
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
    if torso.find("site[@name='imu']") is None:  # the build body places its own, in the phone
        ET.SubElement(torso, "site", name="imu", pos="0 0 0.01", size="0.005")
    sensor = root.find("sensor")
    if sensor is None:
        sensor = ET.SubElement(root, "sensor")
    ET.SubElement(sensor, "accelerometer", name="accel", site="imu")
    ET.SubElement(sensor, "gyro", name="gyro", site="imu")
    if split:
        split_contacts(root)
    if torque_nm:
        root.find("default/position").set("forcerange", f"{-torque_nm} {torque_nm}")
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
    sensor_delay_s: float
    trim: np.ndarray
    imu_rot: np.ndarray
    gyro_bias: np.ndarray
    heading_drift_deg_s: float
    heading_refresh: int


class GrowBotHeadingEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, cfg: Config, render_mode: str | None = None) -> None:
        self.cfg = cfg
        self.render_mode = render_mode
        self._split = bool(cfg.task.surfaces)
        if self._split:
            self._check_surfaces()
        self.model = mujoco.MjModel.from_xml_string(
            body_xml(cfg.task.body, self._split, cfg.task.servo_torque_nm)
        )
        if self._split:
            # Each leg is three geoms in split mode, so MuJoCo prunes its contacts with
            # a per-body box fixed at compile time; a leg lengthened per episode would
            # poke through it and sink into the floor. The bodies are few: check them all.
            self.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_MIDPHASE
        self.data = mujoco.MjData(self.model)
        self.n_substeps = max(1, round(CONTROL_DT / self.model.opt.timestep))
        self.max_steps = round(cfg.task.episode_seconds / CONTROL_DT)
        m = self.model
        if cfg.task.facing not in ("nose", "camera"):
            raise ValueError(f'task.facing must be "nose" or "camera", not {cfg.task.facing!r}')
        self._camera_first = cfg.task.facing == "camera"
        if cfg.task.heading_from not in ("body", "lens"):
            raise ValueError(
                f'task.heading_from must be "body" or "lens", not {cfg.task.heading_from!r}'
            )
        if cfg.task.heading_from == "lens" and not self._camera_first:
            raise ValueError('task.heading_from "lens" needs task.facing "camera"')
        self._lens = cfg.task.heading_from == "lens"
        if cfg.task.speed not in SPEED_MODES:
            raise ValueError(f"task.speed must be one of {SPEED_MODES}, not {cfg.task.speed!r}")
        self.channels = channels_for(cfg)
        self._torso = m.body("base_body").id
        self._legs = [m.body("right_leg").id, m.body("left_leg").id]
        self._leg_geoms = [m.geom("lower_leg_1").id, m.geom("lower_leg_2").id]
        # Every geom of each leg (the mass box, plus the contact halves when split), so a
        # longer or shorter leg moves them all.
        self._leg_all = [[g for g in range(m.ngeom) if m.geom_bodyid[g] == b] for b in self._legs]
        self._scaled_geoms = {g for leg in self._leg_all for g in leg}
        self._surface_geoms = surface_geoms(m) if self._split else {}
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
                "geom_solref",
                "geom_solimp",
                "actuator_gainprm",
                "actuator_biasprm",
                "actuator_forcerange",
            )
        }
        if self._split:
            self._apply_body(self._nominal_body())
        self._nominal_rest_pitch = self._settled_pitch()
        self.frames = cfg.task.history_frames
        n_obs = len(self.channels) * self.frames
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (n_obs,), np.float32)
        self.action_space = gym.spaces.Box(-ACTION_LIMIT, ACTION_LIMIT, (2,), np.float32)
        self._renderer: mujoco.Renderer | None = None
        self.target_deg = 0.0
        # The curriculum's share of the full randomisation, set by the trainer as it goes
        # (randomization.ramp_steps); 1 is the full ranges. It narrows how far off the
        # target starts and jumps too, so an early gait that only walks straight pays.
        self.dr_scale = 1.0
        # When set, scales the target ranges in place of dr_scale: a steered clone wants
        # targets all the way round while the bodies are still near the as-built one.
        self.target_scale: float | None = None

    def _settled_pitch(self) -> float:
        """Resting pitch of the as-built body on neutral legs. A randomised body can rest
        far from it (a nose-heavy one may settle flat, then sit back up after a bump), so
        anything that must not move with that uses this instead."""
        m = self.model
        d = mujoco.MjData(m)
        leg_top = m.body_pos[self._legs[0], 2]
        half = float(m.geom_size[self._leg_geoms[0], 2])
        self._place(d, 2 * half - leg_top + 0.003, np.array([1.0, 0, 0, 0]))
        for _ in range(round(self._settle_s() / m.opt.timestep)):
            mujoco.mj_step(m, d)
        return _pitch_deg(d.xmat[self._torso][6:9])

    def _settle_s(self) -> float:
        return 1.5 if self.cfg.task.start_pitch_deg else 1.0

    def _place(self, d: mujoco.MjData, z: float, q: np.ndarray) -> None:
        """Put the body down: level at height z, or with task.start_pitch_deg, lowered
        tail first. The printed build's weight sits nearly over its feet, so set down
        level it can balance on its soles, which the real one doesn't hold once a leg
        moves; tail first it settles onto its tail as the real one is put down."""
        pitch = math.radians(self.cfg.task.start_pitch_deg)
        if pitch:
            nose_up = np.array([math.cos(pitch / 2), 0.0, -math.sin(pitch / 2), 0.0])
            out = np.zeros(4)
            mujoco.mju_mulQuat(out, q, nose_up)
            q = out
        d.qpos[0:3] = [0.0, 0.0, z]
        d.qpos[3:7] = q
        if pitch:
            # Tilted, the lowest point is the tail's corner, not a sole: set that just
            # clear of the floor. A drop from higher rocks it past its rest, and a body
            # with longer legs then goes right over onto its back.
            mujoco.mj_kinematics(self.model, d)
            d.qpos[2] += 0.001 - self._lowest_point(d)

    def _lowest_point(self, d: mujoco.MjData) -> float:
        m = self.model
        low = math.inf
        for g in range(m.ngeom):
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE or not m.geom_contype[g]:
                continue
            centre, half = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            if g in self._scaled_geoms:  # the aabb is compiled; legs change per episode
                centre, half = np.zeros(3), m.geom_size[g]
            rot = d.geom_xmat[g].reshape(3, 3)
            reach = np.abs(rot[2]) @ half
            low = min(low, d.geom_xpos[g, 2] + rot[2] @ centre - reach)
        return low

    def _check_surfaces(self) -> None:
        task = self.cfg.task
        if len(task.surfaces) != len(SURFACES):
            raise ValueError(f"task.surfaces needs one material each for {SURFACES}")
        if not task.floors:
            raise ValueError("task.floors needs at least one floor")
        for floor in task.floors:
            if floor not in FLOORS:
                raise ValueError(f"unknown floor {floor!r}; known: {sorted(FLOORS)}")
            for mat in task.surfaces:
                if not isinstance(FLOORS[floor].materials.get(mat), float):
                    raise ValueError(f"{mat!r} is not a plain material on {floor!r}")

    # ---------------------------------------------------------------- randomisation
    def _u(self, lo_hi: tuple[float, float]) -> float:
        return float(self.np_random.uniform(lo_hi[0], lo_hi[1]))

    def _u_if(self, lo_hi: tuple[float, float]) -> float:
        """A draw only where the range has width, so a run saved before a draw existed
        (its range backfilled to one value) keeps its random stream: same seed, same
        episodes."""
        return float(lo_hi[0]) if lo_hi[0] == lo_hi[1] else self._u(lo_hi)

    def _apply_body(self, params: dict[str, Any]) -> None:
        m, nom = self.model, self._nominal
        for k, v in nom.items():
            getattr(m, k)[:] = v
        ms = params["mass_scale"]
        m.body_mass[:] = nom["body_mass"] * ms
        m.body_inertia[:] = nom["body_inertia"] * ms
        m.body_ipos[self._torso] = nom["body_ipos"][self._torso] + params["com"]
        self._leg_half = []
        for leg, geom, s, every in zip(
            self._legs, self._leg_geoms, params["leg_scale"], self._leg_all, strict=True
        ):
            half0 = nom["geom_size"][geom, 2]
            half = half0 * s
            fw = params["foot_width"]
            hx, hy = nom["geom_size"][geom, 0] * fw, nom["geom_size"][geom, 1]
            mass = nom["body_mass"][leg] * s * ms
            for g in every:
                m.geom_size[g, 2] = half
                m.geom_pos[g, 2] = -half
                # The split mode's tail-side and nose-side halves narrow with the foot,
                # about its middle, so the foot stays under its mass.
                mid = nom["geom_pos"][geom, 0]
                m.geom_size[g, 0] = nom["geom_size"][g, 0] * fw
                m.geom_pos[g, 0] = mid + (nom["geom_pos"][g, 0] - mid) * fw
                gx, gy = m.geom_size[g, 0], nom["geom_size"][g, 1]
                m.geom_rbound[g] = math.sqrt(gx * gx + gy * gy + half * half)
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
        m.actuator_forcerange[:] = nom["actuator_forcerange"] * params["torque_scale"]
        if self._split:
            floor = FLOORS[params["floor"]]
            for surf, mat in zip(SURFACES, self.cfg.task.surfaces, strict=True):
                k = params["surface_scale"][surf] * params["floor_scale"]
                m.geom_friction[self._surface_geoms[surf], 0] = floor.materials[mat] * k
            m.geom_solref[:] = floor.solref
            m.geom_solimp[:, :3] = floor.solimp
        else:
            m.geom_friction[:, 0] = params["friction"]

    def _near(self, lo_hi: tuple[float, float], nominal: float) -> tuple[float, float]:
        """A range drawn in toward the as-built value by the curriculum (dr_scale)."""
        k = self.dr_scale
        if k >= 1.0:
            return lo_hi
        return (nominal + (lo_hi[0] - nominal) * k, nominal + (lo_hi[1] - nominal) * k)

    def _draw(self, overrides: dict[str, Any], randomize: bool) -> tuple[dict[str, Any], Episode]:
        r = self.cfg.randomization
        rng = self.np_random
        ng = self.model.ngeom
        near, k = self._near, min(1.0, self.dr_scale)
        if randomize:
            shared = self._u(near(r.leg_length_scale, 1.0))
            fr = r.friction
            body = {
                "mass_scale": self._u(near(r.mass_scale, 1.0)),
                "com": np.array([self._u(near(c, 0.0)) for c in (r.com_x_m, r.com_y_m, r.com_z_m)]),
                "leg_scale": [shared * self._u(near(r.leg_asymmetry, 1.0)) for _ in range(2)],
                "gain": self._u(near(r.servo_gain_scale, 1.0)),
                "torque_scale": self._u_if(near(r.servo_torque_scale, 1.0)),
                "foot_width": self._u_if(near(r.foot_width_scale, 1.0)),
                # One value for every geom: MuJoCo takes the larger of two touching
                # geoms' friction, so per-geom draws would skew the floor grippy.
                "friction": np.full(ng, self._u(near(fr, (fr[0] + fr[1]) / 2))),
            }
            ep = Episode(
                slew=rng.uniform(*near(r.servo_slew_rad_s, 9.6), 2),
                lag=rng.uniform(*near(r.servo_lag_s, 0.025), 2),
                delay_s=self._u(near(r.relay_delay_s, 0.04)),
                sensor_delay_s=self._u_if(near(r.sensor_delay_s, 0.0)),
                trim=(
                    rng.uniform(-r.trim_err_rad * k, r.trim_err_rad * k, 2)
                    if r.trim_err_rad
                    else np.zeros(2)
                ),
                imu_rot=_rot(rng.uniform(-r.imu_tilt_rad * k, r.imu_tilt_rad * k, 3)),
                gyro_bias=rng.uniform(-r.gyro_bias_rad_s * k, r.gyro_bias_rad_s * k, 3),
                heading_drift_deg_s=self._u(
                    (-r.heading_drift_deg_per_s * k, r.heading_drift_deg_per_s * k)
                ),
                heading_refresh=int(
                    rng.integers(r.heading_refresh_ticks[0], r.heading_refresh_ticks[1] + 1)
                ),
            )
            if self._split:
                # Each surface drifts on its own (a coating and bare print don't wear
                # together), and the whole floor on top: dust, polish, pile.
                floors = self.cfg.task.floors
                body["floor"] = floors[int(rng.integers(len(floors)))]
                body["surface_scale"] = {
                    s: self._u(near(r.surface_friction_scale, 1.0)) for s in SURFACES
                }
                body["floor_scale"] = self._u(near(r.floor_friction_scale, 1.0))
        else:
            body = self._nominal_body()
            ep = Episode(
                slew=np.full(2, 9.6),
                lag=np.full(2, 0.025),
                delay_s=0.04,
                sensor_delay_s=0.0,
                trim=np.zeros(2),
                imu_rot=np.eye(3),
                gyro_bias=np.zeros(3),
                heading_drift_deg_s=0.0,
                heading_refresh=1,
            )
        # A robustness sweep pins single parameters to the edge of their range.
        for k, v in overrides.items():
            if k == "floor_scale" and self._split:
                body["floor_scale"] = float(v)
            elif k == "friction":
                body["friction"] = np.full(ng, float(v))
            elif k in body:
                body[k] = v
            elif hasattr(ep, k):
                setattr(ep, k, v)
            else:
                raise KeyError(f"unknown override {k!r}")
        return body, ep

    def _nominal_body(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "mass_scale": 1.0,
            "com": np.zeros(3),
            "leg_scale": [1.0, 1.0],
            "gain": 1.0,
            "torque_scale": 1.0,
            "foot_width": 1.0,
            "friction": self._nominal["geom_friction"][:, 0].copy(),
        }
        if self._split:
            body["floor"] = self.cfg.task.floors[0]
            body["surface_scale"] = dict.fromkeys(SURFACES, 1.0)
            body["floor_scale"] = 1.0
        return body

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
        self._place(d, 2 * max(self._leg_half) - leg_top + 0.003, q)
        d.qpos[7:9] = self.np_random.uniform(-0.05, 0.05, 2)
        # The legs hold where a mis-trimmed robot's do, so the stance it settles into, and
        # levels on, is its own.
        d.ctrl[:] = np.clip(self.ep.trim, -CTRL_LIMIT, CTRL_LIMIT)
        for _ in range(round(self._settle_s() / m.opt.timestep)):  # settle onto the feet
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
        self._fixed_tilt = options.get("tilt_cmd")
        self.tilt_cmd = self._draw_tilt()
        # Timers only for the inputs a config uses, so older configs keep their random
        # stream (same seed, same episodes as before these inputs existed).
        if self.cfg.task.tilt_command:
            self._next_tilt = self._u(self.cfg.task.tilt_command_every_s)
        self._set_stance()
        # A folded start: legs tucked, body lying nearly flat. The levelling above was
        # already done on neutral legs, as the app does, so this only changes the pose.
        prob = self.cfg.task.lying_start_prob
        lying = prob > 0 and self.np_random.random() < prob
        legs = np.zeros(2)
        if lying:
            legs[:] = self._u(self.cfg.task.lying_leg_rad)
            d.ctrl[:] = np.clip(legs + self.ep.trim, -CTRL_LIMIT, CTRL_LIMIT)
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
            lim = self.cfg.task.initial_error_deg * self._target_k()
            first_err = float(self.np_random.uniform(-lim, lim))
        self.target_deg = wrap_deg(self.yaw_deg() + first_err)
        self.retarget = bool(options.get("retarget", True))
        self._next_retarget = self._u(self.cfg.task.retarget_every_s)
        pushes = self.cfg.randomization.push_every_s
        self._next_push = self._u(pushes) if pushes[1] > 0 else 0.0
        self._err_obs = 0.0
        self._observe_heading(force=True)
        # "speed_cmd" pins the speed command for the whole episode (evaluation).
        self._fixed_speed = options.get("speed_cmd")
        self.speed_cmd = self._draw_speed()
        if self.cfg.task.speed == "command":
            self._next_speed = self._u(self.cfg.task.speed_command_every_s)
        self.hist = np.zeros((self.frames, len(self.channels)))
        self._imu_queue = np.tile(np.r_[self._imu_raw()], (DELAY_SLOTS, 1))
        self._push_frame()
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg, m, d = self.cfg, self.model, self.data
        a = np.tanh(np.clip(np.asarray(action, dtype=np.float64), -ACTION_LIMIT, ACTION_LIMIT))
        self._actuate(a * cfg.task.action_to_rad)
        p0 = d.qpos[0:2].copy()
        # Headings before the step; the error is taken against the target after any
        # retarget below, so a retarget itself pays no turn reward.
        yaw0, body0 = self.yaw_deg(), self.body_yaw_deg()
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
            lim = cfg.task.retarget_max_deg * self._target_k()
            self.target_deg = wrap_deg(self.target_deg + self._u((-lim, lim)))
            self._next_retarget = t + self._u(cfg.task.retarget_every_s)

        if cfg.task.speed == "command" and t >= self._next_speed:
            self.speed_cmd = self._draw_speed()
            self._next_speed = t + self._u(cfg.task.speed_command_every_s)

        if cfg.task.tilt_command and t >= self._next_tilt:
            self.tilt_cmd = self._draw_tilt()
            self._next_tilt = t + self._u(cfg.task.tilt_command_every_s)
            self._set_stance()

        vel = (d.qpos[0:2] - p0) / CONTROL_DT
        k = min(1.0, CONTROL_DT / max(cfg.reward.speed_window_s, CONTROL_DT))
        self.vel_avg = (1 - k) * self.vel_avg + k * vel
        up = self._up_in_body()
        stance_cos = float(np.dot(up, self._up0))
        tall_cos = float(np.dot(up, self._stance_up))
        r = cfg.reward
        cap = r.speed_cap_m_s
        tr = math.radians(self.target_deg)
        v_along = float(self.vel_avg[0] * math.cos(tr) + self.vel_avg[1] * math.sin(tr))
        speed = float(np.linalg.norm(self.vel_avg))
        if cfg.task.speed == "cap":
            progress = r.progress * min(v_along, cap) / cap
            speed_miss = abs(v_along - cap)
        else:
            want = self.speed_cmd * cap * np.array([math.cos(tr), math.sin(tr)])
            miss = speed_miss = float(np.linalg.norm(self.vel_avg - want))
            if self.speed_cmd == 0.0:
                miss = max(0.0, miss - r.stop_drift_free * cap)
            gain = (float(np.linalg.norm(want)) - miss) / cap
            progress = r.progress * max(-1.0, gain)
        err_score = self.scored_err_deg()
        err0 = self.scored_err_deg(yaw0, body0)
        terms = {
            "progress": progress,
            "heading": r.heading * (1 + math.cos(math.radians(err_score))) / 2,
            "overspeed": -r.overspeed * (max(0.0, speed - cap) / cap) ** 2,
            "tilt": -r.tilt * (1 - tall_cos),
            "action_rate": -r.action_rate * float(np.sum((a - self.prev_action) ** 2)),
            "effort": -r.effort * float(np.sum(a**2)),
        }
        if r.turn:
            terms["turn"] = r.turn * (err0 - err_score)
        if r.wobble:
            # Free-joint angular velocity is in the body frame: x is roll, y is pitch.
            terms["wobble"] = -r.wobble * float(d.qvel[3] ** 2 + d.qvel[4] ** 2)
        if r.stance:
            miss = math.degrees(math.acos(max(-1.0, min(1.0, tall_cos))))
            terms["stance"] = r.stance * math.exp(-((miss / r.stance_tol_deg) ** 2))
        pitch = _pitch_deg(up)
        if r.back_lean:
            # From whichever rest is higher, so a body that happened to settle flat is not
            # charged for sitting in the ordinary stance.
            rest = max(self._rest_pitch, self._nominal_rest_pitch)
            top = cfg.task.stand_lift_deg
            if cfg.task.tilt_command:
                top = max(top, cfg.task.tilt_lift_deg[1])
            limit = rest + top + r.back_lean_margin_deg
            terms["back_lean"] = -r.back_lean * (max(0.0, pitch - limit) / 10.0) ** 2
        fell = stance_cos < math.cos(math.radians(cfg.task.fall_tilt_deg))
        # A tail-heavy body may rest steeper than the line itself; it falls only once it
        # goes clearly past where it rests.
        back_line = max(cfg.task.back_fall_pitch_deg, self._rest_pitch + 10.0)
        if cfg.task.back_fall_pitch_deg and pitch > back_line:
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
            "abs_err_deg": err_score,
            "fell": fell,
            "pitch_deg": pitch,
            # From the same base the stance reward uses, so it reads against want_lift_deg.
            "lift_deg": pitch - max(self._rest_pitch, self._nominal_rest_pitch),
            "speed_miss": speed_miss,
            "speed_cmd": self.speed_cmd,
            "tilt_cmd": self.tilt_cmd,
            "want_lift_deg": self._want_lift,
        }
        return self._obs(), reward, fell, truncated, info

    # ------------------------------------------------------------------- internals
    def _draw_tilt(self) -> float:
        if not self.cfg.task.tilt_command:
            return 1.0
        if self._fixed_tilt is not None:
            return float(self._fixed_tilt)
        if self.np_random.random() < self.cfg.task.tilt_tall_prob:
            return 1.0
        return self._u((-1.0, 1.0))

    def _set_stance(self) -> None:
        """The stance to hold: the commanded camera height, else task.stand_lift_deg."""
        task = self.cfg.task
        if task.tilt_command:
            lo, hi = task.tilt_lift_deg
            self._want_lift = lo + (self.tilt_cmd + 1.0) / 2.0 * (hi - lo)
        else:
            self._want_lift = task.stand_lift_deg
        if self._want_lift or task.tilt_command:
            # From the higher rest, as for back_lean: a body that settled flat in reset still
            # stands in the ordinary stance. Never nose-down: that is not a camera height.
            rest = max(self._rest_pitch, self._nominal_rest_pitch)
            lift = math.radians(max(0.0, rest + self._want_lift))
            self._stance_up = np.array([math.sin(lift), 0.0, math.cos(lift)])
        else:
            self._stance_up = self._up0

    def _draw_speed(self) -> float:
        task = self.cfg.task
        if task.speed == "cap":
            return 1.0
        if task.speed == "still":
            return 0.0
        if self._fixed_speed is not None:
            return float(self._fixed_speed)
        if self.np_random.random() < task.speed_stop_prob:
            return 0.0
        return self._u((task.speed_back_min, 1.0))

    def _actuate(self, cmd_rad: np.ndarray) -> None:
        """The relay and servo between the phone's command and the joint: dropped
        packets, a fractional delay, a slew-rate limit and a first-order lag."""
        ep = self.ep
        drop = self.cfg.randomization.relay_drop_prob * min(1.0, self.dr_scale)
        if self._noisy and self.np_random.random() < drop:
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
        # A mis-trimmed leg holds a little off wherever it's told to go.
        self.data.ctrl[:] = np.clip(self.motor + ep.trim, -CTRL_LIMIT, CTRL_LIMIT)

    def _up_in_body(self) -> np.ndarray:
        # World "up" expressed in the torso frame: the bottom row of the rotation matrix.
        return self.data.xmat[self._torso][6:9].copy()

    def yaw_deg(self) -> float:
        """The heading scored and fed to the policy, degrees CCW from +x: the end that
        walks first (task.facing), or with heading_from "lens" where the camera points."""
        xm = self.data.xmat[self._torso]
        nose = math.degrees(math.atan2(xm[3], xm[0]))
        tail = wrap_deg(nose + 180.0)
        if self._lens:
            return self._lens_deg(xm, tail)
        return tail if self._camera_first else nose

    def lens_dir(self) -> np.ndarray:
        """Where the phone's camera points, as a world unit vector: the torso's up axis."""
        return self.data.xmat[self._torso].reshape(3, 3)[:, 2].copy()

    def want_lens_dir(self) -> np.ndarray:
        """Where the commands ask the camera to point: the facing end on the target, the
        body pitched to the stance (the camera height input's, when the job has one). The
        lens leans toward the tail as the nose lifts, so a nose-first body's lens looks
        away from its target."""
        u = self._stance_up
        lift = math.atan2(u[0], u[2])
        back = self.target_deg if self._camera_first else self.target_deg + 180.0
        b = math.radians(back)
        flat = math.sin(lift)
        return np.array([flat * math.cos(b), flat * math.sin(b), math.cos(lift)])

    def _target_k(self) -> float:
        return min(1.0, self.dr_scale if self.target_scale is None else self.target_scale)

    def scored_err_deg(self, yaw: float | None = None, body: float | None = None) -> float:
        """The heading error the reward pays on, degrees, 0..180, now or for the given
        headings. With heading_from "lens" it is the worse of the lens and body errors: a
        roll swings the lens but never the body, so leaning toward the target earns
        nothing; the body has to turn and the camera has to end up on it."""
        yaw = self.yaw_deg() if yaw is None else yaw
        err = abs(wrap_deg(self.target_deg - yaw))
        if self._lens:
            body = self.body_yaw_deg() if body is None else body
            err = max(err, abs(wrap_deg(self.target_deg - body)))
        return err

    def body_yaw_deg(self) -> float:
        """The tail (camera_first) or nose heading, which no roll or pitch moves."""
        xm = self.data.xmat[self._torso]
        nose = math.degrees(math.atan2(xm[3], xm[0]))
        return wrap_deg(nose + 180.0) if self._camera_first else nose

    @staticmethod
    def _lens_deg(xm: np.ndarray, tail: float) -> float:
        # The lens looks along the torso's up axis, projected flat. Its share of the
        # heading fades with how far it reaches out over the tail end (c): near straight
        # up its compass direction is noise, and tipped sideways or past vertical it would
        # swing toward the nose. Fading on c rather than switching keeps the heading
        # continuous through all of those, so neither the policy's input nor the turn
        # reward ever snaps.
        tr = math.radians(tail)
        lx, ly = xm[2], xm[5]
        c = lx * math.cos(tr) + ly * math.sin(tr)
        s = min(1.0, max(0.0, c / LENS_BLEND))
        if s == 0.0:
            return tail
        return wrap_deg(tail + s * wrap_deg(math.degrees(math.atan2(ly, lx)) - tail))

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

    def _imu_late(self) -> tuple[np.ndarray, np.ndarray]:
        """The IMU reading the policy gets this tick: ep.sensor_delay_s old, between ticks
        interpolated, the way the relay's delay is."""
        acc, gyro = self._imu_raw()
        self._imu_queue = np.concatenate([np.r_[acc, gyro][None], self._imu_queue[:-1]])
        idx = min(self.ep.sensor_delay_s / CONTROL_DT, DELAY_SLOTS - 1.001)
        i0 = int(idx)
        frac = idx - i0
        late = (1 - frac) * self._imu_queue[i0] + frac * self._imu_queue[i0 + 1]
        return late[:3], late[3:]

    def _frame(self) -> np.ndarray:
        r = self.cfg.randomization
        acc, gyro = self._imu_late()
        if self._noisy:
            acc = acc + self.np_random.normal(0, r.accel_noise_m_s2, 3)
            gyro = gyro + self.ep.gyro_bias + self.np_random.normal(0, r.gyro_noise_rad_s, 3)
        phase = 2 * math.pi * self.cfg.task.gait_hz * self.step_index * CONTROL_DT
        cmd = [self._err_obs / 180.0]
        if self.cfg.task.speed == "command":
            cmd.append(self.speed_cmd)
        if self.cfg.task.tilt_command:
            cmd.append(self.tilt_cmd)
        return np.concatenate(
            [acc, gyro, self.prev_action, cmd, [math.sin(phase), math.cos(phase)]]
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
