"""The simulated GrowBot as a phone and a relay chip see it, for driving the JBrain2 creature
page end to end (JBrain2's Dave simulator, ``growbot/sim/``).

Training steps the body in policy actions; the page instead sends absolute servo degrees on
the relay wire and reads the phone's own motion events and camera. So this module speaks
those: ``servos(l, r)`` takes the wire's degrees through the physical wiring (which pin is
which leg, which way each horn turns), ``motion()`` returns what the phone's
``devicemotion``/``deviceorientation`` would report, and ``camera()`` renders what its front
camera sees, a person included. Nothing here knows about the page; the harness does.

    python -m growbot_train.simbody --render-start lying --out lying.png

renders one start pose from outside (needs ``MUJOCO_GL=osmesa`` or ``egl`` headless).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from growbot_train import house
from growbot_train.env import CONTROL_DT, CTRL_LIMIT, LENS_OFFSET_M, body_xml

ASSETS = Path(__file__).parent / "sim_assets"
_HAIR = "hair.png"
ROT90 = -1
# The face plate's centre in the head's frame (the photo side, turned to the robot).
FACE_OFFSET = np.array([-0.085, 0.0, 0.0])
# The nominal servo between the chip's write and the joint, as the training env's
# unrandomised body has it (env.py _draw): a slew limit and a first-order lag. The relay's
# delay is the harness's to add, since the page's own timing is real.
SLEW_RAD_S = 9.6
LAG_S = 0.025
# The phone's front camera as follow.py's CameraSpec has it: the phone lies screen up in
# portrait, so the picture is taller than wide.
CAM_W, CAM_H = 360, 480
CAM_VFOV_DEG = 75.0
STARTS = ("standing", "lying", "back")
SCENES = ("room", "house", "flat")
G = 9.81


@dataclass
class Wiring:
    """How the physical robot's servos answer the wire. The stock build: the right leg's
    horn turns with the wire's degrees, the left leg's against them (body.json's cal is
    L_SIGN -1, R_SIGN +1, which undoes exactly this). ``swap`` crosses the two pins, the
    mistake a bench test catches as a moonwalk."""

    l_sign: int = -1
    r_sign: int = 1
    l_off_deg: float = 0.0
    r_off_deg: float = 0.0
    swap: bool = False

    def joints_rad(self, l_deg: float, r_deg: float) -> np.ndarray:
        """Wire degrees -> joint radians in the model's order, (right, left)."""
        if self.swap:
            l_deg, r_deg = r_deg, l_deg
        right = self.r_sign * math.radians(r_deg - 90.0 - self.r_off_deg)
        left = self.l_sign * math.radians(l_deg - 90.0 - self.l_off_deg)
        return np.array([right, left])


@dataclass
class Person:
    """Someone in the room: where they stand, how high their face is, and how far their
    face turns from looking straight at the robot (0 = straight at it)."""

    x: float = -1.0
    y: float = 0.0
    face_z: float = 1.55
    turn_deg: float = 0.0
    # How much they tip their face down toward the robot: 1 looks right at it, 0 looks
    # straight ahead over it.
    look: float = 1.0
    present: bool = True


@dataclass
class _Servo:
    target: np.ndarray = field(default_factory=lambda: np.zeros(2))
    motor: np.ndarray = field(default_factory=lambda: np.zeros(2))
    limp: bool = True


def _scene_xml(body: str, face_png: str, scene: str = "room") -> str:
    """The body's MJCF plus a camera where the phone's front camera looks out, a room, and
    a person whose face carries the photo. The person is mocap and touches nothing, so the
    physics is the training body's own. ``scene`` "house" swaps the bare room for
    ``house.py``'s furnished two rooms, and "flat" for its hallway and four rooms; their
    walls and furniture the robot bumps into."""
    root = ET.fromstring(body_xml(body))
    torso = root.find(".//body[@name='base_body']")
    assert torso is not None
    imu = torso.find("site[@name='imu']")
    assert imu is not None
    lens_z = float(imu.get("pos", "0 0 0").split()[2]) + LENS_OFFSET_M
    # Looks along the torso's +z (a MuJoCo camera looks down its own -z). Picture up is the
    # phone's top, toward the nose; picture right is the left leg's side, as the phone's
    # unmirrored front-camera frame has it.
    ET.SubElement(
        torso,
        "camera",
        name="front",
        pos=f"0 0 {lens_z:.4f}",
        xyaxes="0 1 0 1 0 0",
        fovy=f"{CAM_VFOV_DEG}",
    )
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(1, asset)
    # A cube texture puts one picture on each face of a box ("left" is the -x face); only
    # the face plate's -x side, the one turned to the robot, carries the photo.
    ET.SubElement(
        asset,
        "texture",
        name="face",
        type="cube",
        fileleft=face_png,
        filefront=_HAIR,
        fileback=_HAIR,
        fileright=_HAIR,
        fileup=_HAIR,
        filedown=_HAIR,
    )
    ET.SubElement(asset, "material", name="face", texture="face")
    ET.SubElement(
        asset,
        "texture",
        name="wall",
        type="cube",
        builtin="checker",
        rgb1="0.82 0.80 0.74",
        rgb2="0.62 0.60 0.55",
        width="64",
        height="64",
    )
    ET.SubElement(asset, "material", name="wall", texture="wall", texrepeat="8 8")
    visual = root.find("visual")
    if visual is None:
        visual = ET.SubElement(root, "visual")
    # The room's own light: a phone camera indoors sees vertical faces lit, not black.
    ET.SubElement(visual, "headlight", ambient="0.45 0.45 0.45", diffuse="0.35 0.35 0.35")
    # Clipping planes scale with the model's extent, which the tiny robot sets: without
    # this the far walls are clipped away.
    ET.SubElement(visual, "map", znear="0.002", zfar="40")
    ET.SubElement(root, "statistic", extent="1", center="0 0 0.3")
    world = root.find("worldbody")
    assert world is not None
    if scene in house.LAYOUTS:
        house.add(root, asset, world, house.LAYOUTS[scene])
    for name, pos, size in (
        ()
        if scene in house.LAYOUTS
        else (
            ("wall_n", "0 3 1.3", "3 0.02 1.3"),
            ("wall_s", "0 -3 1.3", "3 0.02 1.3"),
            ("wall_e", "3 0 1.3", "0.02 3 1.3"),
            ("wall_w", "-3 0 1.3", "0.02 3 1.3"),
            ("ceiling", "0 0 2.6", "3 3 0.02"),
        )
    ):
        ET.SubElement(
            world,
            "geom",
            name=name,
            type="box",
            pos=pos,
            size=size,
            material="wall",
            contype="0",
            conaffinity="0",
        )
    ET.SubElement(world, "light", name="room", pos="0 0 2.4", dir="0 0 -1", diffuse="0.6 0.6 0.6")
    person = ET.SubElement(world, "body", name="person", mocap="true", pos="-1 0 0")
    # The face plate is the photo, square, about a real face's width with hair; the head
    # behind it and the body below give the picture a person's outline.
    ET.SubElement(
        person,
        "geom",
        name="face",
        type="box",
        size="0.004 0.11 0.11",
        pos=" ".join(f"{c:g}" for c in FACE_OFFSET),
        material="face",
        contype="0",
        conaffinity="0",
    )
    ET.SubElement(
        person,
        "geom",
        name="head",
        type="ellipsoid",
        size="0.07 0.08 0.11",
        pos="0 0 0.01",
        rgba="0.25 0.17 0.12 1",
        contype="0",
        conaffinity="0",
    )
    # The body below only turns: the head tips to look down at the robot, the body stays upright.
    trunk = ET.SubElement(world, "body", name="person_body", mocap="true", pos="-1 0 0")
    ET.SubElement(
        trunk,
        "geom",
        name="torso",
        type="capsule",
        fromto="0.03 0 -0.32 0.03 0 -1.7",
        size="0.15",
        rgba="0.35 0.42 0.55 1",
        contype="0",
        conaffinity="0",
    )
    return ET.tostring(root, encoding="unicode")


class SimBody:
    """One simulated robot in one room, stepped by the harness in real control ticks."""

    def __init__(
        self,
        body: str = "build",
        wiring: Wiring | None = None,
        face_png: Path | None = None,
        seed: int = 0,
        scene: str = "room",
    ) -> None:
        if scene not in SCENES:
            raise ValueError(f"scene must be one of {SCENES}, not {scene!r}")
        self.scene = scene
        # The house's ground truth (rooms, walls, doors, furniture), None in the bare room.
        self.layout = house.LAYOUTS.get(scene)
        face = face_png or ASSETS / "face_stylegan.png"
        import imageio.v3 as iio

        img = iio.imread(face)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[:, :, :3]
        side = min(img.shape[:2])
        # Centre-cropped square: a cube texture's faces must be.
        y0, x0 = (img.shape[0] - side) // 2, (img.shape[1] - side) // 2
        img = img[y0 : y0 + side, x0 : x0 + side]
        # The cube's -x face shows its picture a quarter turn round from upright.
        photo = iio.imwrite("<bytes>", np.ascontiguousarray(np.rot90(img, ROT90)), extension=".png")
        # A cube texture's six pictures must be one size.
        assets = {face.name: photo, _HAIR: _flat_png((64, 44, 30), img.shape[0])}
        if scene in house.LAYOUTS:
            assets.update(house.textures(ROT90))
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(body, face.name, scene), assets)
        self.data = mujoco.MjData(self.model)
        self.wiring = wiring or Wiring()
        self.rng = np.random.default_rng(seed)
        m = self.model
        self._torso = m.body("base_body").id
        self._person = m.body_mocapid[m.body("person").id]
        self._trunk = m.body_mocapid[m.body("person_body").id]
        self._face_body = m.body("person").id
        self._accel = m.sensor_adr[m.sensor("accel").id]
        self._gyro = m.sensor_adr[m.sensor("gyro").id]
        self._substeps = max(1, round(CONTROL_DT / m.opt.timestep))
        self._gain = m.actuator_gainprm[:, 0].copy()
        self._bias = m.actuator_biasprm[:, :3].copy()
        self._renderer: mujoco.Renderer | None = None
        self._push: tuple[np.ndarray, float] | None = None
        self.servo = _Servo()
        self.person = Person()
        # A house stands him up where it says he starts; a reset with no x, y is the origin.
        if self.layout is None:
            self.reset()
        else:
            sx, sy, syaw = self.layout.start
            self.reset(yaw_deg=syaw, x=sx, y=sy)

    # ------------------------------------------------------------------ setup

    def reset(
        self, start: str = "standing", yaw_deg: float = 0.0, x: float = 0.0, y: float = 0.0
    ) -> None:
        """Put the robot down. "standing": on its feet at its resting stance, set down
        tail first as the real one is. "lying": flat on its belly, legs folded up along
        the body. "back": on its back, legs in the air. ``x``, ``y`` is where on the floor,
        ``yaw_deg`` which way its nose points (0 = +x, so the front camera, at the tail
        end, looks toward -x)."""
        if start not in STARTS:
            raise ValueError(f"start must be one of {STARTS}, not {start!r}")
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        self._push = None
        yaw = math.radians(yaw_deg)
        if start == "standing":
            pitch, roll, legs = math.radians(35.0), 0.0, np.zeros(2)
        elif start == "lying":
            pitch, roll, legs = 0.0, 0.0, np.radians([-88.0, -88.0])
        else:
            pitch, roll, legs = 0.0, math.pi, np.zeros(2)
        q = np.zeros(4)
        # Nose-up pitch is a negative turn about the body's y axis.
        # Extrinsic (world axes): roll, then pitch, then yaw about the world's vertical, so a
        # turned start keeps its stance. Intrinsic order turned the yaw about the pitched
        # body, which tipped a 90 degree start onto its side.
        mujoco.mju_euler2Quat(q, np.array([roll, -pitch, yaw]), "XYZ")
        d.qpos[0:3] = [x, y, 0.3]
        d.qpos[3:7] = q
        d.qpos[7:9] = legs
        mujoco.mj_kinematics(m, d)
        d.qpos[2] += 0.002 - self._lowest_point()
        self.servo = _Servo(target=legs.copy(), motor=legs.copy(), limp=start != "standing")
        self._hold(legs)
        # Settle: a standing robot holds neutral, a lying one rests limp, as a switched-on
        # robot does before the page sends anything.
        for _ in range(round(1.0 / CONTROL_DT)):
            self.tick()
        d.time = 0.0
        self.place_person(self.person)

    def _lowest_point(self) -> float:
        m, d = self.model, self.data
        low = math.inf
        robot = m.body_rootid[self._torso]
        for g in range(m.ngeom):
            # The robot's own solid parts only: the house's walls and furniture collide too.
            if m.body_rootid[m.geom_bodyid[g]] != robot or not m.geom_contype[g]:
                continue
            rot = d.geom_xmat[g].reshape(3, 3)
            reach = np.abs(rot[2]) @ m.geom_size[g]
            low = min(low, d.geom_xpos[g, 2] - reach)
        return low

    def place_person(self, p: Person) -> None:
        """Stand the person at (x, y), facing the robot turned by ``turn_deg``; absent,
        they are moved far behind a wall."""
        self.person = p
        d = self.data
        if not p.present:
            d.mocap_pos[self._person] = d.mocap_pos[self._trunk] = [0.0, 50.0, -10.0]
            mujoco.mj_kinematics(self.model, d)
            return
        d.mocap_pos[self._person] = d.mocap_pos[self._trunk] = [p.x, p.y, p.face_z]
        rx, ry = d.qpos[0] - p.x, d.qpos[1] - p.y
        # The face plate sits on the head's -x side, so turn -x toward the robot, and tip
        # it down to the robot on the floor, as someone looking at it does.
        yaw = math.atan2(-ry, -rx) + math.radians(p.turn_deg)
        down = math.atan2(p.face_z - d.qpos[2], max(math.hypot(rx, ry), 0.05))
        q = np.zeros(4)
        mujoco.mju_euler2Quat(q, np.array([yaw, -down * p.look, 0.0]), "zyx")
        d.mocap_quat[self._person] = q
        body_yaw = math.atan2(-ry, -rx)
        d.mocap_quat[self._trunk] = [math.cos(body_yaw / 2), 0.0, 0.0, math.sin(body_yaw / 2)]
        # So a picture taken before the next step already shows them where they now stand.
        mujoco.mj_kinematics(self.model, d)

    # ------------------------------------------------------------------ the wire

    def servos(self, l_deg: float, r_deg: float) -> None:
        """A write of both servos in wire degrees (0..180, 90 neutral)."""
        legs = self.wiring.joints_rad(
            float(np.clip(l_deg, 0.0, 180.0)), float(np.clip(r_deg, 0.0, 180.0))
        )
        self.servo.target = np.clip(legs, -CTRL_LIMIT, CTRL_LIMIT)
        if self.servo.limp:
            # A released servo picks up from wherever its leg hangs.
            self.servo.motor = self.data.qpos[7:9].copy()
            self.servo.limp = False

    def release(self) -> None:
        """Both servos go limp: no torque until the next write."""
        self.servo.limp = True

    def _hold(self, legs: np.ndarray) -> None:
        m = self.model
        if self.servo.limp:
            m.actuator_gainprm[:, 0] = 0.0
            m.actuator_biasprm[:, :3] = 0.0
            self.data.ctrl[:] = 0.0
        else:
            m.actuator_gainprm[:, 0] = self._gain
            m.actuator_biasprm[:, :3] = self._bias
            self.data.ctrl[:] = legs

    # ------------------------------------------------------------------ disturbances

    def push(self, force_n: tuple[float, float, float], secs: float = 0.1) -> None:
        """A shove on the torso, world-frame newtons, held for ``secs``."""
        self._push = (np.asarray(force_n, dtype=float), secs)

    def lift(self, height_m: float = 0.1) -> None:
        """Move the robot straight up, keeping its attitude, and let go: it drops back."""
        self.data.qpos[2] += height_m
        self.data.qvel[:] = 0.0

    # ------------------------------------------------------------------ stepping

    def tick(self) -> None:
        """One control tick (CONTROL_DT) of servo model and physics."""
        m, d, s = self.model, self.data, self.servo
        if not s.limp:
            step = SLEW_RAD_S * CONTROL_DT
            slewed = np.clip(s.target, s.motor - step, s.motor + step)
            alpha = CONTROL_DT / (LAG_S + CONTROL_DT)
            s.motor = (1 - alpha) * s.motor + alpha * slewed
        self._hold(s.motor)
        for _ in range(self._substeps):
            if self._push is not None:
                f, left = self._push
                d.xfrc_applied[self._torso, :3] = f
                self._push = (f, left - m.opt.timestep) if left > m.opt.timestep else None
            else:
                d.xfrc_applied[self._torso, :3] = 0.0
            mujoco.mj_step(m, d)
        if not self.person.present:
            return
        # The person keeps facing the robot as it moves.
        self.place_person(self.person)

    # ------------------------------------------------------------------ what the phone reads

    def _torso_rot(self) -> np.ndarray:
        return self.data.xmat[self._torso].reshape(3, 3)

    def motion(self) -> dict[str, Any]:
        """What the phone reports this instant, in the page's own conventions (JBrain2
        growbot-box.js ``bodyAccel``/``bodyGyro`` invert exactly this): device x is the
        body's right side (-y), device y the nose (+x), z out of the screen;
        ``accelerationIncludingGravity`` in m/s^2 reads +9.81 on z lying screen up, and
        ``rotationRate`` is deg/s with alpha about device x, beta about y and gamma about
        z, as the phones the walks were tuned on report it."""
        d = self.data
        a = d.sensordata[self._accel : self._accel + 3]
        w = np.degrees(d.sensordata[self._gyro : self._gyro + 3])
        accel = {"x": -float(a[1]), "y": float(a[0]), "z": float(a[2])}
        # devicemotion's `acceleration` is the same reading with gravity taken out.
        lin = a - self._torso_rot().T @ np.array([0.0, 0.0, G])
        linear = {"x": -float(lin[1]), "y": float(lin[0]), "z": float(lin[2])}
        rate = {"alpha": -float(w[1]), "beta": float(w[0]), "gamma": float(w[2])}
        # Device axes in the world, columns x, y, z.
        rb = self._torso_rot()
        rd = np.column_stack([-rb[:, 1], rb[:, 0], rb[:, 2]])
        return {"accel": accel, "linear": linear, "rate": rate, "orientation": _w3c_euler(rd)}

    def state(self) -> dict[str, Any]:
        """Where the robot is and how it sits, for scoring a run (never fed to the page)."""
        d = self.data
        rb = self._torso_rot()
        up = rb[2]  # world z in the body frame
        nose = math.degrees(math.atan2(rb[1, 0], rb[0, 0]))
        pitch = math.degrees(math.atan2(up[0], up[2]))
        roll = math.degrees(math.atan2(-up[1], up[2]))
        legs = d.qpos[7:9]
        # The page's wire degrees that would hold these legs, so a run reads in its terms.
        w = self.wiring
        r_deg = 90.0 + w.r_off_deg + math.degrees(legs[0]) / w.r_sign
        l_deg = 90.0 + w.l_off_deg + math.degrees(legs[1]) / w.l_sign
        if w.swap:
            l_deg, r_deg = r_deg, l_deg
        return {
            "t": round(float(d.time), 3),
            "x": float(d.qpos[0]),
            "y": float(d.qpos[1]),
            "z": float(d.qpos[2]),
            "nose_deg": nose,
            "pitch_deg": pitch,
            "roll_deg": roll,
            "screen_up": bool(up[2] > 0.0),
            "legs_wire_deg": [round(l_deg, 1), round(r_deg, 1)],
            "limp": bool(self.servo.limp),
            "face": self.face_in_view(),
        }

    def face_in_view(self) -> dict[str, Any]:
        """Where the person's face falls in the camera's picture (0..1 across and down)
        and how far away it is: the ground truth a detector's box is checked against."""
        if not self.person.present:
            return {"visible": False}
        m, d = self.model, self.data
        cam = m.camera("front").id
        pos = d.cam_xpos[cam]
        rot = d.cam_xmat[cam].reshape(3, 3)
        face = d.xpos[self._face_body] + d.xmat[self._face_body].reshape(3, 3) @ FACE_OFFSET
        v = rot.T @ (face - pos)
        dist = float(np.linalg.norm(face - pos))
        if v[2] >= 0:
            return {"visible": False, "dist_m": dist}
        f = 0.5 * CAM_H / math.tan(math.radians(CAM_VFOV_DEG) / 2)
        u = 0.5 + f * v[0] / -v[2] / CAM_W
        w = 0.5 - f * v[1] / -v[2] / CAM_H
        inside = bool(0.0 <= u <= 1.0 and 0.0 <= w <= 1.0)
        return {"visible": inside, "u": round(u, 3), "v": round(w, 3), "dist_m": round(dist, 3)}

    def camera(self, fmt: str = "jpeg") -> bytes:
        """The front camera's picture, encoded (``jpeg`` or ``png``)."""
        import imageio.v3 as iio

        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, CAM_H, CAM_W)
        self._renderer.update_scene(self.data, camera="front")
        rgb = self._renderer.render()
        buf = io.BytesIO()
        iio.imwrite(buf, rgb, extension=f".{'jpg' if fmt == 'jpeg' else fmt}")
        return buf.getvalue()

    def outside_view(self, azimuth: float = 135.0, elevation: float = -20.0) -> np.ndarray:
        """An onlooker's view of the robot, for reports."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, CAM_H, CAM_W)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = self.data.qpos[0:3]
        cam.distance, cam.azimuth, cam.elevation = 0.45, azimuth, elevation
        self._renderer.update_scene(self.data, cam)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def _flat_png(rgb: tuple[int, int, int], size: int) -> bytes:
    import imageio.v3 as iio

    return iio.imwrite("<bytes>", np.full((size, size, 3), rgb, dtype=np.uint8), extension=".png")


def _w3c_euler(r: np.ndarray) -> dict[str, float]:
    """``deviceorientation``'s alpha/beta/gamma (degrees, intrinsic Z-X'-Y') for a device
    whose axes, as columns, are ``r`` in a z-up world: beta in [-180, 180), gamma in
    [-90, 90), alpha in [0, 360)."""
    beta = math.atan2(r[2, 1], math.hypot(r[2, 0], r[2, 2]))
    gamma = math.atan2(-r[2, 0], r[2, 2])
    if gamma >= math.pi / 2 or gamma < -math.pi / 2:
        gamma -= math.copysign(math.pi, gamma)
        beta = math.copysign(math.pi, beta) - beta
    cb = math.cos(beta)
    if abs(cb) < 1e-9:
        alpha = math.atan2(r[1, 0], r[0, 0])
    else:
        alpha = math.atan2(-r[0, 1] / cb, r[1, 1] / cb)
    if beta >= math.pi:
        beta -= 2 * math.pi
    return {
        "alpha": math.degrees(alpha) % 360.0,
        "beta": math.degrees(beta),
        "gamma": math.degrees(gamma),
    }


# ---------------------------------------------------------------------- JSON lines


def serve(sim: SimBody, inp: Any = sys.stdin, out: Any = sys.stdout) -> None:
    """Drive a SimBody over JSON lines, one request per line, one reply per request:

        {"op": "reset", "start": "lying", "yaw_deg": 0}
        {"op": "servos", "l": 90, "r": 90}      {"op": "release"}
        {"op": "tick", "n": 1}                  -> {"motion": ..., "state": ...}
        {"op": "camera"}                        -> {"jpeg_b64": ...}
        {"op": "person", "x": 1, "y": 0, "face_z": 0.45, "turn_deg": 0, "present": true}
        {"op": "push", "force": [0, 2, 0], "secs": 0.1}     {"op": "lift", "m": 0.1}
        {"op": "wiring", "swap": true, ...}

    so a harness in any language steps the same body the policies trained on."""
    import base64

    for line in inp:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            op = req.get("op")
            reply: dict[str, Any] = {"ok": True}
            if op == "reset":
                sim.reset(
                    req.get("start", "standing"),
                    float(req.get("yaw_deg", 0.0)),
                    float(req.get("x", 0.0)),
                    float(req.get("y", 0.0)),
                )
                reply["state"] = sim.state()
            elif op == "servos":
                sim.servos(float(req["l"]), float(req["r"]))
            elif op == "release":
                sim.release()
            elif op == "tick":
                for _ in range(int(req.get("n", 1))):
                    sim.tick()
                reply["motion"] = sim.motion()
                reply["state"] = sim.state()
            elif op == "camera":
                reply["jpeg_b64"] = base64.b64encode(sim.camera()).decode()
            elif op == "person":
                fields = {
                    k: req[k]
                    for k in ("x", "y", "face_z", "turn_deg", "look", "present")
                    if k in req
                }
                sim.place_person(Person(**fields))
            elif op == "push":
                sim.push(tuple(req["force"]), float(req.get("secs", 0.1)))
            elif op == "lift":
                sim.lift(float(req.get("m", 0.1)))
            elif op == "wiring":
                fields = {
                    k: req[k]
                    for k in ("l_sign", "r_sign", "l_off_deg", "r_off_deg", "swap")
                    if k in req
                }
                sim.wiring = Wiring(**{**sim.wiring.__dict__, **fields})
            else:
                reply = {"ok": False, "error": f"unknown op {op!r}"}
        except Exception as e:  # one bad request (or no GL for a camera) answers, never kills it
            reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(reply) + "\n")
        out.flush()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--body", default="build")
    ap.add_argument("--serve", action="store_true", help="JSON lines on stdin/stdout")
    ap.add_argument("--scene", choices=SCENES, default="room")
    ap.add_argument("--render-start", choices=STARTS, help="render one start pose and exit")
    ap.add_argument("--out", default="start.png")
    ap.add_argument("--camera", action="store_true", help="render the front camera instead")
    args = ap.parse_args(argv)
    sim = SimBody(args.body, scene=args.scene)
    if args.serve:
        serve(sim)
        return
    if args.render_start:
        import imageio.v3 as iio

        sim.reset(args.render_start)
        print(json.dumps(sim.state()))
        img = sim.outside_view() if not args.camera else iio.imread(sim.camera("png"))
        iio.imwrite(args.out, img)
        sim.close()


if __name__ == "__main__":
    main()
