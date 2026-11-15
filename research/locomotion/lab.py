"""A test bench for tail-first locomotion: the GrowBot body with chosen contact materials
on each surface, driven by an open-loop gait through the same relay and servo model the
trainer uses, scored on how fast it travels toward the camera (the tail end).

Each surface carries its own material (growbot_train/surfaces.py), so a dipped foot and
a bare tail can differ. The bench can also give a surface a ratchet (directional)
friction, the way angled bristles or a climbing-skin strip slide one way and bite the
other, which the training env doesn't model. Open-loop gaits here run hundreds of times
faster than PPO training, so this is where to compare body changes.
"""

from __future__ import annotations

import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from growbot_train.surfaces import (  # noqa: E402
    FLOORS,
    SURFACES,
    Ratchet,
    split_contacts,
    surface_geoms,
)

EXPORTS = (FLOORS, SURFACES, Ratchet)  # re-exported for search.py and the tests
BODIES = HERE.parents[1] / "growbot_train" / "bodies"
# The printed build with a phone on top (make_body.py); "rollout" is upstream's sim body.
DEFAULT_BODY = "build"
CONTROL_DT = 0.04  # the app runner's 25 Hz tick
DELAY_SLOTS = 6

# Nose-up pitch that counts as going over. The legs brace it from behind, so it stands
# taller than it rests; past vertical it topples forward onto the screen. A little short
# of vertical, for margin.
FALL_PITCH_DEG = 85.0


@dataclass
class Scenario:
    name: str
    surfaces: dict[str, str]  # surface -> material name in FLOORS
    note: str = ""
    # Servo travel either side of straight out, radians. The MG90S swings 90 degrees each
    # way; the trained policies use 1.0 (task.action_to_rad), which caps how tall it stands.
    swing_rad: float = 1.5
    # Camera height: the average pitch (degrees nose-up) the gait must hold, 0 for none.
    min_pitch_deg: float = 0.0
    # The other way: the average pitch it must stay under (a low waddle), 0 for none...
    max_pitch_deg: float = 0.0
    # ...and the highest it may rear on any step (an unchecked n1 vaulted to 59), 0 for none.
    peak_pitch_deg: float = 0.0
    # Which way counts as forward: +1 tail (camera) first, -1 nose first.
    direction: float = 1.0
    # Leg angle (radians) held while it is set down, and the gait eases in from. Long
    # rocker legs at 0 tip it onto its nose; tucked they let it sit on its tail.
    rest_leg_rad: float = 0.0
    # Foot shape (a reprinted leg): "flat" (stock), "round" (a half-round tip), "bevel"
    # (the sole cut at foot_deg; positive leaves the nose-side edge lowest, negative the
    # tail-side edge), or "toe_nose" / "toe_tail" (an L: a toe sticking out foot_deg mm
    # that way along the floor, the heel half rounded). leg_extra_m lengthens the leg.
    # "cam" is a curved sole spanning foot_deg either side of the leg's line, its distance
    # from the axle growing cam_m_per_rad toward the nose (negative: toward the tail), so
    # the leg is longer or shorter depending on its angle to the ground; at 0 it is an arc
    # around the axle, a wheel rim. "edge_out" / "edge_in" (Jeff): the foot's outer (or
    # inner) 4 mm, across its thickness, chamfered up by foot_deg so it only touches when
    # the body rolls toward that leg; for these "toe_nose" is that edge strip's material
    # and "toe_tail" the main sole's.
    foot: str = "flat"
    foot_deg: float = 0.0
    leg_extra_m: float = 0.0
    cam_m_per_rad: float = 0.0
    # Where a cam foot's sole changes from its tail-side to its nose-side material, as an
    # angle from the leg's line (degrees, + toward the nose); 0 splits it down the middle.
    split_deg: float = 0.0


@dataclass
class Domain:
    """One draw of the robot and floor. `Domain()` is the as-designed robot."""

    mass_scale: float = 1.0
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    leg_scale: tuple[float, float] = (1.0, 1.0)
    gain: float = 1.0
    slew: tuple[float, float] = (9.6, 9.6)
    lag: tuple[float, float] = (0.025, 0.025)
    delay_s: float = 0.04
    drop_prob: float = 0.0
    floor: str = "hard"
    floor_scale: float = 1.0
    surface_scale: dict[str, float] = field(default_factory=lambda: dict.fromkeys(SURFACES, 1.0))
    seed: int = 0


def random_domain(rng: np.random.Generator, floor: str = "hard", com_x: float = 0.03) -> Domain:
    """The trainer's randomisation ranges (configs/camera_walk.toml), plus each surface's
    friction drawn on its own, since a coating and a tape don't drift together."""
    shared = rng.uniform(0.85, 1.15)
    return Domain(
        mass_scale=rng.uniform(0.80, 1.25),
        com=(rng.uniform(-com_x, com_x), rng.uniform(-0.015, 0.015), rng.uniform(-0.01, 0.015)),
        leg_scale=(shared * rng.uniform(0.96, 1.04), shared * rng.uniform(0.96, 1.04)),
        gain=rng.uniform(0.75, 1.25),
        slew=tuple(rng.uniform(8.7, 10.5, 2)),
        lag=tuple(rng.uniform(0.015, 0.035, 2)),
        delay_s=rng.uniform(0.02, 0.09),
        drop_prob=0.08,
        floor=floor,
        floor_scale=rng.uniform(0.7, 1.3),
        surface_scale={s: rng.uniform(0.7, 1.3) for s in SURFACES},
        seed=int(rng.integers(1 << 30)),
    )


def domains(
    n: int, seed: int, floors: tuple[str, ...] = ("hard", "carpet"), com_x: float = 0.03
) -> list[Domain]:
    """n draws, alternating floors. `com_x` is the fore-aft centre-of-mass spread (m):
    the trainer's 3 cm, or less for a build whose weights are known."""
    rng = np.random.default_rng(seed)
    return [random_domain(rng, floors[i % len(floors)], com_x) for i in range(n)]


def _sole(foot: str, deg: float, hx: float, reach: float, x: float) -> float:
    """Height of the sole under the point x (along the leg's width, + toward the nose)."""
    if foot == "flat":
        return -reach
    if foot == "round":
        return -reach + hx - math.sqrt(max(0.0, hx * hx - x * x))
    if foot == "bevel":
        t = math.tan(math.radians(abs(deg)))
        return -reach + ((hx - x) if deg > 0 else (x + hx)) * t
    raise ValueError(f"unknown foot {foot!r}")


TOE_THICK_M = 0.008


def _half_profile(
    foot, deg, hx, reach, x0, x1, half, cam=0.0, split_deg=0.0
) -> list[tuple[float, float]]:
    """The outline (x, z) of one half of the leg, from the axle down; its convex hull is
    the contact shape (so an L toe gets a gusset back up to the leg, as printed)."""
    if foot == "cam":
        a = math.radians(deg)
        # Each piece keeps at least a degree of sole so neither collapses to a sliver.
        sp = max(-a + 0.0175, min(a - 0.0175, math.radians(split_deg)))
        phis = np.linspace(-a, sp, 9) if half == "toe_tail" else np.linspace(sp, a, 9)
        arc = [((reach + cam * p) * math.sin(p), -(reach + cam * p) * math.cos(p)) for p in phis]
        return [(0.0, 0.0), *arc]
    if foot.startswith("toe_"):
        toe_side = "toe_nose" if foot == "toe_nose" else "toe_tail"
        if half == toe_side:
            sign = 1.0 if toe_side == "toe_nose" else -1.0
            tip = sign * (hx + deg / 1000.0)
            up = -reach + TOE_THICK_M
            return [(0.0, 0.0), (sign * hx, 0.0), (tip, up), (tip, -reach), (0.0, -reach)]
        foot = "round"  # the heel
    xs = np.linspace(x0, x1, 9)
    return [(x0, 0.0), (x1, 0.0)] + [(x, _sole(foot, deg, hx, reach, x)) for x in xs[::-1]]


def shape_feet(
    root: ET.Element, foot: str, deg: float, extra: float, cam: float = 0.0, split: float = 0.0
) -> None:
    """Swap each foot's two contact halves for convex prisms with the chosen sole, keeping
    the halves' names (and so their materials)."""
    if foot == "flat" and not extra:
        return
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    if foot.startswith("edge_"):
        _edge_feet(root, asset, foot, deg, extra)
        return
    for i, side in enumerate(("right_leg", "left_leg")):
        leg = root.find(f".//body[@name='{side}']")
        mass_geom = leg.find("geom[@name='lower_leg_1']" if i == 0 else "geom[@name='lower_leg_2']")
        hx, hy, hz = (float(v) for v in mass_geom.get("size").split())
        reach = 2 * hz + extra
        mass_geom.set("size", f"{hx} {hy} {reach / 2}")
        mass_geom.set("pos", f"0 0 {-reach / 2}")
        for half, (x0, x1) in (("toe_tail", (-hx, 0.0)), ("toe_nose", (0.0, hx))):
            g = leg.find(f"geom[@name='{half}_{i}']")
            leg.remove(g)
            pts = _half_profile(foot, deg, hx, reach, x0, x1, half, cam, split)
            verts = " ".join(f"{x} {y} {z}" for x, z in pts for y in (-hy, hy))
            name = f"{half}_{i}_mesh"
            ET.SubElement(asset, "mesh", name=name, vertex=verts)
            ET.SubElement(leg, "geom", name=f"{half}_{i}", type="mesh", mesh=name, mass="1e-6")


EDGE_M = 0.004


def _edge_feet(root, asset, foot: str, deg: float, extra: float) -> None:
    t = math.tan(math.radians(deg))
    for i, side in enumerate(("right_leg", "left_leg")):
        leg = root.find(f".//body[@name='{side}']")
        mass_geom = leg.find(f"geom[@name='lower_leg_{i + 1}']")
        hx, hy, hz = (float(v) for v in mass_geom.get("size").split())
        reach = 2 * hz + extra
        mass_geom.set("size", f"{hx} {hy} {reach / 2}")
        mass_geom.set("pos", f"0 0 {-reach / 2}")
        out = -1.0 if i == 0 else 1.0  # the right leg sits at -y
        sign = out if foot == "edge_out" else -out
        edge0 = sign * (hy - EDGE_M)  # where the chamfer starts, across the thickness
        main = [(-sign * hy, 0.0), (edge0, 0.0), (edge0, -reach), (-sign * hy, -reach)]
        # Half a millimetre proud of the sole at its inner side, so level it stays clear.
        lift = 0.0005
        strip = [(edge0, 0.0), (sign * hy, 0.0), (sign * hy, -reach + lift + EDGE_M * t),
                 (edge0, -reach + lift)]  # fmt: skip
        for name, poly in ((f"toe_tail_{i}", main), (f"toe_nose_{i}", strip)):
            leg.remove(leg.find(f"geom[@name='{name}']"))
            verts = " ".join(f"{x} {y} {z}" for y, z in poly for x in (-hx, hx))
            ET.SubElement(asset, "mesh", name=f"{name}_mesh", vertex=verts)
            ET.SubElement(leg, "geom", name=name, type="mesh", mesh=f"{name}_mesh", mass="1e-6")


def build_xml(body: str, dom: Domain, torque_nm: float, scen: Scenario | None = None) -> str:
    path = HERE / body if body.endswith(".xml") else BODIES / f"{body}.xml"
    root = ET.fromstring(path.read_text())
    floor = root.find("worldbody/geom[@name='floor']")
    fl = FLOORS[dom.floor]
    # A contact's softness is the mix of its two geoms'; give every geom the floor's.
    root.find("default/geom").set("solref", " ".join(map(str, fl.solref)))
    root.find("default/geom").set("solimp", " ".join(map(str, fl.solimp)))
    floor.set("solref", " ".join(map(str, fl.solref)))
    floor.set("solimp", " ".join(map(str, fl.solimp)))
    # MG90S stall torque is about 0.2 N m; the upstream model allows 1.5.
    root.find("default/position").set("forcerange", f"{-torque_nm} {torque_nm}")
    torso = root.find(".//body[@name='base_body']")
    inertial = torso.find("inertial")
    mass = float(inertial.get("mass")) * dom.mass_scale
    inertial.set("mass", str(mass))
    ipos = np.array([float(v) for v in inertial.get("pos").split()]) + np.array(dom.com)
    inertial.set("pos", " ".join(map(str, ipos)))
    for key in ("diaginertia", "fullinertia"):
        if inertial.get(key):
            vals = [float(v) * dom.mass_scale for v in inertial.get(key).split()]
            inertial.set(key, " ".join(map(str, vals)))
    for i, side in enumerate(("right_leg", "left_leg")):
        g = torso.find(f"body[@name='{side}']/geom")
        hx, hy, hz = (float(v) for v in g.get("size").split())
        px = float(g.get("pos").split()[0])
        hz *= dom.leg_scale[i]
        g.set("size", f"{hx} {hy} {hz}")
        g.set("pos", f"{px} 0 {-hz}")
        g.set("mass", str(float(g.get("mass")) * dom.leg_scale[i] * dom.mass_scale))
    split_contacts(root)
    if scen is not None:
        shape_feet(
            root, scen.foot, scen.foot_deg, scen.leg_extra_m, scen.cam_m_per_rad, scen.split_deg
        )
    return ET.tostring(root, encoding="unicode")


class Bench:
    """One compiled robot + floor + materials; `run` plays a gait on it."""

    def __init__(self, scen: Scenario, dom: Domain, body: str = DEFAULT_BODY, torque_nm=0.25):
        self.scen, self.dom = scen, dom
        self.m = mujoco.MjModel.from_xml_string(build_xml(body, dom, torque_nm, scen))
        self.m.actuator_gainprm[:, 0] *= dom.gain
        self.m.actuator_biasprm[:, 1:3] *= dom.gain
        self.d = mujoco.MjData(self.m)
        self.torso = self.m.body("base_body").id
        self.geoms = surface_geoms(self.m)
        self.ratchets: dict[int, tuple[float, float]] = {}
        for s, gids in self.geoms.items():
            k = dom.surface_scale[s] * dom.floor_scale
            material = FLOORS[dom.floor].materials[scen.surfaces[s]]
            for g in gids:
                if isinstance(material, Ratchet):
                    self.ratchets[g] = (material.free * k, material.grip * k)
                    self.m.geom_friction[g, 0] = material.grip * k
                else:
                    self.m.geom_friction[g, 0] = material * k
        self._force = np.zeros(6)
        self.start_pitch_deg = 25.0

    def _update_ratchets(self) -> None:
        """Pick each ratchet surface's friction for the next step from which way it slips
        tail-ward or nose-ward along the body (by velocity, or by the friction force when
        it is near standstill)."""
        m, d = self.m, self.d
        tail_dir = -d.xmat[self.torso].reshape(3, 3)[:, 0]
        tail_dir[2] = 0.0
        n = np.linalg.norm(tail_dir)
        if n < 1e-6:
            return
        tail_dir /= n
        want: dict[int, bool] = {}
        vel = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g = c.geom1 if c.geom1 in self.ratchets else c.geom2
            if g not in self.ratchets:
                continue
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, g, vel, 0)
            r = c.pos - d.geom_xpos[g]
            v = vel[3:] + np.cross(vel[:3], r)
            slip = float(v @ tail_dir)
            if abs(slip) > 1e-3:
                free = slip > 0
            else:
                mujoco.mj_contactForce(m, d, i, self._force)
                frame = c.frame.reshape(3, 3)
                f_world = frame.T @ self._force[:3]
                if c.geom2 != g:  # force is on geom2; flip to act on g
                    f_world = -f_world
                # Friction pushing the surface nose-ward means it is being driven tail-ward.
                free = float(f_world @ tail_dir) < 0
            want[g] = want.get(g, True) and free
        for g, (lo, hi) in self.ratchets.items():
            if g in want:
                m.geom_friction[g, 0] = lo if want[g] else hi

    def reset(self) -> None:
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        leg_top = m.body_pos[m.body("right_leg").id, 2]
        half = max(m.geom_size[m.geom(f"lower_leg_{i}").id, 2] for i in (1, 2))
        # Lowered tail first, as the robot is set down: the printed build's centre of mass
        # sits nearly over its feet, so dropped level it can balance on its soles, which
        # the real one doesn't hold once a leg moves.
        pitch = math.radians(self.start_pitch_deg)
        d.qpos[3:7] = [math.cos(pitch / 2), 0.0, -math.sin(pitch / 2), 0.0]
        d.qpos[2] = 2 * half - leg_top + 0.003 + 0.09 * math.sin(pitch)
        rest = self.scen.rest_leg_rad if self.scen else 0.0
        if rest:
            # Held with tucked legs: lift clear of the floor first so the long feet
            # don't start buried in it.
            d.qpos[2] += 0.03
            d.qpos[7:9] = rest
            d.ctrl[:] = rest
        for _ in range(round(1.5 / m.opt.timestep)):
            self._step()

    def _step(self) -> None:
        if self.ratchets:
            self._update_ratchets()
        mujoco.mj_step(self.m, self.d)

    def run(self, gait, seconds: float = 8.0, ramp_s: float = 1.0) -> dict[str, float]:
        """Play `gait(t) -> (right, left)` radians from the settled stance. Distances are
        along and across the tail direction the robot started with."""
        m, d, dom = self.m, self.d, self.dom
        self.reset()
        rng = np.random.default_rng(dom.seed)
        xm = d.xmat[self.torso].reshape(3, 3)
        fwd = -xm[:2, 0] / np.linalg.norm(xm[:2, 0])
        left = np.array([-fwd[1], fwd[0]])
        p0 = d.qpos[:2].copy()
        yaw0 = math.atan2(xm[1, 0], xm[0, 0])
        up0 = xm[2].copy()
        p_start = math.degrees(math.atan2(up0[0], up0[2]))
        if p_start > FALL_PITCH_DEG or abs(math.degrees(math.asin(up0[1]))) > 45:
            # This draw of the robot can't even sit on its tail (a far-back centre of mass
            # tips it over when set down): no gait's fault, so not counted.
            return dict.fromkeys(
                (
                    "v_fwd",
                    "v_side",
                    "yaw_deg_s",
                    "max_tilt",
                    "max_pitch",
                    "mean_pitch",
                    "lifted",
                    "fell",
                ),
                0.0,
            ) | {"bad_start": 1.0}
        motor = np.full(2, self.scen.rest_leg_rad)
        queue = np.full((DELAY_SLOTS, 2), self.scen.rest_leg_rad)
        slew = np.array(dom.slew) * CONTROL_DT
        alpha = CONTROL_DT / (np.array(dom.lag) + CONTROL_DT)
        idx = min(dom.delay_s / CONTROL_DT, DELAY_SLOTS - 1.001)
        i0, frac = int(idx), idx - int(idx)
        sub = round(CONTROL_DT / m.opt.timestep)
        n_ticks = round(seconds / CONTROL_DT)
        max_tilt = max_pitch = pitch_sum = 0.0
        pitch_n = lifted = 0
        body = set(self.geoms["tail"]) | set(self.geoms["belly"])
        fell = False
        lim = self.scen.swing_rad
        for k in range(n_ticks):
            t = k * CONTROL_DT
            rest = self.scen.rest_leg_rad
            cmd = np.clip(np.asarray(gait(t), dtype=float), -lim, lim)
            cmd = rest + (cmd - rest) * min(1.0, t / ramp_s)
            if rng.random() < dom.drop_prob:
                cmd = queue[0]
            queue = np.concatenate([cmd[None], queue[:-1]])
            delayed = (1 - frac) * queue[i0] + frac * queue[i0 + 1]
            motor = (1 - alpha) * motor + alpha * np.clip(delayed, motor - slew, motor + slew)
            d.ctrl[:] = motor
            for _ in range(sub):
                self._step()
            up = d.xmat[self.torso].reshape(3, 3)[2]
            tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up @ up0)))))
            pitch = math.degrees(math.atan2(up[0], up[2]))
            roll = math.degrees(math.asin(max(-1.0, min(1.0, float(up[1])))))
            max_tilt, max_pitch = max(max_tilt, tilt), max(max_pitch, pitch)
            if k * CONTROL_DT >= ramp_s:
                pitch_sum += pitch
                pitch_n += 1
                # Whole body off the floor, standing on its feet alone (a vault or a hop).
                con = d.contact[: d.ncon]
                lifted += not any(c.geom1 in body or c.geom2 in body for c in con)
            # Past vertical it goes over onto the screen; a big roll is over on its side.
            if pitch > FALL_PITCH_DEG or abs(roll) > 45 or pitch < -20:
                fell = True
                break
        dp = d.qpos[:2] - p0
        xm = d.xmat[self.torso].reshape(3, 3)
        yaw = math.atan2(xm[1, 0], xm[0, 0])
        dur = max(CONTROL_DT, (k + 1) * CONTROL_DT)
        return {
            "v_fwd": float(dp @ fwd) / dur,
            "v_side": float(dp @ left) / dur,
            "yaw_deg_s": math.degrees((yaw - yaw0 + math.pi) % (2 * math.pi) - math.pi) / dur,
            "max_tilt": max_tilt,
            "max_pitch": max_pitch,
            "mean_pitch": pitch_sum / max(1, pitch_n),
            "lifted": lifted / max(1, pitch_n),
            "fell": float(fell),
            "bad_start": 0.0,
        }


class KeyframeGait:
    """Each leg's angle traced through K evenly spaced keyframes per cycle, linearly
    interpolated and repeated at `hz`. Covers in-phase and alternating strokes, uneven
    phasing, quick-snap/slow-push (stick-slip) and small fast buzzes alike."""

    K = 8
    HZ = (0.5, 5.0)

    def __init__(self, x: np.ndarray, swing: float):
        self.hz = self.HZ[0] + (self.HZ[1] - self.HZ[0]) / (1 + math.exp(-x[0]))
        self.keys = np.tanh(x[1:].reshape(2, self.K)) * swing

    @classmethod
    def dim(cls) -> int:
        return 1 + 2 * cls.K

    def __call__(self, t: float) -> np.ndarray:
        ph = (t * self.hz % 1.0) * self.K
        i = int(ph)
        f = ph - i
        j = (i + 1) % self.K
        return (1 - f) * self.keys[:, i] + f * self.keys[:, j]

    def describe(self) -> dict:
        return {"hz": round(self.hz, 3), "keys_rad": np.round(self.keys, 3).tolist()}


# What a fall costs, in m/s of speed: more than the fastest gait earns, so no speed is
# worth toppling on some of the robots. (The first sweep used 0.05 and its fast gaits
# fell on a third to a half of the held-out robots.)
FALL_COST = 0.3


def score(
    results: list[dict[str, float]],
    min_pitch_deg: float = 0.0,
    max_pitch_deg: float = 0.0,
    direction: float = 1.0,
    peak_pitch_deg: float = 0.0,
) -> float:
    """What the search climbs: speed the scenario's way, a little drift cost (a policy
    steers that out, but a gait that mostly spins is not a walk), and falling is ruinous.
    With a height to hold, each degree the average pitch is past it costs 0.1 cm/s."""
    s = [
        direction * r["v_fwd"]
        - 0.3 * abs(r["v_side"])
        - 0.0002 * abs(r["yaw_deg_s"])
        - r["fell"] * FALL_COST
        - 0.001 * max(0.0, min_pitch_deg - r["mean_pitch"])
        - 0.001 * (max(0.0, r["mean_pitch"] - max_pitch_deg) if max_pitch_deg else 0.0)
        - 0.001 * (max(0.0, r["max_pitch"] - peak_pitch_deg) if peak_pitch_deg else 0.0)
        for r in results
    ]
    return float(np.mean(s))
