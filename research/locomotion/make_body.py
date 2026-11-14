"""Write growbot_train/bodies/build.xml: the GrowBot as printed from upstream's STLs
(hardware/print/ at britcruise9/GrowBot), with a phone on top.

The upstream sim body is a 4.5 mm plate with legs hung 9 mm above its underside. The
printed body is a 163 x 75 mm tray closed by a backplate, about 30 mm deep, with both
MG90S servos in a cradle at mid-length, so the hip sits about 18 mm above the underside
and each leg reaches 74 mm from axle to tip. Measured from the STLs:

    body tray   163.4 x 75.4 x 25.0 mm (141 along its bottom), 38.3 cm^3, centroid 8.9 mm
                below its rim plane
    backplate   168 x 80 mm lid, 44.5 cm^3, servo axle 2.6 mm above the tray rim
    leg         21 x 13 mm bar, axle to tip 74 mm, 21.1 cm^3

Masses are estimates (PLA 1.24 g/cm^3, thin-walled parts printed nearly solid, the chunky
legs about half full), and the phone is a Pixel 10 in an OtterBox Defender Pro XT case
(254 g) centred on the backplate, camera at the nose end, batteries at the tail end as
upstream's BUILD.md says. The tray's ends are cut back at the bottom, one end more than
the other (ENDS). The bench randomises mass and centre of mass around these anyway.

    python research/locomotion/make_body.py   # rewrites growbot_train/bodies/build.xml
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

HALF = np.array([0.0817, 0.0385, 0.0149])  # the closed body's bounding box (m, half extents)
TRAY_TOP = HALF[2] - 0.0048  # tray rim, under the backplate
PLATE_HALF = (0.084, 0.040)  # the backplate overhangs the tray a little all round
# The tray's ends are cut back at the bottom (Jeff's print, 2026-11-14). Measured from the
# STL's outline along its length: (end at the bottom, end where the cut meets the wall,
# height of that meeting above the bottom), metres. One end is cut longer and shallower
# (about 30 degrees from upright) than the other (about 45). The batteries, and so the
# tail, sit at the steep end: the side wall's one opening, for the board's power switch,
# is in the shallow half (STL y -36 to -26 mm), and the board sits opposite the
# batteries (Jeff, 2026-11-14).
ENDS = {
    "shallow": (0.0715, 0.0805, 0.0156),
    "steep": (0.0686, 0.0801, 0.0115),
}
TAIL_END = "steep"
# The tail surface: the backplate's overhang, the cut and the first few mm of the bottom.
TAIL_PAD_M = round(PLATE_HALF[0] - ENDS[TAIL_END][0] + 0.005, 4)
NOSE_END = "steep" if TAIL_END == "shallow" else "shallow"
AXLE_Z = -HALF[2] + 0.0185  # servo axle height in the body frame (origin at box centre)
LEG_REACH = 0.074
LEG_Y = 0.0485  # leg centre: backplate wall at 40 mm, servo spline and horn, half the leg
LEG_MASS = 0.015
# A Pixel 10 (204 g) in an OtterBox Defender Pro XT (50 g; 163.8 x 84.1 x 14.2 mm over
# the camera bar, from Google's store listing), screen out, centred on the backplate.
PHONE = {"size": (0.1638, 0.0841, 0.0142), "mass": 0.254}

# (name, mass kg, centre xyz m, full size xyz m) in the body frame: x toward the nose
# (the end the legs push toward easily), z out of the backplate (where the phone sits).
PARTS = [
    ("tray", 0.043, (0.0, 0.0, -0.0079), (0.163, 0.075, 0.025)),
    ("backplate", 0.047, (0.0, 0.0, 0.0114), (0.168, 0.080, 0.006)),
    ("servos", 0.027, (0.0, 0.0, AXLE_Z), (0.023, 0.077, 0.012)),
    ("pico_and_driver", 0.016, (0.055, 0.0, -0.010), (0.052, 0.026, 0.012)),
    ("aa_x4_and_holder", 0.085, (-0.055, 0.0, -0.007), (0.058, 0.062, 0.015)),
    ("phone", PHONE["mass"], (0.0, 0.0, HALF[2] + PHONE["size"][2] / 2), PHONE["size"]),
]


def torso_inertial() -> tuple[float, np.ndarray, np.ndarray]:
    m = sum(p[1] for p in PARTS)
    com = sum(p[1] * np.array(p[2]) for p in PARTS) / m
    inertia = np.zeros((3, 3))
    for _, mi, c, s in PARTS:
        a, b, h = s
        box = mi / 12 * np.diag([b * b + h * h, a * a + h * h, a * a + b * b])
        r = np.array(c) - com
        inertia += box + mi * (r @ r * np.eye(3) - np.outer(r, r))
    return m, com, inertia


def _box(name: str, x0: float, x1: float, z0: float, z1: float, hy: float) -> str:
    return (
        f'      <geom name="{name}" type="box" size="{(x1 - x0) / 2:.5f} {hy:.5f} '
        f'{(z1 - z0) / 2:.5f}" pos="{(x0 + x1) / 2:.5f} 0 {(z0 + z1) / 2:.5f}" '
        'rgba="0.15 0.15 0.15 1"/>'
    )


def _cut(name: str, sign: float, end: tuple[float, float, float], hy: float) -> str:
    """A box lying on the cut face, reaching inward (into the tray), so its outer face is
    the cut and nothing pokes out of the outline. Deep enough to fill the wedge between
    the cut, the core's end and the wall above, so the model has no hollow."""
    bottom, top, height = end
    z_bot = -HALF[2]
    a = np.array([sign * bottom, z_bot])
    b = np.array([sign * top, z_bot + height])
    length = float(np.linalg.norm(b - a))
    along = (b - a) / length
    inward = np.array([-along[1], along[0]])
    if inward @ -((a + b) / 2) < 0:  # point it toward the middle of the body
        inward = -inward
    thick = 0.0042
    c = (a + b) / 2 + inward * thick
    # The box's x axis runs along the cut: rotate about y by the cut's angle.
    ang = -math.atan2(along[1], along[0])
    return (
        f'      <geom name="{name}" type="box" size="{length / 2:.5f} {hy:.5f} {thick:.5f}" '
        f'pos="{c[0]:.5f} 0 {c[1]:.5f}" euler="0 {math.degrees(ang):.4f} 0" '
        'rgba="0.15 0.15 0.15 1"/>'
    )


def shell() -> str:
    """The tray and backplate as boxes, ends cut back at the bottom."""
    hy = 0.0377
    tb, tt, th = ENDS[TAIL_END]
    nb, nt, nh = ENDS[NOSE_END]
    z_bot = -HALF[2]
    return "\n".join(
        [
            _box("torso_core", -tb, nb, z_bot, TRAY_TOP, hy),
            _box("torso_tail_wall", -tt, -tb, z_bot + th, TRAY_TOP, hy),
            _box("torso_nose_wall", nb, nt, z_bot + nh, TRAY_TOP, hy),
            _cut("torso_tail_cut", -1.0, ENDS[TAIL_END], hy),
            _cut("torso_nose_cut", 1.0, ENDS[NOSE_END], hy),
            _box("torso_plate", -PLATE_HALF[0], PLATE_HALF[0], TRAY_TOP, HALF[2], PLATE_HALF[1]),
        ]
    )


def xml() -> str:
    m, com, inertia = torso_inertial()
    full = " ".join(f"{v:.8g}" for v in (inertia[0, 0], inertia[1, 1], inertia[2, 2],
                                         inertia[0, 1], inertia[0, 2], inertia[1, 2]))  # fmt: skip
    leg_half = LEG_REACH / 2
    legs = "\n".join(
        f"""      <body name="{name}" pos="0 {sign * LEG_Y} {AXLE_Z:.5f}">
        <joint name="joint_{i}" type="hinge" axis="0 1 0" limited="true" range="-90 90"/>
        <geom name="lower_leg_{i}" type="box" size="0.0105 0.0065 {leg_half}" pos="0 0 {-leg_half}" mass="{LEG_MASS}" rgba="0.1 0.1 0.1 1"/>
      </body>"""  # noqa: E501
        for i, name, sign in ((1, "right_leg", -1), (2, "left_leg", 1))
    )
    ph = PHONE["size"]
    return f"""<mujoco model="GrowBot printed build">
  <!-- Generated by research/locomotion/make_body.py from upstream's STLs; edit that. -->
  <!-- Torso {m * 1000:.0f} g incl. phone, CoM {com[0] * 1000:+.1f} / {com[2] * 1000:+.1f} mm (x / z). -->
  <option gravity="0 0 -9.81" timestep="0.005" integrator="RK4" solver="CG" iterations="10" ls_iterations="10"/>
  <default>
    <geom friction="1.2 0.1 0.1" solref="0.005 1" solimp="0.99 0.99 0.01" condim="3"/>
    <joint damping="0.03" armature="0.002"/>
    <position kp="0.75" kv="0.05" ctrlrange="-1.57 1.57" forcerange="-1.5 1.5"/>
  </default>
  <custom>
    <!-- How far in from the backplate's end the tail runs: past the cut, onto the edge it rests on. -->
    <numeric name="tail_pad_m" data="{TAIL_PAD_M}"/>
  </custom>
  <worldbody>
    <light name="sun" pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="0 0 0.05" rgba="0.76 0.87 0.70 1"/>
    <body name="base_body" pos="0 0 0.3">
      <joint name="root_joint" type="free"/>
      <inertial pos="{com[0]:.6f} {com[1]:.6f} {com[2]:.6f}" mass="{m:.6f}" fullinertia="{full}"/>
{shell()}
      <site name="imu" pos="0 0 {HALF[2] + ph[2] / 2}" size="0.005"/>
      <geom name="phone" type="box" size="{ph[0] / 2} {ph[1] / 2} {ph[2] / 2}" pos="0 0 {HALF[2] + ph[2] / 2}" contype="0" conaffinity="0" mass="0" rgba="0.3 0.3 0.35 1"/>
{legs}
    </body>
  </worldbody>
  <actuator>
    <position name="servo_1" joint="joint_1"/>
    <position name="servo_2" joint="joint_2"/>
  </actuator>
</mujoco>
"""  # noqa: E501


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[2] / "growbot_train" / "bodies" / "build.xml"
    out.write_text(xml())
    m, com, _ = torso_inertial()
    x, z = com[0] * 1e3, com[2] * 1e3
    print(f"wrote {out}: torso {m * 1000:.0f} g, CoM x {x:+.1f} z {z:+.1f} mm")
