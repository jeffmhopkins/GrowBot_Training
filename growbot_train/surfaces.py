"""Contact materials: what each part of the robot that touches the floor is made of, and
how each material grips each kind of floor.

MuJoCo takes the larger of two touching geoms' friction, so one friction for every geom
(the original setup) means the floor sets the grip of every contact: a dipped foot and
a bare tail can't differ. With surfaces split out, the floor is frictionless and each
surface carries its own material's coefficient against the floor, so they can.

The surfaces: "tail" is the last 15 mm of the body, the edge it rests on; "belly" the
rest of the underside and the nose. Each foot is split down the middle of its width
(along the body): "toe_tail" is the half, sole and face, on the tail side of the leg,
"toe_nose" the half on the nose side, so dipping one face of a foot can be modelled.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

SURFACES = ("tail", "belly", "toe_tail", "toe_nose")
TAIL_PAD_M = 0.015


@dataclass(frozen=True)
class Ratchet:
    """Directional friction: `free` when the surface slides tail-ward, `grip` when it
    slides or is pushed nose-ward (angled bristles, a climbing-skin strip). Only the
    research bench models these; the training env takes plain materials."""

    free: float
    grip: float


Material = float | Ratchet


@dataclass(frozen=True)
class Floor:
    """A floor: each material's friction against it, and how soft it is."""

    name: str
    materials: dict[str, Material]
    # MuJoCo contact softness, (time constant s, damping ratio) and solimp. Hard floors
    # keep upstream's stiff contact; carpet pile gives a few millimetres and soaks up energy.
    solref: tuple[float, float] = (0.005, 1.0)
    solimp: tuple[float, float, float] = (0.99, 0.99, 0.01)


# Sliding friction of each material against each floor: rough mid-range figures, which
# the randomisation scales per surface and per floor because real values move a lot with
# dust, finish and pile. "rubber" is liquid dip or a rubber band. "sim" is what the
# upstream model gives every surface.
FLOORS: dict[str, Floor] = {
    "hard": Floor(
        "hard",
        {
            "ptfe": 0.12,  # PTFE (Teflon) tape or a furniture glide
            "felt": 0.25,
            "pla": 0.35,  # bare 3D print on sealed wood, laminate or a table
            "tape": 0.45,  # packing tape
            "rubber": 1.0,
            "skin": Ratchet(free=0.25, grip=0.75),
            "skin_weak": Ratchet(free=0.35, grip=0.6),
            "sim": 1.2,
        },
    ),
    "carpet": Floor(
        "carpet",
        {
            "ptfe": 0.25,
            "felt": 0.5,
            "pla": 0.45,
            "tape": 0.5,
            "rubber": 0.9,
            "skin": Ratchet(free=0.3, grip=1.0),  # the pile catches bristles harder
            "skin_weak": Ratchet(free=0.4, grip=0.8),
            "sim": 1.2,
        },
        solref=(0.02, 1.0),
        solimp=(0.6, 0.95, 0.006),
    ),
}


def split_contacts(root: ET.Element) -> None:
    """Rewrite a body MJCF in place for per-surface materials: a frictionless floor, the
    torso's contact box split into tail and belly, and each leg's box kept for its mass
    but replaced for contact by a tail-side and a nose-side half (toe_tail_<i>,
    toe_nose_<i>, i = 0 right, 1 left)."""
    root.find("worldbody/geom[@name='floor']").set("friction", "0 0 0")
    torso = root.find(".//body[@name='base_body']")
    plate = torso.find("geom[@name='torso_geom']")
    sx, sy, sz = (float(v) for v in plate.get("size").split())
    px, py, pz = (float(v) for v in (plate.get("pos") or "0 0 0").split())
    torso.remove(plate)
    half_tail = TAIL_PAD_M / 2
    ET.SubElement(
        torso, "geom", name="tail", type="box", size=f"{half_tail} {sy} {sz}",
        pos=f"{px - sx + half_tail} {py} {pz}", rgba="0.9 0.5 0.2 1",
    )  # fmt: skip
    ET.SubElement(
        torso, "geom", name="belly", type="box", size=f"{sx - half_tail} {sy} {sz}",
        pos=f"{px + half_tail} {py} {pz}", rgba="0.7 0.7 0.7 1",
    )  # fmt: skip
    for i, side in enumerate(("right_leg", "left_leg")):
        leg = torso.find(f"body[@name='{side}']")
        g = leg.find("geom")
        hx, hy, hz = (float(v) for v in g.get("size").split())
        gx, gy, gz = (float(v) for v in g.get("pos").split())
        g.set("contype", "0")
        g.set("conaffinity", "0")
        for half, sign in (("toe_tail", -1), ("toe_nose", 1)):
            ET.SubElement(
                leg, "geom", name=f"{half}_{i}", type="box", mass="1e-6",
                size=f"{hx / 2} {hy} {hz}", pos=f"{gx + sign * hx / 2} {gy} {gz}",
            )  # fmt: skip


def surface_geoms(model) -> dict[str, list[int]]:
    """Geom ids of each surface in a model built with split_contacts."""
    return {
        "tail": [model.geom("tail").id],
        "belly": [model.geom("belly").id],
        "toe_tail": [model.geom(f"toe_tail_{i}").id for i in range(2)],
        "toe_nose": [model.geom(f"toe_nose_{i}").id for i in range(2)],
    }
