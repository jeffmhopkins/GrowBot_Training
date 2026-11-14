"""Contact materials: what each part of the robot that touches the floor is made of, and
how each material grips each kind of floor.

MuJoCo takes the larger of two touching geoms' friction, so one friction for every geom
(the original setup) means the floor sets the grip of every contact: a dipped foot and
a bare tail can't differ. With surfaces split out, the floor is frictionless and each
surface carries its own material's coefficient against the floor, so they can.

The surfaces: "tail" is the last 15 mm of the body, the end it rests on (on the printed
build, the cut-back bottom edge and the face above it); "belly" the rest of the underside
and the nose. Each foot is split down the middle of its width
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


def _x_extent(g: ET.Element) -> tuple[float, float, bool]:
    """A torso box's reach along the body (min x, max x), and whether it is axis-aligned
    (only those are split; a tilted one, such as an end's cut face, goes whole by its
    centre)."""
    sx = float(g.get("size").split()[0])
    px = float((g.get("pos") or "0 0 0").split()[0])
    return px - sx, px + sx, g.get("euler") is None and g.get("quat") is None


def split_contacts(root: ET.Element) -> None:
    """Rewrite a body MJCF in place for per-surface materials: a frictionless floor, the
    torso's contact boxes split into tail (tail_<k>) and belly (belly_<k>) where they
    cross the line TAIL_PAD_M in from the tail end, and each leg's box kept for its mass
    but replaced for contact by a tail-side and a nose-side half (toe_tail_<i>,
    toe_nose_<i>, i = 0 right, 1 left)."""
    root.find("worldbody/geom[@name='floor']").set("friction", "0 0 0")
    torso = root.find(".//body[@name='base_body']")
    shell = [
        g for g in torso.findall("geom")
        if g.get("type") == "box" and g.get("contype") != "0"
    ]  # fmt: skip
    line = min(_x_extent(g)[0] for g in shell) + TAIL_PAD_M
    n = {"tail": 0, "belly": 0}

    def add(kind: str, like: ET.Element, x0: float, x1: float) -> None:
        _, sy, sz = like.get("size").split()
        _, py, pz = (like.get("pos") or "0 0 0").split()
        rgba = "0.9 0.5 0.2 1" if kind == "tail" else like.get("rgba", "0.7 0.7 0.7 1")
        ET.SubElement(
            torso, "geom", name=f"{kind}_{n[kind]}", type="box",
            size=f"{(x1 - x0) / 2} {sy} {sz}", pos=f"{(x0 + x1) / 2} {py} {pz}", rgba=rgba,
        )  # fmt: skip
        n[kind] += 1

    for g in shell:
        torso.remove(g)
        x0, x1, aligned = _x_extent(g)
        if aligned and x0 < line < x1:
            add("tail", g, x0, line)
            add("belly", g, line, x1)
            continue
        kind = "tail" if (x0 + x1) / 2 < line or (aligned and x1 <= line) else "belly"
        if aligned:
            add(kind, g, x0, x1)
        else:
            g.set("name", f"{kind}_{n[kind]}")
            if kind == "tail":
                g.set("rgba", "0.9 0.5 0.2 1")
            n[kind] += 1
            torso.append(g)
    for i, side in enumerate(("right_leg", "left_leg")):
        leg = torso.find(f"body[@name='{side}']")
        g = leg.find("geom")
        hx, hy, hz = (float(v) for v in g.get("size").split())
        gx, gy, gz = (float(v) for v in g.get("pos").split())
        g.set("contype", "0")
        g.set("conaffinity", "0")
        # Drawn by its two halves instead: three coincident boxes flicker in a viewer.
        rgba = g.get("rgba", "0.5 0.5 0.5 1")
        g.set("rgba", "0 0 0 0")
        for half, sign in (("toe_tail", -1), ("toe_nose", 1)):
            ET.SubElement(
                leg, "geom", name=f"{half}_{i}", type="box", mass="1e-6", rgba=rgba,
                size=f"{hx / 2} {hy} {hz}", pos=f"{gx + sign * hx / 2} {gy} {gz}",
            )  # fmt: skip


def surface_geoms(model) -> dict[str, list[int]]:
    """Geom ids of each surface in a model built with split_contacts."""
    return {
        **{
            kind: [g for g in range(model.ngeom) if model.geom(g).name.startswith(kind + "_")]
            for kind in ("tail", "belly")
        },
        "toe_tail": [model.geom(f"toe_tail_{i}").id for i in range(2)],
        "toe_nose": [model.geom(f"toe_nose_{i}").id for i in range(2)],
    }
