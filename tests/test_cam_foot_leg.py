"""The printable cam-foot leg: its sole is the bench's winning foot, the dip recess is where
and as deep as asked, the hub's voids come from the stock leg untouched, the pair are
mirror images, and nothing overhangs printed outer face down.

A stand-in stock leg (a bar with a slot and a screw hole) keeps the test offline; the real
STL's hash is checked when make_leg.py fetches it."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

mf = pytest.importorskip("manifold3d")
geometry = pytest.importorskip("shapely.geometry")

_PATH = Path(__file__).resolve().parents[1] / "hardware" / "cam_foot_leg" / "make_leg.py"
_spec = importlib.util.spec_from_file_location("make_leg", _PATH)
leg_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(leg_mod)


def _stand_in_stock() -> "mf.Manifold":
    bar = mf.Manifold.cube([21.0, 13.0, 84.5]).translate([-10.5, -13.0, -74.0])
    slot = mf.Manifold.cube([7.0, 9.6, 21.0]).translate([-3.5, -13.1, -18.0])
    hole = mf.Manifold.cylinder(14.0, 2.5, 2.5, 48).rotate([90.0, 0.0, 0.0]).translate([0, 0.5, 0])
    return bar - slot - hole


@pytest.fixture(scope="module")
def stock():
    return _stand_in_stock()


@pytest.fixture(scope="module")
def legs(stock):
    return {side: leg_mod.build(stock, 0.75, side) for side in ("left", "right")}


def test_sole_matches_the_benchs_winning_foot() -> None:
    assert leg_mod.sole_radius(-leg_mod.ARC) == pytest.approx(73.46, abs=0.01)
    assert leg_mod.sole_radius(0.0) == pytest.approx(67.41, abs=0.01)
    assert leg_mod.sole_radius(leg_mod.ARC) == pytest.approx(61.37, abs=0.01)
    bare_arc_mm = (leg_mod.SPLIT + leg_mod.ARC) * leg_mod.sole_radius(-leg_mod.ARC)
    assert bare_arc_mm == pytest.approx(2.6, abs=0.05)


def test_only_the_dipped_stretch_is_recessed_and_by_the_dip() -> None:
    outline = leg_mod.outline(0.75)
    for phi, short in ((-0.97 * leg_mod.ARC, 0.0), (0.0, 0.75), (0.9 * leg_mod.ARC, 0.75)):
        x, z = leg_mod._polar(1.0, phi)
        inside = [
            r for r in np.arange(55.0, 80.0, 0.01) if outline.contains(geometry.Point(r * x, r * z))
        ]
        reach = max(inside)
        assert reach == pytest.approx(leg_mod.sole_radius(phi) - short, abs=0.03)


def test_both_legs_are_closed_solids_and_mirror_images(legs) -> None:
    left, right = legs["left"], legs["right"]
    for leg in (left, right):
        assert leg.status() == mf.Error.NoError
        assert leg.genus() >= 9  # the screw hole and at least eight cells
    assert left.volume() == pytest.approx(right.volume(), rel=1e-3)
    # Same shape mirrored: only the engraved letter differs.
    mirrored = left.mirror([1.0, 0.0, 0.0])
    assert (mirrored - right).volume() + (right - mirrored).volume() < 40.0
    lb, rb = left.bounding_box(), right.bounding_box()
    assert lb[0] == pytest.approx(-rb[3], abs=1e-4) and lb[3] == pytest.approx(-rb[0], abs=1e-4)
    # The left leg's longer tail edge is at -x, toward the tail.
    assert lb[0] < -25.0 and 21.0 < lb[3] < 22.0


def test_hub_voids_are_the_stock_legs_exactly(legs, stock) -> None:
    (x0, x1), _, (z0, z1) = leg_mod.HUB_BOX
    box = mf.Manifold.cube([x1 - x0, 13.0, z1 - z0]).translate([x0, -13.0, z0])
    for side, leg in legs.items():
        ref = stock if side == "left" else stock.mirror([1.0, 0.0, 0.0])
        new, old = leg ^ box, ref ^ box
        assert (new - old).volume() + (old - new).volume() == pytest.approx(0.0, abs=1e-3)


def test_nothing_overhangs_printed_outer_face_down(legs) -> None:
    mesh = legs["left"].to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    tri = v[np.asarray(mesh.tri_verts)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    ny = n[:, 1] / np.maximum(2 * area, 1e-12)
    cy = tri[:, :, 1].mean(axis=1)
    # Faces looking at the bed (+y) above the bed itself, past the chamfer's layer steps.
    over = (ny > 0.5) & (cy < -leg_mod.CHAMFER - 0.05)
    assert area[over].sum() == pytest.approx(0.0, abs=1e-6)


def test_writes_a_binary_stl(tmp_path, legs) -> None:
    path = tmp_path / "leg.stl"
    leg_mod.write_stl(legs["left"], path, "t")
    data = path.read_bytes()
    n = int.from_bytes(data[80:84], "little")
    assert len(data) == 84 + 50 * n
    back = leg_mod.read_stl(path)
    assert back.volume() == pytest.approx(legs["left"].volume(), rel=1e-4)
    assert math.isfinite(back.volume())
