"""The printable cam-foot leg: its sole is the bench's winning foot, the dip recess is where
and as deep as asked, the hub's voids come from the stock leg untouched, the pair are
mirror images with the horn slot facing out on both, and printed inner face down nothing
overhangs.

A stand-in stock leg (a bar with a slot and a screw hole) keeps the test offline; the real
STL's hash is checked when make_leg.py fetches it."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

mf = pytest.importorskip("manifold3d")
geometry = pytest.importorskip("shapely.geometry")
affinity = pytest.importorskip("shapely.affinity")

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
    outline = leg_mod.face(0.75)
    for phi, short in ((-0.97 * leg_mod.ARC, 0.0), (0.0, 0.75), (0.9 * leg_mod.ARC, 0.75)):
        x, z = leg_mod._polar(1.0, phi)
        inside = [
            r for r in np.arange(55.0, 80.0, 0.01) if outline.contains(geometry.Point(r * x, r * z))
        ]
        reach = max(inside)
        assert reach == pytest.approx(leg_mod.sole_radius(phi) - short, abs=0.03)


def test_only_the_sole_can_reach_the_floor() -> None:
    prof = leg_mod.face(0.75)
    # Slivers from the two curves' different sampling only; nothing pokes out 0.02 mm.
    assert prof.intersection(leg_mod._below_sole(0.75)).buffer(-0.02).is_empty


def test_outer_face_is_sculpted_only_on_the_shaft() -> None:
    prof = leg_mod.face(0.75)
    x = np.array([0.0, 6.0, -2.0, 7.5])
    z = np.array([-10.0, -18.0, -64.0, -40.0])
    t = leg_mod.thickness(prof, x, z, 0.75)
    # The hub (round the horn slot) and the sole keep the full 13 mm.
    assert t[:3] == pytest.approx([leg_mod.THICK] * 3, abs=0.01)
    # Mid-shaft by the edge, it's rounded over.
    assert t[3] < leg_mod.THICK - 1.0
    assert t[3] > leg_mod.THICK - leg_mod.ROUND_DROP - leg_mod.WAIST_DEPTH - 0.01


def test_printed_leg_keeps_full_thickness_at_the_hub_and_sole(legs) -> None:
    # On the built leg, not just thickness(): the face's blur must not carve these.
    for x, z in ((6.0, -18.0), (-2.0, -64.0), (8.0, 3.0)):
        probe = mf.Manifold.cube([0.2, 15.0, 0.2]).translate([x - 0.1, -14.0, z - 0.1])
        assert (legs["right"] ^ probe).bounding_box()[1] == pytest.approx(-leg_mod.THICK, abs=1e-4)


def test_both_legs_are_closed_solids_and_mirror_images(legs) -> None:
    left, right = legs["left"], legs["right"]
    for leg in (left, right):
        assert leg.status() == mf.Error.NoError
        assert leg.genus() == 2  # the screw hole and the marrow slot
    assert left.volume() == pytest.approx(right.volume(), rel=1e-3)
    # Same shape mirrored: only the engraved letter differs.
    mirrored = left.mirror([1.0, 0.0, 0.0])
    assert (mirrored - right).volume() + (right - mirrored).volume() < 40.0
    lb, rb = left.bounding_box(), right.bounding_box()
    assert lb[0] == pytest.approx(-rb[3], abs=1e-4) and lb[3] == pytest.approx(-rb[0], abs=1e-4)
    # The right leg keeps the frame (slot out at -y, the robot's right): its longer tail
    # edge is at -x, toward the tail. The left's is mirrored to +x.
    assert rb[0] < -25.0 and 21.0 < rb[3] < 22.0


def test_hub_voids_are_the_stock_legs_exactly(legs, stock) -> None:
    (x0, x1), _, (z0, z1) = leg_mod.HUB_BOX
    box = mf.Manifold.cube([x1 - x0, 13.0, z1 - z0]).translate([x0, -13.0, z0])
    for side, leg in legs.items():
        ref = stock if side == "right" else stock.mirror([1.0, 0.0, 0.0])
        new, old = leg ^ box, ref ^ box
        assert (new - old).volume() + (old - new).volume() == pytest.approx(0.0, abs=1e-3)


def _faces(leg):
    mesh = leg.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3]
    tri = v[np.asarray(mesh.tri_verts)]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(n, axis=1) / 2
    return n / np.maximum(2 * area, 1e-12)[:, None], area, tri.mean(axis=1)


@pytest.mark.parametrize("side", ["left", "right"])
def test_prints_inner_face_down_with_nothing_overhanging(legs, side) -> None:
    flat = leg_mod.print_pose(legs[side])
    assert flat.bounding_box()[2] == pytest.approx(0.0, abs=1e-6)
    assert flat.bounding_box()[5] == pytest.approx(leg_mod.THICK, abs=1e-6)
    n, area, c = _faces(flat)
    # The bed face is the flat inner face, not the sculpted one.
    on_bed = (n[:, 2] < -0.99) & (c[:, 2] < 1e-6)
    inner = (_faces(legs[side])[0][:, 1] > 0.99) & (np.abs(_faces(legs[side])[2][:, 1]) < 1e-6)
    assert area[on_bed].sum() == pytest.approx(area[inner].sum(), rel=1e-6)
    # Faces looking at the bed above it, past the chamfer's steps and the engraved label.
    over = (n[:, 2] < -0.5) & (c[:, 2] > leg_mod.CHAMFER + 0.05)
    assert area[over].sum() == pytest.approx(0.0, abs=1e-6)


def test_labels_stay_clear_of_the_edge_and_the_spline_hole() -> None:
    prof = leg_mod.face(0.75)
    inner = prof.buffer(-leg_mod.CHAMFER)
    hole = geometry.Point(0, 0).buffer(3.6)
    for side in ("left", "right"):
        mark = leg_mod._label(side, prof)
        assert mark.area > 5.0
        assert mark.difference(inner).is_empty
        assert mark.distance(hole) > 1.0


@pytest.mark.parametrize("side", ["left", "right"])
def test_labels_read_right_way_round_on_the_fitted_inner_face(side) -> None:
    mark = leg_mod._label(side, leg_mod.face(0.75))
    # Seen from +y +x runs leftward; the left leg's mirror flips its label once more.
    seen = affinity.scale(mark, xfact=-1 if side == "right" else 1, origin=(0, 0))
    x0, y0, _, y1 = seen.bounds
    # L and R both have a solid upright stem on the reader's left.
    stem = seen.intersection(geometry.box(x0, y0, x0 + 0.8, y1))
    assert stem.area > 0.85 * 0.8 * (y1 - y0)


def test_writes_a_binary_stl(tmp_path, legs) -> None:
    path = tmp_path / "leg.stl"
    leg_mod.write_stl(legs["right"], path, "t")
    data = path.read_bytes()
    n = int.from_bytes(data[80:84], "little")
    assert len(data) == 84 + 50 * n
    back = leg_mod.read_stl(path)
    assert back.volume() == pytest.approx(legs["right"].volume(), rel=1e-4)
    assert math.isfinite(back.volume())
