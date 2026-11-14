"""Build the printable cam-foot leg pair: the locomotion bench's winning curved sole on
upstream's leg hub, so it screws onto the MG90S horn exactly like the stock leg.

    python hardware/cam_foot_leg/make_leg.py                  # writes out/ beside this file
    python hardware/cam_foot_leg/make_leg.py --dip 1.0 --out /tmp/legs

The sole is research/locomotion/results/codesign.json's foot: an arc 20.85 deg either side
of the leg's line whose distance from the axle is 67.41 mm on the line and shrinks
16.61 mm per radian toward the nose (73.46 mm at the tail edge, 61.37 mm at the nose edge).
Everything but the tail-most 2.04 deg of it gets rubber dip, so that stretch is printed
--dip mm short and the coat brings it back to the simulated shape.

The hub isn't redrawn: every void of upstream's leg within the hub box (the tapered horn
slot, the spline bore and the screw hole) is cut out of the new leg, so the horn fit is
upstream's to the facet. Upstream's STL is fetched at a pinned commit and checked by hash.

Frame (mm): the axle at the origin, the leg hanging down -z, the outer (screw-head) face
at y = 0 and the servo face, where the horn slot opens, at y = -13. In the left leg's
file +x points to the nose; the right leg is its mirror image. The STLs are written lying
flat for printing, outer face down (--leg-frame keeps this frame): every void then widens
upward, so nothing overhangs and no supports are needed.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import urllib.request
from pathlib import Path

import manifold3d as mf
import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

HERE = Path(__file__).resolve().parent

UPSTREAM_COMMIT = "f1dd1b20b4a6849d1ec09a087ea6aa81957dd14f"
UPSTREAM_LEG = "hardware/print/growbot_diy_leg_v1_r01.stl"
UPSTREAM_URL = (
    f"https://raw.githubusercontent.com/britcruise9/GrowBot/{UPSTREAM_COMMIT}/{UPSTREAM_LEG}"
)
UPSTREAM_SHA256 = "1286fc2fafeb6501a802c9f61263103603dc600dbf05a6333ef3424b79b142bb"

# The winning foot (codesign.json), in mm and radians.
ARC = math.radians(20.84583511740862)
SOLE_MID = 74.0 - 6.586099236617389  # stock reach plus leg_extra_m
CAM = -16.610477744656828  # mm per radian toward the nose
SPLIT = math.radians(-18.80622635547998)  # dipped from here to the nose edge

THICK = 13.0
HUB_R = 10.5  # the stock leg's rounded top, centred on the axle
# Upstream's horn slot, bore and screw hole all sit inside this box (x, y, z ranges).
HUB_BOX = ((-5.0, 5.0), (-THICK - 0.5, 0.5), (-20.0, 6.0))

CHAMFER = 0.8  # on every edge of both faces, as 0.2 mm layer steps
SOLE_BAND = 6.0  # solid rim behind the sole
FLANK_WALL = 4.0
HUB_KEEP = 25.0  # solid round the hub: the slot reaches 19.1 mm below the axle
SPOKE_W = 4.0
ARCH_R = 44.0
WINDOW_ROUND = 2.5
LABEL_DEPTH = 0.6


def sole_radius(phi: float) -> float:
    return SOLE_MID + CAM * phi


def _polar(r: float, phi: float) -> tuple[float, float]:
    """(x, z) at distance r from the axle, phi from straight down, + toward the nose."""
    return r * math.sin(phi), -r * math.cos(phi)


def _tangent_point(p: tuple[float, float], r: float, side: float) -> tuple[float, float]:
    """Where the line from p touches the circle of radius r round the origin, on the
    outside of the leg (side -1: tail flank, +1: nose flank)."""
    d = math.hypot(*p)
    base = math.atan2(p[1], p[0])
    off = math.acos(r / d)
    cands = [(r * math.cos(base + s * off), r * math.sin(base + s * off)) for s in (1, -1)]
    return max(cands, key=lambda c: side * c[0])


def sole_points(dip: float, n: int = 80) -> list[tuple[float, float]]:
    """The sole from tail edge to nose edge, the dipped stretch dip mm short of the
    simulated curve, with a square step where the dip begins (a line to dip to)."""
    tail = [_polar(sole_radius(p), p) for p in np.linspace(-ARC, SPLIT, max(3, n // 10))]
    nose = [_polar(sole_radius(p) - dip, p) for p in np.linspace(SPLIT, ARC, n)]
    return tail + nose


def outline(dip: float) -> Polygon:
    """The leg's side profile: the hub circle and the sole joined by straight flanks
    tangent to both, the sole's corners eased just enough not to chip."""
    sole = sole_points(dip)
    t_tail = _tangent_point(sole[0], HUB_R, -1.0)
    t_nose = _tangent_point(sole[-1], HUB_R, 1.0)
    a0 = math.atan2(t_nose[1], t_nose[0])
    a1 = math.atan2(t_tail[1], t_tail[0])
    if a1 < a0:
        a1 += 2 * math.pi
    cap = [(HUB_R * math.cos(a), HUB_R * math.sin(a)) for a in np.linspace(a0, a1, 60)]
    poly = Polygon(sole + cap).buffer(0)
    return poly.buffer(-0.6, join_style="round").buffer(0.6, join_style="round")


def windows(profile: Polygon) -> Polygon:
    """Through-cuts between radial spokes: load at the sole runs straight to the axle along
    the spokes and flanks, the rim and hub stay solid."""
    inner = profile.buffer(-FLANK_WALL, join_style="mitre")
    sole_keep = Polygon(
        [(0.0, 0.0)]
        + [_polar(sole_radius(p) - SOLE_BAND, p) for p in np.linspace(-ARC - 0.2, ARC + 0.2, 80)]
    )
    region = inner.intersection(sole_keep).difference(Point(0, 0).buffer(HUB_KEEP, 128))

    # One spoke down the middle of the open area, from hub to sole, and a ring crossing it
    # concentric with the axle, so each flank is braced halfway along.
    def mid(r: float) -> tuple[float, float]:
        ring = region.intersection(Point(0, 0).buffer(r, 256).exterior)
        pts = np.asarray(ring.convex_hull.exterior.coords)
        angles = [math.atan2(x, -z) for x, z in pts]
        a = (min(angles) + max(angles)) / 2
        return _polar(r, a)

    p0, p1 = np.array(mid(HUB_KEEP + 2.0)), np.array(mid(SOLE_MID - SOLE_BAND - 3.0))
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    spoke = LineString([tuple(p0 - 20 * d), tuple(p1 + 20 * d)]).buffer(
        SPOKE_W / 2, cap_style="flat"
    )
    ring = (
        Point(0, 0)
        .buffer(ARCH_R + SPOKE_W / 2, 256)
        .difference(Point(0, 0).buffer(ARCH_R - SPOKE_W / 2, 256))
    )
    spokes = spoke.union(ring)
    cut = region.difference(spokes)
    return cut.buffer(-WINDOW_ROUND, join_style="round").buffer(WINDOW_ROUND, join_style="round")


def _letter(side: str) -> Polygon:
    """A stroke L or R 5.5 mm tall, laid out for reading on the servo face."""
    if side == "left":
        strokes = [LineString([(0, 5), (0, 0), (3, 0)])]
    else:
        bowl = [(0, 5), (2, 5)] + [
            (2 + 1.25 * math.cos(a), 3.75 + 1.25 * math.sin(a))
            for a in np.linspace(math.pi / 2, -math.pi / 2, 12)
        ]
        strokes = [LineString([(0, 0), (0, 5)]), LineString([*bowl, (0, 2.5)]),
                   LineString([(1.6, 2.5), (3.2, 0)])]  # fmt: skip
    mark = unary_union([s.buffer(0.45, cap_style="square", join_style="mitre") for s in strokes])
    return affinity.scale(mark, 1.1, 1.1, origin=(0, 0))


def _xsec(poly: Polygon) -> mf.CrossSection:
    polys = []
    for p in getattr(poly, "geoms", [poly]):
        if p.is_empty:
            continue
        polys.append(np.asarray(p.exterior.coords[:-1]))
        polys.extend(np.asarray(h.coords[:-1]) for h in p.interiors)
    return mf.CrossSection(polys, mf.FillRule.EvenOdd)


def _slab(poly: Polygon, y0: float, y1: float) -> mf.Manifold:
    """poly (in x, z) extruded across the thickness from y0 down to y1 (y1 < y0)."""
    m = mf.Manifold.extrude(_xsec(poly), y0 - y1)
    # Extrusion runs along +w; rotating +90 deg about x sends (u, v, w) to (u, -w, v).
    return m.rotate([90.0, 0.0, 0.0]).translate([0.0, y0, 0.0])


def body(face: Polygon) -> mf.Manifold:
    """face extruded through the thickness, every edge on both faces chamfered in layer
    steps (0.2 mm, so the chamfer prints exactly as modelled)."""
    steps = round(CHAMFER / 0.2)
    h = CHAMFER / steps
    parts = [_slab(face, -CHAMFER, -THICK + CHAMFER)]
    for k in range(steps):
        inset = CHAMFER - k * h - h / 2
        f = face.buffer(-inset, join_style="mitre")
        parts.append(_slab(f, -k * h, -(k + 1) * h))
        parts.append(_slab(f, -THICK + (k + 1) * h, -THICK + k * h))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)


def read_stl(path: Path) -> mf.Manifold:
    data = path.read_bytes()
    n = int.from_bytes(data[80:84], "little")
    rec = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]),
                        count=n, offset=84)  # fmt: skip
    verts, idx = np.unique(rec["v"].reshape(-1, 3), axis=0, return_inverse=True)
    mesh = mf.Mesh(
        vert_properties=verts.astype(np.float32), tri_verts=idx.reshape(-1, 3).astype(np.uint32)
    )
    leg = mf.Manifold(mesh)
    if leg.status() != mf.Error.NoError:
        raise ValueError(f"{path} isn't a closed solid: {leg.status()}")
    return leg


def upstream_leg(cache: Path) -> mf.Manifold:
    path = cache / Path(UPSTREAM_LEG).name
    if not path.exists():
        cache.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(UPSTREAM_URL, timeout=60) as r:
            path.write_bytes(r.read())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != UPSTREAM_SHA256:
        raise ValueError(f"{path} isn't upstream's leg at {UPSTREAM_COMMIT[:8]} (sha256 {digest})")
    return read_stl(path)


def hub_voids(stock: mf.Manifold) -> mf.Manifold:
    (x0, x1), (y0, y1), (z0, z1) = HUB_BOX
    box = mf.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    return box - stock


def build(stock: mf.Manifold, dip: float = 0.75, side: str = "left") -> mf.Manifold:
    """The leg for one side; side 'right' is the mirror image of 'left'."""
    face = outline(dip)
    leg = body(face.difference(windows(face))) - hub_voids(stock)
    # The label sits on the servo face beside the slot, reading upright with the leg
    # hanging; the right leg's is pre-mirrored so it reads correctly once mirrored.
    mark = _letter(side)
    if side == "right":
        mark = affinity.scale(mark, xfact=-1, origin=(0, 0))
    mark = affinity.translate(mark, 6.0 if side == "left" else -6.0, -16.0)
    leg = leg - _slab(mark, -THICK + LABEL_DEPTH, -THICK - 0.5)
    if side == "right":
        leg = leg.mirror([1.0, 0.0, 0.0])
    return leg


def write_stl(leg: mf.Manifold, path: Path, name: str) -> None:
    mesh = leg.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    tri = v[np.asarray(mesh.tri_verts)]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    rec["n"], rec["v"] = normals, tri
    header = name.encode()[:80].ljust(80, b" ")
    path.write_bytes(header + len(tri).to_bytes(4, "little") + rec.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dip", type=float, default=0.75, help="rubber dip thickness, mm")
    ap.add_argument("--out", type=Path, default=HERE / "out")
    ap.add_argument("--cache", type=Path, default=HERE / ".upstream")
    ap.add_argument(
        "--leg-frame", action="store_true", help="keep the leg frame instead of lying it flat"
    )
    args = ap.parse_args()
    stock = upstream_leg(args.cache)
    args.out.mkdir(parents=True, exist_ok=True)
    for side in ("left", "right"):
        leg = build(stock, args.dip, side)
        path = args.out / f"cam_foot_leg_{side}.stl"
        # Flat for the slicer: outer face on the bed (z = 0), servo face up, sole toward -y.
        out = leg if args.leg_frame else leg.rotate([-90.0, 0.0, 0.0])
        write_stl(out, path, f"GrowBot cam-foot leg ({side}), dip {args.dip} mm")
        print(f"{path}: {leg.volume() / 1000:.1f} cm^3 (stock {stock.volume() / 1000:.1f})")


if __name__ == "__main__":
    main()
