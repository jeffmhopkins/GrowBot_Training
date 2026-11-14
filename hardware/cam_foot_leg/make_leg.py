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

Between them runs a femur: a shaft flaring from the hub into the sole, an S-shaped marrow
slot down it, and its outer face sculpted like bone (thickness()).

Frame (mm): the axle at the origin, the leg hanging down -z, the inner face (against the
servo, where the spline comes through) at y = 0, and the outer face, where the horn slot
opens and the horn drops in, at y = -13. The outer face is the sculpted one. In the right
leg's file +x points to the nose; the left leg is its mirror image. The STLs are written
lying flat for printing, inner face down (--leg-frame keeps this frame): every void
widens upward and the sculpted face is a heightfield on top, so no supports are needed.
The only roofs are the chamfers' 0.2 mm steps and the engraved letter's 1 mm-wide strokes,
which bridge.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import urllib.request
from pathlib import Path

import manifold3d as mf
import numpy as np
import shapely
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
SOLE_BAND = 5.0  # solid rim behind the sole
HUB_WALL = 4.5  # round the horn slot, a little more than the stock leg's 3.5 at its tip
# The femur: a shaft from the hub flaring into the sole, its marrow slot an S.
SHAFT = ((0.0, -15.0), (-4.0, -40.0), (-1.0, -62.0))  # quadratic Bezier, hub to sole
SHAFT_W = (18.0, 12.5, 46.0)  # width at the hub, mid-shaft and the sole
SLOT_W = (3.5, 4.5, 7.0)
FILLET = 6.0  # where the shaft meets the hub and the sole
# The outer face is sculpted like bone: rounded over toward every edge, and the shaft's
# middle dished a little, easing back to the full 13 mm at the hub and along the sole.
ROUND_R = 5.0  # how far in from each edge the rounding reaches
ROUND_DROP = 3.5  # how far the outer face falls at the very edge
WAIST_DEPTH = 1.6
FULL_BLEND = 8.0  # mm over which the sculpting fades into the full-thickness hub and sole
GRID = 0.3  # heightfield step, mm
BLUR = 0.9  # mm, the sculpt's smoothing (_blur)
LABEL_DEPTH = 0.6
LABEL_AT = (5.6, -8.0)  # the left leg's, before mirroring


def sole_radius(phi: float) -> float:
    return SOLE_MID + CAM * phi


def _polar(r: float, phi: float) -> tuple[float, float]:
    """(x, z) at distance r from the axle, phi from straight down, + toward the nose."""
    return r * math.sin(phi), -r * math.cos(phi)


def sole_points(dip: float, n: int = 80) -> list[tuple[float, float]]:
    """The sole from tail edge to nose edge, the dipped stretch dip mm short of the
    simulated curve, with a square step where the dip begins (a line to dip to)."""
    tail = [_polar(sole_radius(p), p) for p in np.linspace(-ARC, SPLIT, max(3, n // 10))]
    nose = [_polar(sole_radius(p) - dip, p) for p in np.linspace(SPLIT, ARC, n)]
    return tail + nose


def _boss() -> Polygon:
    """The solid round the hub: the horn slot's outline grown by its wall, merged into the
    stock leg's hub circle as one teardrop."""
    slot = Polygon([(-4.4, 4.4), (4.4, 4.4), (4.4, -19.1), (-4.4, -19.1)])
    return slot.buffer(HUB_WALL).union(Point(0, 0).buffer(HUB_R + 2.0, 128)).convex_hull


def _sole_band(dip: float) -> Polygon:
    inner = [_polar(sole_radius(p) - SOLE_BAND, p) for p in np.linspace(ARC, -ARC, 60)]
    return Polygon(sole_points(dip) + inner).buffer(0)


def _below_sole(dip: float) -> Polygon:
    """Everything beyond the sole curve, extended past its ends, so nothing but the sole
    ever reaches the floor."""
    tail = [_polar(sole_radius(p), p) for p in np.linspace(-ARC - 0.6, -ARC, 12)[:-1]]
    nose = [_polar(sole_radius(p) - dip, p) for p in np.linspace(ARC, ARC + 0.6, 12)[1:]]
    edge = tail + sole_points(dip) + nose
    return Polygon(edge + [_polar(300.0, ARC + 0.6), _polar(300.0, -ARC - 0.6)]).buffer(0)


def _bezier(p0, p1, p2, n: int = 48) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n)[:, None]
    p0, p1, p2 = map(np.asarray, (p0, p1, p2))
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2


def _stroke(pts: np.ndarray, widths: tuple[float, float, float]) -> Polygon:
    """A smooth band along pts whose width eases from widths[0] through widths[1] to
    widths[2]: circles along the path, each neighbouring pair hulled."""
    w0, wm, w1 = widths
    t = np.linspace(0.0, 1.0, len(pts))
    w = (1 - t) ** 2 * w0 + 2 * (1 - t) * t * wm + t**2 * w1
    discs = [Point(*p).buffer(r / 2, 32) for p, r in zip(pts, w, strict=True)]
    return unary_union([discs[i].union(discs[i + 1]).convex_hull for i in range(len(discs) - 1)])


def face(dip: float) -> Polygon:
    """The side profile: a femur from the hub boss flaring into the sole band, filleted
    where they meet, with an S-shaped marrow slot down the shaft."""
    shaft = _stroke(_bezier(*SHAFT), SHAFT_W)
    solid = unary_union([shaft, _boss(), _sole_band(dip)])
    solid = solid.buffer(FILLET, 64).buffer(-FILLET, 64).buffer(-0.6).buffer(0.6)
    solid = solid.difference(_below_sole(dip)).union(_sole_band(dip))
    t = np.linspace(0.0, 1.0, 40)
    slot = np.column_stack([-1.5 - 2.2 * np.sin(t * math.pi * 1.5), -28.0 - 29.0 * t])
    return solid.difference(_stroke(slot, SLOT_W))


def _smooth(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _carving(prof: Polygon, x: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """How deep the outer face is carved before blending, and the weight (1 at the hub and
    along the sole) that blends it back to full thickness."""
    pts = shapely.points(x, z)
    # Signed: negative outside the profile, so the curve carries on smoothly past the edge
    # instead of folding back into a ridge there.
    edge = shapely.distance(pts, prof.boundary)
    edge = np.where(shapely.contains_xy(prof, x, z), edge, -edge)
    roundover = ROUND_DROP * np.clip(1.0 - edge / ROUND_R, 0.0, None) ** 2
    waist = WAIST_DEPTH * np.exp(-(((z + 40.0) / 11.0) ** 2)) * _smooth(edge / 6.0)
    hub = 1.0 - _smooth(shapely.distance(pts, _boss()) / FULL_BLEND)
    r = np.hypot(x, z)
    phi = np.arctan2(x, -z)
    to_sole = (np.vectorize(sole_radius)(phi) - SOLE_BAND) - r
    sole = 1.0 - _smooth(to_sole / FULL_BLEND)
    return roundover + waist, np.maximum(hub, sole)


def thickness(prof: Polygon, x: np.ndarray, z: np.ndarray, dip: float) -> np.ndarray:
    """How far the solid reaches from the inner face at each (x, z): THICK at the hub and
    along the sole, rounded over toward the shaft's edges and its slot, and dished a
    little mid-shaft, like the waist of a long bone. The printed face blurs the carving
    before blending it out, so it differs from this by a few tenths of a millimetre on the
    shaft and not at all at the hub and sole (_heightfield)."""
    carve, full = _carving(prof, x, z)
    return THICK - carve * (1.0 - full)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur on the grid, edges held. Distances to a polygon kink at its vertices
    and round the slot's ends, which reads as ripple on a curved face; this irons it out."""
    k = np.arange(-math.ceil(3 * sigma), math.ceil(3 * sigma) + 1)
    w = np.exp(-0.5 * (k / sigma) ** 2)
    w /= w.sum()
    n = len(k) // 2
    for axis in (0, 1):
        pad = [(n, n) if i == axis else (0, 0) for i in range(2)]
        p = np.pad(a, pad, mode="edge")
        a = sum(wi * np.take(p, range(i, i + a.shape[axis]), axis=axis) for i, wi in enumerate(w))
    return a


def _heightfield(prof: Polygon, dip: float) -> mf.Manifold:
    """The solid between the flat inner face and the sculpted outer face: a closed grid
    mesh, flat beyond the inner face and following thickness() on the outer side, except
    that the carving is blurred (_blur) before it's blended out, and the face is lifted
    clear where nothing is carved. A heightfield has no overhang anywhere, so with the
    inner face on the bed the sculpting prints as plain top layers."""
    x0, z0, x1, z1 = prof.bounds
    xs = np.arange(x0 - 1.0, x1 + 1.0 + GRID, GRID)
    zs = np.arange(z0 - 1.0, z1 + 1.0 + GRID, GRID)
    gx, gz = np.meshgrid(xs, zs, indexing="ij")
    carve, full = (v.reshape(gx.shape) for v in _carving(prof, gx.ravel(), gz.ravel()))
    drop = _blur(carve, BLUR / GRID) * (1.0 - full)
    # Lift the face clear of y = -THICK where nothing is carved, smoothly: a cut-off here
    # left the grid's staircase as a seam round the hub and along the sole.
    top = -THICK + drop - 0.5 * np.exp(-drop / 0.05)
    nx, nz = gx.shape
    verts = np.concatenate(
        [
            np.column_stack([gx.ravel(), top.ravel(), gz.ravel()]),
            np.column_stack([gx.ravel(), np.full(gx.size, 1.0), gz.ravel()]),
        ]
    )
    idx = np.arange(nx * nz).reshape(nx, nz)
    off = nx * nz
    a, b, c, d = idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]
    tris = [np.stack([a, b, c], -1), np.stack([a, c, d], -1)]  # top, facing -y
    tris += [np.stack([a, c, b], -1) + off, np.stack([a, d, c], -1) + off]  # bottom, +y

    def wall(line: np.ndarray) -> np.ndarray:
        """Quads down the grid's rim; walked the same way round as the top's edges, so
        each quad is wound against its top triangle's shared edge."""
        p, q = line[:-1], line[1:]
        return np.concatenate([np.stack([q + off, q, p], -1), np.stack([p + off, q + off, p], -1)])

    rim = [idx[:, 0], idx[-1, :], idx[::-1, -1], idx[0, ::-1]]
    sides = [wall(line) for line in rim]
    tri = np.concatenate([t.reshape(-1, 3) for t in tris] + sides).astype(np.uint32)
    hf = mf.Manifold(mf.Mesh(vert_properties=verts.astype(np.float32), tri_verts=tri))
    if hf.status() != mf.Error.NoError:
        raise ValueError(f"heightfield isn't a closed solid: {hf.status()}")
    return hf


def _letter(side: str) -> Polygon:
    """A stroke L or R 5.5 mm tall, as read with the leg hanging."""
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


def build(stock: mf.Manifold, dip: float = 0.75, side: str = "right") -> mf.Manifold:
    """The leg for one side; side 'left' is the mirror image of 'right'. The horn slot
    faces out from the body on both, so the right leg (on the robot's -y side) keeps this
    frame and the left is mirrored for its nose to point forward too."""
    prof = face(dip)
    leg = (body(prof) ^ _heightfield(prof, dip)) - hub_voids(stock)
    leg = leg - _slab(_label(side, prof), 0.5, -LABEL_DEPTH)
    if side == "left":
        leg = leg.mirror([1.0, 0.0, 0.0])
    return leg


def _label(side: str, prof: Polygon) -> Polygon:
    """L or R for the inner face beside the spline hole, out of sight once fitted. It reads
    right way round looking at that face, so it's drawn mirrored on the right leg (seen
    from +y, +x runs leftward) and plain on the left, which the mirror then flips."""
    mark = _letter(side)
    if side == "right":
        mark = affinity.scale(mark, xfact=-1, origin=(0, 0))
    mark = affinity.translate(mark, LABEL_AT[0] * (-1 if side == "right" else 1), LABEL_AT[1])
    # Never into the edge's chamfer, whatever the profile does beside the hub.
    return mark.intersection(prof.buffer(-CHAMFER - 0.4))


def print_pose(leg: mf.Manifold) -> mf.Manifold:
    """Flat for the slicer: inner face on the bed (z = 0), sculpted outer face up."""
    return leg.rotate([-90.0, 0.0, 0.0])


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
    for side in ("right", "left"):
        leg = build(stock, args.dip, side)
        path = args.out / f"cam_foot_leg_{side}.stl"
        out = leg if args.leg_frame else print_pose(leg)
        write_stl(out, path, f"GrowBot cam-foot leg ({side}), dip {args.dip} mm")
        print(f"{path}: {leg.volume() / 1000:.1f} cm^3 (stock {stock.volume() / 1000:.1f})")


if __name__ == "__main__":
    main()
