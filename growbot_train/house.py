"""Furnished houses for the sim body, so JBrain2's Dave simulator can ask Dave to go and
find something, or leave him alone to wander, and score where he actually went.

``SimBody(scene="house")`` is two rooms joined by a doorway: a living room (where he starts)
with a piano in its far corner, a couch, a bookshelf, a plant and a lamp, and a kitchen
behind him with a fridge, a counter, a table and the cat. ``scene="flat"`` is the next
level: a hallway joining four rooms, doors out of line with each other, a couch in the way
of the straight line out of the living room, and the piano in a study at the far end. The
walls and the furniture's solid parts collide with the robot, so a wall stops him; a rug and
the table top (above his head) don't.

Everything is plain boxes, cylinders and spheres with drawn textures (no photos), shaped so
a vision model reads them as what they are from a phone a few centimetres off the floor.
A ``Layout`` (``LAYOUTS[scene]``) holds the ground truth a scenario scores against and a
report draws: its rooms, walls, doors and furniture. The module-level names are the two-room
house's, as they always were.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

# The house's footprint (x0, y0, x1, y1), metres. The robot starts at the origin facing -x,
# toward the living room's far wall; the kitchen is behind him through a doorway.
OUTER = (-3.0, -2.2, 2.6, 2.2)
DIVIDE_X = 0.6
DOOR = (-0.45, 0.45)  # the doorway's span in y, in the dividing wall
WALL_H = 2.4
WALL_T = 0.03
ROOMS: dict[str, tuple[float, float, float, float]] = {
    "living room": (OUTER[0], OUTER[1], DIVIDE_X, OUTER[3]),
    "kitchen": (DIVIDE_X, OUTER[1], OUTER[2], OUTER[3]),
}
TEX = 128


@dataclass(frozen=True)
class Item:
    """One piece of furniture: its centre on the floor, its footprint's half sizes, which
    way its front faces (degrees from +x, counter-clockwise), and its map colour."""

    name: str
    room: str
    x: float
    y: float
    hx: float
    hy: float
    face_deg: float
    rgb: tuple[int, int, int]
    kind: str = ""  # the builder, when it isn't the name

    @property
    def build(self) -> str:
        return self.kind or self.name


@dataclass(frozen=True)
class Door:
    """A gap in a wall: its centre, which way the wall runs through it ("x" or "y"), its half
    width, and the two rooms it joins."""

    name: str
    x: float
    y: float
    along: str
    half: float
    rooms: tuple[str, str]


@dataclass(frozen=True)
class Layout:
    """One house: footprint, rooms, wall segments (name, ax, ay, bx, by, material), doors,
    furniture, a light per room, and where he starts (x, y, nose yaw degrees)."""

    name: str
    OUTER: tuple[float, float, float, float]
    ROOMS: dict[str, tuple[float, float, float, float]]
    WALLS: tuple[tuple[str, float, float, float, float, str], ...]
    DOORS: tuple[Door, ...]
    ITEMS: tuple[Item, ...]
    lights: tuple[tuple[float, float], ...]
    start: tuple[float, float, float] = (0.0, 0.0, 0.0)
    ITEM: dict[str, Item] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ITEM", {i.name: i for i in self.ITEMS})

    def room_at(self, x: float, y: float) -> str:
        for name, (x0, y0, x1, y1) in self.ROOMS.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                return name
        return ""

    def distance_to(self, name: str, x: float, y: float) -> float:
        """How far (x, y) is from the item's footprint (0 inside it), metres."""
        return _distance(self.ITEM[name], x, y)

    def route(self, a: str, b: str) -> list[Door]:
        """The doors from room a to room b, fewest first (empty when they are the same)."""
        seen, todo = {a}, [(a, [])]
        while todo:
            room, path = todo.pop(0)
            if room == b:
                return path
            for d in self.DOORS:
                if room in d.rooms:
                    nxt = d.rooms[1] if d.rooms[0] == room else d.rooms[0]
                    if nxt not in seen:
                        seen.add(nxt)
                        todo.append((nxt, [*path, d]))
        return []


ITEMS: tuple[Item, ...] = (
    # Front faces south (-y), into the room, in the far left corner from the start.
    Item("piano", "living room", -2.15, 1.85, 0.75, 0.32, -90.0, (25, 25, 28)),
    Item("piano bench", "living room", -2.15, 1.25, 0.4, 0.18, -90.0, (40, 30, 25)),
    Item("couch", "living room", -1.2, -1.72, 1.05, 0.45, 90.0, (150, 45, 40)),
    Item("bookshelf", "living room", -2.77, -0.4, 0.2, 0.5, 0.0, (120, 80, 45)),
    Item("plant", "living room", -0.2, 1.9, 0.2, 0.2, -90.0, (60, 140, 60)),
    Item("lamp", "living room", -2.65, -1.85, 0.15, 0.15, 45.0, (230, 210, 150)),
    Item("rug", "living room", -1.3, 0.0, 0.9, 0.6, 0.0, (70, 90, 140)),
    Item("fridge", "kitchen", 2.22, -1.75, 0.35, 0.4, 180.0, (225, 228, 232)),
    Item("counter", "kitchen", 2.27, 0.4, 0.3, 0.9, 180.0, (200, 190, 170)),
    Item("table", "kitchen", 1.45, 1.35, 0.45, 0.6, 0.0, (150, 110, 70)),
    # Curled up by the table, facing the doorway: out of sight from the living room.
    Item("cat", "kitchen", 1.3, 0.55, 0.22, 0.12, 180.0, (215, 125, 50)),
)
ITEM = {i.name: i for i in ITEMS}
_X0, _Y0, _X1, _Y1 = OUTER
TWO_ROOM = Layout(
    name="house",
    OUTER=OUTER,
    ROOMS=ROOMS,
    WALLS=(
        ("house_wall_w", _X0, _Y0, _X0, _Y1, "house_living"),
        ("house_wall_n_living", _X0, _Y1, DIVIDE_X, _Y1, "house_living"),
        ("house_wall_s_living", _X0, _Y0, DIVIDE_X, _Y0, "house_living"),
        ("house_wall_n_kitchen", DIVIDE_X, _Y1, _X1, _Y1, "house_kitchen"),
        ("house_wall_s_kitchen", DIVIDE_X, _Y0, _X1, _Y0, "house_kitchen"),
        ("house_wall_e", _X1, _Y0, _X1, _Y1, "house_kitchen"),
        ("house_divide_s", DIVIDE_X, _Y0, DIVIDE_X, DOOR[0], "house_living"),
        ("house_divide_n", DIVIDE_X, DOOR[1], DIVIDE_X, _Y1, "house_living"),
    ),
    DOORS=(
        Door(
            "kitchen door",
            DIVIDE_X,
            (DOOR[0] + DOOR[1]) / 2,
            "y",
            (DOOR[1] - DOOR[0]) / 2,
            ("living room", "kitchen"),
        ),
    ),
    ITEMS=ITEMS,
    lights=((1.6, 0.0), (-1.8, 0.0)),
)

# The flat: a hallway across the middle, the living room (where he starts) and the kitchen
# south of it with no door between them, a bedroom and a study north. Every door is off line
# from the last, so no room is reached by walking straight.
_FY = 0.6  # the hallway's walls at y = -0.6 and 0.6
_FLAT_OUTER = (-4.0, -3.0, 4.0, 3.0)
FLAT = Layout(
    name="flat",
    OUTER=_FLAT_OUTER,
    ROOMS={
        "living room": (-4.0, -3.0, -0.5, -_FY),
        "kitchen": (-0.5, -3.0, 4.0, -_FY),
        "hallway": (-4.0, -_FY, 4.0, _FY),
        "bedroom": (-4.0, _FY, 0.5, 3.0),
        "study": (0.5, _FY, 4.0, 3.0),
    },
    WALLS=(
        ("house_wall_w", -4.0, -3.0, -4.0, 3.0, "house_hall"),
        ("house_wall_e", 4.0, -3.0, 4.0, 3.0, "house_hall"),
        ("house_wall_s_living", -4.0, -3.0, -0.5, -3.0, "house_living"),
        ("house_wall_s_kitchen", -0.5, -3.0, 4.0, -3.0, "house_kitchen"),
        ("house_wall_n_bedroom", -4.0, 3.0, 0.5, 3.0, "house_bedroom"),
        ("house_wall_n_study", 0.5, 3.0, 4.0, 3.0, "house_study"),
        # The hallway's south side: the living room's door, then the kitchen's far along.
        ("house_hall_s_0", -4.0, -_FY, -1.95, -_FY, "house_living"),
        ("house_hall_s_1", -1.05, -_FY, 2.55, -_FY, "house_kitchen"),
        ("house_hall_s_2", 3.45, -_FY, 4.0, -_FY, "house_kitchen"),
        # Its north side: the bedroom's door at the west end, the study's in the middle.
        ("house_hall_n_0", -4.0, _FY, -3.65, _FY, "house_bedroom"),
        ("house_hall_n_1", -2.75, _FY, 0.75, _FY, "house_bedroom"),
        ("house_hall_n_2", 1.65, _FY, 4.0, _FY, "house_study"),
        ("house_divide_south", -0.5, -3.0, -0.5, -_FY, "house_living"),
        ("house_divide_north", 0.5, _FY, 0.5, 3.0, "house_study"),
    ),
    DOORS=(
        Door("living room door", -1.5, -_FY, "x", 0.45, ("living room", "hallway")),
        Door("kitchen door", 3.0, -_FY, "x", 0.45, ("kitchen", "hallway")),
        Door("bedroom door", -3.2, _FY, "x", 0.45, ("bedroom", "hallway")),
        Door("study door", 1.2, _FY, "x", 0.45, ("study", "hallway")),
    ),
    ITEMS=(
        # Across the straight line from his start to the living room's door.
        Item("couch", "living room", -2.0, -1.5, 1.05, 0.45, 90.0, (150, 45, 40)),
        Item("lamp", "living room", -0.8, -2.75, 0.15, 0.15, 45.0, (230, 210, 150)),
        Item("rug", "living room", -2.2, -2.3, 0.9, 0.5, 0.0, (70, 90, 140)),
        Item("fridge", "kitchen", 3.62, -2.55, 0.35, 0.4, 180.0, (225, 228, 232)),
        Item("counter", "kitchen", 1.5, -2.64, 0.9, 0.3, 90.0, (200, 190, 170)),
        Item("table", "kitchen", 1.0, -1.5, 0.45, 0.6, 0.0, (150, 110, 70)),
        Item("bookshelf", "bedroom", -3.77, 2.0, 0.2, 0.5, 0.0, (120, 80, 45)),
        Item("plant", "bedroom", -0.2, 2.75, 0.2, 0.2, -90.0, (60, 140, 60)),
        Item("cat", "bedroom", -1.6, 1.9, 0.22, 0.12, 0.0, (215, 125, 50)),
        Item("piano", "study", 2.5, 2.65, 0.75, 0.32, -90.0, (25, 25, 28)),
        Item("piano bench", "study", 2.5, 2.05, 0.4, 0.18, -90.0, (40, 30, 25)),
    ),
    lights=((-2.2, -1.8), (1.7, -1.8), (0.0, 0.0), (-1.7, 1.8), (2.2, 1.8)),
    # In the living room's south-west corner, his camera on the far (east) wall: the couch
    # stands between him and the door.
    start=(-3.3, -2.4, 180.0),
)
LAYOUTS = {"house": TWO_ROOM, "flat": FLAT}


def room_at(x: float, y: float) -> str:
    return TWO_ROOM.room_at(x, y)


def distance_to(name: str, x: float, y: float) -> float:
    """How far (x, y) is from the two-room house's item's footprint (0 inside it), metres."""
    return TWO_ROOM.distance_to(name, x, y)


def _distance(it: Item, x: float, y: float) -> float:
    a = math.radians(it.face_deg)
    dx, dy = x - it.x, y - it.y
    # Into the item's own frame (front along u), where its footprint is axis-aligned.
    u = dx * math.cos(a) + dy * math.sin(a)
    v = -dx * math.sin(a) + dy * math.cos(a)
    hu, hv = _half(it)
    return math.hypot(max(abs(u) - hu, 0.0), max(abs(v) - hv, 0.0))


def _half(it: Item) -> tuple[float, float]:
    """The footprint's half sizes along (front, side) in the item's frame. Items are listed
    with their half sizes in the world's x and y as they stand."""
    a = round(it.face_deg / 90.0) % 2
    return (it.hy, it.hx) if a else (it.hx, it.hy)


# ---------------------------------------------------------------------- textures


def _img(rgb: tuple[int, int, int]) -> np.ndarray:
    return np.full((TEX, TEX, 3), rgb, dtype=np.uint8)


def _piano_front() -> np.ndarray:
    """An upright piano's face above the keys: glossy black, a music stand with open sheet
    music, and a gold maker's line."""
    im = _img((18, 18, 20))
    im[: TEX // 12] = (30, 30, 34)
    s0, s1 = int(TEX * 0.1), int(TEX * 0.42)
    im[s0:s1, int(TEX * 0.22) : int(TEX * 0.78)] = (238, 234, 222)
    im[s0:s1, TEX // 2 - 1 : TEX // 2 + 1] = (160, 155, 145)
    for r in range(s0 + 6, s1 - 4, 6):
        im[r, int(TEX * 0.25) : int(TEX * 0.75)] = (40, 40, 40)
        for c in range(int(TEX * 0.27), int(TEX * 0.74), 9):
            im[r - 2 : r + 1, c : c + 2] = (20, 20, 20)
    im[int(TEX * 0.47) : int(TEX * 0.49), int(TEX * 0.4) : int(TEX * 0.6)] = (200, 165, 70)
    return im


def _keys() -> np.ndarray:
    """A keyboard seen from the player's side: white keys with black ones in twos and threes."""
    im = _img((245, 243, 236))
    n = 26
    w = TEX / n
    for k in range(n + 1):
        im[:, min(int(k * w), TEX - 1)] = (90, 90, 90)
    for k in range(n):
        if k % 7 in (2, 6):
            continue
        c = int((k + 1) * w - w * 0.3)
        im[: int(TEX * 0.6), c : c + max(2, int(w * 0.6))] = (15, 15, 15)
    return im


def _key_fronts() -> np.ndarray:
    im = _img((240, 238, 230))
    for c in range(0, TEX, 5):
        im[:, c] = (120, 120, 120)
    return im


def _books() -> np.ndarray:
    """Four shelves of coloured book spines in a wooden frame."""
    rng = np.random.default_rng(7)
    im = _img((110, 72, 40))
    shelves = 4
    h = TEX // shelves
    for s in range(shelves):
        y0, y1 = s * h + 4, (s + 1) * h - 3
        x = 5
        while x < TEX - 8:
            w = int(rng.integers(3, 8))
            top = y0 + int(rng.integers(0, 6))
            col = tuple(int(c) for c in rng.integers(40, 230, 3))
            im[top:y1, x : x + w] = col
            x += w + 1
        im[y1 : y1 + 3] = (80, 50, 28)
    return im


def _fridge() -> np.ndarray:
    im = _img((228, 230, 234))
    im[int(TEX * 0.34) : int(TEX * 0.36)] = (150, 152, 156)
    im[:, :2] = im[:, -2:] = (190, 192, 196)
    return im


def _cabinets() -> np.ndarray:
    """Kitchen base cabinets: doors with knobs and a drawer row."""
    im = _img((205, 195, 172))
    for c in range(0, TEX, TEX // 4):
        im[:, c : c + 2] = (130, 120, 100)
    im[TEX // 4 : TEX // 4 + 2] = (130, 120, 100)
    for c in range(TEX // 8, TEX, TEX // 4):
        im[TEX // 8 - 2 : TEX // 8 + 2, c - 4 : c + 4] = (60, 60, 60)
        im[TEX // 2 - 3 : TEX // 2 + 3, c + 10 : c + 13] = (60, 60, 60)
    return im


def _rug() -> np.ndarray:
    im = _img((70, 90, 140))
    b = TEX // 10
    im[b:-b, b:-b] = (180, 160, 110)
    im[2 * b : -2 * b, 2 * b : -2 * b] = (120, 50, 50)
    return im


def _plain(rgb: tuple[int, int, int]) -> np.ndarray:
    return _img(rgb)


# (texture name) -> the picture on the item's front face, and its other faces' colour.
_FACES = {
    "piano_front": (_piano_front, (18, 18, 20)),
    "piano_keys": (_key_fronts, (240, 238, 230)),
    "books": (_books, (110, 72, 40)),
    "fridge": (_fridge, (228, 230, 234)),
    "cabinets": (_cabinets, (205, 195, 172)),
}


def textures(rot90: int) -> dict[str, bytes]:
    """The PNGs the house's textures load, keyed by the file names its MJCF names. ``rot90``
    turns a front picture the way the sim's cube textures show a -x face upright."""
    import imageio.v3 as iio

    out: dict[str, bytes] = {}

    def png(a: np.ndarray) -> bytes:
        return iio.imwrite("<bytes>", np.ascontiguousarray(a), extension=".png")

    for name, (draw, side) in _FACES.items():
        out[f"house_{name}.png"] = png(np.rot90(draw(), rot90))
        out[f"house_{name}_side.png"] = png(_plain(side))
    out["house_keys_top.png"] = png(_keys())
    out["house_keys_top_side.png"] = png(_plain((15, 15, 15)))
    out["house_rug.png"] = png(_rug())
    return out


# ---------------------------------------------------------------------- MJCF


def _quat(deg: float) -> str:
    a = math.radians(deg) / 2.0
    return f"{math.cos(a):.6f} 0 0 {math.sin(a):.6f}"


def _geom(parent: ET.Element, solid: bool, **kw: str) -> None:
    if not solid:
        kw.update(contype="0", conaffinity="0")
    ET.SubElement(parent, "geom", **kw)


def add(root: ET.Element, asset: ET.Element, world: ET.Element, layout: Layout = TWO_ROOM) -> None:
    """Add the house's walls, materials and furniture to the scene."""
    del root
    for name in _FACES:
        ET.SubElement(
            asset,
            "texture",
            name=f"house_{name}",
            type="cube",
            fileleft=f"house_{name}.png",
            fileright=f"house_{name}_side.png",
            filefront=f"house_{name}_side.png",
            fileback=f"house_{name}_side.png",
            fileup=f"house_{name}_side.png",
            filedown=f"house_{name}_side.png",
        )
        ET.SubElement(asset, "material", name=f"house_{name}", texture=f"house_{name}")
    ET.SubElement(
        asset,
        "texture",
        name="house_keys_top",
        type="cube",
        fileleft="house_keys_top_side.png",
        fileright="house_keys_top_side.png",
        filefront="house_keys_top.png",
        fileback="house_keys_top_side.png",
        fileup="house_keys_top_side.png",
        filedown="house_keys_top_side.png",
    )
    ET.SubElement(asset, "material", name="house_keys_top", texture="house_keys_top")
    ET.SubElement(asset, "texture", name="house_rug", type="2d", file="house_rug.png")
    ET.SubElement(asset, "material", name="house_rug", texture="house_rug")
    ET.SubElement(asset, "material", name="house_living", rgba="0.86 0.80 0.66 1")
    ET.SubElement(asset, "material", name="house_kitchen", rgba="0.72 0.82 0.86 1")
    ET.SubElement(asset, "material", name="house_hall", rgba="0.80 0.78 0.74 1")
    ET.SubElement(asset, "material", name="house_bedroom", rgba="0.78 0.74 0.86 1")
    ET.SubElement(asset, "material", name="house_study", rgba="0.70 0.80 0.68 1")

    _walls(world, layout)
    for it in layout.ITEMS:
        body = ET.SubElement(
            world,
            "body",
            name=f"house_{it.name.replace(' ', '_')}",
            pos=f"{it.x:g} {it.y:g} 0",
            quat=_quat(it.face_deg + 180.0),
        )
        _BUILD[it.build](body, _half(it))
    for k, (lx, ly) in enumerate(layout.lights):
        ET.SubElement(
            world,
            "light",
            name=f"house_light_{k}",
            pos=f"{lx:g} {ly:g} 2.3",
            dir="0 0 -1",
            diffuse="0.5 0.5 0.5",
        )


def _walls(world: ET.Element, layout: Layout) -> None:
    x0, y0, x1, y1 = layout.OUTER
    h = WALL_H / 2
    t = WALL_T

    def wall(name: str, ax: float, ay: float, bx: float, by: float, mat: str) -> None:
        cx, cy = (ax + bx) / 2, (ay + by) / 2
        sx, sy = max(abs(bx - ax) / 2, t), max(abs(by - ay) / 2, t)
        _geom(
            world,
            True,
            name=name,
            type="box",
            pos=f"{cx:g} {cy:g} {h:g}",
            size=f"{sx:g} {sy:g} {h:g}",
            material=mat,
        )

    for name, ax, ay, bx, by, mat in layout.WALLS:
        wall(name, ax, ay, bx, by, mat)
    # A lintel over each doorway, above his head.
    for k, d in enumerate(layout.DOORS):
        sx, sy = (d.half, t) if d.along == "x" else (t, d.half)
        _geom(
            world,
            False,
            name=f"house_lintel_{k}",
            type="box",
            pos=f"{d.x:g} {d.y:g} {(2.0 + WALL_H) / 2:g}",
            size=f"{sx:g} {sy:g} {(WALL_H - 2.0) / 2:g}",
            material="house_hall",
        )
    _geom(
        world,
        False,
        name="house_ceiling",
        type="box",
        pos=f"{(x0 + x1) / 2:g} {(y0 + y1) / 2:g} {WALL_H:g}",
        size=f"{(x1 - x0) / 2:g} {(y1 - y0) / 2:g} 0.02",
        rgba="0.93 0.93 0.9 1",
    )


# Each builder works in the item's own frame: its front is local -x, its footprint is
# (hx, hy) along local (x, y), and the floor is z = 0.


def _piano(b: ET.Element, hs: tuple[float, float]) -> None:
    hx, hy = hs
    case_hx = hx * 0.55
    case_cx = hx - case_hx
    _geom(
        b,
        True,
        name="piano_case",
        type="box",
        pos=f"{case_cx:g} 0 0.62",
        size=f"{case_hx:g} {hy:g} 0.62",
        material="house_piano_front",
    )
    _geom(
        b,
        True,
        name="piano_keybed",
        type="box",
        pos=f"{case_cx - case_hx - 0.13:g} 0 0.7",
        size=f"0.13 {hy - 0.04:g} 0.045",
        material="house_piano_keys",
    )
    _geom(
        b,
        False,
        name="piano_keys",
        type="box",
        pos=f"{case_cx - case_hx - 0.13:g} 0 0.747",
        size=f"0.12 {hy - 0.08:g} 0.003",
        material="house_keys_top",
    )
    for s in (-1, 1):
        _geom(
            b,
            True,
            name=f"piano_toe_{s}",
            type="box",
            pos=f"{case_cx - case_hx - 0.13:g} {s * (hy - 0.05):g} 0.33",
            size="0.03 0.03 0.33",
            rgba="0.07 0.07 0.08 1",
        )
    _geom(
        b,
        False,
        name="piano_lid",
        type="box",
        pos=f"{case_cx:g} 0 1.25",
        size=f"{case_hx + 0.01:g} {hy + 0.01:g} 0.012",
        rgba="0.1 0.1 0.11 1",
    )


def _bench(b: ET.Element, hs: tuple[float, float]) -> None:
    hx, hy = hs
    _geom(
        b,
        False,
        name="bench_top",
        type="box",
        pos="0 0 0.46",
        size=f"{hx:g} {hy:g} 0.03",
        rgba="0.16 0.12 0.1 1",
    )
    for sx in (-1, 1):
        for sy in (-1, 1):
            _geom(
                b,
                True,
                name=f"bench_leg_{sx}_{sy}",
                type="box",
                pos=f"{sx * (hx - 0.03):g} {sy * (hy - 0.03):g} 0.22",
                size="0.02 0.02 0.22",
                rgba="0.16 0.12 0.1 1",
            )


def _couch(b: ET.Element, hs: tuple[float, float]) -> None:
    hx, hy = hs
    red, dark = "0.6 0.17 0.15 1", "0.45 0.12 0.1 1"
    _geom(
        b,
        True,
        name="couch_base",
        type="box",
        pos="0.02 0 0.2",
        size=f"{hx - 0.02:g} {hy:g} 0.2",
        rgba=dark,
    )
    _geom(
        b,
        True,
        name="couch_back",
        type="box",
        pos=f"{hx - 0.1:g} 0 0.55",
        size=f"0.1 {hy:g} 0.3",
        rgba=red,
    )
    for s in (-1, 1):
        _geom(
            b,
            True,
            name=f"couch_arm_{s}",
            type="box",
            pos=f"0 {s * (hy - 0.08):g} 0.36",
            size=f"{hx:g} 0.08 0.16",
            rgba=red,
        )
    for k, y in enumerate((-hy / 2 + 0.04, hy / 2 - 0.04)):
        _geom(
            b,
            False,
            name=f"couch_cushion_{k}",
            type="box",
            pos="-0.05 " + f"{y:g} 0.44",
            size=f"{hx - 0.15:g} {hy / 2 - 0.12:g} 0.05",
            rgba="0.7 0.22 0.2 1",
        )
        _geom(
            b,
            False,
            name=f"couch_pillow_{k}",
            type="ellipsoid",
            pos=f"0.18 {y:g} 0.62",
            size="0.08 0.2 0.16",
            rgba="0.92 0.82 0.5 1",
        )


def _bookshelf(b: ET.Element, hs: tuple[float, float]) -> None:
    hx, hy = hs
    _geom(
        b,
        True,
        name="bookshelf",
        type="box",
        pos="0 0 0.95",
        size=f"{hx:g} {hy:g} 0.95",
        material="house_books",
    )


def _plant(b: ET.Element, hs: tuple[float, float]) -> None:
    del hs
    _geom(
        b,
        True,
        name="plant_pot",
        type="cylinder",
        pos="0 0 0.15",
        size="0.15 0.15",
        rgba="0.72 0.38 0.22 1",
    )
    for k, (x, y, z, r) in enumerate(
        ((0, 0, 0.5, 0.2), (0.12, 0.05, 0.75, 0.16), (-0.1, -0.06, 0.8, 0.15), (0, 0.1, 1.0, 0.13))
    ):
        _geom(
            b,
            False,
            name=f"plant_leaf_{k}",
            type="ellipsoid",
            pos=f"{x:g} {y:g} {z:g}",
            size=f"{r:g} {r * 0.9:g} {r * 1.3:g}",
            rgba="0.2 0.55 0.22 1",
        )


def _lamp(b: ET.Element, hs: tuple[float, float]) -> None:
    del hs
    _geom(
        b,
        True,
        name="lamp_base",
        type="cylinder",
        pos="0 0 0.02",
        size="0.15 0.02",
        rgba="0.2 0.2 0.2 1",
    )
    _geom(
        b,
        True,
        name="lamp_pole",
        type="cylinder",
        pos="0 0 0.8",
        size="0.015 0.78",
        rgba="0.25 0.25 0.25 1",
    )
    _geom(
        b,
        False,
        name="lamp_shade",
        type="cylinder",
        pos="0 0 1.55",
        size="0.2 0.15",
        rgba="0.95 0.88 0.65 1",
    )


def _rug_build(b: ET.Element, hs: tuple[float, float]) -> None:
    hx, hy = hs
    _geom(
        b,
        False,
        name="rug",
        type="box",
        pos="0 0 0.002",
        size=f"{hx:g} {hy:g} 0.002",
        material="house_rug",
    )


def _fridge_build(b: ET.Element, hs: tuple[float, float]) -> None:
    hx, hy = hs
    _geom(
        b,
        True,
        name="fridge",
        type="box",
        pos="0 0 0.9",
        size=f"{hx:g} {hy:g} 0.9",
        material="house_fridge",
    )
    for k, (z, half) in enumerate(((1.45, 0.18), (0.75, 0.3))):
        _geom(
            b,
            False,
            name=f"fridge_handle_{k}",
            type="box",
            pos=f"{-hx - 0.03:g} {hy - 0.08:g} {z:g}",
            size=f"0.015 0.015 {half:g}",
            rgba="0.6 0.62 0.65 1",
        )


def _counter(b: ET.Element, hs: tuple[float, float]) -> None:
    hx, hy = hs
    _geom(
        b,
        True,
        name="counter",
        type="box",
        pos="0 0 0.44",
        size=f"{hx:g} {hy:g} 0.44",
        material="house_cabinets",
    )
    _geom(
        b,
        False,
        name="counter_top",
        type="box",
        pos="-0.02 0 0.9",
        size=f"{hx + 0.02:g} {hy:g} 0.02",
        rgba="0.3 0.3 0.32 1",
    )
    _geom(
        b,
        False,
        name="kettle",
        type="cylinder",
        pos="0 0.4 1.0",
        size="0.08 0.09",
        rgba="0.75 0.2 0.15 1",
    )


def _table(b: ET.Element, hs: tuple[float, float]) -> None:
    hx, hy = hs
    wood = "0.55 0.4 0.25 1"
    _geom(
        b,
        False,
        name="table_top",
        type="box",
        pos="0 0 0.74",
        size=f"{hx:g} {hy:g} 0.02",
        rgba=wood,
    )
    for sx in (-1, 1):
        for sy in (-1, 1):
            _geom(
                b,
                True,
                name=f"table_leg_{sx}_{sy}",
                type="box",
                pos=f"{sx * (hx - 0.04):g} {sy * (hy - 0.04):g} 0.36",
                size="0.025 0.025 0.36",
                rgba=wood,
            )
    _geom(
        b,
        False,
        name="fruit_bowl",
        type="ellipsoid",
        pos="0 0 0.8",
        size="0.14 0.14 0.05",
        rgba="0.9 0.6 0.15 1",
    )


def _cat(b: ET.Element, hs: tuple[float, float]) -> None:
    """A ginger cat lying down, head up toward its front (local -x), tail curled behind."""
    del hs
    fur, dark = "0.84 0.49 0.2 1", "0.55 0.3 0.12 1"
    _geom(
        b,
        True,
        name="cat_body",
        type="ellipsoid",
        pos="0.03 0 0.09",
        size="0.17 0.1 0.09",
        rgba=fur,
    )
    _geom(b, True, name="cat_head", type="sphere", pos="-0.15 0 0.19", size="0.075", rgba=fur)
    for s in (-1, 1):
        _geom(
            b,
            False,
            name=f"cat_ear_{s}",
            type="box",
            pos=f"-0.15 {s * 0.045:g} 0.27",
            size="0.012 0.022 0.03",
            euler=f"{s * 15} 0 0",
            rgba=dark,
        )
        _geom(
            b,
            False,
            name=f"cat_eye_{s}",
            type="sphere",
            pos=f"-0.22 {s * 0.03:g} 0.2",
            size="0.012",
            rgba="0.15 0.6 0.2 1",
        )
        _geom(
            b,
            False,
            name=f"cat_paw_{s}",
            type="ellipsoid",
            pos=f"-0.17 {s * 0.05:g} 0.025",
            size="0.05 0.025 0.025",
            rgba=fur,
        )
    _geom(
        b,
        False,
        name="cat_nose",
        type="sphere",
        pos="-0.224 0 0.18",
        size="0.01",
        rgba="0.85 0.45 0.5 1",
    )
    _geom(
        b,
        False,
        name="cat_tail",
        type="capsule",
        fromto="0.17 0 0.04 0.1 0.13 0.03",
        size="0.022",
        rgba=dark,
    )


_BUILD = {
    "piano": _piano,
    "piano bench": _bench,
    "couch": _couch,
    "bookshelf": _bookshelf,
    "plant": _plant,
    "lamp": _lamp,
    "rug": _rug_build,
    "fridge": _fridge_build,
    "counter": _counter,
    "table": _table,
    "cat": _cat,
}
