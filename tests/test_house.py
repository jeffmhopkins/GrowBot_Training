import math

import mujoco
import numpy as np
import pytest

from growbot_train import house
from growbot_train.simbody import SimBody


@pytest.fixture(scope="module")
def sim() -> SimBody:
    return SimBody(scene="house")


def test_rooms_and_distances() -> None:
    assert house.room_at(0.0, 0.0) == "living room"
    assert house.room_at(1.5, -1.0) == "kitchen"
    assert house.room_at(9.0, 0.0) == ""
    piano = house.ITEM["piano"]
    assert house.distance_to("piano", piano.x, piano.y) == 0.0
    # Straight out from its front (south), half a metre past the footprint.
    assert house.distance_to("piano", piano.x, piano.y - piano.hy - 0.5) == pytest.approx(0.5)
    assert house.distance_to("piano", piano.x - piano.hx - 0.3, piano.y) == pytest.approx(0.3)


def test_every_item_stands_inside_its_room() -> None:
    for it in house.ITEMS:
        x0, y0, x1, y1 = house.ROOMS[it.room]
        # Against a wall is inside; a millimetre over it is not.
        assert x0 - 1e-6 <= it.x - it.hx and it.x + it.hx <= x1 + 1e-6, it.name
        assert y0 - 1e-6 <= it.y - it.hy and it.y + it.hy <= y1 + 1e-6, it.name


def test_the_start_is_clear_and_he_stands(sim: SimBody) -> None:
    sim.reset("standing")
    st = sim.state()
    assert st["screen_up"] and 30 < st["pitch_deg"] < 50
    for it in house.ITEMS:
        if it.name != "rug":
            assert house.distance_to(it.name, st["x"], st["y"]) > 0.5, it.name


@pytest.mark.parametrize("yaw", [90.0, 180.0, -135.0])
def test_a_turned_start_keeps_its_stance(sim: SimBody, yaw: float) -> None:
    sim.reset("standing", yaw_deg=yaw, x=-1.0, y=0.5)
    st = sim.state()
    assert math.isclose(st["nose_deg"] % 360, yaw % 360, abs_tol=1.0)
    assert 30 < st["pitch_deg"] < 50 and abs(st["roll_deg"]) < 1
    assert st["x"] == pytest.approx(-1.0, abs=0.03) and st["y"] == pytest.approx(0.5, abs=0.03)


def test_a_wall_stops_him(sim: SimBody) -> None:
    # Set down just inside the west wall and shoved toward it: he ends against it, not
    # through it, as the bare room's walls (which collide with nothing) would let him.
    x0 = house.OUTER[0]
    sim.reset("lying", x=x0 + 0.12, y=1.0)
    d = sim.data
    for _ in range(50):
        d.qvel[0] = -1.0
        sim.tick()
    assert sim.state()["x"] > x0
    wall = sim.model.geom("house_wall_w").id
    robot = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "base_body")
    root = sim.model.body_rootid[robot]
    hit = any(
        wall in (c.geom1, c.geom2)
        and root
        in (
            sim.model.body_rootid[sim.model.geom_bodyid[c.geom1]],
            sim.model.body_rootid[sim.model.geom_bodyid[c.geom2]],
        )
        for c in d.contact[: d.ncon]
    )
    assert hit or sim.state()["x"] < x0 + 0.1


def test_the_camera_sees_the_room(sim: SimBody) -> None:
    import imageio.v3 as iio

    sim.reset("standing")
    img = np.asarray(iio.imread(sim.camera()))
    # The bookshelf straight ahead: many distinct colours, not a flat wall.
    assert len(np.unique(img.reshape(-1, 3) // 32, axis=0)) > 20


def test_the_bare_room_is_unchanged() -> None:
    plain = SimBody()
    names = {
        mujoco.mj_id2name(plain.model, mujoco.mjtObj.mjOBJ_GEOM, g)
        for g in range(plain.model.ngeom)
    }
    assert "wall_n" in names and not any(n and n.startswith("house_") for n in names)
    with pytest.raises(ValueError):
        SimBody(scene="castle")


def test_the_server_sets_him_down_anywhere_in_the_house() -> None:
    import io
    import json

    from growbot_train.simbody import serve

    sim = SimBody(scene="house")
    out = io.StringIO()
    serve(sim, io.StringIO('{"op": "reset", "start": "standing", "x": 1.4, "y": -0.5}\n'), out)
    st = json.loads(out.getvalue())["state"]
    assert abs(st["x"] - 1.4) < 0.05 and abs(st["y"] + 0.5) < 0.05


def test_nothing_is_buried_in_a_wall() -> None:
    # The walls are 6 cm thick on OUTER, so their inner faces sit 3 cm inside it.
    x0, y0, x1, y1 = house.OUTER
    for it in house.ITEMS:
        hx, hy = it.hx, it.hy  # half sizes in the world's x and y
        assert x0 + 0.03 - 1e-6 <= it.x - hx and it.x + hx <= x1 - 0.03 + 1e-6, it.name
        assert y0 + 0.03 - 1e-6 <= it.y - hy and it.y + hy <= y1 - 0.03 + 1e-6, it.name


@pytest.mark.parametrize("name", sorted(house.LAYOUTS))
def test_each_house_hangs_together(name: str) -> None:
    lay = house.LAYOUTS[name]
    for it in lay.ITEMS:
        x0, y0, x1, y1 = lay.ROOMS[it.room]
        # Off the walls' inner faces, 3 cm in from the room's edges.
        assert x0 + 0.03 - 1e-6 <= it.x - it.hx and it.x + it.hx <= x1 - 0.03 + 1e-6, it.name
        assert y0 + 0.03 - 1e-6 <= it.y - it.hy and it.y + it.hy <= y1 - 0.03 + 1e-6, it.name
    solid = [it for it in lay.ITEMS if it.name != "rug"]
    for i, a in enumerate(solid):
        for b in solid[i + 1 :]:
            apart = abs(a.x - b.x) >= a.hx + b.hx or abs(a.y - b.y) >= a.hy + b.hy
            assert apart, (a.name, b.name)
    # Every room reaches every other, and no wall stands in a doorway.
    for a in lay.ROOMS:
        for b in lay.ROOMS:
            assert a == b or lay.route(a, b), (a, b)
    for d in lay.DOORS:
        assert {lay.room_at(d.x + 0.1, d.y + 0.1), lay.room_at(d.x - 0.1, d.y - 0.1)} == set(
            d.rooms
        ), d.name
        for _, ax, ay, bx, by, _ in lay.WALLS:
            on = (
                min(ax, bx) - 1e-6 <= d.x <= max(ax, bx) + 1e-6
                and min(ay, by) - 1e-6 <= d.y <= max(ay, by) + 1e-6
            )
            assert not on, d.name
    x, y, _ = lay.start
    assert lay.room_at(x, y) == "living room"
    assert all(lay.distance_to(it.name, x, y) > 0.4 for it in solid)


def test_the_flat_stands_him_up_at_its_start() -> None:
    # With no reset of its own, a house puts him down at its start, not the origin.
    sb = SimBody(scene="flat")
    x, y, yaw = sb.layout.start
    st = sb.state()
    assert 30 < st["pitch_deg"] < 50 and abs(st["x"] - x) < 0.05
    # From the living room to the study is out through its door and in through the study's.
    assert [d.name for d in sb.layout.route("living room", "study")] == [
        "living room door",
        "study door",
    ]
    assert SimBody().layout is None
    with pytest.raises(ValueError):
        sb.layout.route("living room", "attic")
