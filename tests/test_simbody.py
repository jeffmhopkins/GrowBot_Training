import io
import json
import math

import numpy as np
import pytest

from growbot_train.simbody import Person, SimBody, Wiring, _w3c_euler, serve


@pytest.fixture(scope="module")
def sim() -> SimBody:
    return SimBody()


def _app_body(motion: dict) -> tuple[np.ndarray, np.ndarray]:
    """JBrain2 growbot-box.js bodyAccel/bodyGyro, verbatim in numpy: what the page makes of
    the phone's events before levelling them."""
    a, r = motion["accel"], motion["rate"]
    acc = np.array([a["y"], -a["x"], a["z"]])
    gyro = np.radians([r["beta"], -r["alpha"], r["gamma"]])
    return acc, gyro


def test_stock_wiring_undoes_the_runners_calibration() -> None:
    # growbot_runner.js: l = 90 + L_SIGN * deg(aL), r = 90 + R_SIGN * deg(aR), body.json's
    # cal L_SIGN -1, R_SIGN +1. The training env drives (right, left) = action * to_rad,
    # so the wire must land on exactly those joints.
    w = Wiring()
    for a_r, a_l in [(0.3, -0.2), (-0.7, 0.5), (0.0, 0.0)]:
        left = 90 + -1 * math.degrees(a_l)
        right = 90 + 1 * math.degrees(a_r)
        np.testing.assert_allclose(w.joints_rad(left, right), [a_r, a_l], atol=1e-12)


def test_swapped_pins_cross_the_legs() -> None:
    straight, crossed = Wiring(), Wiring(swap=True)
    np.testing.assert_allclose(crossed.joints_rad(120, 80), straight.joints_rad(80, 120))
    assert not np.allclose(crossed.joints_rad(120, 80), straight.joints_rad(120, 80))


def test_standing_robot_reads_gravity_like_the_training_env(sim: SimBody) -> None:
    sim.reset("standing")
    st = sim.state()
    assert st["screen_up"] and 30 < st["pitch_deg"] < 50 and abs(st["roll_deg"]) < 1
    acc, gyro = _app_body(sim.motion())
    # The page's body frame is the training env's (env.py CHANNELS: z up, x the nose), so
    # its unlevelled accelerometer is the sim's own sensor.
    d = sim.data
    np.testing.assert_allclose(acc, d.sensordata[sim._accel : sim._accel + 3], atol=1e-9)
    np.testing.assert_allclose(gyro, d.sensordata[sim._gyro : sim._gyro + 3], atol=1e-9)
    assert np.linalg.norm(acc) == pytest.approx(9.81, abs=0.05)
    # Nose up: gravity leans onto the body's -x, as a nose-up phone reads it.
    assert acc[0] > 4


def test_spinning_left_reads_a_positive_yaw_rate(sim: SimBody) -> None:
    sim.reset("lying")
    sim.lift(0.3)  # in the air, so the floor's friction can't stop the spin
    sim.data.qvel[5] = 1.0  # rad/s about world z, counter-clockwise from above
    sim.tick()
    _, gyro = _app_body(sim.motion())
    assert gyro[2] > 0.5


def test_orientation_follows_the_w3c_angles() -> None:
    flat = _w3c_euler(np.eye(3))
    assert flat["beta"] == pytest.approx(0) and flat["gamma"] == pytest.approx(0)
    # Top edge raised 30 degrees: beta +30.
    t = math.radians(30)
    tilt = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    assert _w3c_euler(tilt)["beta"] == pytest.approx(30)
    # Face down: beta wraps to 180, gamma stays in range.
    down = _w3c_euler(np.diag([1.0, -1.0, -1.0]))
    assert abs(down["beta"]) == pytest.approx(180) and abs(down["gamma"]) < 1e-9


def test_the_stock_get_up_stands_it_from_lying(sim: SimBody) -> None:
    sim.reset("lying")
    st = sim.state()
    assert st["z"] < 0.02 and abs(st["pitch_deg"]) < 5 and st["limp"]

    def smooth(x: float) -> float:
        x = min(1.0, max(0.0, x))
        return x * x * (3 - 2 * x)

    # Upstream's get-up (JBrain2 getup-wire.js): fold both legs to -80 degrees, then glide
    # back to neutral, on its own -1/-1 calibration.
    for i in range(int(4 / 0.04)):
        t = i * 0.04
        h = -80 * smooth(t / 0.7) if t < 1 else -80 * (1 - smooth(t - 1)) if t < 2 else 0.0
        sim.servos(90 - h, 90 + h)
        sim.tick()
    st = sim.state()
    assert st["z"] > 0.045 and 30 < st["pitch_deg"] < 50 and not st["limp"]


def test_on_its_back_it_lies_screen_down(sim: SimBody) -> None:
    sim.reset("back")
    assert not sim.state()["screen_up"]


def test_release_lets_the_legs_go(sim: SimBody) -> None:
    sim.reset("standing")
    sim.servos(30, 150)
    sim.release()
    for _ in range(10):
        sim.tick()
    assert sim.state()["limp"]
    np.testing.assert_allclose(sim.data.actuator_force, 0.0)


def test_a_shove_knocks_it_over(sim: SimBody) -> None:
    sim.reset("standing")
    # A light one rocks it and it settles back on its feet; a hard one rolls it over.
    sim.push((0.0, 6.0, 0.0), 0.15)
    rocked = 0.0
    for _ in range(50):
        sim.tick()
        rocked = max(rocked, abs(sim.state()["roll_deg"]))
    assert rocked > 15 and abs(sim.state()["roll_deg"]) < 5
    sim.push((0.0, 20.0, 0.0), 0.15)
    for _ in range(50):
        sim.tick()
    assert abs(sim.state()["roll_deg"]) > 60


def test_the_face_sits_where_the_camera_looks(sim: SimBody) -> None:
    sim.reset("standing")
    # Behind the tail end, which the camera faces, a face at sitting height is in view.
    sim.place_person(Person(x=-0.8, y=0.0, face_z=0.6))
    f = sim.face_in_view()
    assert f["visible"] and f["u"] == pytest.approx(0.5, abs=0.02)
    # Off to the robot's left (+y), it shows right of centre, as the phone's unmirrored
    # front camera has it (JBrain2 growbot-follow.js, THE FRAME).
    sim.place_person(Person(x=-0.8, y=0.3, face_z=0.6))
    assert sim.face_in_view()["u"] > 0.6
    # At the nose end it is out of view.
    sim.place_person(Person(x=0.8, y=0.0, face_z=0.6))
    assert not sim.face_in_view()["visible"]
    sim.place_person(Person(present=False))
    assert not sim.face_in_view()["visible"]


def test_camera_renders_when_a_gl_is_available(sim: SimBody) -> None:
    sim.reset("standing")
    sim.place_person(Person(x=-0.7, face_z=0.5))
    try:
        jpeg = sim.camera()
    except Exception as e:  # no OpenGL here: MUJOCO_GL=osmesa or egl renders headless
        pytest.skip(f"no offscreen GL: {e}")
    assert jpeg[:2] == b"\xff\xd8"


def test_json_lines_drive_the_same_body() -> None:
    sim = SimBody()
    reqs = [
        {"op": "reset", "start": "standing"},
        {"op": "servos", "l": 90, "r": 90},
        {"op": "tick", "n": 3},
        {"op": "person", "x": -1.0, "y": 0.2},
        {"op": "wiring", "swap": True},
        {"op": "nope"},
        {"op": "servos"},
    ]
    out = io.StringIO()
    serve(sim, io.StringIO("\n".join(json.dumps(r) for r in reqs) + "\n"), out)
    replies = [json.loads(line) for line in out.getvalue().splitlines()]
    assert [r["ok"] for r in replies] == [True] * 5 + [False, False]
    assert replies[0]["state"]["screen_up"]
    assert set(replies[2]["motion"]) == {"accel", "rate", "orientation"}
    assert sim.wiring.swap
