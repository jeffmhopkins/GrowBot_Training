import math
from pathlib import Path

import numpy as np
import pytest

from growbot_train import follow
from growbot_train.follow import Camera, CameraSpec, Follower, Hold, Person, Sighting, Tracker
from growbot_train.policy import JsonPolicy

POLICIES = Path(__file__).parent.parent / "policies"
LOOK = POLICIES / "look-v6" / "policy_look.json"
NOSE = next((POLICIES / "nose-walk-1").glob("policy_*.json"))


def test_person_stays_in_the_room_and_walks_at_a_walking_pace() -> None:
    p = Person(seed=3)
    last = p.pos.copy()
    for _ in range(3000):
        p.step(0.04)
        assert 0.0 < np.linalg.norm(p.pos) <= p.ring_m[1] + 1e-9
        assert np.linalg.norm(p.pos - last) <= p.speed_m_s[1] * 0.04 + 1e-9
        last = p.pos.copy()


def test_the_same_seed_walks_the_same_path() -> None:
    a, b = Person(seed=5), Person(seed=5)
    for _ in range(500):
        a.step(0.04)
        b.step(0.04)
    assert np.allclose(a.pos, b.pos) and a.face_z == b.face_z


def test_the_camera_sees_only_inside_its_field_of_view() -> None:
    cam = Camera(CameraSpec(hfov_deg=60, vfov_deg=40), np.random.default_rng(0))
    assert cam.in_view(Sighting(0.0, 0.0, 1.0, 29.0, 19.0))
    assert not cam.in_view(Sighting(0.0, 0.0, 1.0, 31.0, 0.0))
    assert not cam.in_view(Sighting(0.0, 0.0, 1.0, 0.0, 21.0))
    assert not cam.in_view(Sighting(0.0, 0.0, 1.0, 0.0, 0.0, in_front=False))


def test_the_camera_reports_late_by_its_latency() -> None:
    spec = CameraSpec(latency_s=(0.12, 0.12), dropout_at_1m=0.0, dropout_per_m=0.0)
    cam = Camera(spec, np.random.default_rng(0))
    lag = round(0.12 / 0.04)
    seen = [cam.frame(Sighting(0.0, 0.0, 1.0)) for _ in range(lag + 1)]
    assert all(s is None for s in seen[:lag]) and seen[lag] is not None


def test_a_far_face_drops_out_more_than_a_near_one() -> None:
    spec = CameraSpec(latency_s=(0.04, 0.04), dropout_at_1m=0.05, dropout_per_m=0.2)

    def misses(dist: float) -> int:
        cam = Camera(spec, np.random.default_rng(1))
        return sum(cam.frame(Sighting(0.0, 0.0, dist)) is None for _ in range(2000))

    assert misses(3.0) > 2 * misses(1.0)


def test_the_tracker_keeps_a_remembered_bearing_on_the_same_spot_as_the_robot_turns() -> None:
    tr = Tracker()
    tr.update(Sighting(20.0, 0.0, 1.0), turned_deg=0.0)
    out = tr.update(None, turned_deg=15.0)
    assert out is not None and out.bearing_deg == pytest.approx(5.0)


def test_the_tracker_says_lost_after_a_short_gap() -> None:
    tr = Tracker(lost_after_s=0.5)
    tr.update(Sighting(0.0, 0.0, 1.0), 0.0)
    for _ in range(round(0.5 / 0.04) + 1):
        tr.update(None, 0.0)
    assert tr.lost


def test_the_follower_walks_only_once_roughly_facing_the_person() -> None:
    f = Follower(set_distance_m=1.0, frames=50)
    assert f.command(Sighting(10.0, 0.0, 2.0), lost=False)[1] == 1.0
    assert f.command(Sighting(90.0, 0.0, 2.0), lost=False)[1] == 0.0


def test_the_follower_backs_away_when_too_close_whatever_the_heading() -> None:
    f = Follower(set_distance_m=1.0)
    assert f.command(Sighting(170.0, 0.0, 0.3), lost=False)[1] < 0.0


def test_a_far_person_behind_never_makes_it_back_up() -> None:
    f = Follower(set_distance_m=1.0)
    assert f.command(Sighting(180.0, 0.0, 2.0), lost=False)[1] >= 0.0


def test_stay_stops_walking_but_keeps_turning_and_tilting() -> None:
    f = Follower(frames=50)
    f.voice("stay")
    yaw, speed, tilt = f.command(Sighting(40.0, 10.0, 3.0), lost=False)
    assert yaw == 40.0 and speed == 0.0 and tilt < 1.0
    f.voice("follow me")
    assert f.command(Sighting(0.0, 0.0, 3.0), lost=False)[1] == 1.0


def test_a_face_above_the_lens_lowers_the_camera_height() -> None:
    # Taller tips the lens toward the horizon, so looking up means going lower.
    f = Follower(tilt=0.0)
    f.command(Sighting(0.0, 20.0, 1.0), lost=False)
    assert f.tilt < 0.0


def test_voice_steps_the_set_distance_within_its_limits() -> None:
    f = Follower(set_distance_m=1.0, near_m=0.5, far_m=1.5, step_m=0.25)
    f.voice("come closer")
    assert f.set_distance_m == pytest.approx(0.75)
    for _ in range(5):
        f.voice("go further back")
    assert f.set_distance_m == pytest.approx(1.5)
    f.voice("come")
    assert f.set_distance_m == pytest.approx(0.5)
    with pytest.raises(ValueError):
        f.voice("dance")


def test_lost_spins_the_way_they_went_and_keeps_spinning_that_way() -> None:
    f = Follower(search_deg=90.0)
    assert f.command(None, lost=True) == (90.0, 0.0, f.tilt)
    f.command(Sighting(-20.0, 0.0, 1.0), lost=False)
    # As the robot turns, the remembered bearing swings past zero; the spin must not
    # flip with it, or it would stall facing the spot they left from.
    for remembered in (-20.0, -5.0, 3.0, 40.0):
        yaw, speed, _ = f.command(Sighting(remembered, 0.0, 1.0), lost=True)
        assert yaw == -90.0 and speed == 0.0


def test_the_camera_height_is_held_while_the_person_is_lost() -> None:
    f = Follower(tilt=0.3)
    f.command(Sighting(0.0, 30.0, 1.0), lost=True)
    assert f.tilt == 0.3


def test_the_true_sighting_of_a_face_dead_ahead_has_no_bearing() -> None:
    policy = JsonPolicy(LOOK)
    env = follow.env_for(policy)
    env.reset(seed=1, options={"retarget": False, "nominal": True})
    p = Person(seed=0)
    yaw = math.radians(env.yaw_deg())
    cam, _ = follow.lens_pose(env)
    p.pos = cam[:2] + 1.5 * np.array([math.cos(yaw), math.sin(yaw)])
    assert follow.true_sighting(env, p).bearing_deg == pytest.approx(0.0, abs=1e-6)


def _put_ahead_of_camera(env, p: Person, metres: float) -> None:
    head = math.radians(follow.camera_heading_deg(env))
    cam, _ = follow.lens_pose(env)
    p.pos = cam[:2] + metres * np.array([math.cos(head), math.sin(head)])


def test_nose_first_the_camera_looks_out_of_the_tail() -> None:
    # The nose walk's heading is its nose's; the phone's camera faces the other way, and
    # a person in front of the camera must read as dead ahead, not half a turn away.
    env = follow.env_for(JsonPolicy(NOSE))
    env.reset(seed=1, options={"retarget": False, "nominal": True})
    assert follow.camera_heading_deg(env) == pytest.approx(follow.wrap_deg(env.yaw_deg() + 180.0))
    p = Person(seed=0)
    _put_ahead_of_camera(env, p, 1.5)
    assert follow.true_sighting(env, p).bearing_deg == pytest.approx(0.0, abs=1e-6)


def test_the_floor_distance_is_shorter_than_the_line_to_a_raised_face() -> None:
    env = follow.env_for(JsonPolicy(LOOK))
    env.reset(seed=1, options={"retarget": False, "nominal": True})
    p = Person(seed=0)
    _put_ahead_of_camera(env, p, 1.0)
    assert follow.floor_distance_m(env, p) == pytest.approx(1.0, abs=1e-6)
    assert follow.true_sighting(env, p).distance_m > 1.2


def _angle_to(env, p: Person, heading_deg: float) -> float:
    cam, _ = follow.lens_pose(env)
    d = p.pos - cam[:2]
    return abs(follow.wrap_deg(heading_deg - math.degrees(math.atan2(d[1], d[0]))))


@pytest.mark.parametrize("off_deg", [0.0, 25.0, -25.0])
def test_the_nose_walk_is_aimed_with_its_nose_away_from_the_person(off_deg: float) -> None:
    env = follow.env_for(JsonPolicy(NOSE))
    env.reset(seed=1, options={"retarget": False, "nominal": True})
    head = math.radians(follow.camera_heading_deg(env) + off_deg)
    cam, _ = follow.lens_pose(env)
    p = Person(seed=0)
    p.pos = cam[:2] + 1.5 * np.array([math.cos(head), math.sin(head)])
    target = follow.skill_target_deg(env, follow.true_sighting(env, p).bearing_deg)
    assert _angle_to(env, p, target) == pytest.approx(180.0, abs=1e-6)
    env_look = follow.env_for(JsonPolicy(LOOK))
    env_look.reset(seed=1, options={"retarget": False, "nominal": True})
    p.pos = follow.lens_pose(env_look)[0][:2] + 1.5 * np.array([math.cos(head), math.sin(head)])
    target = follow.skill_target_deg(env_look, follow.true_sighting(env_look, p).bearing_deg)
    assert _angle_to(env_look, p, target) == pytest.approx(0.0, abs=1e-6)


def test_the_camera_distance_converts_back_to_the_floor_distance() -> None:
    env = follow.env_for(JsonPolicy(LOOK))
    env.reset(seed=1, options={"retarget": False, "nominal": True})
    p = Person(seed=0)
    for metres in (0.7, 1.0, 2.5):
        _put_ahead_of_camera(env, p, metres)
        floor = follow.floor_from_camera(env, follow.true_sighting(env, p)).distance_m
        assert floor == pytest.approx(follow.floor_distance_m(env, p), abs=1e-6)


def test_a_face_ahead_but_off_the_tipped_up_lens_axis_is_seen_where_the_image_puts_it() -> None:
    # The lens looks well above the horizon, so a face 40 degrees round the compass sits
    # nearer the middle of the image than 40 degrees: in view of a 60 degree camera.
    env = follow.env_for(JsonPolicy(LOOK))
    env.reset(seed=1, options={"retarget": False, "nominal": True})
    assert follow.lens_elevation_deg(env) > 30.0
    head = math.radians(follow.camera_heading_deg(env) + 40.0)
    cam, _ = follow.lens_pose(env)
    p = Person(seed=0)
    p.pos = cam[:2] + 1.0 * np.array([math.cos(head), math.sin(head)])
    s = follow.true_sighting(env, p)
    assert s.bearing_deg == pytest.approx(40.0, abs=1e-6)
    assert 0.0 < s.image_x_deg < 40.0 and s.in_front


def test_the_camera_height_holds_on_a_tick_with_no_new_frame() -> None:
    f = Follower(tilt=0.3)
    f.command(Sighting(0.0, 30.0, 1.0), lost=False, fresh=False)
    assert f.tilt == 0.3


def test_a_short_follow_run_on_look_scores_every_tick() -> None:
    out = follow.run(JsonPolicy(LOOK), seed=1, seconds=4.0, nominal=True)
    s = out["summary"]
    assert len(out["trace"]) == round(4.0 / 0.04) or s["fell"] == 1.0
    for key in ("in_view_pct", "aim_err_deg", "pitch_err_deg", "distance_err_m"):
        assert np.isfinite(s[key])
    # Look never takes a speed: the run must not have asked its env to walk.
    assert all(abs(r["robot_speed"]) < 0.1 for r in out["trace"])


def _settle(h: Hold, acts: list[np.ndarray], bearing: float = 2.0) -> np.ndarray:
    out = np.zeros(2)
    for a in acts:
        out = h.action(lambda a=a: a, bearing, standing=True, lost=False, tilt=1.0)
    return out


def test_hold_freezes_on_the_average_of_the_last_second_once_settled_on_target() -> None:
    h = Hold(settle_s=1.0)
    n = h.window
    acts = [np.array([math.sin(i), -math.sin(i)]) for i in range(n + 5)]
    _settle(h, acts[: n - 1])
    assert h.held is None
    _settle(h, acts[n - 1 :])
    # It holds the moment it has a full second on target, and keeps that pose.
    assert h.held is not None
    # Averaged as servo targets, after the env's tanh, and handed back in its input.
    assert np.allclose(np.tanh(h.held), np.mean(np.tanh(acts[:n]), axis=0))

    # Held: the policy isn't asked, and the same pose comes back every tick.
    def boom() -> np.ndarray:
        raise AssertionError("the policy was asked while held")

    out = h.action(boom, 10.0, standing=True, lost=False, tilt=1.0)
    assert np.array_equal(out, h.held)


def test_hold_averages_a_saturated_command_as_the_servo_saw_it() -> None:
    # The env clips its input at ACTION_LIMIT, so a command of 7 drives the servo no
    # further than one of 4; averaging raw inputs would push the pose too far.
    h = Hold(window=4, settle_s=0.1)
    _settle(h, [np.array([7.0, 0.0]), np.array([-4.0, 0.0])] * 2)
    assert h.held is not None and abs(h.held[0]) < 1e-6


def test_hold_waits_for_the_robot_to_settle_and_to_be_asked_to_stand() -> None:
    h = Hold(settle_s=1.0)
    acts = [np.ones(2)] * (2 * h.window)
    _settle(h, acts, bearing=8.0)
    assert h.held is None
    for a in acts:
        h.action(lambda a=a: a, 2.0, standing=False, lost=False, tilt=1.0)
    assert h.held is None


@pytest.mark.parametrize(
    ("bearing", "standing", "lost", "tilt"),
    [
        (16.0, True, False, 1.0),
        (2.0, False, False, 1.0),
        (2.0, True, True, 1.0),
        (2.0, True, False, 0.5),
    ],
)
def test_hold_lets_go_when_the_person_moves_or_the_robot_is_asked_for_more(
    bearing: float, standing: bool, lost: bool, tilt: float
) -> None:
    h = Hold()
    _settle(h, [np.ones(2)] * (2 * h.window))
    assert h.held is not None
    out = h.action(lambda: np.full(2, 7.0), bearing, standing=standing, lost=lost, tilt=tilt)
    assert h.held is None and np.array_equal(out, np.full(2, 7.0))


def test_hold_stays_between_its_enter_and_exit_angles() -> None:
    h = Hold(enter_deg=6.0, exit_deg=15.0)
    _settle(h, [np.ones(2)] * (2 * h.window))
    h.action(lambda: np.zeros(2), 12.0, standing=True, lost=False, tilt=1.0)
    assert h.held is not None


def test_hold_stops_look_creeping_on_target_and_it_still_turns_after() -> None:
    policy = JsonPolicy(LOOK)
    free = follow.hold_check(policy, 0, hold=False, nominal=True)
    held = follow.hold_check(policy, 0, hold=True, nominal=True)
    assert free is not None and held is not None and held["fell"] == 0.0
    assert held["creep_cm_s"] < 0.1 < free["creep_cm_s"]
    assert held["turn_err_deg"] < 20.0


def _seen(distance: float, bearing: float = 0.0) -> Sighting:
    return Sighting(bearing_deg=bearing, pitch_deg=0.0, distance_m=distance)


def test_the_follower_keeps_distance_on_the_average_not_each_waddle() -> None:
    f = Follower(set_distance_m=1.0)
    # Each waddle swings the camera's distance 0.3 m either side of a true 1.05 m.
    speeds = []
    for k in range(250):
        swing = 0.3 if (k // 12) % 2 else -0.3
        speeds.append(f.command(_seen(1.05 + swing), lost=False)[1])
    assert f.distance == pytest.approx(1.05, abs=0.06)
    assert speeds[-100:] == [0.0] * 100


def test_one_long_frame_never_starts_a_walk() -> None:
    f = Follower(set_distance_m=1.0)
    # Just found, the first frame reads a waddle's 0.25 m long: no walk yet...
    assert f.command(_seen(1.25), lost=False)[1] == 0.0
    # ...and a second of true readings averages it away before one could start.
    speeds = [f.command(_seen(1.0), lost=False)[1] for _ in range(50)]
    assert speeds == [0.0] * 50


def test_backing_keeps_its_15_cm_band() -> None:
    f = Follower(set_distance_m=1.0)
    f.frames, f.distance = 50, 0.88
    assert f.command(_seen(0.88), lost=False)[1] == 0.0
    f.distance = 0.8
    assert f.command(_seen(0.8), lost=False)[1] < 0.0


def test_a_walk_once_started_closes_the_gap_to_within_the_stop_band() -> None:
    f = Follower(set_distance_m=1.0)
    f.frames = 50
    assert f.command(_seen(1.2), lost=False)[1] == 1.0
    # Inside the 10 cm band but not yet within 5 cm: it keeps walking...
    f.distance = 1.08
    assert f.command(_seen(1.08), lost=False)[1] == 1.0
    f.distance = 1.04
    assert f.command(_seen(1.04), lost=False)[1] == 0.0
    # ...but standing, the same 8 cm doesn't start a walk.
    f.distance = 1.08
    assert f.command(_seen(1.08), lost=False)[1] == 0.0


def test_a_walk_lets_go_of_the_heading_only_ten_degrees_wider() -> None:
    f = Follower(set_distance_m=1.0)
    f.frames, f.distance = 50, 1.5
    assert f.command(_seen(1.5, bearing=35.0), lost=False)[1] == 0.0
    assert f.command(_seen(1.5, bearing=20.0), lost=False)[1] == 1.0
    assert f.command(_seen(1.5, bearing=35.0), lost=False)[1] == 1.0
    assert f.command(_seen(1.5, bearing=45.0), lost=False)[1] == 0.0


def test_losing_the_person_forgets_their_distance() -> None:
    f = Follower()
    f.command(_seen(2.0), lost=False)
    f.command(_seen(2.0), lost=True)
    assert f.distance is None and not f.walking
