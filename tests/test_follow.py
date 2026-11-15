import math
from pathlib import Path

import numpy as np
import pytest

from growbot_train import follow
from growbot_train.follow import Camera, CameraSpec, Follower, Person, Sighting, Tracker
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
    assert cam.in_view(Sighting(29.0, 19.0, 1.0))
    assert not cam.in_view(Sighting(31.0, 0.0, 1.0))
    assert not cam.in_view(Sighting(0.0, 21.0, 1.0))


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
    f = Follower(set_distance_m=1.0)
    assert f.command(Sighting(10.0, 0.0, 2.0), lost=False)[1] == 1.0
    assert f.command(Sighting(90.0, 0.0, 2.0), lost=False)[1] == 0.0


def test_the_follower_backs_away_when_too_close_whatever_the_heading() -> None:
    f = Follower(set_distance_m=1.0)
    assert f.command(Sighting(170.0, 0.0, 0.3), lost=False)[1] < 0.0


def test_a_person_behind_never_makes_it_back_up() -> None:
    f = Follower(set_distance_m=1.0)
    assert f.command(Sighting(180.0, 0.0, 2.0), lost=False)[1] >= 0.0


def test_stay_stops_walking_but_keeps_turning_and_tilting() -> None:
    f = Follower()
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


def test_the_nose_walk_is_only_ever_asked_to_back_away() -> None:
    out = follow.run(JsonPolicy(NOSE), seed=1, seconds=3.0, nominal=True)
    # Its speed input is how fast to back away: a forward command must reach it as 0,
    # and a backing one as its size, never with the sign flipped.
    for r in out["trace"]:
        assert r["skill_speed"] == pytest.approx(max(0.0, -r["cmd_speed"]))
    assert np.isfinite(out["summary"]["distance_err_m"])


def test_a_short_follow_run_on_look_scores_every_tick() -> None:
    out = follow.run(JsonPolicy(LOOK), seed=1, seconds=4.0, nominal=True)
    s = out["summary"]
    assert len(out["trace"]) == round(4.0 / 0.04) or s["fell"] == 1.0
    for key in ("in_view_pct", "aim_err_deg", "pitch_err_deg", "distance_err_m"):
        assert np.isfinite(s[key])
    # Look never takes a speed: the run must not have asked its env to walk.
    assert all(abs(r["robot_speed"]) < 0.1 for r in out["trace"])
