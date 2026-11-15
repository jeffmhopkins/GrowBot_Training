import math
from pathlib import Path

import numpy as np
import pytest

from growbot_train import follow
from growbot_train.evaluate import run_episode
from growbot_train.follow import Person
from growbot_train.policy import JsonPolicy

POLICIES = Path(__file__).parent.parent / "policies"
LOOK = JsonPolicy(POLICIES / "look-v9" / "policy_look.json")
NOSE = JsonPolicy(next((POLICIES / "nose-walk-1").glob("policy_*.json")))


def _env(policy: JsonPolicy, frame: float = 0.0, err: float = 0.0):
    env = follow.env_for(policy)
    env.cfg.reward.frame = frame
    env.reset(seed=1, options={"retarget": False, "nominal": True, "target_error_deg": err})
    return env


def _face_as_person(env) -> Person:
    p = Person(seed=0)
    az = math.radians(env.person_deg())
    p.pos = env.data.qpos[0:2] + env._face_dist * np.array([math.cos(az), math.sin(az)])
    p.face_z = env._face_z
    return p


@pytest.mark.parametrize("policy", [LOOK, NOSE])
def test_the_virtual_face_lands_where_the_tracker_would_see_it(policy) -> None:
    env = _env(policy, err=20.0)
    x, y, _ = env.face_image_deg()
    s = follow.true_sighting(env, _face_as_person(env))
    assert (x, y) == pytest.approx((s.image_x_deg, s.image_y_deg), abs=1e-6)


def test_a_nose_first_skill_keeps_the_person_behind_its_nose() -> None:
    env = _env(NOSE)
    assert env.person_deg() == pytest.approx(follow.wrap_deg(env.target_deg + 180.0))
    assert _env(LOOK).person_deg() == pytest.approx(_env(LOOK).target_deg)


def test_a_face_dead_ahead_is_in_frame_and_one_behind_is_not() -> None:
    assert _env(LOOK, err=0.0).face_image_deg()[2]
    assert not _env(LOOK, err=180.0).face_image_deg()[2]


def test_the_face_is_redrawn_in_range_per_target_without_moving_other_draws() -> None:
    env = _env(LOOK)
    r = env.cfg.reward
    seen = set()
    for t in range(20):
        env._new_target(float(t))
        assert r.frame_distance_m[0] <= env._face_dist <= r.frame_distance_m[1]
        assert r.frame_face_height_m[0] <= env._face_z <= r.frame_face_height_m[1]
        seen.add(env._face_dist)
    assert len(seen) == 20
    before = env.np_random.bit_generator.state
    env._new_target(99.0)
    assert env.np_random.bit_generator.state == before


def _frame_term(env) -> float:
    _, _, _, _, info = env.step(np.zeros(2))
    return info["terms"].get("frame", 0.0)


def test_frame_pays_in_frame_only_and_centres_only_when_standing() -> None:
    assert _frame_term(_env(LOOK, frame=1.0, err=0.0)) > 0.0
    assert _frame_term(_env(LOOK, frame=1.0, err=180.0)) == 0.0
    assert _frame_term(_env(LOOK, frame=0.0, err=0.0)) == 0.0
    # Standing, 20 degrees across a 60-wide picture pays less than dead centre.
    env = _env(LOOK, frame=1.0, err=20.0)
    _, _, _, _, info = env.step(np.zeros(2))
    assert abs(info["image_x_deg"]) > 10.0
    x = info["image_x_avg_deg"]
    assert x == pytest.approx(info["image_x_deg"], abs=2.0)
    assert info["terms"]["frame"] == pytest.approx(1.0 - (x / 30.0) ** 2)
    # Walking, anywhere in the picture pays in full.
    walk = _env(NOSE, frame=1.0, err=0.0)
    walk.speed_cmd = 1.0
    _, _, _, _, info = walk.step(np.zeros(2))
    assert info["in_frame"] and info["terms"]["frame"] == 1.0


def test_evaluate_reports_time_in_frame() -> None:
    env = follow.env_for(LOOK)
    env.max_steps = 50
    m = run_episode(env, LOOK, seed=0, options={"target_error_deg": 0.0, "nominal": True})
    assert 0.0 <= m["frame_pct"] <= 100.0 and m["img_x"] >= 0.0


def test_a_stops_aim_is_averaged_so_a_periodic_sway_cancels() -> None:
    env = _env(LOOK, frame=1.0, err=0.0)
    # A waddle's sway: 20 degrees either side, once a second.
    tick = iter(range(10**6))
    env.face_image_deg = lambda: (20.0 * math.sin(2 * math.pi * next(tick) * 0.04), 0.0, True)
    env._img_x_avg = 0.0
    paid = []
    for _ in range(150):
        _, _, _, _, info = env.step(np.zeros(2))
        paid.append(info["terms"]["frame"])
    # Frame to frame it swings up to 20 degrees off (worth 0.56), but the average stays
    # within a few degrees of centre, so the aim pays nearly in full throughout.
    assert abs(info["image_x_avg_deg"]) < 4.0
    assert min(paid[50:]) > 0.98
