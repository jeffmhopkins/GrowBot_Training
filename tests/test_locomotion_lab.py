"""The locomotion bench's physics hold up: nothing moves on its own, a ratchet with equal
directions behaves like plain friction, it reproduces the known nose-first walk, and a
tail ratchet turns rocking into tail-first travel."""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "locomotion"))
lab = pytest.importorskip("lab")
surfaces = pytest.importorskip("growbot_train.surfaces")


def _scen(material: str) -> "lab.Scenario":
    return lab.Scenario("t", dict.fromkeys(lab.SURFACES, material))


def _paddle(t: float) -> tuple[float, float]:
    s = 0.8 * math.sin(2 * math.pi * t)
    return (s, -s)


@pytest.fixture
def even_ratchet(monkeypatch):
    hard = surfaces.FLOORS["hard"]
    mats = dict(hard.materials, even=surfaces.Ratchet(0.35, 0.35))
    monkeypatch.setitem(surfaces.FLOORS, "hard", surfaces.Floor("hard", mats))


def test_still_legs_stay_put_even_with_ratchets():
    r = lab.Bench(_scen("skin"), lab.Domain()).run(lambda t: (0.0, 0.0), 3.0)
    assert abs(r["v_fwd"]) < 1e-4 and not r["fell"]


def test_symmetric_ratchet_matches_plain_friction(even_ratchet):
    plain = lab.Bench(_scen("pla"), lab.Domain()).run(_paddle, 3.0)
    ratchet = lab.Bench(_scen("even"), lab.Domain()).run(_paddle, 3.0)
    assert ratchet["v_fwd"] == pytest.approx(plain["v_fwd"], abs=1e-3)


def test_upstream_body_paddles_nose_first():
    # Upstream body, friction and torque: the known nose-first walk, about 3 cm/s.
    bench = lab.Bench(_scen("sim"), lab.Domain(), body="rollout", torque_nm=1.5)
    assert bench.run(_paddle, 4.0)["v_fwd"] < -0.015


def test_tail_ratchet_turns_rocking_into_tail_first_travel():
    def rock(t: float) -> tuple[float, float]:
        s = 0.5 * math.sin(2 * math.pi * 2 * t)
        return (s, s)

    plain = lab.Bench(_scen("pla"), lab.Domain()).run(rock, 4.0)
    surf = {"tail": "skin", "belly": "pla", "toe_tail": "skin", "toe_nose": "skin"}
    ratchet = lab.Bench(lab.Scenario("r", surf), lab.Domain()).run(rock, 4.0)
    assert ratchet["v_fwd"] > plain["v_fwd"] + 0.005


def test_printed_build_rests_on_its_tail():
    bench = lab.Bench(_scen("pla"), lab.Domain())
    bench.reset()
    up = bench.d.xmat[bench.torso].reshape(3, 3)[2]
    assert 30 < math.degrees(math.atan2(up[0], up[2])) < 45
