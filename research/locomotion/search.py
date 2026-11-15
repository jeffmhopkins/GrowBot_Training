"""Search the best tail-first gait for each contact-material scenario, then test it on
robots and floors it never trained on.

    python research/locomotion/search.py                 # every scenario
    python research/locomotion/search.py --only m2 m7    # some of them
    python research/locomotion/search.py --quick         # a fast smoke run

Each scenario writes results/<name>.json: the gait, its speed on the as-designed robot,
and its spread over held-out random robots. `report.py` turns those into the table.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from lab import (
    DEFAULT_BODY,
    SURFACES,
    Bench,
    Domain,
    KeyframeGait,
    Scenario,
    domains,
    score,
)

OUT = Path(__file__).parent / "results"


def S(name: str, note: str, tail, belly, toe_tail, toe_nose, **kw) -> Scenario:
    surf = dict(zip(SURFACES, (tail, belly, toe_tail, toe_nose), strict=True))
    return Scenario(name, surf, note, **kw)


SCENARIOS: list[Scenario] = [
    S("s0", "Upstream sim friction everywhere (1.2), the current training setup", *["sim"] * 4),
    S("m1", "Bare print everywhere", "pla", "pla", "pla", "pla"),
    S("m2", "Dipped legs, bare body", "pla", "pla", "rubber", "rubber"),
    S("m3", "Dipped body, bare legs", "rubber", "rubber", "pla", "pla"),
    S("m4", "Dipped legs, PTFE tape on tail and belly", "ptfe", "ptfe", "rubber", "rubber"),
    S("m5", "Bare legs, PTFE tape on tail and belly", "ptfe", "ptfe", "pla", "pla"),
    S("m6", "Dipped everything", *["rubber"] * 4),
    S(
        "m7",
        "Rubber on the tail-side face of each foot, PTFE on its nose-side face; PTFE tail",
        "ptfe",
        "ptfe",
        "rubber",
        "ptfe",
    ),  # fmt: skip
    S(
        "m8",
        "Rubber on the nose-side face of each foot, PTFE on its tail-side face; PTFE tail",
        "ptfe",
        "ptfe",
        "ptfe",
        "rubber",
    ),  # fmt: skip
    S("m9", "Felt pad on tail and belly, dipped legs", "felt", "felt", "rubber", "rubber"),
    S("r1", "Bristle/skin ratchet on the tail, bare legs", "skin", "pla", "pla", "pla"),
    S("r2", "Bristle/skin ratchet on the tail, dipped legs", "skin", "ptfe", "rubber", "rubber"),
    S("r3", "Ratchet on both feet, PTFE tail", "ptfe", "ptfe", "skin", "skin"),
    S("r4", "Ratchet on feet and tail", "skin", "ptfe", "skin", "skin"),
    S("r5", "Weak ratchet on feet and tail", "skin_weak", "ptfe", "skin_weak", "skin_weak"),
    # Standing tall (camera nearer level): the gait must average at least 47 degrees nose-up
    # (the printed build tops out near 50 degrees standing still: see the report).
    S("t1", "TALL: bare print everywhere", *["pla"] * 4, min_pitch_deg=47),
    S("t2", "TALL: dipped legs, bare body", "pla", "pla", "rubber", "rubber", min_pitch_deg=47),
    S(
        "t3",
        "TALL: dipped legs, PTFE tape on the tail edge and belly",
        "ptfe",
        "ptfe",
        "rubber",
        "rubber",
        min_pitch_deg=47,
    ),  # fmt: skip
    S("t4", "TALL: dipped body, bare legs", "rubber", "rubber", "pla", "pla", min_pitch_deg=47),
    S("t5", "TALL: ratchet on feet and tail", "skin", "ptfe", "skin", "skin", min_pitch_deg=47),
]
# Jeff's constraint (2026-11-14): keep the build stock, liquid dip rubber is the only
# allowed change. Partial dips count: one face of each foot, or only the tail edge.
STOCK: list[Scenario] = [
    S("d1", "Stock: bare print everywhere", *["pla"] * 4),
    S("d2", "Dip the feet (whole leg tip)", "pla", "pla", "rubber", "rubber"),
    S("d3", "Dip the body underside (tail edge and belly)", "rubber", "rubber", "pla", "pla"),
    S("d4", "Dip everything that touches the floor", *["rubber"] * 4),
    S("d5", "Dip only the nose-side face of each foot", "pla", "pla", "pla", "rubber"),
    S("d6", "Dip only the tail-side face of each foot", "pla", "pla", "rubber", "pla"),
    S("d7", "Dip only the tail edge", "rubber", "pla", "pla", "pla"),
    S("d8", "Dip the nose-side foot faces and the tail edge", "rubber", "pla", "pla", "rubber"),
    S("d9", "Dip the belly but not the tail edge", "pla", "rubber", "pla", "pla"),
    S("dt1", "TALL: stock", *["pla"] * 4, min_pitch_deg=47),
    S("dt2", "TALL: dip the feet", "pla", "pla", "rubber", "rubber", min_pitch_deg=47),
    S("dt3", "TALL: dip the nose-side foot faces", "pla", "pla", "pla", "rubber", min_pitch_deg=47),
]
# Reprinted legs (Jeff, 2026-11-14): the body stays stock, the foot changes shape, and
# any part of it may be dipped. See lab.Scenario.foot.
F = {"belly": "pla", "tail": "pla"}
FEET: list[Scenario] = [
    Scenario("f1", {**F, "toe_tail": "pla", "toe_nose": "pla"}, "Round tip, bare", foot="round"),
    Scenario("f2", {**F, "toe_tail": "rubber", "toe_nose": "rubber"}, "Round tip, dipped",
             foot="round"),
    Scenario("f3", {**F, "toe_tail": "pla", "toe_nose": "pla"},
             "Sole bevelled 25 deg, nose-side edge lowest, bare", foot="bevel", foot_deg=25),
    Scenario("f4", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "Sole bevelled 25 deg, nose-side edge lowest, that edge dipped", foot="bevel",
             foot_deg=25),
    Scenario("f5", {**F, "toe_tail": "pla", "toe_nose": "pla"},
             "Sole bevelled 25 deg, tail-side edge lowest, bare", foot="bevel", foot_deg=-25),
    Scenario("f6", {**F, "toe_tail": "rubber", "toe_nose": "pla"},
             "Sole bevelled 25 deg, tail-side edge lowest, that edge dipped", foot="bevel",
             foot_deg=-25),
    Scenario("f7", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "L foot: 20 mm toe toward the nose, toe dipped, round bare heel",
             foot="toe_nose", foot_deg=20),
    Scenario("f8", {**F, "toe_tail": "rubber", "toe_nose": "pla"},
             "L foot: 20 mm toe toward the tail, toe dipped, round bare heel",
             foot="toe_tail", foot_deg=20),
    Scenario("f9", {**F, "toe_tail": "rubber", "toe_nose": "rubber"},
             "Legs 10 mm longer, feet dipped", leg_extra_m=0.01),
    Scenario("f10", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "Legs 10 mm longer, nose-side foot face dipped", leg_extra_m=0.01),
    Scenario("f11", {**F, "toe_tail": "rubber", "toe_nose": "rubber"},
             "Legs 10 mm shorter, feet dipped", leg_extra_m=-0.01),
    Scenario("f12", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "L foot toward the nose, toe dipped, legs 10 mm longer", foot="toe_nose",
             foot_deg=20, leg_extra_m=0.01),
    # Curved "cam" feet (Jeff): the sole's distance from the axle changes with the angle.
    Scenario("f13", {**F, "toe_tail": "rubber", "toe_nose": "rubber"},
             "Curved foot, an arc round the axle (+-20 deg, a wheel rim), dipped",
             foot="cam", foot_deg=20),
    Scenario("f14", {**F, "toe_tail": "rubber", "toe_nose": "rubber"},
             "Cam foot, longer toward the nose (+30 mm/rad), dipped", foot="cam",
             foot_deg=20, cam_m_per_rad=0.03),
    Scenario("f15", {**F, "toe_tail": "rubber", "toe_nose": "rubber"},
             "Cam foot, longer toward the tail (-30 mm/rad), dipped", foot="cam",
             foot_deg=20, cam_m_per_rad=-0.03),
    Scenario("f16", {**F, "toe_tail": "pla", "toe_nose": "pla"},
             "Cam foot, longer toward the nose (+30 mm/rad), bare", foot="cam",
             foot_deg=20, cam_m_per_rad=0.03),
    # Jeff's pick: cam foot, grippy toe, bare (slippery) heel, both ways round.
    Scenario("f17", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "Cam foot longer toward the nose, nose-side (toe) dipped, heel bare",
             foot="cam", foot_deg=20, cam_m_per_rad=0.03),
    Scenario("f18", {**F, "toe_tail": "rubber", "toe_nose": "pla"},
             "Cam foot longer toward the tail, tail-side (toe) dipped, heel bare",
             foot="cam", foot_deg=20, cam_m_per_rad=-0.03),
    Scenario("f19", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "Cam foot longer toward the tail, nose-side (heel) dipped, toe bare",
             foot="cam", foot_deg=20, cam_m_per_rad=-0.03),
    # Jeff: a rubber edge across the foot's thickness, chamfered so it only bites when the
    # body rolls onto that leg; level, both feet stand on bare print.
    Scenario("f20", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "Outer edge of each foot chamfered 5 deg and dipped, sole bare",
             foot="edge_out", foot_deg=5),
    Scenario("f21", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "Inner edge of each foot chamfered 5 deg and dipped, sole bare",
             foot="edge_in", foot_deg=5),
]  # fmt: skip
# Jeff (2026-11-14): rocker feet that grow a long way toward the tail on longer legs. A
# hard backward push should lift the whole body onto its feet and rock it over them tail
# first, the long tail-side sole keeping it from going over, then set it down further on:
# a hop. Which part to dip is open, so each is tried.
VAULT: list[Scenario] = [
    Scenario("v1", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "Rocker foot (+-30 deg, 50 mm/rad toward the tail), legs +15 mm, nose side dipped",
             foot="cam", foot_deg=30, cam_m_per_rad=-0.05, leg_extra_m=0.015, rest_leg_rad=-1.0),
    Scenario("v2", {**F, "toe_tail": "rubber", "toe_nose": "pla"},
             "Rocker foot (+-30 deg, 50 mm/rad toward the tail), legs +15 mm, tail side dipped",
             foot="cam", foot_deg=30, cam_m_per_rad=-0.05, leg_extra_m=0.015, rest_leg_rad=-1.0),
    Scenario("v3", {**F, "toe_tail": "rubber", "toe_nose": "rubber"},
             "Rocker foot (+-30 deg, 50 mm/rad toward the tail), legs +15 mm, all dipped",
             foot="cam", foot_deg=30, cam_m_per_rad=-0.05, leg_extra_m=0.015, rest_leg_rad=-1.0),
    Scenario("v4", {**F, "toe_tail": "pla", "toe_nose": "rubber"},
             "Big rocker foot (+-40 deg, 70 mm/rad toward the tail), legs +25 mm, nose side dipped",
             foot="cam", foot_deg=40, cam_m_per_rad=-0.07, leg_extra_m=0.025, rest_leg_rad=-1.0),
]  # fmt: skip
# A clip Jeff found (2026-11-14) of someone else's GrowBot hopping camera-first: stood
# nearly upright on its tail, legs swung back as props, kicking alternately so it rocks
# side to side. Set down with the legs at +0.6 rad this body levers itself up to ~58
# degrees, its tallest stance; the gait must keep it above 50.
WADDLE: list[Scenario] = [
    Scenario("w1", {**F, "toe_tail": "pla", "toe_nose": "pla"},
             "Tall waddle: stood up on its tail, stock bare feet", min_pitch_deg=50,
             rest_leg_rad=0.6),
    Scenario("w2", {**F, "toe_tail": "rubber", "toe_nose": "rubber"},
             "Tall waddle: stood up on its tail, dipped feet", min_pitch_deg=50,
             rest_leg_rad=0.6),
    Scenario("w3", {**F, "toe_tail": "rubber", "toe_nose": "rubber"},
             "Tall waddle, dipped feet, no height floor (just stood up to start)",
             rest_leg_rad=0.6),
]  # fmt: skip
# Jeff (2026-11-15): backing away from the camera is the nose-first low waddle, the body
# dropped lower than its resting stance (about 35-40 degrees nose-up as set down) to do it,
# like the very first policy. Stock bare legs, the servos' whole swing.
NOSE: list[Scenario] = [
    Scenario("n1", {**F, "toe_tail": "pla", "toe_nose": "pla"},
             "Nose-first low waddle, stock bare feet", direction=-1.0, max_pitch_deg=25.0),
]  # fmt: skip
SCENARIOS += STOCK + FEET + VAULT + WADDLE + NOSE
BY_NAME = {s.name: s for s in SCENARIOS}

# Filled in each worker by _init so every candidate reuses compiled models.
_BENCHES: list[Bench] = []
_SECONDS = 8.0


def _init(scen_name: str, doms: list[Domain], torque: float, seconds: float, body: str) -> None:
    global _BENCHES, _SECONDS
    _BENCHES = [Bench(BY_NAME[scen_name], d, body=body, torque_nm=torque) for d in doms]
    _SECONDS = seconds


def _eval(x: np.ndarray) -> float:
    """Half the average floor's score, half the worst floor's: one gait has to do on
    both hard floors and carpet, since the robot won't know which it is on."""
    gait = KeyframeGait(x, _BENCHES[0].scen.swing_rad)
    per: dict[str, list[dict]] = {}
    for b in _BENCHES:
        per.setdefault(b.dom.floor, []).append(b.run(gait, _SECONDS))
    sc = _BENCHES[0].scen
    s = [score(rs, sc.min_pitch_deg, sc.max_pitch_deg, sc.direction) for rs in per.values()]
    return 0.5 * float(np.mean(s)) + 0.5 * min(s)


def _eval_detail(args) -> dict:
    x, i = args
    gait = KeyframeGait(np.asarray(x), _BENCHES[i].scen.swing_rad)
    return _BENCHES[i].run(gait, _SECONDS)


def seeds(rng: np.random.Generator) -> list[np.ndarray]:
    """Hand-made starting gaits (raw, pre-squash parameters): in-phase rock, alternating
    paddle, and quick-snap/slow-push shapes, at a few speeds."""
    K = KeyframeGait.K
    out = []
    ph = np.arange(K) / K
    for hz_raw in (-1.5, 0.0, 1.5):
        for shape in ("sin", "saw", "saw_rev"):
            if shape == "sin":
                w = np.sin(2 * math.pi * ph)
            elif shape == "saw":
                w = 2 * ph - 1
            else:
                w = 1 - 2 * ph
            for lag in (0, K // 2, K // 4):
                right = 0.8 * w
                left = 0.8 * np.roll(w, lag)
                out.append(np.concatenate([[hz_raw], np.arctanh(right), np.arctanh(left)]))
    rng.shuffle(out)
    return out


def cma(f, x0: np.ndarray, sigma: float, pop: int, gens: int, pool: Pool, log=None):
    """A plain (mu/mu_w, lambda) CMA-ES, maximising f."""
    n = len(x0)
    mu = pop // 2
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w /= w.sum()
    mueff = 1 / np.sum(w**2)
    cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
    cs = (mueff + 2) / (n + mueff + 5)
    c1 = 2 / ((n + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (n + 1)) - 1) + cs
    chin = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
    mean, C = x0.copy(), np.eye(n)
    pc, ps = np.zeros(n), np.zeros(n)
    rng = np.random.default_rng(int(abs(x0).sum() * 1e6) % (1 << 31))
    best_x, best_f = x0.copy(), -np.inf
    for g in range(gens):
        eigval, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(eigval, 1e-20))
        z = rng.standard_normal((pop, n))
        y = z * D @ B.T
        xs = mean + sigma * y
        xs = np.clip(xs, -4, 4)
        fs = np.array(pool.map(f, list(xs)))
        order = np.argsort(-fs)
        if fs[order[0]] > best_f:
            best_f, best_x = float(fs[order[0]]), xs[order[0]].copy()
        old = mean
        mean = w @ xs[order[:mu]]
        ymean = (mean - old) / sigma
        invsqrt = B @ np.diag(1 / D) @ B.T
        ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * invsqrt @ ymean
        hsig = np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (g + 1))) / chin < 1.4 + 2 / (
            n + 1
        )
        pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mueff) * ymean
        yk = (xs[order[:mu]] - old) / sigma
        C = (
            (1 - c1 - cmu) * C
            + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
            + cmu * (yk.T * w) @ yk
        )
        sigma *= math.exp((cs / damps) * (np.linalg.norm(ps) / chin - 1))
        if log and g % 10 == 0:
            log(f"    gen {g:3d}  best {best_f * 100:6.2f}  gen-best {fs[order[0]] * 100:6.2f}")
    return best_x, best_f


def _stats(rs: list[dict]) -> dict:
    rs = [r for r in rs if not r.get("bad_start")]
    v = np.array([r["v_fwd"] for r in rs]) * 100
    return {
        "mean_cm_s": float(v.mean()),
        "p10_cm_s": float(np.percentile(v, 10)),
        "min_cm_s": float(v.min()),
        "falls": int(sum(r["fell"] for r in rs)),
        "n": len(rs),
        "side_cm_s": float(np.mean([abs(r["v_side"]) for r in rs]) * 100),
        "yaw_deg_s": float(np.mean([abs(r["yaw_deg_s"]) for r in rs])),
        "lifted": float(np.mean([r.get("lifted", 0.0) for r in rs])),
    }


def run_scenario(scen: Scenario, args) -> dict:
    t0 = time.time()
    nominal_doms = [Domain(floor="hard"), Domain(floor="carpet")]
    train_doms = nominal_doms + domains(args.train_domains, seed=1, com_x=args.com_x)
    test_doms = domains(args.test_domains, seed=2, com_x=args.com_x)
    rng = np.random.default_rng(0)
    init = (args.torque, args.seconds, args.body)
    print(f"[{scen.name}] {scen.note}", flush=True)
    with Pool(args.workers, _init, (scen.name, train_doms, *init)) as pool:
        # Screen the hand-made seeds and a batch of random gaits, then climb from the best few.
        cands = seeds(rng) + [rng.normal(0, 1.2, KeyframeGait.dim()) for _ in range(args.random)]
        fs = np.array(pool.map(_eval, cands))
        starts = [cands[i] for i in np.argsort(-fs)[: args.restarts]]
        prior = OUT / f"{scen.name}{args.warm}.json" if args.warm is not None else None
        if prior and prior.exists():
            # Carry on from an earlier search's best gait, retuned on these robots.
            starts[0] = np.array(json.loads(prior.read_text())["x"])
        print(f"    screened {len(cands)}: best {fs.max() * 100:.2f} cm/s-score", flush=True)
        best_x, best_f = None, -np.inf
        for k, x0 in enumerate(starts):
            x, fx = cma(_eval, x0, 0.6, args.pop, args.gens, pool, log=print)
            print(f"  restart {k}: {fx * 100:.2f}", flush=True)
            if fx > best_f:
                best_x, best_f = x, fx
    with Pool(args.workers, _init, (scen.name, test_doms + nominal_doms, *init)) as pool:
        allr = pool.map(_eval_detail, [(best_x, i) for i in range(len(test_doms) + 2)])
    test, nominal = allr[:-2], allr[-2:]
    gait = KeyframeGait(best_x, scen.swing_rad)
    out = {
        "name": scen.name,
        "note": scen.note,
        "surfaces": dict(scen.surfaces),
        "body": args.body,
        "torque_nm": args.torque,
        "com_x_m": args.com_x,
        "train_domains": args.train_domains,
        "train_score": best_f,
        "nominal": {"hard": nominal[0], "carpet": nominal[1]},
        "test": {
            "all": _stats(test),
            **{
                f: _stats([r for r, d in zip(test, test_doms, strict=True) if d.floor == f])
                for f in ("hard", "carpet")
            },
        },  # fmt: skip
        "gait": gait.describe(),
        "x": best_x.tolist(),
        "minutes": (time.time() - t0) / 60,
    }
    t = out["test"]
    print(
        f"[{scen.name}] nominal hard {nominal[0]['v_fwd'] * 100:.2f} / carpet "
        f"{nominal[1]['v_fwd'] * 100:.2f} cm/s; held-out hard {t['hard']['mean_cm_s']:.2f} "
        f"(p10 {t['hard']['p10_cm_s']:.2f}), carpet {t['carpet']['mean_cm_s']:.2f} "
        f"(p10 {t['carpet']['p10_cm_s']:.2f}), falls {t['all']['falls']}/{len(test)}, "
        f"{gait.describe()['hz']} Hz, {out['minutes']:.1f} min",
        flush=True,
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-domains", type=int, default=4)
    p.add_argument("--test-domains", type=int, default=32)
    p.add_argument("--random", type=int, default=40)
    p.add_argument("--restarts", type=int, default=2)
    p.add_argument("--pop", type=int, default=20)
    p.add_argument("--gens", type=int, default=40)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--torque", type=float, default=0.25, help="servo torque limit, N m")
    p.add_argument("--body", default=DEFAULT_BODY, help='"build" (printed build) or "rollout"')
    p.add_argument("--tag", default="", help="suffix for the result files")
    p.add_argument("--warm", help='start from results/<name><WARM>.json\'s gait ("" = no tag)')
    p.add_argument("--com-x", type=float, default=0.03, help="fore-aft CoM spread, m")
    p.add_argument("--stock", action="store_true", help="only the stock + dip scenarios")
    p.add_argument("--feet", action="store_true", help="only the reprinted-foot scenarios")
    p.add_argument("--vault", action="store_true", help="only the rocker-foot vault scenarios")
    p.add_argument("--waddle", action="store_true", help="only the tall waddle scenarios")
    p.add_argument("--nose", action="store_true", help="only the nose-first scenarios")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.train_domains, args.test_domains, args.random = 2, 8, 8
        args.restarts, args.pop, args.gens = 1, 8, 3
    OUT.mkdir(exist_ok=True)
    group = SCENARIOS
    for flag, scens in (
        (args.stock, STOCK),
        (args.feet, FEET),
        (args.vault, VAULT),
        (args.waddle, WADDLE),
        (args.nose, NOSE),
    ):
        if flag:
            group = scens
            break
    for scen in group:
        if args.only and scen.name not in args.only:
            continue
        res = run_scenario(scen, args)
        (OUT / f"{scen.name}{args.tag}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
