"""Tune a cam foot and its gait together: the search moves the leg motion and the foot's
shape at once, inside printable limits.

    python research/locomotion/codesign.py --from f13 f17     # start from those results

Foot variables (each squashed into its range):
    arc      how far the curved sole reaches either side of the leg's line, 8 to 25 deg
    cam      how fast the sole's distance from the axle grows toward the nose, m/rad,
             -0.04 to 0.04 (negative: grows toward the tail; 0: an arc round the axle)
    extra    leg length change, -10 to +15 mm
    dip      how much of the sole is dipped: +0.3 dips the nose-side 30% of the arc,
             -0.3 the tail-side 30%, 0 none, +-1 all of it

Writes results/codesign.json (same shape as search.py's, plus the foot it chose).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from lab import DEFAULT_BODY, Bench, Domain, KeyframeGait, Scenario, domains, score
from search import BY_NAME, _stats, cma

OUT = Path(__file__).parent / "results"
NG = KeyframeGait.dim()


def _sig(v: float) -> float:
    return 1 / (1 + math.exp(-v))


def foot(y: np.ndarray) -> dict:
    arc = 8 + 17 * _sig(y[0])
    dip = math.tanh(y[3])
    # The split sits where the dipped share of the arc begins (nose side for dip > 0).
    split = arc - 2 * arc * dip if dip >= 0 else -arc + 2 * arc * -dip
    return {
        "arc_deg": arc,
        "cam_m_per_rad": 0.04 * math.tanh(y[1]),
        "leg_extra_m": -0.01 + 0.025 * _sig(y[2]),
        "dip": dip,
        "split_deg": split,
    }


def scenario(y: np.ndarray) -> Scenario:
    f = foot(y)
    nose_rubber = f["dip"] >= 0
    surf = {
        "tail": "pla",
        "belly": "pla",
        "toe_nose": "rubber" if nose_rubber else "pla",
        "toe_tail": "pla" if nose_rubber else "rubber",
    }
    return Scenario(
        "codesign", surf, "Cam foot and gait tuned together", foot="cam",
        foot_deg=f["arc_deg"], leg_extra_m=f["leg_extra_m"],
        cam_m_per_rad=f["cam_m_per_rad"], split_deg=f["split_deg"],
    )  # fmt: skip


def foot_raw(arc: float, cam: float, extra: float, dip: float) -> np.ndarray:
    """The raw variables for a given foot (to start from a fixed-foot result)."""

    def logit(p: float) -> float:
        p = min(max(p, 1e-3), 1 - 1e-3)
        return math.log(p / (1 - p))

    return np.array(
        [
            logit((arc - 8) / 17),
            math.atanh(max(-0.999, min(0.999, cam / 0.04))),
            logit((extra + 0.01) / 0.025),
            math.atanh(max(-0.99, min(0.99, dip))),
        ]
    )


_DOMS: list[Domain] = []
_CFG: tuple = ()


def _init(doms, body, torque, seconds) -> None:
    global _DOMS, _CFG
    _DOMS, _CFG = doms, (body, torque, seconds)


def _run_all(x: np.ndarray) -> list[dict]:
    body, torque, seconds = _CFG
    scen = scenario(x[NG:])
    gait = KeyframeGait(x[:NG], scen.swing_rad)
    return [Bench(scen, d, body=body, torque_nm=torque).run(gait, seconds) for d in _DOMS]


def _eval(x: np.ndarray) -> float:
    per: dict[str, list[dict]] = {}
    for d, r in zip(_DOMS, _run_all(x), strict=True):
        per.setdefault(d.floor, []).append(r)
    s = [score(rs) for rs in per.values()]
    return 0.5 * float(np.mean(s)) + 0.5 * min(s)


def _detail(args) -> dict:
    x, i = args
    body, torque, seconds = _CFG
    scen = scenario(np.asarray(x)[NG:])
    gait = KeyframeGait(np.asarray(x)[:NG], scen.swing_rad)
    return Bench(scen, _DOMS[i], body=body, torque_nm=torque).run(gait, seconds)


def start_from(name: str) -> np.ndarray:
    r = json.loads((OUT / f"{name}.json").read_text())
    s = BY_NAME[name]
    nose, tail = (s.surfaces[k] == "rubber" for k in ("toe_nose", "toe_tail"))
    dip = 0.99 if nose and tail else 0.5 if nose else -0.5 if tail else 0.0
    y = foot_raw(s.foot_deg, s.cam_m_per_rad, s.leg_extra_m, dip)
    return np.concatenate([np.array(r["x"]), y])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="starts", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--train-domains", type=int, default=8)
    p.add_argument("--test-domains", type=int, default=32)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--gens", type=int, default=50)
    p.add_argument("--seconds", type=float, default=8.0)
    p.add_argument("--torque", type=float, default=0.25)
    p.add_argument("--com-x", type=float, default=0.02)
    p.add_argument("--body", default=DEFAULT_BODY)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    t0 = time.time()
    nominal = [Domain(floor="hard"), Domain(floor="carpet")]
    train = nominal + domains(args.train_domains, seed=1, com_x=args.com_x)
    test = domains(args.test_domains, seed=2, com_x=args.com_x)
    init = (args.body, args.torque, args.seconds)
    best_x, best_f = None, -np.inf
    with Pool(args.workers, _init, (train, *init)) as pool:
        for name in args.starts:
            x0 = start_from(name)
            print(f"[codesign] from {name}: {pool.map(_eval, [x0])[0] * 100:.2f}", flush=True)
            x, fx = cma(_eval, x0, 0.4, args.pop, args.gens, pool, log=print)
            print(f"[codesign] from {name} -> {fx * 100:.2f}  {foot(x[NG:])}", flush=True)
            if fx > best_f:
                best_x, best_f = x, fx
    with Pool(args.workers, _init, (test + nominal, *init)) as pool:
        allr = pool.map(_detail, [(best_x, i) for i in range(len(test) + 2)])
    res, nom = allr[:-2], allr[-2:]
    scen = scenario(best_x[NG:])
    gait = KeyframeGait(best_x[:NG], scen.swing_rad)
    out = {
        "name": f"codesign{args.tag}",
        "note": "Cam foot and gait tuned together (" + ", ".join(args.starts) + ")",
        "surfaces": scen.surfaces,
        "scenario": {
            "foot": scen.foot,
            "foot_deg": scen.foot_deg,
            "leg_extra_m": scen.leg_extra_m,
            "cam_m_per_rad": scen.cam_m_per_rad,
            "split_deg": scen.split_deg,
        },  # fmt: skip
        "foot": foot(best_x[NG:]),
        "body": args.body,
        "torque_nm": args.torque,
        "com_x_m": args.com_x,
        "train_score": best_f,
        "nominal": {"hard": nom[0], "carpet": nom[1]},
        "test": {
            "all": _stats(res),
            **{
                f: _stats([r for r, d in zip(res, test, strict=True) if d.floor == f])
                for f in ("hard", "carpet")
            },
        },  # fmt: skip
        "gait": gait.describe(),
        "x": best_x.tolist(),
        "minutes": (time.time() - t0) / 60,
    }
    (OUT / f"codesign{args.tag}.json").write_text(json.dumps(out, indent=1))
    t = out["test"]
    print(
        f"[codesign] nominal hard {nom[0]['v_fwd'] * 100:.2f} / carpet "
        f"{nom[1]['v_fwd'] * 100:.2f}; held-out hard {t['hard']['mean_cm_s']:.2f} "
        f"(p10 {t['hard']['p10_cm_s']:.2f}), carpet {t['carpet']['mean_cm_s']:.2f} "
        f"(p10 {t['carpet']['p10_cm_s']:.2f}), falls {t['all']['falls']}/{t['all']['n']}; "
        f"foot {out['foot']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
