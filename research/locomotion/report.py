"""Print the results tables (markdown) from a folder of search results, plus each
gait's speed on the as-designed robot as the floor gets slicker or grippier.

    python research/locomotion/report.py                    # results/
    python research/locomotion/report.py results/sweep1     # another sweep
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from lab import Bench, Domain, KeyframeGait, Scenario
from search import BY_NAME

HERE = Path(__file__).parent
SCALES = (0.7, 1.0, 1.3)


def scenario_of(res: dict) -> Scenario:
    if res["name"] in BY_NAME:
        return BY_NAME[res["name"]]
    # A co-design result names no fixed scenario: it carries the foot it chose.
    return Scenario(res["name"], res["surfaces"], res["note"], **res["scenario"])


def floor_sweep(res: dict) -> dict[str, list[float]]:
    scen = scenario_of(res)
    gait = KeyframeGait(np.array(res["x"]), scen.swing_rad)
    body = res["body"].removesuffix(".xml")
    out = {}
    for floor in ("hard", "carpet"):
        out[floor] = [
            Bench(scen, replace(Domain(), floor=floor, floor_scale=k), body, res["torque_nm"])
            .run(gait)["v_fwd"] * 100
            for k in SCALES
        ]  # fmt: skip
    return out


def main() -> None:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    # Ranked by the worse floor's held-out mean: it has to work on both.
    rows.sort(key=lambda r: -min(r["test"]["hard"]["mean_cm_s"], r["test"]["carpet"]["mean_cm_s"]))
    print("Speeds tail first, cm/s. *Nominal* is the as-designed robot; *held-out* is the")
    print("mean (and 10th percentile) over random robots the gait was not tuned on.\n")
    print(
        "| | Setup | Gait | Nominal hard | Nominal carpet | Held-out hard (p10) "
        "| Held-out carpet (p10) | Falls | Mean pitch |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in rows:
        t, n = r["test"], r["nominal"]
        print(
            f"| {r['name']} | {r['note']} | {r['gait']['hz']:.1f} Hz "
            f"| {n['hard']['v_fwd'] * 100:.2f} | {n['carpet']['v_fwd'] * 100:.2f} "
            f"| {t['hard']['mean_cm_s']:.2f} ({t['hard']['p10_cm_s']:.2f}) "
            f"| {t['carpet']['mean_cm_s']:.2f} ({t['carpet']['p10_cm_s']:.2f}) "
            f"| {t['all']['falls']}/{t['all']['n']} "
            f"| {n['hard']['mean_pitch']:.0f}° |"
        )
    print("\nAs-designed robot, floor friction scaled x" + " / x".join(map(str, SCALES)) + ":\n")
    print("| | Hard | Carpet |")
    print("| --- | --- | --- |")
    for r in rows:
        fs = floor_sweep(r)
        cell = {f: " / ".join(f"{v:.2f}" for v in fs[f]) for f in fs}
        print(f"| {r['name']} | {cell['hard']} | {cell['carpet']} |")


if __name__ == "__main__":
    main()
