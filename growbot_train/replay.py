"""Record a policy walking and write a 3D replay page that opens in any browser.

    python -m growbot_train.replay runs/my_run/policy_heading_walk.json
    # -> runs/my_run/replay.html (one file; open it on a laptop or a phone)

The page needs no simulator, graphics drivers or install: the walks are simulated here,
and the page only plays back where each part of the robot was, tick by tick, with the
target arrow, the path, and the heading error and speed. ``view.py`` is the live,
steerable alternative when there is a screen attached to the machine.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from growbot_train.env import CONTROL_DT, GrowBotHeadingEnv, wrap_deg
from growbot_train.evaluate import env_for
from growbot_train.policy import JsonPolicy

TEMPLATE = Path(__file__).parent / "replay_template.html"
PLACEHOLDER = "/*REPLAY_DATA*/null"

# (title, starting heading error, seconds between 90-degree target turns or 0, nominal body)
SCENARIOS: tuple[tuple[str, float, float, bool], ...] = (
    ("Target ahead", 0.0, 0.0, True),
    ("Target 90 degrees left", 90.0, 0.0, True),
    ("Target behind", 180.0, 0.0, True),
    ("Target turns every 6 s", 0.0, 6.0, True),
    ("Random body: target 90 right", -90.0, 0.0, False),
    ("Random body: target turns every 6 s", 0.0, 6.0, False),
)


def _geoms(env: GrowBotHeadingEnv) -> list[int]:
    m = env.model
    return [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX]


def record(
    env: GrowBotHeadingEnv,
    policy: JsonPolicy,
    seed: int,
    start_error: float,
    seconds: float,
    turn_every: float,
    nominal: bool,
) -> dict[str, Any]:
    obs, _ = env.reset(
        seed=seed,
        options={"retarget": False, "target_error_deg": start_error, "nominal": nominal},
    )
    env.max_steps = 10**9  # a replay may run longer than a training episode
    m, d = env.model, env.data
    ids = _geoms(env)
    shapes = [
        {
            "size": [round(float(v), 5) for v in m.geom_size[g]],
            "rgba": [round(float(v), 3) for v in m.geom_rgba[g]],
            "visual": bool(m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0),
        }
        for g in ids
    ]
    q = np.zeros(4)

    def snapshot(info: dict[str, Any] | None) -> list[Any]:
        parts = []
        for g in ids:
            mujoco.mju_mat2Quat(q, d.geom_xmat[g])
            parts.append([round(float(v), 5) for v in (*d.geom_xpos[g], *q)])
        return [
            parts,
            round(env.target_deg, 2),
            round(info["abs_err_deg"], 1)
            if info
            else round(abs(wrap_deg(env.target_deg - env.yaw_deg())), 1),
            round(info["v_along"] * 100, 2) if info else 0.0,
            round(info["speed"] * 100, 2) if info else 0.0,
        ]

    frames = [snapshot(None)]
    fell = False
    every = round(turn_every / CONTROL_DT) if turn_every else 0
    for i in range(1, round(seconds / CONTROL_DT) + 1):
        if every and i % every == 0:
            env.target_deg = wrap_deg(env.target_deg + 90.0)
        obs, _, term, _, info = env.step(policy.env_action(obs))
        frames.append(snapshot(info))
        if term:
            fell = True
            break
    return {"shapes": shapes, "frames": frames, "fell": fell}


def build(
    policy_path: Path,
    seconds: float,
    seed: int,
    body: str | None,
    scenarios: tuple[tuple[str, float, float, bool], ...] = SCENARIOS,
    quiet: bool = False,
) -> dict[str, Any]:
    policy = JsonPolicy(policy_path)
    env = env_for(policy, body)
    episodes = []
    for i, (title, err, turn, nominal) in enumerate(scenarios):
        ep = record(env, policy, seed + i, err, seconds, turn, nominal)
        ep["title"] = title
        episodes.append(ep)
        if not quiet:
            fell = " (fell)" if ep["fell"] else ""
            print(f"{title}: {len(ep['frames']) * CONTROL_DT:.1f} s{fell}")
    return {
        "policy": policy_path.name,
        "body": env.cfg.task.body,
        "dt": CONTROL_DT,
        "speed_cap_cm_s": round(env.cfg.reward.speed_cap_m_s * 100, 2),
        "episodes": episodes,
    }


def write_html(data: dict[str, Any], out: Path) -> None:
    page = TEMPLATE.read_text()
    if PLACEHOLDER not in page:
        raise RuntimeError("replay template has lost its data placeholder")
    # "</" can't appear inside a <script> block, so escape it in the embedded JSON.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    out.write_text(page.replace(PLACEHOLDER, blob))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Write a browser 3D replay of a policy")
    ap.add_argument("policy", help="policy JSON (export.py output)")
    ap.add_argument("--out", help="HTML file to write (default: replay.html beside the policy)")
    ap.add_argument("--seconds", type=float, default=20.0, help="length of each walk")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--body", help="rollout / olie")
    args = ap.parse_args(argv)
    policy_path = Path(args.policy)
    out = Path(args.out) if args.out else policy_path.with_name("replay.html")
    write_html(build(policy_path, args.seconds, args.seed, args.body), out)
    print(f"wrote {out} ({math.ceil(out.stat().st_size / 1024)} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
