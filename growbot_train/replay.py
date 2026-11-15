"""Record a policy walking and write a 3D replay page that opens in any browser.

    python -m growbot_train.replay runs/my_run/policy_heading_walk.json
    # -> runs/my_run/replay.html (one file; open it on a laptop or a phone)

The page needs no simulator, graphics drivers or install: the walks are simulated here,
and the page only plays back where each part of the robot was, tick by tick, with the
command arrows (heading and camera height, speed), the camera's actual aim and velocity
beside them, the path, and the heading error and speed. ``view.py`` is the live,
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

# (title, starting heading error, seconds between 90-degree target turns or 0, nominal body).
# build() pulls each one into the range the policy trained on (see _within).
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
    # A see-through geom is there only for its mass (a leg's box, drawn by its contact
    # halves); drawn as well it flickers against them.
    return [
        g
        for g in range(m.ngeom)
        if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_rgba[g, 3] > 0
    ]


def record(
    env: GrowBotHeadingEnv,
    policy: JsonPolicy,
    seed: int,
    start_error: float,
    seconds: float,
    turn_every: float,
    nominal: bool,
    turn_deg: float = 90.0,
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

    def vec(v: np.ndarray, k: float = 1.0) -> list[float]:
        return [round(float(x) * k, 3) for x in v]

    def snapshot(info: dict[str, Any] | None) -> list[Any]:
        parts = []
        for g in ids:
            mujoco.mju_mat2Quat(q, d.geom_xmat[g])
            parts.append([round(float(v), 5) for v in (*d.geom_xpos[g], *q)])
        return [
            parts,
            round(env.target_deg, 2),
            round(info["abs_err_deg"], 1) if info else round(env.scored_err_deg(), 1),
            round(info["v_along"] * 100, 2) if info else 0.0,
            round(info["speed"] * 100, 2) if info else 0.0,
            # The inputs, and what the robot did with them: the speed command (signed;
            # below 0 backs away), the camera height command, where the commands ask the
            # camera to point and where it points, and the averaged velocity in cm/s.
            round(env.speed_cmd, 3),
            round(env.tilt_cmd, 3),
            vec(env.want_lens_dir()),
            vec(env.lens_dir()),
            vec(env.vel_avg, 100),
        ]

    frames = [snapshot(None)]
    fell = False
    every = round(turn_every / CONTROL_DT) if turn_every else 0
    for i in range(1, round(seconds / CONTROL_DT) + 1):
        if every and i % every == 0:
            # Left then right, so the target stays within one turn of where it began: a
            # one-way series ends up behind a policy that never trains on that.
            sign = 1.0 if (i // every) % 2 else -1.0
            env.target_deg = wrap_deg(env.target_deg + sign * turn_deg)
        obs, _, term, _, info = env.step(policy.env_action(obs))
        frames.append(snapshot(info))
        if term:
            fell = True
            break
    return {"shapes": shapes, "frames": frames, "fell": fell}


def _within(
    scen: tuple[str, float, float, bool], cfg: Any
) -> tuple[str, float, float, bool, float]:
    """A scenario as this policy's training saw it: a start no wider than its first
    targets and turns no bigger than its retargets (lift and fall speed trains within
    60 degrees; a 90-degree preset showed it walking away from a target it never sees)."""
    title, err, turn, nominal = scen
    lim = cfg.task.initial_error_deg
    step = min(90.0, cfg.task.retarget_max_deg)
    if abs(err) > lim:
        err = math.copysign(lim, err)
        side = "left" if err > 0 else "right"
        title = title.replace(title.split(": ")[-1], f"target {lim:g} degrees {side}")
        title = title[0].upper() + title[1:]
    if turn and step != 90.0:
        title = title.replace("turns every", f"turns {step:g} degrees every")
    return title, err, turn, nominal, step


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
    for i, scen in enumerate(scenarios):
        title, err, turn, nominal, turn_deg = _within(scen, env.cfg)
        ep = record(env, policy, seed + i, err, seconds, turn, nominal, turn_deg)
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
        # Which inputs the policy takes, so the page draws only those.
        "inputs": {
            "speed": env.cfg.task.speed == "command",
            "camera_height": bool(env.cfg.task.tilt_command),
            "camera_first": env.cfg.task.facing == "camera",
        },
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
