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
from collections.abc import Callable
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
    ("Speed steps: full, half, stop, full", 0.0, 0.0, True),
    ("Like training: random body, targets and speeds", 0.0, 0.0, False),
    ("Like training: another random body", 0.0, 0.0, False),
)
# Walks that drive more than the target: the speed command stepped every 6 s (only for
# a policy with a speed input), or the training draws themselves (first target, moves,
# speeds) instead of a scripted target.
SPEED_STEPS = {"Speed steps: full, half, stop, full": (1.0, 0.5, 0.0, 1.0)}
LIKE_TRAINING = {
    "Like training: random body, targets and speeds",
    "Like training: another random body",
}


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
    alternate: bool = False,
    speeds: tuple[float, ...] = (),
    like_training: bool = False,
) -> dict[str, Any]:
    options: dict[str, Any] = {"retarget": like_training, "nominal": nominal}
    if not like_training:
        options["target_error_deg"] = start_error
    if speeds:
        options["speed_cmd"] = speeds[0]
    obs, _ = env.reset(seed=seed, options=options)
    env.max_steps = 10**9  # a replay may run longer than a training episode
    shapes, snapshot = recorder(env)
    frames = [snapshot(None)]
    fell = False
    every = round(turn_every / CONTROL_DT) if turn_every else 0
    step_every = round(6.0 / CONTROL_DT)
    for i in range(1, round(seconds / CONTROL_DT) + 1):
        if speeds and i % step_every == 0:
            # The env reads its pinned speed on its own redraws too, so pin both.
            k = min(i // step_every, len(speeds) - 1)
            env._fixed_speed = env.speed_cmd = speeds[k]
        if every and i % every == 0:
            # For a policy whose targets stay in front, left then right, so the target
            # stays within one turn of where it began: a one-way series ends up behind.
            sign = -1.0 if alternate and (i // every) % 2 == 0 else 1.0
            env.target_deg = wrap_deg(env.target_deg + sign * turn_deg)
        obs, _, term, _, info = env.step(policy.env_action(obs))
        frames.append(snapshot(info))
        if term:
            fell = True
            break
    return {"shapes": shapes, "frames": frames, "fell": fell}


def recorder(
    env: GrowBotHeadingEnv,
) -> tuple[list[dict[str, Any]], Callable[[dict[str, Any] | None], list[Any]]]:
    """The robot's drawn parts, and a function giving one tick's frame: where each part
    is, the commands, and what the robot did with them. Shared with the master's replay
    (growbot_train.master), which drives the env itself."""
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

    return shapes, snapshot


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
        # Out of range on the left (90, or behind) goes to the right edge, so the two
        # edges both get a walk rather than one twice.
        err = -lim if err > 0 and abs(err) > 90 else math.copysign(lim, err)
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
    has_speed = env.cfg.task.speed == "command"
    for i, scen in enumerate(scenarios):
        speeds = SPEED_STEPS.get(scen[0], ())
        if speeds and not has_speed:
            continue
        like = scen[0] in LIKE_TRAINING
        title, err, turn, nominal, turn_deg = _within(scen, env.cfg)
        narrow = env.cfg.task.initial_error_deg < 180.0
        ep = record(
            env, policy, seed + i, err, seconds, turn, nominal, turn_deg, narrow, speeds, like
        )
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
