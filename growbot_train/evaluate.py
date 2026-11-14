"""Score an exported policy the way the robot will use it.

    python -m growbot_train.evaluate runs/my_run/policy_heading_walk.json
    python -m growbot_train.evaluate runs/my_run/policy_heading_walk.json --sweep

Runs the JSON file itself (not the training checkpoint) on randomised bodies, starting
with the target dead ahead, to the side, and behind, and reports per case:

    toward   average speed along the target direction (m/s); higher is better
    speed    average speed in any direction; should sit at or under the cap
    over%    share of time the averaged speed was more than 10% over the cap
    err      average heading error over the second half, degrees; lower is better
    turn_s   seconds until the error first dropped under 20 degrees
    falls    episodes that ended in a fall

``--sweep`` repeats it with one body parameter pinned at a time to the edge of its
training range, the robustness check upstream runs before shipping a policy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from growbot_train import config as config_mod
from growbot_train.env import CONTROL_DT, GrowBotHeadingEnv
from growbot_train.policy import JsonPolicy

START_ERRORS = (0.0, 45.0, -45.0, 90.0, -90.0, 180.0)


def env_for(policy: JsonPolicy, body: str | None = None) -> GrowBotHeadingEnv:
    cfg = config_mod.from_dict(policy.config) if policy.config else config_mod.Config()
    cfg.task.history_frames = policy.frames
    cfg.task.gait_hz = policy.gait_hz
    cfg.task.action_to_rad = float(policy.data["obs_spec"]["action_to_rad"])
    if body:
        cfg.task.body = body
    return GrowBotHeadingEnv(cfg)


def run_episode(
    env: GrowBotHeadingEnv, policy: JsonPolicy, seed: int, options: dict[str, Any]
) -> dict[str, float]:
    obs, _ = env.reset(seed=seed, options={"retarget": False, **options})
    cap = env.cfg.reward.speed_cap_m_s
    along, speed, errs, over = [], [], [], 0
    turn_s = float("nan")
    fell = False
    done = False
    while not done:
        obs, _, term, trunc, info = env.step(policy.env_action(obs))
        along.append(info["v_along"])
        speed.append(info["speed"])
        errs.append(info["abs_err_deg"])
        over += info["speed"] > 1.1 * cap
        if np.isnan(turn_s) and info["abs_err_deg"] < 20:
            turn_s = env.step_index * CONTROL_DT
        fell = bool(term)
        done = term or trunc
    half = len(errs) // 2
    return {
        "toward": float(np.mean(along)),
        "speed": float(np.mean(speed)),
        "over_pct": 100.0 * over / len(speed),
        "err": float(np.mean(errs[half:])),
        "turn_s": turn_s,
        "fell": float(fell),
    }


def score(
    env: GrowBotHeadingEnv,
    policy: JsonPolicy,
    episodes: int,
    overrides: dict[str, Any] | None = None,
    nominal: bool = False,
    errors: tuple[float, ...] = START_ERRORS,
) -> dict[str, dict[str, float]]:
    out = {}
    for err in errors:
        runs = [
            run_episode(
                env,
                policy,
                seed=1000 + i,
                options={"target_error_deg": err, "overrides": overrides or {}, "nominal": nominal},
            )
            for i in range(episodes)
        ]
        agg = {k: float(np.nanmean([r[k] for r in runs])) for k in runs[0] if k != "fell"}
        agg["falls"] = float(sum(r["fell"] for r in runs))
        agg["episodes"] = float(episodes)
        out[f"{err:+.0f}"] = agg
    return out


def sweep_cases(cfg: config_mod.Config) -> dict[str, dict[str, Any]]:
    r = cfg.randomization
    return {
        "light": {"mass_scale": r.mass_scale[0]},
        "heavy": {"mass_scale": r.mass_scale[1]},
        "short_legs": {"leg_scale": [r.leg_length_scale[0]] * 2},
        "long_legs": {"leg_scale": [r.leg_length_scale[1]] * 2},
        "weak_servos": {"gain": r.servo_gain_scale[0]},
        "strong_servos": {"gain": r.servo_gain_scale[1]},
        "slippery": {"friction": r.friction[0]},
        "grippy": {"friction": r.friction[1]},
        "slow_relay": {"delay_s": r.relay_delay_s[1]},
        "nose_heavy": {"com": np.array([r.com_x_m[1], 0.0, 0.0])},
        "tail_heavy": {"com": np.array([r.com_x_m[0], 0.0, 0.0])},
    }


def _table(title: str, rows: dict[str, dict[str, float]], cap: float) -> str:
    lines = [
        f"-- {title} (speed cap {cap:.3f} m/s)",
        f"{'start':>6} {'toward':>8} {'speed':>7} {'over%':>6} {'err':>6} "
        f"{'turn_s':>7} {'falls':>6}",
    ]
    for name, m in rows.items():
        lines.append(
            f"{name:>6} {m['toward']:8.4f} {m['speed']:7.4f} {m['over_pct']:6.1f} "
            f"{m['err']:6.1f} {m['turn_s']:7.2f} {int(m['falls']):>3}/{int(m['episodes'])}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate an exported policy JSON")
    ap.add_argument("policy", help="policy JSON (export.py output)")
    ap.add_argument("--episodes", type=int, default=6, help="bodies per starting error")
    ap.add_argument("--body", help="evaluate on another body (rollout / olie)")
    ap.add_argument("--sweep", action="store_true", help="also run the robustness sweep")
    ap.add_argument("--json", help="write all numbers here as JSON")
    args = ap.parse_args(argv)

    policy = JsonPolicy(args.policy)
    env = env_for(policy, args.body)
    cap = env.cfg.reward.speed_cap_m_s
    report: dict[str, Any] = {"policy": str(args.policy), "speed_cap_m_s": cap}
    report["randomized"] = score(env, policy, args.episodes)
    print(_table("randomised bodies", report["randomized"], cap), flush=True)
    report["nominal"] = score(env, policy, 1, nominal=True)
    print(_table("nominal body, no noise", report["nominal"], cap), flush=True)
    if args.sweep:
        report["sweep"] = {}
        for name, ov in sweep_cases(env.cfg).items():
            rows = score(
                env, policy, max(1, args.episodes // 2), overrides=ov, errors=(0.0, 90.0, 180.0)
            )
            report["sweep"][name] = rows
            print(_table(f"sweep: {name}", rows, cap), flush=True)

    # One summary line the launcher's results page picks up.
    rnd = report["randomized"].values()
    falls = sum(m["falls"] for m in rnd)
    total = sum(m["episodes"] for m in rnd)
    toward = float(np.mean([m["toward"] for m in rnd]))
    err = float(np.mean([m["err"] for m in rnd]))
    print(
        f"eval summary: toward {toward:.4f} m/s, heading error {err:.1f} deg, "
        f"falls {int(falls)}/{int(total)}",
        flush=True,
    )
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, default=float) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
