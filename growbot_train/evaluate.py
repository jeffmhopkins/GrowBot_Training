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

A policy with a speed input also gets a speed-tracking table: what it walks at, target
dead ahead, for commands of 0, 1/3, 2/3 and all of the cap.

``--sweep`` repeats it with one body parameter pinned at a time to the edge of its
training range (drawn in by randomization.ramp_to), the robustness check upstream runs
before shipping a policy.
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
    cfg = config_mod.from_dict(policy.config, saved=True) if policy.config else config_mod.Config()
    cfg.task.history_frames = policy.frames
    cfg.task.gait_hz = policy.gait_hz
    cfg.task.action_to_rad = float(policy.data["obs_spec"]["action_to_rad"])
    if body:
        cfg.task.body = body
    return GrowBotHeadingEnv(cfg)


def run_episode(
    env: GrowBotHeadingEnv, policy: JsonPolicy, seed: int, options: dict[str, Any]
) -> dict[str, float]:
    # Full speed ahead, standing tall, for a policy that takes those commands, so its
    # numbers compare with the others'.
    start = {"retarget": False, "speed_cmd": 1.0, "tilt_cmd": 1.0, **options}
    obs, _ = env.reset(seed=seed, options=start)
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


# The speed commands a speed-input policy is checked at, as shares of the cap: does it
# walk at the speed it's asked for, and stop when asked?
SPEED_STEPS = (0.0, 0.33, 0.67, 1.0)


def speed_tracking(
    env: GrowBotHeadingEnv, policy: JsonPolicy, episodes: int
) -> dict[str, dict[str, float]]:
    cap = env.cfg.reward.speed_cap_m_s
    out = {}
    for cmd in SPEED_STEPS:
        runs = [
            run_episode(env, policy, 2000 + i, {"target_error_deg": 0.0, "speed_cmd": cmd})
            for i in range(episodes)
        ]
        out[f"{cmd:.2f}"] = {
            "want": cmd * cap,
            "toward": float(np.mean([r["toward"] for r in runs])),
            "speed": float(np.mean([r["speed"] for r in runs])),
            "falls": float(sum(r["fell"] for r in runs)),
            "episodes": float(episodes),
        }
    return out


def sweep_cases(cfg: config_mod.Config) -> dict[str, dict[str, Any]]:
    r = cfg.randomization
    k = min(1.0, r.ramp_to)

    # Each edge drawn in toward the as-built value as far as training's were (ramp_to).
    def lo(rng: tuple[float, float], nominal: float = 1.0) -> float:
        return nominal + (rng[0] - nominal) * k

    def hi(rng: tuple[float, float], nominal: float = 1.0) -> float:
        return nominal + (rng[1] - nominal) * k

    floors = {f: {"floor": f} for f in cfg.task.floors} if cfg.task.surfaces else {}
    # With surfaces, the floor's slickness is drawn from its own range: test its edges.
    fr = r.floor_friction_scale if cfg.task.surfaces else r.friction
    key = "floor_scale" if cfg.task.surfaces else "friction"
    fr_mid = 1.0 if cfg.task.surfaces else sum(r.friction) / 2
    return {
        **floors,
        "light": {"mass_scale": lo(r.mass_scale)},
        "heavy": {"mass_scale": hi(r.mass_scale)},
        "short_legs": {"leg_scale": [lo(r.leg_length_scale)] * 2},
        "long_legs": {"leg_scale": [hi(r.leg_length_scale)] * 2},
        "weak_servos": {"gain": lo(r.servo_gain_scale)},
        "strong_servos": {"gain": hi(r.servo_gain_scale)},
        "slippery": {key: lo(fr, fr_mid)},
        "grippy": {key: hi(fr, fr_mid)},
        "slow_relay": {"delay_s": hi(r.relay_delay_s, 0.04)},
        "slow_sensors": {"sensor_delay_s": hi(r.sensor_delay_s, 0.0)},
        "weak_torque": {"torque_scale": lo(r.servo_torque_scale)},
        "narrow_feet": {"foot_width": lo(r.foot_width_scale)},
        "mis_trimmed": {"trim": np.array([r.trim_err_rad * k, -r.trim_err_rad * k])},
        "nose_heavy": {"com": np.array([hi(r.com_x_m, 0.0), 0.0, 0.0])},
        "tail_heavy": {"com": np.array([lo(r.com_x_m, 0.0), 0.0, 0.0])},
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
    if env.cfg.task.speed == "command":
        report["speed_tracking"] = rows = speed_tracking(env, policy, args.episodes)
        print(f"-- speed tracking, randomised bodies (cap {cap:.3f} m/s)", flush=True)
        print(f"{'cmd':>6} {'want':>8} {'toward':>8} {'speed':>7} {'falls':>6}", flush=True)
        for name, m in rows.items():
            print(
                f"{name:>6} {m['want']:8.4f} {m['toward']:8.4f} {m['speed']:7.4f} "
                f"{int(m['falls']):>3}/{int(m['episodes'])}",
                flush=True,
            )
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
