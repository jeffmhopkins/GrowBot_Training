"""Turn a trained run into the JSON policy the JBrain2 app's runner loads.

    python -m growbot_train.export runs/my_run            # -> runs/my_run/policy_heading_walk.json

The target is ``growbot_runner.js``'s ``Walker`` (the ``segments`` dialect the app's
``policy_drrma.json`` uses): ``(obs - mean) / std``, then each layer ``x @ W + b`` with
its activation, then ``tanh`` on the output. ``W`` is stored input-major
(``W[input][output]``), which is how that runner indexes it.

Every export is checked: the JSON is read back, run through a plain-numpy copy of the
runner's arithmetic, and compared with the trained network on random observations. The
result goes to ``export_verify.log``, which ends ``EXPORT OK`` only when they match.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from stable_baselines3 import PPO

from growbot_train import config as config_mod
from growbot_train.env import CONTROL_DT, LENS_BLEND, channels_for
from growbot_train.policy import runner_forward

FORMAT = "growbot-policy"
TOLERANCE = 1e-5


def _layers(model: PPO) -> list[dict[str, Any]]:
    mods = [m for m in model.policy.mlp_extractor.policy_net if isinstance(m, torch.nn.Linear)]
    acts = [m for m in model.policy.mlp_extractor.policy_net if not isinstance(m, torch.nn.Linear)]
    if not all(isinstance(a, torch.nn.Tanh) for a in acts) or len(acts) != len(mods):
        raise ValueError("expected a Linear/Tanh policy trunk")
    out = []
    for lin in [*mods, model.policy.action_net]:
        out.append(
            {
                "W": lin.weight.detach().double().numpy().T.tolist(),
                "b": lin.bias.detach().double().numpy().tolist(),
                "act": "tanh" if lin is not model.policy.action_net else "linear",
            }
        )
    return out


def _residual_spec(cfg: config_mod.Config) -> dict[str, Any]:
    """A residual-gait policy's legs are the gait plus its output, on a clock that runs
    at the speed command's size; the runner has to know both."""
    if not cfg.task.residual_gait:
        return {}
    from growbot_train.gait import Keyframes

    g = Keyframes.load_repo(cfg.task.residual_gait)
    return {
        "residual_gait": {
            "hz": g.hz,
            "keys_rad": g.keys.round(4).tolist(),
            "residual_rad": cfg.task.residual_rad,
            "meaning": (
                "leg command (rad) = gait(phase) + output x residual_rad, eased in over "
                "the first second; the gait_sin/cos phase advances at frequency x "
                "|speed command| (a stop holds it), not at a fixed rate"
            ),
        }
    }


def build_policy(model: PPO, norm: Any, cfg: config_mod.Config, name: str) -> dict[str, Any]:
    mean = np.asarray(norm.obs_rms.mean, dtype=np.float64)
    std = np.sqrt(np.asarray(norm.obs_rms.var, dtype=np.float64) + norm.epsilon)
    n = mean.shape[0]
    frames = cfg.task.history_frames
    front = "nose" if cfg.task.facing == "nose" else "camera end (its tail)"
    channels = channels_for(cfg)
    if n != frames * len(channels):
        raise ValueError(f"normaliser has {n} inputs, config expects {frames * len(channels)}")
    policy: dict[str, Any] = {
        "format": FORMAT,
        "version": 1,
        "name": name,
        "obs_size": n,
        "act_size": 2,
        "activation": "tanh",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "layers": _layers(model),
        "obs_spec": {
            "control_hz": round(1 / CONTROL_DT),
            "action_to_rad": cfg.task.action_to_rad,
            "action_map": {"0": "right_leg", "1": "left_leg"},
            "imu_frame": {
                "up": "+z",
                "note": (
                    "levelled body frame: after the legs-neutral stance settles, rotate "
                    "the phone's readings by the smallest rotation that makes the "
                    "gravity it measures read +z (so x stays toward the nose, y to the "
                    "robot's left); accel is accelerationIncludingGravity in m/s^2, "
                    "gyro in rad/s; IMU_SIGN must be [1, 1, 1] for this policy"
                ),
            },
            "segments": [
                {
                    "channels": list(channels),
                    "frames": frames,
                    "order": "newest_first",
                    "layout": "frame_major",
                }
            ],
            "cycles": {
                "gait_sin": {"type": "sin", "frequency": cfg.task.gait_hz},
                "gait_cos": {"type": "cos", "frequency": cfg.task.gait_hz},
            },
            **_residual_spec(cfg),
            # The runner has no heading channel yet, so the heading error rides the
            # existing cmd_yaw input; this block tells the app what to put there.
            "cmd_yaw": {
                "meaning": "heading_error",
                "scale": "degrees / 180",
                "range": [-1, 1],
                "sign": (
                    "positive when the target is counter-clockwise of the robot's "
                    f"{front} seen from above, i.e. turn left"
                ),
                # Which end walks first and counts as "ahead" for the error: "nose", or
                # "camera" (the tail end, where the phone's front camera looks).
                "facing": cfg.task.facing,
                # "body": from that end's direction. "lens": from where the phone's rear
                # camera points (the torso's up axis, projected flat; it fades into the
                # tail end as it nears straight up or tips sideways) — what a face's
                # left-right position in the camera image gives directly.
                "heading_from": cfg.task.heading_from,
                # With "lens": the lens's share is clamp(c / lens_fade, 0, 1), c being how
                # far its flat axis reaches out over the tail end; the app must fade alike.
                "lens_fade": LENS_BLEND,
            },
        },
        "training": {
            "config": cfg.to_dict(),
            "speed_cap_m_s": cfg.reward.speed_cap_m_s,
            "speed": cfg.task.speed,
        },
    }
    if cfg.task.speed == "command":
        policy["obs_spec"]["cmd_forward"] = {
            "meaning": "speed_command",
            "range": [cfg.task.speed_back_min, 1],
            "scale": (
                f"+1 walks toward the target at {cfg.reward.speed_cap_m_s * 100:g} cm/s, "
                "0 stands still and keeps turning to face it, below 0 backs away "
                "(trained down to the range's low end)"
            ),
        }
    if cfg.task.tilt_command:
        lo, hi = cfg.task.tilt_lift_deg
        policy["obs_spec"]["cmd_pitch"] = {
            "meaning": "camera_height",
            "range": [-1, 1],
            "scale": (
                f"-1 holds the body {-lo:g} degrees flatter than its resting stance (the "
                f"camera looks almost straight up), +1 {hi:g} degrees taller (the camera "
                "looks furthest toward the horizon), linear in between"
            ),
        }
    return policy


def verify(policy: dict[str, Any], model: PPO, n: int = 256, seed: int = 0) -> float:
    """Largest gap between the JSON's actions and the trained network's, on raw
    observations scattered around the normaliser's own range."""
    rng = np.random.default_rng(seed)
    mean = np.asarray(policy["mean"])
    std = np.asarray(policy["std"])
    raw = mean + std * rng.normal(0, 1.5, (n, mean.shape[0]))
    ours = runner_forward(policy, raw)
    normed = torch.as_tensor((raw - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        dist = model.policy.get_distribution(normed)
        theirs = torch.tanh(dist.mode()).double().numpy()
    return float(np.max(np.abs(ours - theirs)))


def export(
    model_path: Path, norm_path: Path, cfg: config_mod.Config, out: Path, log: Path | None = None
) -> Path:
    model = PPO.load(model_path, device="cpu")
    with open(norm_path, "rb") as fh:
        norm = pickle.load(fh)
    policy = build_policy(model, norm, cfg, f"growbot-{cfg.name}")
    out.write_text(json.dumps(policy, separators=(",", ":")) + "\n")
    reread = json.loads(out.read_text())
    gap = verify(reread, model)
    ok = gap < TOLERANCE
    lines = [
        f"policy: {out.name}",
        f"obs_size: {reread['obs_size']}",
        f"max |json - network|: {gap:.3e} (tolerance {TOLERANCE:g})",
        "EXPORT OK" if ok else "EXPORT MISMATCH",
    ]
    if log is not None:
        log.write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    if not ok:
        raise RuntimeError(f"exported policy differs from the network by {gap:.3e}")
    return out


def export_run(run: Path, model_name: str = "final_model.zip") -> Path:
    cfg = config_mod.from_dict(json.loads((run / "config.json").read_text()), saved=True)
    model_path = run / model_name
    if model_path.name.startswith("model_"):
        norm_path = model_path.with_name(
            model_path.name.replace("model_", "vecnormalize_")
        ).with_suffix(".pkl")
    else:
        norm_path = run / "vecnormalize.pkl"
    return export(
        model_path, norm_path, cfg, run / f"policy_{cfg.name}.json", log=run / "export_verify.log"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export a run to the app's policy JSON")
    ap.add_argument("run", help="the run folder train.py wrote")
    ap.add_argument(
        "--model",
        default="final_model.zip",
        help="which model in the run (e.g. checkpoints/model_4000000.zip)",
    )
    args = ap.parse_args(argv)
    export_run(Path(args.run), args.model)
    return 0


if __name__ == "__main__":
    sys.exit(main())
