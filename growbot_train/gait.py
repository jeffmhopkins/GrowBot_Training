"""Open-loop gaits, and warm-starting a policy from one.

A gait is each leg's angle traced through K evenly spaced keyframes per cycle, linearly
interpolated and repeated at `hz` (the shape research/locomotion/search.py optimises).
Saved as JSON: {"hz": 2.0, "keys_rad": [[right leg x K], [left leg x K]]}.

Why warm-start: PPO starting from Gaussian noise around standing still never stumbles
on a coordinated tail-first scoot (the first camera walk runs sat at zero speed for tens
of millions of steps). Cloning a gait the search already found, before PPO, starts it
from a moving robot; PPO then adds steering, speed and the camera height on top. The
gait is driven by the policy's own gait_sin/cos clock, so the app's runner needs nothing
new.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


class Keyframes:
    def __init__(self, hz: float, keys_rad: list[list[float]]) -> None:
        self.hz = float(hz)
        self.keys = np.asarray(keys_rad, dtype=float)
        if self.keys.ndim != 2 or self.keys.shape[0] != 2 or self.keys.shape[1] < 2:
            raise ValueError("keys_rad must be two lists (right leg, left leg) of 2+ angles")

    @classmethod
    def load(cls, path: str | Path) -> Keyframes:
        data = json.loads(Path(path).read_text())
        g = data.get("gait", data)  # a search result file or a bare gait
        return cls(g["hz"], g["keys_rad"])

    @classmethod
    def load_repo(cls, path: str | Path) -> Keyframes:
        """A gait named in a config: as given, or relative to the repo when the run
        starts somewhere else."""
        p = Path(path)
        if not p.is_absolute() and not p.exists():
            p = Path(__file__).resolve().parents[1] / p
        return cls.load(p)

    def __call__(self, t: float) -> np.ndarray:
        k = self.keys.shape[1]
        ph = (t * self.hz % 1.0) * k
        i = int(ph)
        f = ph - i
        return (1 - f) * self.keys[:, i] + f * self.keys[:, (i + 1) % k]


def _steer(want: np.ndarray, vec, deadband_deg: float) -> np.ndarray:
    # Steer on the heading error the policy observes (noise, drift and stale readings
    # included), so each label is a function of what the policy can see.
    out = want.copy()
    for i, err in enumerate(vec.get_attr("_err_obs")):
        if abs(err) < deadband_deg:
            out[i] = 0.0
        elif err < 0:
            out[i] = want[i, ::-1]
    return out


def _tilt_offset(vec, offset: tuple[float, ...]) -> np.ndarray:
    lo, hi = offset
    tilt = np.array(vec.get_attr("tilt_cmd"), dtype=float)
    return lo + (tilt + 1.0) / 2.0 * (hi - lo)


def _labels(want: np.ndarray, vec, cfg) -> np.ndarray:
    """The gait's leg commands per env, steered and shifted for camera height as set."""
    if cfg.train.warm_gait_steer:
        want = _steer(want, vec, cfg.train.warm_gait_deadband_deg)
    if cfg.train.warm_gait_tilt_offset:
        want = want + _tilt_offset(vec, tuple(cfg.train.warm_gait_tilt_offset))[:, None]
    return want


def clone_gait(model, vec, cfg, gait: Keyframes, samples: int, epochs: int, log=print) -> float:
    """Fit the policy's mean action to the gait, from states the gait itself visits (with
    some action noise so it also sees how to get back onto the gait). Also warms the
    observation normaliser. Returns the final mean squared error (pre-squash action).

    With cfg.train.warm_gait_steer the gait is a counter-clockwise spin, cloned mirrored
    (legs swapped) toward a clockwise target and as zero action near the target. With
    cfg.train.warm_gait_tilt_offset both legs are shifted with the camera height
    command, so the clone already holds the asked camera height."""
    import torch

    from growbot_train.env import CONTROL_DT

    if abs(gait.hz - cfg.task.gait_hz) > 1e-6:
        raise ValueError(
            f"the gait runs at {gait.hz} Hz but task.gait_hz is {cfg.task.gait_hz}: the "
            "policy's clock must tick at the gait's rate"
        )
    rng = np.random.default_rng(cfg.train.seed)
    lim = math.tanh(3.0)  # leave the policy room either side of the gait
    raw, targets = [], []
    offset = tuple(cfg.train.warm_gait_tilt_offset)
    if offset and len(offset) != 2:
        raise ValueError("train.warm_gait_tilt_offset is two numbers: at height -1 and +1")
    if offset and not cfg.task.tilt_command:
        raise ValueError("train.warm_gait_tilt_offset needs task.tilt_command")
    if cfg.train.warm_gait_steer:
        # Targets all the way round, even while the curriculum keeps the bodies narrow.
        vec.set_attr("target_scale", 1.0)
    command = cfg.task.speed == "command"
    if command:
        # The full spread of speed commands, though the labels ignore them: on the one
        # full-speed command the curriculum starts on, the normaliser would learn a near-zero
        # spread for that input and blow up the first slower command PPO sees.
        cmd_was = vec.get_attr("cmd_scale")
        vec.set_attr("cmd_scale", 1.0)
    vec.reset()
    while len(raw) * vec.num_envs < samples:
        t = np.array(vec.get_attr("step_index"), dtype=float) * CONTROL_DT
        want = np.array([gait(ti) for ti in t]) / cfg.task.action_to_rad
        want = _labels(want, vec, cfg)
        a = np.arctanh(np.clip(want, -lim, lim))
        raw.append(vec.get_original_obs().copy())
        targets.append(a)
        vec.step((a + rng.normal(0.0, 0.3, a.shape)).astype(np.float32))
    if cfg.train.warm_gait_steer:
        vec.set_attr("target_scale", None)
    if command:
        for i, k in enumerate(cmd_was):
            vec.set_attr("cmd_scale", k, indices=i)
    obs = torch.as_tensor(vec.normalize_obs(np.concatenate(raw)), dtype=torch.float32)
    act = torch.as_tensor(np.concatenate(targets), dtype=torch.float32)
    policy = model.policy
    params = list(policy.mlp_extractor.policy_net.parameters()) + list(
        policy.action_net.parameters()
    )
    opt = torch.optim.Adam(params, lr=1e-3)
    n = len(obs)
    loss_v = float("nan")
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, 2048):
            idx = perm[i : i + 2048]
            mean = policy.get_distribution(obs[idx]).distribution.mean
            loss = torch.mean((mean - act[idx]) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        loss_v = total / n
        if epoch % 5 == 0 or epoch == epochs - 1:
            log(f"   clone epoch {epoch:3d}: mse {loss_v:.4f}")
    return loss_v
