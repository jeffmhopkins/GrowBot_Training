"""Run an exported policy JSON exactly as the phone does, so evaluation and the viewer
test the file that ships rather than the training checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from growbot_train import config as config_mod
from growbot_train.env import CHANNELS, channels_for


def runner_forward(policy: dict[str, Any], obs: np.ndarray) -> np.ndarray:
    """The app runner's arithmetic (growbot_runner.js runForward), in float64 like
    JavaScript numbers: normalise, each layer, then tanh on the output."""
    x = (np.asarray(obs, dtype=np.float64) - np.asarray(policy["mean"])) / np.asarray(policy["std"])
    for layer in policy["layers"]:
        x = x @ np.asarray(layer["W"]) + np.asarray(layer["b"])
        act = layer.get("act")
        if act == "tanh":
            x = np.tanh(x)
        elif act == "relu":
            x = np.maximum(x, 0)
        elif act == "swish":
            x = x / (1 + np.exp(-x))
        elif act not in (None, "linear"):
            raise ValueError(f"unexpected activation {act}")
    return np.tanh(x)


class JsonPolicy:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = json.loads(self.path.read_text())
        seg = self.data["obs_spec"]["segments"]
        cfg = config_mod.from_dict(self.config) if self.config else config_mod.Config()
        expected = channels_for(cfg) if self.config else CHANNELS
        if len(seg) != 1 or tuple(seg[0]["channels"]) != expected:
            raise ValueError(f"{self.path} was not trained for this environment's inputs")
        self.frames = int(seg[0]["frames"])
        self.gait_hz = float(self.data["obs_spec"]["cycles"]["gait_sin"]["frequency"])
        # Convert to arrays once; the per-tick forward pass then stays cheap.
        self.mean = np.asarray(self.data["mean"])
        self.std = np.asarray(self.data["std"])
        self.layers = [
            (np.asarray(L["W"]), np.asarray(L["b"]), L.get("act")) for L in self.data["layers"]
        ]

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Leg commands in [-1, 1] (right, left)."""
        fast = {
            "mean": self.mean,
            "std": self.std,
            "layers": [{"W": W, "b": b, "act": a} for W, b, a in self.layers],
        }
        return runner_forward(fast, obs)

    def env_action(self, obs: np.ndarray) -> np.ndarray:
        # The environment squashes its input with tanh, so hand it the pre-squash value.
        return np.arctanh(np.clip(self.act(obs), -0.999999, 0.999999))

    @property
    def config(self) -> dict[str, Any]:
        return self.data.get("training", {}).get("config", {})
