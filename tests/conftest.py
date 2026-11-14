from __future__ import annotations

from pathlib import Path

import pytest
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from growbot_train.config import Config
from growbot_train.env import GrowBotHeadingEnv
from growbot_train.export import export_run
from growbot_train.train import CLIP_OBS


@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A real (if hopeless) trained run, small enough for CI: exercises the same save
    layout train.py writes, so the exporter and evaluator are tested end to end."""
    run = tmp_path_factory.mktemp("run")
    cfg = Config()
    cfg.task.episode_seconds = 2.0
    cfg.train.policy_layers = (16, 16)
    cfg.train.value_layers = (16, 16)
    cfg.save(run / "config.json")
    vec = VecNormalize(DummyVecEnv([lambda: GrowBotHeadingEnv(cfg)]), clip_obs=CLIP_OBS)
    model = PPO(
        "MlpPolicy",
        vec,
        n_steps=64,
        batch_size=64,
        n_epochs=1,
        seed=0,
        policy_kwargs={"net_arch": {"pi": [16, 16], "vf": [16, 16]}},
        device="cpu",
    )
    model.learn(128)
    model.save(run / "final_model.zip")
    vec.save(str(run / "vecnormalize.pkl"))
    export_run(run)
    return run
