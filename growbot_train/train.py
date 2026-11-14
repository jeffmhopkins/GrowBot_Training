"""Train the heading-following walk with PPO on the CPU, one simulator per core.

    python -m growbot_train.train --out runs/my_run
    python -m growbot_train.train --out runs/quick --steps 200000 --set reward.speed_cap_m_s=0.04

Writes into ``--out``: the config it used, periodic checkpoints, the final model and its
observation normaliser, TensorBoard logs, and the exported app policy JSON.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecMonitor,
    VecNormalize,
)

from growbot_train import config as config_mod
from growbot_train.env import GrowBotHeadingEnv
from growbot_train.live import Live

# The app's runner does not clip, so the normaliser must not either: a wide bound keeps
# training and the phone computing the same numbers.
CLIP_OBS = 1000.0


def make_env(cfg: config_mod.Config, rank: int, seed: int):
    def _init() -> GrowBotHeadingEnv:
        env = GrowBotHeadingEnv(cfg)
        env.reset(seed=seed + rank)
        return env

    return _init


def build_vec_env(cfg: config_mod.Config, n_envs: int, seed: int) -> VecMonitor:
    fns = [make_env(cfg, i, seed) for i in range(n_envs)]
    vec = SubprocVecEnv(fns, start_method="fork") if n_envs > 1 else DummyVecEnv(fns)
    return VecMonitor(vec)


def ramp_scale(cfg: config_mod.Config, steps: int) -> float:
    """The curriculum's share of the full randomisation after `steps` (1 = full)."""
    r = cfg.randomization
    if r.ramp_steps <= 0:
        return 1.0
    return min(1.0, r.ramp_from + (1.0 - r.ramp_from) * steps / r.ramp_steps)


class Curriculum(BaseCallback):
    """Widens the randomisation as training goes (randomization.ramp_steps)."""

    def __init__(self, cfg: config_mod.Config) -> None:
        super().__init__()
        self.cfg = cfg

    def _on_rollout_start(self) -> None:
        k = ramp_scale(self.cfg, self.num_timesteps)
        self.training_env.set_attr("dr_scale", k)
        self.logger.record("walk/dr_scale", k)

    def _on_step(self) -> bool:
        return True


class Diagnostics(BaseCallback):
    """Logs what the walk is actually doing (speed along the target, heading error,
    falls, each reward term) so a reward change can be judged by more than one number."""

    def __init__(self, live: Live | None = None) -> None:
        super().__init__()
        self.live = live
        self.sums: dict[str, float] = defaultdict(float)
        self.n = 0
        self.falls = 0
        self.t0 = time.time()

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            self.n += 1
            self.sums["v_along"] += info["v_along"]
            self.sums["speed"] += info["speed"]
            self.sums["abs_err_deg"] += info["abs_err_deg"]
            self.sums["lift_deg"] += info["lift_deg"]
            self.sums["want_lift_deg"] += info["want_lift_deg"]
            self.sums["speed_miss"] += info["speed_miss"]
            self.sums["drift_m"] += info["drift_m"]
            self.falls += int(info["fell"])
            for k, v in info["terms"].items():
                self.sums["term_" + k] += v
        return True

    def _on_rollout_end(self) -> None:
        if not self.n:
            return
        for k, v in self.sums.items():
            self.logger.record(f"walk/{k}", v / self.n)
        self.logger.record("walk/falls_per_1k_steps", 1000 * self.falls / self.n)
        self.logger.record(
            "time/steps_per_second", self.num_timesteps / max(1e-6, time.time() - self.t0)
        )
        if self.live is not None:
            self.live.update(
                self.num_timesteps,
                {
                    "v_along": self.sums["v_along"] / self.n,
                    "speed": self.sums["speed"] / self.n,
                    "abs_err_deg": self.sums["abs_err_deg"] / self.n,
                    "falls_per_1k_steps": 1000 * self.falls / self.n,
                    "lift_deg": self.sums["lift_deg"] / self.n,
                    "want_lift_deg": self.sums["want_lift_deg"] / self.n,
                    "speed_miss": self.sums["speed_miss"] / self.n,
                    "drift_m": self.sums["drift_m"] / self.n,
                },
            )
        self.sums.clear()
        self.n = 0
        self.falls = 0


class LiveReplay(BaseCallback):
    """Refresh the live folder's replay and latest policy from the current network, more
    often than checkpoints are saved, so a phone sees the walk change as it learns."""

    def __init__(self, every: int, live: Live) -> None:
        super().__init__()
        self.every = every
        self.live = live
        self.next = every

    def _init_callback(self) -> None:
        self.next = (self.model.num_timesteps // self.every + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next:
            self.next = (self.num_timesteps // self.every + 1) * self.every
            self.live.snapshot(self.model, self.num_timesteps)
        return True


class Checkpoint(BaseCallback):
    def __init__(self, every: int, folder: Path) -> None:
        super().__init__()
        self.every = every
        self.folder = folder
        self.next = every

    def _init_callback(self) -> None:
        # A resumed run starts mid-count; the next save is the next boundary after it.
        self.next = (self.model.num_timesteps // self.every + 1) * self.every

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next:
            self.next += self.every
            self.folder.mkdir(parents=True, exist_ok=True)
            self.model.save(self.folder / f"model_{self.num_timesteps}.zip")
            vec = self.model.get_vec_normalize_env()
            if vec is not None:
                vec.save(str(self.folder / f"vecnormalize_{self.num_timesteps}.pkl"))
        return True


def default_n_envs() -> int:
    return max(1, os.cpu_count() or 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/heading_walk.toml")
    ap.add_argument("--out", required=True, help="run folder to write into")
    ap.add_argument("--steps", type=int, help="override train.total_steps")
    ap.add_argument("--envs", type=int, help="override train.n_envs")
    ap.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=VALUE")
    ap.add_argument(
        "--resume",
        help="a checkpoints/model_*.zip or final_model.zip to continue from; the run's "
        "config.json is reused",
    )
    args = ap.parse_args(argv)

    out = Path(args.out)
    # A resumed run keeps the settings it was started with (its own config.json), so a
    # one-off --set from the first launch isn't silently dropped; new --set still apply.
    saved = out / "config.json"
    cfg = config_mod.load(saved if args.resume and saved.exists() else args.config, args.set)
    if args.steps:
        cfg.train.total_steps = args.steps
    if args.envs:
        cfg.train.n_envs = args.envs
    t = cfg.train
    n_envs = t.n_envs or default_n_envs()
    out.mkdir(parents=True, exist_ok=True)
    cfg.save(out / "config.json")
    # The learner is small; a few threads are plenty and leave the cores to the sims.
    torch.set_num_threads(max(1, min(8, (os.cpu_count() or 2) // 4)))
    print(f"== training {cfg.name}: {t.total_steps:,} steps on {n_envs} simulators ==", flush=True)

    vec = build_vec_env(cfg, n_envs, t.seed)
    if args.resume:
        resume = Path(args.resume)
        norm_path = (
            resume.with_name(resume.name.replace("model_", "vecnormalize_")).with_suffix(".pkl")
            if resume.name.startswith("model_")
            else resume.with_name("vecnormalize.pkl")
        )
        vec = VecNormalize.load(str(norm_path), vec)
        vec.training = True
        model = PPO.load(resume, env=vec, device="cpu")
        reset = False
        # learn() resets the envs before the first rollout callback, so a resumed run
        # would spend its first episodes at full randomisation without this.
        vec.set_attr("dr_scale", ramp_scale(cfg, model.num_timesteps))
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=CLIP_OBS, gamma=t.gamma)
        model = PPO(
            "MlpPolicy",
            vec,
            n_steps=t.n_steps,
            batch_size=min(t.batch_size, t.n_steps * n_envs),
            n_epochs=t.n_epochs,
            learning_rate=t.learning_rate,
            gamma=t.gamma,
            gae_lambda=t.gae_lambda,
            clip_range=t.clip_range,
            ent_coef=t.ent_coef,
            policy_kwargs={
                "net_arch": {"pi": list(t.policy_layers), "vf": list(t.value_layers)},
                "activation_fn": torch.nn.Tanh,
                "log_std_init": t.log_std_init,
            },
            tensorboard_log=str(out / "tb"),
            seed=t.seed,
            device="cpu",
            verbose=1,
        )
        reset = True
        # The clone samples the robots the curriculum starts on.
        vec.set_attr("dr_scale", ramp_scale(cfg, 0))
        if t.warm_gait:
            from growbot_train.gait import Keyframes, clone_gait

            path = Path(t.warm_gait)
            if not path.exists():
                path = Path(__file__).resolve().parents[1] / t.warm_gait
            print(f"== cloning the gait in {path} before PPO ==", flush=True)
            mse = clone_gait(
                model, vec, cfg, Keyframes.load(path), t.warm_gait_samples, t.warm_gait_epochs
            )
            print(f"== cloned: mse {mse:.4f} ==", flush=True)

    # The live folder (progress, latest replay and policy) is what a phone watches.
    live = Live(out, cfg, t.total_steps, start_steps=0 if reset else model.num_timesteps)
    callbacks = CallbackList(
        [
            Curriculum(cfg),
            Diagnostics(live),
            Checkpoint(t.checkpoint_every_steps, out / "checkpoints"),
            LiveReplay(t.live_replay_every_steps, live),
        ]
    )
    try:
        model.learn(
            total_timesteps=t.total_steps,
            callback=callbacks,
            reset_num_timesteps=reset,
            tb_log_name="ppo",
        )
    except BaseException:
        live.mark("failed")
        raise
    live.mark("finishing")
    model.save(out / "final_model.zip")
    vec.save(str(out / "vecnormalize.pkl"))
    vec.close()
    print(f"== saved {out / 'final_model.zip'} ==", flush=True)

    from growbot_train.export import export_run

    policy_path = export_run(out)
    print(f"== exported {policy_path} ==", flush=True)
    return 0


if __name__ == "__main__":
    np.set_printoptions(precision=3, suppress=True)
    sys.exit(main())
