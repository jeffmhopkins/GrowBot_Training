"""Many simulators per worker process.

SB3's SubprocVecEnv gives every simulator its own process and pipe. Each tick the
learner sends one action down each pipe and waits for every reply, and on the 32-thread
box that round trip cost more than the physics: training ran about 11,000 steps a
second with the cores 14% busy. Here each worker steps a block of simulators per round
trip, so the pipe cost is paid once per block instead of once per simulator.

Same contract as SubprocVecEnv (auto-reset, terminal_observation, TimeLimit.truncated,
seeds and options passed to the next reset), so the trainer and callbacks see no change.
"""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Sequence
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    CloudpickleWrapper,
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)


def _step_one(
    env: gym.Env, action: np.ndarray, reset_info: dict
) -> tuple[Any, float, bool, dict, dict]:
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    info["TimeLimit.truncated"] = truncated and not terminated
    if done:
        info["terminal_observation"] = obs
        obs, reset_info = env.reset()
    return obs, float(reward), done, info, reset_info


def _worker(remote: Any, parent_remote: Any, fns_wrapper: CloudpickleWrapper) -> None:
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    envs: list[gym.Env] = [fn() for fn in fns_wrapper.var]
    # As SB3: an env that didn't reset this step reports its last reset's info again.
    last_reset: list[dict] = [{} for _ in envs]
    try:
        _serve(remote, envs, last_reset, is_wrapped)
    except KeyboardInterrupt:
        pass


def _serve(remote: Any, envs: list[gym.Env], last_reset: list[dict], is_wrapped: Any) -> None:
    while True:
        try:
            cmd, data = remote.recv()
        except EOFError:
            break
        if cmd == "step":
            out = [_step_one(env, a, ri) for env, a, ri in zip(envs, data, last_reset, strict=True)]
            obs, rews, dones, infos, reset_infos = zip(*out, strict=True)
            last_reset[:] = reset_infos
            remote.send((np.stack(obs), np.array(rews), np.array(dones), infos, reset_infos))
        elif cmd == "reset":
            seeds, options = data
            out = []
            for env, seed, opt in zip(envs, seeds, options, strict=True):
                out.append(env.reset(seed=seed, **({"options": opt} if opt else {})))
            obs, reset_infos = zip(*out, strict=True)
            last_reset[:] = reset_infos
            remote.send((np.stack(obs), reset_infos))
        elif cmd == "get_spaces":
            remote.send((envs[0].observation_space, envs[0].action_space))
        elif cmd == "has_attr":
            name, local = data
            out = []
            for i in local:
                try:
                    envs[i].get_wrapper_attr(name)
                    out.append(True)
                except AttributeError:
                    out.append(False)
            remote.send(out)
        elif cmd == "get_attr":
            name, local = data
            remote.send([envs[i].get_wrapper_attr(name) for i in local])
        elif cmd == "set_attr":
            name, value, local = data
            for i in local:
                setattr(envs[i], name, value)
            remote.send(None)
        elif cmd == "env_method":
            name, args, kwargs, local = data
            remote.send([envs[i].get_wrapper_attr(name)(*args, **kwargs) for i in local])
        elif cmd == "is_wrapped":
            cls, local = data
            remote.send([is_wrapped(envs[i], cls) for i in local])
        elif cmd == "close":
            for env in envs:
                env.close()
            remote.close()
            break
        else:
            raise NotImplementedError(cmd)


class BlockSubprocVecEnv(VecEnv):
    """`env_fns` split into `workers` contiguous blocks, one process per block."""

    def __init__(self, env_fns: list[Callable[[], gym.Env]], workers: int) -> None:
        n = len(env_fns)
        workers = max(1, min(workers, n))
        bounds = np.linspace(0, n, workers + 1).round().astype(int)
        self._blocks = [range(bounds[w], bounds[w + 1]) for w in range(workers)]
        self._where = [(w, i - blk.start) for w, blk in enumerate(self._blocks) for i in blk]
        ctx = mp.get_context("fork")
        self.remotes = []
        self.processes = []
        # Each pipe is made just before its worker forks, and its child end closed right
        # after, so no other worker inherits it. Otherwise a dead worker's pipe stays
        # open in its siblings, the learner never sees EOF, and training hangs instead
        # of failing.
        for blk in self._blocks:
            remote, work = ctx.Pipe()
            self.remotes.append(remote)
            fns = CloudpickleWrapper([env_fns[i] for i in blk])
            p = ctx.Process(target=_worker, args=(work, remote, fns), daemon=True)
            p.start()
            self.processes.append(p)
            work.close()
        self.waiting = False
        self.closed = False
        self.remotes[0].send(("get_spaces", None))
        obs_space, act_space = self.remotes[0].recv()
        super().__init__(n, obs_space, act_space)

    def step_async(self, actions: np.ndarray) -> None:
        for remote, blk in zip(self.remotes, self._blocks, strict=True):
            remote.send(("step", actions[blk.start : blk.stop]))
        self.waiting = True

    def step_wait(self) -> VecEnvStepReturn:
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos, reset_infos = zip(*results, strict=True)
        self.reset_infos = [ri for block in reset_infos for ri in block]
        return (
            np.concatenate(obs),
            np.concatenate(rews),
            np.concatenate(dones),
            [i for block in infos for i in block],
        )

    def reset(self) -> VecEnvObs:
        for remote, blk in zip(self.remotes, self._blocks, strict=True):
            seeds = [self._seeds[i] for i in blk]
            options = [self._options[i] for i in blk]
            remote.send(("reset", (seeds, options)))
        results = [remote.recv() for remote in self.remotes]
        obs, reset_infos = zip(*results, strict=True)
        self.reset_infos = [ri for block in reset_infos for ri in block]
        self._reset_seeds()
        self._reset_options()
        return np.concatenate(obs)

    def _by_worker(self, indices: VecEnvIndices) -> dict[int, list[int]]:
        out: dict[int, list[int]] = {}
        for i in self._get_indices(indices):
            w, local = self._where[i]
            out.setdefault(w, []).append(local)
        return out

    def _ask(self, cmd: str, indices: VecEnvIndices, *payload: Any) -> list[Any]:
        groups = self._by_worker(indices)
        for w, local in groups.items():
            self.remotes[w].send((cmd, (*payload, local)))
        replies = {w: self.remotes[w].recv() for w in groups}
        # Back in the order the indices were asked for.
        order = list(self._get_indices(indices))
        flat = {
            self._blocks[w].start + loc: val
            for w, local in groups.items()
            for loc, val in zip(local, replies[w] or [None] * len(local), strict=True)
        }
        return [flat[i] for i in order]

    def has_attr(self, attr_name: str) -> bool:
        return all(self._ask("has_attr", None, attr_name))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        return self._ask("get_attr", indices, attr_name)

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        self._ask("set_attr", indices, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args: Any,
        indices: VecEnvIndices = None,
        **method_kwargs: Any,
    ) -> list[Any]:
        return self._ask("env_method", indices, method_name, method_args, method_kwargs)

    def env_is_wrapped(
        self, wrapper_class: type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> list[bool]:
        return self._ask("is_wrapped", indices, wrapper_class)

    def get_images(self) -> Sequence[np.ndarray | None]:
        return [None] * self.num_envs

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.processes:
            p.join()
        self.closed = True
