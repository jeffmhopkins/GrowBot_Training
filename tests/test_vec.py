from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv

from growbot_train.vec import BlockSubprocVecEnv


class Counter(gym.Env):
    """Ends after `limit` steps (truncated) or when the action is negative (terminated)."""

    observation_space = gym.spaces.Box(-np.inf, np.inf, (2,), np.float32)
    action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

    def __init__(self, ident: int, limit: int = 3) -> None:
        self.ident = ident
        self.limit = limit
        self.t = 0
        self.tag = "none"

    def _obs(self) -> np.ndarray:
        return np.array([self.ident, self.t], np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return self._obs(), {"seed": seed, "options": options}

    def step(self, action):
        self.t += 1
        terminated = bool(action[0] < 0)
        truncated = self.t >= self.limit
        return self._obs(), float(self.ident + self.t), terminated, truncated, {}

    def scaled(self, k: int) -> int:
        return self.ident * k


class Crashes(Counter):
    def step(self, action):
        raise RuntimeError("simulator blew up")


def _fns(n: int, limit: int = 3):
    return [lambda i=i: Counter(i, limit) for i in range(n)]


def _run(vec, actions_list):
    out = [vec.reset()]
    for actions in actions_list:
        obs, rew, done, infos = vec.step(actions)
        resets = [sorted(r.items(), key=str) for r in vec.reset_infos]
        out.append((obs, rew, done, [sorted(i.items(), key=str) for i in infos], resets))
    return out


def test_it_steps_exactly_like_subproc() -> None:
    rng = np.random.default_rng(0)
    actions = [rng.uniform(-0.3, 1.0, (7, 1)).astype(np.float32) for _ in range(9)]
    block = BlockSubprocVecEnv(_fns(7), workers=3)
    sub = SubprocVecEnv(_fns(7), start_method="fork")
    try:
        got, want = _run(block, actions), _run(sub, actions)
        np.testing.assert_array_equal(got[0], want[0])
        for (go, gr, gd, gi, gz), (wo, wr, wd, wi, wz) in zip(got[1:], want[1:], strict=True):
            np.testing.assert_array_equal(go, wo)
            np.testing.assert_array_equal(gr, wr)
            np.testing.assert_array_equal(gd, wd)
            assert repr(gi) == repr(wi)
            assert repr(gz) == repr(wz)
    finally:
        block.close()
        sub.close()


def test_auto_reset_keeps_the_terminal_observation() -> None:
    vec = BlockSubprocVecEnv(_fns(4), workers=2)
    try:
        vec.reset()
        a = np.ones((4, 1), np.float32)
        a[2] = -1.0
        obs, _, done, infos = vec.step(a)
        assert done.tolist() == [False, False, True, False]
        np.testing.assert_array_equal(infos[2]["terminal_observation"], [2, 1])
        assert infos[2]["TimeLimit.truncated"] is False
        np.testing.assert_array_equal(obs[2], [2, 0])
        vec.step(np.ones((4, 1), np.float32))
        _, _, done, infos = vec.step(np.ones((4, 1), np.float32))
        assert done.tolist() == [True, True, False, True]
        assert infos[0]["TimeLimit.truncated"] is True
    finally:
        vec.close()


def test_seeds_and_options_reach_the_right_simulator() -> None:
    vec = BlockSubprocVecEnv(_fns(5), workers=2)
    try:
        vec.seed(10)
        vec.set_options([{"k": i} for i in range(5)])
        vec.reset()
        assert [ri["seed"] for ri in vec.reset_infos] == [10, 11, 12, 13, 14]
        assert [ri["options"] for ri in vec.reset_infos] == [{"k": i} for i in range(5)]
    finally:
        vec.close()


def test_attributes_and_methods_come_back_in_the_order_asked() -> None:
    vec = BlockSubprocVecEnv(_fns(6), workers=4)
    try:
        assert vec.get_attr("ident") == list(range(6))
        assert vec.get_attr("ident", indices=[5, 0, 3]) == [5, 0, 3]
        vec.set_attr("tag", "x", indices=[1, 4])
        assert vec.get_attr("tag") == ["none", "x", "none", "none", "x", "none"]
        assert vec.env_method("scaled", 10, indices=[4, 2]) == [40, 20]
        assert vec.env_method("scaled", k=2) == [0, 2, 4, 6, 8, 10]
        assert vec.env_is_wrapped(gym.Wrapper) == [False] * 6
        assert vec.has_attr("ident") is True
        assert vec.has_attr("nope") is False
    finally:
        vec.close()


def test_more_workers_than_simulators_is_one_each() -> None:
    vec = BlockSubprocVecEnv(_fns(2), workers=8)
    try:
        assert len(vec.processes) == 2
        assert vec.reset().shape == (2, 2)
    finally:
        vec.close()
        vec.close()


def test_a_crash_in_a_later_worker_fails_the_step_instead_of_hanging() -> None:
    # Before each pipe was made just before its fork, a sibling worker held a dead
    # worker's pipe open and the learner waited forever on it.
    import multiprocessing as mp

    def run(q) -> None:
        fns = _fns(6)
        fns[5] = lambda: Crashes(5)
        vec = BlockSubprocVecEnv(fns, workers=3)
        q.put([w.pid for w in vec.processes])
        vec.reset()
        try:
            vec.step(np.ones((6, 1), np.float32))
        except EOFError:
            q.put("failed")

    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=run, args=(q,))
    p.start()
    workers = q.get(timeout=20)
    p.join(20)
    hung = p.is_alive()
    if hung:
        # Its workers too: orphaned, they'd hold each other's pipes open forever.
        import os
        import signal

        for pid in [p.pid, *workers]:
            os.kill(pid, signal.SIGKILL)
    assert not hung
    assert q.get(timeout=1) == "failed"


def test_build_vec_env_picks_block_only_when_workers_share() -> None:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    from growbot_train.config import Config
    from growbot_train.train import build_vec_env

    cfg = Config()
    for n, workers, kind in [
        (4, 2, BlockSubprocVecEnv),
        (2, 2, SubprocVecEnv),
        (2, None, SubprocVecEnv),
        (1, 1, DummyVecEnv),
    ]:
        vec = build_vec_env(cfg, n, seed=0, workers=workers)
        try:
            assert type(vec.venv) is kind
            assert vec.num_envs == n
        finally:
            vec.close()
