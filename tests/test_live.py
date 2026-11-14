from __future__ import annotations

import json
from pathlib import Path

from growbot_train.config import Config
from growbot_train.live import HISTORY_MAX, Live, finish


def _progress(run: Path) -> dict:
    return json.loads((run / "live" / "progress.json").read_text())


def test_progress_is_written_from_the_start_and_on_each_update(tmp_path: Path) -> None:
    live = Live(tmp_path, Config(), total_steps=1000)
    doc = _progress(tmp_path)
    assert doc["state"] == "training" and doc["percent"] == 0
    live.update(
        250, {"v_along": 0.02, "speed": 0.025, "abs_err_deg": 40.0, "falls_per_1k_steps": 0.5}
    )
    doc = _progress(tmp_path)
    assert doc["percent"] == 25.0
    labels = {s["label"]: s["value"] for s in doc["summary"]}
    assert labels["Toward target"].startswith("2.0 cm/s")
    assert labels["Heading error"] == "40°"
    assert doc["history"][-1]["steps"] == 250


def test_history_stays_bounded(tmp_path: Path) -> None:
    live = Live(tmp_path, Config(), total_steps=10**6)
    m = {"v_along": 0.0, "speed": 0.0, "abs_err_deg": 0.0, "falls_per_1k_steps": 0.0}
    for i in range(3 * HISTORY_MAX):
        live.update(i, m)
    assert len(live.history) <= HISTORY_MAX
    assert live.history[-1]["steps"] == 3 * HISTORY_MAX - 1


def test_snapshot_and_finish(tiny_run: Path, tmp_path: Path) -> None:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from growbot_train.env import GrowBotHeadingEnv

    cfg = Config()
    vec = VecNormalize.load(
        str(tiny_run / "vecnormalize.pkl"), DummyVecEnv([lambda: GrowBotHeadingEnv(cfg)])
    )
    model = PPO.load(tiny_run / "final_model.zip", env=vec, device="cpu")
    run = tmp_path / "run"
    live = Live(run, cfg, total_steps=128)
    live.snapshot(model, 128)
    assert (run / "live" / "policy_latest.json").exists()
    page = (run / "live" / "replay.html").read_text()
    assert "Checkpoint at 128 steps" in page

    # The run's end swaps in the final files and marks it done.
    (run / "policy_heading_walk.json").write_text("{}")
    (run / "replay.html").write_text("<html>final</html>")
    (run / "eval.txt").write_text("eval summary: toward 0.0100 m/s, heading error 20.0 deg\n")
    finish(run)
    doc = _progress(run)
    assert doc["state"] == "done" and doc["percent"] == 100 and doc["eta_s"] is None
    assert (run / "live" / "replay.html").read_text() == "<html>final</html>"
    assert (run / "live" / "policy_final.json").exists()
    assert not (run / "live" / "policy_latest.json").exists()
    assert any(s["label"] == "Final evaluation" for s in doc["summary"])
