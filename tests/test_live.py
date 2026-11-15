from __future__ import annotations

import json
import time
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
    assert '"live":true' in page  # the open page refreshes itself

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


def test_resume_keeps_history_and_counts_from_the_old_steps(tmp_path: Path) -> None:
    m = {"v_along": 0.01, "speed": 0.01, "abs_err_deg": 30.0, "falls_per_1k_steps": 0.0}
    first = Live(tmp_path, Config(), total_steps=1000)
    first.update(1000, m)
    resumed = Live(tmp_path, Config(), total_steps=1000, start_steps=1000)
    assert [h["steps"] for h in resumed.history] == [1000]
    resumed.update(1500, m)
    doc = _progress(tmp_path)
    assert doc["percent"] == 75.0
    assert [h["steps"] for h in doc["history"]] == [1000, 1500]


def test_states_after_training(tmp_path: Path) -> None:
    from growbot_train.live import failed

    live = Live(tmp_path, Config(), total_steps=1000)
    live.mark("finishing")
    assert _progress(tmp_path)["state"] == "finishing"
    assert _progress(tmp_path)["eta_s"] is None
    failed(tmp_path)
    assert _progress(tmp_path)["state"] == "failed"


def test_progress_write_failure_never_stops_training(tmp_path: Path, monkeypatch) -> None:
    import growbot_train.live as live_mod

    live = Live(tmp_path, Config(), total_steps=1000)

    def boom(*_a: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(live_mod, "_write", boom)
    live.update(10, {"v_along": 0.0, "speed": 0.0, "abs_err_deg": 0.0, "falls_per_1k_steps": 0.0})


def test_finish_without_progress_or_eval(tmp_path: Path) -> None:
    (tmp_path / "policy_heading_walk.json").write_text("{}")
    finish(tmp_path)
    doc = _progress(tmp_path)
    assert doc["state"] == "done"
    assert doc["title"] == f"Training {tmp_path.name}"
    assert (tmp_path / "live" / "policy_final.json").exists()


def test_live_files_match_what_the_launcher_serves(tiny_run: Path) -> None:
    import re

    allowed = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}\.(json|html|txt|csv|png|gz)$")
    finish(tiny_run)
    names = [p.name for p in (tiny_run / "live").iterdir() if not p.name.startswith(".")]
    assert names and all(allowed.match(n) for n in names), names


def test_commanded_runs_show_speed_miss_and_the_commanded_height(tmp_path: Path) -> None:
    cfg = Config()
    cfg.task.speed = "command"
    cfg.task.tilt_command = True
    cfg.reward.stance = 0.5
    live = Live(tmp_path, cfg, total_steps=1000)
    m = {"v_along": 0.0, "speed": 0.01, "abs_err_deg": 10.0, "falls_per_1k_steps": 0.0}
    live.update(10, {**m, "speed_miss": 0.004, "lift_deg": -2.0, "want_lift_deg": -3.0})
    labels = {s["label"]: s["value"] for s in _progress(tmp_path)["summary"]}
    assert "Toward target" not in labels
    assert labels["Off commanded speed"].startswith("0.4 cm/s")
    assert labels["Standing taller"] == "-2.0° (aim -3°)"


def test_the_cpu_meter_counts_the_run_and_its_trainer() -> None:
    from growbot_train.live import CpuMeter

    meter = CpuMeter()
    end = time.monotonic() + 0.5
    while time.monotonic() < end:  # keep this process busy for a moment
        pass
    got = meter.read()
    assert 0.2 < got["trainer_cpu"] <= 1.5
    assert got["cpu_cores"] >= got["trainer_cpu"] * 0.9
