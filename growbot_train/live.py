"""A run's ``live/`` folder: what a phone can look at while training is still going.

The trainer keeps it current. JBrain's Launcher shows it on the job's page:

    progress.json       how far along, an estimate of the time left, and how the walk is
                        doing right now (rewritten after every batch of experience)
    replay.html         the 3D replay of the latest checkpoint (rewritten at each one)
    policy_latest.json  that checkpoint as an app policy, ready to download

Its ``state`` is "training", then "finishing" while the run evaluates and draws the final
replay, then "done" once ``python -m growbot_train.live finish runs/<name>`` swaps in the
final policy, the full replay and the evaluation ("failed" if the run stops on an error).
``progress.json``'s ``title``/``state``/``percent``/``eta_s``/``summary`` shape is what the
Launcher reads; the ``history`` list is for anything that wants to chart the run.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from growbot_train import config as config_mod

LIVE = "live"
HISTORY_MAX = 400
# The live replay is every walk but the duplicate random-body ones, 24 s each (owner,
# 2026-11-14: 12 s was too short to see a walk settle; 2026-11-15: three walks showed too
# little of what it trains on). The pause it costs is timed in the run log.
LIVE_SCENARIOS = (
    "Target ahead",
    "Target 90 degrees left",
    "Target behind",
    "Target turns every 6 s",
    "Speed steps: full, half, stop, full",
    "Like training: random body, targets and speeds",
    "Like training: another random body",
)
LIVE_SECONDS = 24.0


def _write(path: Path, text: str) -> None:
    # Write-then-rename, so a page reading mid-update never sees half a file.
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _cpu_seconds(pid: str) -> tuple[int, float] | None:
    """(process group, CPU seconds used) of one process, from /proc; None if it's gone or
    there is no /proc (macOS)."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            fields = fh.read().rsplit(")", 1)[1].split()
        return int(fields[2]), (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, IndexError, ValueError):
        return None


class CpuMeter:
    """How many cores the run is really keeping busy: the CPU time its process group (the
    trainer and every simulator worker) used since the last reading, over the wall time.
    The trainer's own share says whether one serial process is what holds the run back
    (owner, 2026-11-15: two runs side by side still showed the box ~10% busy)."""

    def __init__(self) -> None:
        self.last: tuple[float, float, float] | None = None
        self.read()

    def read(self) -> dict[str, float]:
        try:
            group = os.getpgrp()
            pids = [p for p in os.listdir("/proc") if p.isdigit()]
        except (AttributeError, OSError):
            return {}
        total = 0.0
        for pid in pids:
            got = _cpu_seconds(pid)
            if got is not None and got[0] == group:
                total += got[1]
        mine = _cpu_seconds("self")
        now = time.monotonic()
        prev, self.last = self.last, (now, total, mine[1] if mine else 0.0)
        if prev is None or now <= prev[0]:
            return {}
        wall = now - prev[0]
        return {
            "cpu_cores": max(0.0, total - prev[1]) / wall,
            "trainer_cpu": max(0.0, self.last[2] - prev[2]) / wall,
        }


def _usable() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class Live:
    def __init__(
        self, run: Path, cfg: config_mod.Config, total_steps: int, start_steps: int = 0
    ) -> None:
        self.folder = run / LIVE
        self.folder.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        self.total = start_steps + total_steps
        self.start_steps = start_steps
        self.t0 = time.time()
        self.history: list[dict[str, float]] = []
        if start_steps:  # a resumed run keeps its chart
            old = _load(self.folder / "progress.json")
            self.history = [h for h in old.get("history", []) if h.get("steps", 0) <= start_steps]
        self.replay_steps: int | None = None
        self.latest: dict[str, float] = {}
        self.steps = start_steps
        self._save()

    def update(self, steps: int, metrics: dict[str, float]) -> None:
        self.steps = steps
        self.latest = metrics
        self.history.append({"steps": steps, **{k: round(v, 5) for k, v in metrics.items()}})
        if len(self.history) > HISTORY_MAX:  # keep it small: drop every other old point
            self.history = self.history[: HISTORY_MAX // 2 : 2] + self.history[HISTORY_MAX // 2 :]
        self._save()

    def snapshot(self, model: Any, steps: int) -> None:
        """Export the current network and write its replay. Never fatal: a failure here is
        logged and training carries on."""
        try:
            from growbot_train.export import build_policy
            from growbot_train.replay import SCENARIOS, build, write_html

            vec = model.get_vec_normalize_env()
            policy = build_policy(model, vec, self.cfg, f"growbot-{self.cfg.name}-{steps}")
            path = self.folder / "policy_latest.json"
            _write(path, json.dumps(policy, separators=(",", ":")) + "\n")
            chosen = tuple(s for s in SCENARIOS if s[0] in LIVE_SCENARIOS)
            t0 = time.monotonic()
            data = build(path, LIVE_SECONDS, 0, None, chosen, quiet=True)
            print(f"live: replay at {steps} took {time.monotonic() - t0:.1f} s", flush=True)
            data["note"] = f"Checkpoint at {steps:,} steps, while training"
            data["live"] = True
            tmp = self.folder / ".replay.html.tmp"
            write_html(data, tmp)
            os.replace(tmp, self.folder / "replay.html")
            self.replay_steps = steps
            self._save()
        except Exception as exc:  # noqa: BLE001 - the live view must never stop a run
            print(f"live: snapshot at {steps} failed: {exc}", flush=True)

    def mark(self, state: str) -> None:
        """Training has stopped: "finishing" (evaluation and the final replay still to come)
        or "failed". Either way there is no time left to estimate."""
        self._save(state)

    def _save(self, state: str = "training") -> None:
        try:
            self._write_progress(state)
        except Exception as exc:  # noqa: BLE001 - the live view must never stop a run
            print(f"live: progress not written: {exc}", flush=True)

    def _write_progress(self, state: str) -> None:
        done = max(0, self.steps - self.start_steps)
        elapsed = time.time() - self.t0
        rate = done / elapsed if elapsed > 0 else 0.0
        left = max(0, self.total - self.steps)
        cap = self.cfg.reward.speed_cap_m_s * 100
        summary = [
            {"label": "Steps", "value": f"{self.steps:,} of {self.total:,}"},
            {"label": "Speed", "value": f"{rate:,.0f} steps/s"},
        ]
        m = self.latest
        if m:
            if self.cfg.task.speed == "cap" or "speed_miss" not in m:
                summary.append(
                    {
                        "label": "Toward target",
                        "value": f"{m['v_along'] * 100:.1f} cm/s (cap {cap:.1f})",
                    }
                )
            else:
                # Stops and back-aways are mixed in, so the useful number is how far off
                # the commanded speed it walks.
                summary.append(
                    {
                        "label": "Off commanded speed",
                        "value": f"{m['speed_miss'] * 100:.1f} cm/s (cap {cap:.1f})",
                    }
                )
            summary += [
                {"label": "Heading error", "value": f"{m['abs_err_deg']:.0f}°"},
                {"label": "Falls", "value": f"{m['falls_per_1k_steps']:.1f} per 1,000 steps"},
            ]
            if self.cfg.reward.anchor and "drift_m" in m:
                summary.append(
                    {"label": "Wandered while stopped", "value": f"{m['drift_m'] * 100:.1f} cm"}
                )
            if "cpu_cores" in m:
                summary.append(
                    {
                        "label": "CPU in use",
                        "value": f"{m['cpu_cores']:.1f} of {_usable()} cores (trainer "
                        f"{m['trainer_cpu']:.2f}, learning {100 * m.get('learn_frac', 0):.0f}%"
                        " of the time)",
                    }
                )
            if self.cfg.reward.stance and "lift_deg" in m:
                want = m.get("want_lift_deg", self.cfg.task.stand_lift_deg)
                summary.append(
                    {
                        "label": "Standing taller",
                        "value": f"{m['lift_deg']:+.1f}° (aim {want:+.0f}°)",
                    }
                )
        if self.replay_steps is not None:
            summary.append({"label": "Latest replay", "value": f"at {self.replay_steps:,} steps"})
        doc = {
            "title": f"Training {self.cfg.name}",
            "state": state,
            "percent": round(100 * min(1.0, self.steps / self.total), 1) if self.total else 0,
            "eta_s": round(left / rate) if rate > 0 and state == "training" else None,
            "elapsed_s": round(elapsed),
            "updated_at": _now(),
            "summary": summary,
            "history": self.history,
        }
        _write(self.folder / "progress.json", json.dumps(doc, indent=1) + "\n")


def finish(run: Path) -> None:
    """The run is over: the live folder gets the final policy, full replay and scores."""
    folder = run / LIVE
    folder.mkdir(parents=True, exist_ok=True)
    for src, dest in (
        (next(iter(sorted(run.glob("policy_*.json"))), None), "policy_final.json"),
        (run / "replay.html", "replay.html"),
        (run / "eval.txt", "eval.txt"),
    ):
        if src is not None and src.exists():
            shutil.copyfile(src, folder / f".{dest}.tmp")
            os.replace(folder / f".{dest}.tmp", folder / dest)
    (folder / "policy_latest.json").unlink(missing_ok=True)
    progress = folder / "progress.json"
    doc = _load(progress) or {"title": f"Training {run.name}", "summary": []}
    summary = [s for s in doc.get("summary", []) if s["label"] != "Latest replay"]
    tail = [line for line in _lines(run / "eval.txt") if line.startswith("eval summary:")]
    if tail:
        summary.append({"label": "Final evaluation", "value": tail[-1][len("eval summary: ") :]})
    if (run / "replay.html").exists():
        summary.append({"label": "Replay", "value": "final policy"})
    doc.update(state="done", percent=100, eta_s=None, updated_at=_now(), summary=summary)
    _write(progress, json.dumps(doc, indent=1) + "\n")


def failed(run: Path) -> None:
    """The run stopped with an error before it finished: say so, so a phone watching it
    doesn't wait on a "training" that will never move again."""
    progress = run / LIVE / "progress.json"
    doc = _load(progress)
    if not doc or doc.get("state") == "done":
        return
    doc.update(state="failed", eta_s=None, updated_at=_now())
    _write(progress, json.dumps(doc, indent=1) + "\n")


def _load(path: Path) -> dict[str, Any]:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return doc if isinstance(doc, dict) else {}


def _lines(path: Path) -> list[str]:
    return path.read_text().splitlines() if path.exists() else []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Finish a run's live folder")
    ap.add_argument("action", choices=["finish", "failed"])
    ap.add_argument("run")
    args = ap.parse_args(argv)
    (finish if args.action == "finish" else failed)(Path(args.run))
    print(f"live: {Path(args.run) / LIVE} marked {args.action}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
