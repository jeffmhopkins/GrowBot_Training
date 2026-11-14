"""jobs.toml passes the same checks JBrain's Launcher applies when it reads it
(jlaunch_ctl/catalog.py in JBrain2), so a bad entry fails CI here instead of a Refresh on
the box."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from growbot_train.config import load

ROOT = Path(__file__).resolve().parent.parent
NAME = re.compile(r"^growbot_[a-z0-9_]{1,56}$")
CONFIG = re.compile(r"^configs/[A-Za-z0-9_-]{1,64}\.toml$")
KEYS = {"name", "title", "config", "est_hours", "notes", "disk_gb"}
LIMITS = {"title": 120, "est_hours": 20, "notes": 1000}


def _jobs() -> list[dict]:
    text = (ROOT / "jobs.toml").read_text()
    assert len(text.encode()) <= 64 * 1024
    data = tomllib.loads(text)
    assert set(data) <= {"job"}
    assert len(data["job"]) <= 50
    return data["job"]


def test_every_job_is_well_formed() -> None:
    names = set()
    for job in _jobs():
        assert set(job) <= KEYS, job
        assert {"name", "title", "config"} <= set(job), job
        assert NAME.fullmatch(job["name"])
        assert job["name"] not in names, f"{job['name']} is listed twice"
        names.add(job["name"])
        assert job["title"].strip()
        for key, limit in LIMITS.items():
            assert len(job.get(key, "")) <= limit, (job["name"], key)
        disk = job.get("disk_gb", 3)
        assert isinstance(disk, int) and not isinstance(disk, bool) and 1 <= disk <= 100


def test_every_config_exists_and_loads() -> None:
    for job in _jobs():
        assert CONFIG.fullmatch(job["config"]), job["config"]
        path = ROOT / job["config"]
        assert path.is_file() and not path.is_symlink(), job["config"]
        load(path)
