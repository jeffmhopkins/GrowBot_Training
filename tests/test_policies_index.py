"""policies/index.toml passes the same checks JBrain's Launcher applies when it reads it
(jlaunch_ctl/policy_index.py in JBrain2), and every policy it lists loads."""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from growbot_train.policy import JsonPolicy

ROOT = Path(__file__).resolve().parent.parent
SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
JOB = re.compile(r"^growbot_[a-z0-9_]{1,56}$")
FILE = re.compile(r"^policies/[a-z0-9][a-z0-9_-]{0,47}/[A-Za-z0-9_-]{1,64}\.json$")
RUN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
KEYS = {"slug", "title", "job", "file", "run", "notes"}


def _policies() -> list[dict]:
    text = (ROOT / "policies" / "index.toml").read_text()
    assert len(text.encode()) <= 64 * 1024
    data = tomllib.loads(text)
    assert set(data) <= {"policy"}
    assert len(data.get("policy", [])) <= 50
    return data.get("policy", [])


def test_every_entry_is_well_formed() -> None:
    slugs = set()
    total = 0
    for p in _policies():
        assert set(p) <= KEYS and {"slug", "title", "job", "file"} <= set(p), p
        assert SLUG.fullmatch(p["slug"]) and p["slug"] not in slugs, p["slug"]
        slugs.add(p["slug"])
        assert JOB.fullmatch(p["job"]), p["job"]
        assert " ".join(p["title"].split()) and len(p["title"]) <= 60, p["title"]
        assert len(p.get("notes", "")) <= 1000
        assert "run" not in p or RUN.fullmatch(p["run"])
        assert FILE.fullmatch(p["file"]), p["file"]
        path = ROOT / p["file"]
        assert path.is_file() and not path.is_symlink(), p["file"]
        size = path.stat().st_size
        assert size <= 4 * 1024 * 1024, p["file"]
        total += size
    assert total <= 64 * 1024 * 1024


def test_every_listed_policy_loads() -> None:
    for p in _policies():
        JsonPolicy(ROOT / p["file"])
