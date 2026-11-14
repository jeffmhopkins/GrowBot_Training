from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from growbot_train.env import CHANNELS
from growbot_train.evaluate import env_for, run_episode
from growbot_train.policy import JsonPolicy, runner_forward

# What growbot_runner.js's Walker accepts (its _validate), restated so a change to the
# exporter that the app would refuse fails here first.
RUNNER_KNOWN = {
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "cmd_forward",
    "cmd_yaw",
    "cmd_pitch",
    "euler_roll",
    "euler_pitch",
    "euler_yaw",
}


def test_export_verifies(tiny_run: Path) -> None:
    log = (tiny_run / "export_verify.log").read_text().strip().splitlines()
    assert log[-1] == "EXPORT OK"


def test_policy_json_satisfies_the_app_runner(tiny_run: Path) -> None:
    p = json.loads((tiny_run / "policy_heading_walk.json").read_text())
    spec = p["obs_spec"]
    seg = spec["segments"][0]
    assert seg["order"] == "newest_first" and seg["layout"] == "frame_major"
    assert p["obs_size"] == len(seg["channels"]) * seg["frames"] == len(p["mean"]) == len(p["std"])
    assert all(s > 0 for s in p["std"])
    assert len(p["layers"][0]["W"]) == p["obs_size"]
    assert spec["control_hz"] > 0 and spec["imu_frame"]["up"] == "+z"
    for c in seg["channels"]:
        cy = re.match(r"^cycle:(.+)$", c)
        pa = re.match(r"^prev_action_(\d+)$", c)
        assert (
            c in RUNNER_KNOWN
            or (pa and int(pa[1]) < p["act_size"])
            or (cy and spec["cycles"][cy[1]]["type"] in ("sin", "cos"))
        ), c
    for layer in p["layers"]:
        assert layer["act"] in ("tanh", "linear")
        assert len(layer["W"][0]) == len(layer["b"])


def test_json_forward_is_what_the_policy_does(tiny_run: Path) -> None:
    pol = JsonPolicy(tiny_run / "policy_heading_walk.json")
    obs = np.random.default_rng(0).normal(size=(4, len(CHANNELS) * pol.frames))
    np.testing.assert_allclose(pol.act(obs), runner_forward(pol.data, obs))
    assert np.all(np.abs(pol.act(obs)) <= 1)


def test_evaluate_runs_the_exported_file(tiny_run: Path) -> None:
    pol = JsonPolicy(tiny_run / "policy_heading_walk.json")
    env = env_for(pol)
    m = run_episode(env, pol, seed=0, options={"target_error_deg": 90.0})
    assert set(m) == {"toward", "speed", "over_pct", "err", "turn_s", "fell"}


def test_replay_page_embeds_the_walks(tiny_run: Path, tmp_path: Path) -> None:
    from growbot_train.replay import PLACEHOLDER, build, write_html

    data = build(tiny_run / "policy_heading_walk.json", seconds=1.0, seed=0, body=None)
    out = tmp_path / "replay.html"
    write_html(data, out)
    page = out.read_text()
    assert PLACEHOLDER not in page
    blob = page.split("const DATA = ", 1)[1].split(";\n", 1)[0]
    loaded = json.loads(blob.replace("<\\/", "</"))
    first = loaded["episodes"][0]
    # One pose (xyz + quaternion) per box geom per tick, ticks at 25 Hz.
    assert len(first["frames"]) == 26
    assert all(len(pose) == 7 for pose in first["frames"][0][0])
    assert len(first["frames"][0][0]) == len(first["shapes"])


def test_speed_command_policy_declares_cmd_forward(tiny_speed_run: Path) -> None:
    p = json.loads((tiny_speed_run / "policy_heading_walk.json").read_text())
    chans = p["obs_spec"]["segments"][0]["channels"]
    assert chans[chans.index("cmd_yaw") + 1] == "cmd_forward"
    assert p["obs_size"] == len(chans) * p["obs_spec"]["segments"][0]["frames"]
    assert p["obs_spec"]["cmd_forward"]["meaning"] == "speed_command"
    assert p["training"]["speed"] == "command"
    assert chans[chans.index("cmd_forward") + 1] == "cmd_pitch"
    assert p["obs_spec"]["cmd_pitch"]["meaning"] == "camera_height"
    pol = JsonPolicy(tiny_speed_run / "policy_heading_walk.json")
    assert run_episode(env_for(pol), pol, seed=0, options={})["fell"] in (0.0, 1.0)
