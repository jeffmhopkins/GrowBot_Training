"""The exported file, run by the JBrain2 app's own runner, gives the same leg commands
as Python. Needs node and a JBrain2 checkout: set GROWBOT_SITE to its growbot/site."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from growbot_train.evaluate import env_for
from growbot_train.policy import JsonPolicy

SITE = os.environ.get("GROWBOT_SITE")
pytestmark = pytest.mark.skipif(
    not SITE or not shutil.which("node"), reason="set GROWBOT_SITE and install node"
)


@pytest.mark.parametrize("run", ["tiny_run", "tiny_speed_run"])
def test_app_runner_matches(run: str, request: pytest.FixtureRequest, tmp_path: Path) -> None:
    policy_path = request.getfixturevalue(run) / "policy_heading_walk.json"
    pol = JsonPolicy(policy_path)
    env = env_for(pol)
    obs, _ = env.reset(
        seed=5, options={"target_error_deg": 60.0, "speed_cmd": 0.6, "tilt_cmd": -0.4}
    )
    frames, actions = [], []
    for _ in range(40):
        frame = obs[: len(env.channels)]
        frames.append(frame.astype(float).tolist())
        a = pol.act(obs)
        actions.append(a.tolist())
        obs, _, term, _, _ = env.step(pol.env_action(obs))
        if term:
            break
    case = tmp_path / "case.json"
    case.write_text(json.dumps({"frames": frames, "channels": list(env.channels)}))
    script = Path(__file__).parent / "js" / "runner_parity.mjs"
    out = subprocess.run(
        ["node", str(script), str(Path(SITE) / "growbot_runner.js"), str(policy_path), str(case)],
        check=True,
        capture_output=True,
        text=True,
    )
    js = np.array(json.loads(out.stdout))
    # float32 observations on the Python side vs float64 in JS: agreement to ~1e-5.
    np.testing.assert_allclose(js, np.array(actions), atol=1e-4)
