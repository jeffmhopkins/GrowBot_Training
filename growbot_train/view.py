"""Watch a policy walk.

    python -m growbot_train.view runs/my_run/policy_heading_walk.json
    python -m growbot_train.view runs/my_run/policy_heading_walk.json --video walk.mp4

In the window: Left / Right arrow turn the target 30 degrees, Up points it back at the
robot's nose, Backspace restarts on a new random body, N toggles the nominal body. The
red arrow is the target direction; the console prints speed and heading error once a
second. ``--video`` renders without a window (set MUJOCO_GL=egl on a headless machine).
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import mujoco
import numpy as np

from growbot_train.env import CONTROL_DT, GrowBotHeadingEnv, wrap_deg
from growbot_train.evaluate import env_for
from growbot_train.policy import JsonPolicy

KEY_RIGHT, KEY_LEFT, KEY_UP, KEY_BACKSPACE, KEY_N = 262, 263, 265, 259, 78


def _target_arrow(scn: mujoco.MjvScene, env: GrowBotHeadingEnv, keep: int = 0) -> None:
    """Draw the target arrow after the first ``keep`` geoms (the viewer's own overlay
    scene starts empty; a renderer's scene already holds the robot)."""
    if keep >= scn.maxgeom:
        return
    scn.ngeom = keep
    pos = env.data.qpos[0:3].copy()
    pos[2] += 0.06
    t = math.radians(env.target_deg)
    tip = pos + 0.15 * np.array([math.cos(t), math.sin(t), 0.0])
    g = scn.geoms[keep]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3),
        np.zeros(3),
        np.eye(3).flatten(),
        np.array([1, 0.1, 0.1, 1], dtype=np.float32),
    )
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.006, pos, tip)
    scn.ngeom = keep + 1


def interactive(env: GrowBotHeadingEnv, policy: JsonPolicy, seed: int) -> None:
    import mujoco.viewer

    state = {"reset": False, "turn": 0.0, "ahead": False, "nominal": False}

    def on_key(key: int) -> None:
        if key == KEY_LEFT:
            state["turn"] += 30.0
        elif key == KEY_RIGHT:
            state["turn"] -= 30.0
        elif key == KEY_UP:
            state["ahead"] = True
        elif key == KEY_BACKSPACE:
            state["reset"] = True
        elif key == KEY_N:
            state["nominal"] = not state["nominal"]
            state["reset"] = True

    obs, _ = env.reset(seed=seed, options={"retarget": False})
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as v:
        v.cam.distance, v.cam.elevation = 0.7, -25.0
        last_print = 0.0
        while v.is_running():
            t0 = time.perf_counter()
            if state["reset"]:
                seed += 1
                obs, _ = env.reset(
                    seed=seed, options={"retarget": False, "nominal": state["nominal"]}
                )
                state["reset"] = False
                print(f"-- new body (seed {seed}{', nominal' if state['nominal'] else ''})")
            if state["turn"]:
                env.target_deg = wrap_deg(env.target_deg + state["turn"])
                state["turn"] = 0.0
            if state["ahead"]:
                env.target_deg = env.yaw_deg()
                state["ahead"] = False
            obs, _, term, trunc, info = env.step(policy.env_action(obs))
            if term:
                print("-- fell over; restarting")
                obs, _ = env.reset(seed=seed, options={"retarget": False})
            elif trunc:  # keep walking past the training episode length
                env.step_index = 0
            v.cam.lookat[:] = env.data.qpos[0:3]
            with v.lock():
                _target_arrow(v.user_scn, env)
            v.sync()
            now = time.time()
            if now - last_print > 1.0:
                last_print = now
                print(
                    f"toward {info['v_along'] * 100:5.2f} cm/s   speed "
                    f"{info['speed'] * 100:5.2f} cm/s   heading error "
                    f"{info['abs_err_deg']:5.1f} deg"
                )
            time.sleep(max(0.0, CONTROL_DT - (time.perf_counter() - t0)))


def record(
    env: GrowBotHeadingEnv,
    policy: JsonPolicy,
    seed: int,
    path: str,
    seconds: float,
    turn_every: float,
) -> None:
    import imageio.v2 as imageio

    obs, _ = env.reset(seed=seed, options={"retarget": False})
    renderer = mujoco.Renderer(env.model, 480, 640)
    cam = mujoco.MjvCamera()
    cam.distance, cam.azimuth, cam.elevation = 0.7, 135.0, -25.0
    steps = round(seconds / CONTROL_DT)
    with imageio.get_writer(path, fps=round(1 / CONTROL_DT)) as out:
        for i in range(steps):
            if turn_every and i and i % round(turn_every / CONTROL_DT) == 0:
                env.target_deg = wrap_deg(env.target_deg + 90.0)
            obs, _, term, _, _ = env.step(policy.env_action(obs))
            if term:
                break
            cam.lookat[:] = env.data.qpos[0:3]
            renderer.update_scene(env.data, cam)
            _target_arrow(renderer.scene, env, keep=renderer.scene.ngeom)
            out.append_data(renderer.render())
    renderer.close()
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Watch an exported policy walk")
    ap.add_argument("policy")
    ap.add_argument("--body", help="rollout / olie")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--video", help="write an mp4 instead of opening a window")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument(
        "--turn-every",
        type=float,
        default=5.0,
        help="video only: turn the target 90 degrees this often (0 = never)",
    )
    args = ap.parse_args(argv)
    policy = JsonPolicy(args.policy)
    env = env_for(policy, args.body)
    if args.video:
        record(env, policy, args.seed, args.video, args.seconds, args.turn_every)
    else:
        interactive(env, policy, args.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
