from __future__ import annotations

import json
import math

import numpy as np
import pytest

from growbot_train.config import Config
from growbot_train.env import CHANNELS, CONTROL_DT, GrowBotHeadingEnv, wrap_deg

NCH = len(CHANNELS)


def _env(**task: object) -> GrowBotHeadingEnv:
    cfg = Config()
    for k, v in task.items():
        setattr(cfg.task, k, v)
    return GrowBotHeadingEnv(cfg)


def ch(obs: np.ndarray, name: str, frame: int = 0) -> float:
    return float(obs[frame * NCH + CHANNELS.index(name)])


def test_wrap() -> None:
    assert wrap_deg(190) == -170
    assert wrap_deg(-190) == 170
    assert wrap_deg(0) == 0


@pytest.mark.parametrize("body", ["rollout", "olie"])
def test_standing_still_is_stable_and_reads_gravity(body: str) -> None:
    env = _env(body=body)
    obs, _ = env.reset(seed=0, options={"nominal": True, "retarget": False})
    assert obs.shape == (NCH * env.frames,)
    acc = np.array([ch(obs, "accel_x"), ch(obs, "accel_y"), ch(obs, "accel_z")])
    assert abs(np.linalg.norm(acc) - 9.81) < 0.3
    # Levelled at the stance like the app does: gravity reads straight +z even though
    # this body rests nose-up.
    assert abs(acc[0]) < 0.3 and abs(acc[1]) < 0.3
    for _ in range(125):  # 5 s of zero commands: nothing falls over
        obs, _, term, _, info = env.step(np.zeros(2))
        assert not term
    assert info["speed"] < 0.005


def test_frame_layout_matches_the_app_runner() -> None:
    env = _env()
    obs, _ = env.reset(
        seed=1, options={"nominal": True, "retarget": False, "target_error_deg": 36.0}
    )
    # First frame: no previous action, clock at phase zero, older frames still zero
    # (the runner starts its ring with zero frames too).
    assert ch(obs, "prev_action_0") == 0 and ch(obs, "prev_action_1") == 0
    assert ch(obs, "cycle:gait_sin") == pytest.approx(0.0)
    assert ch(obs, "cycle:gait_cos") == pytest.approx(1.0)
    assert ch(obs, "cmd_yaw") == pytest.approx(36.0 / 180.0, abs=1e-6)
    assert not np.any(obs[NCH:])

    raw = np.array([0.3, -0.7])
    obs, *_ = env.step(raw)
    assert ch(obs, "prev_action_0") == pytest.approx(math.tanh(0.3))
    assert ch(obs, "prev_action_1") == pytest.approx(math.tanh(-0.7))
    phase = 2 * math.pi * env.cfg.task.gait_hz * CONTROL_DT
    assert ch(obs, "cycle:gait_sin") == pytest.approx(math.sin(phase))
    # The previous newest frame shifted one slot older.
    assert ch(obs, "cmd_yaw", frame=1) == pytest.approx(36.0 / 180.0, abs=1e-6)


@pytest.mark.parametrize("err", [30.0, -30.0, 179.0])
def test_heading_error_sign(err: float) -> None:
    env = _env()
    obs, _ = env.reset(seed=2, options={"nominal": True, "target_error_deg": err})
    # Positive = target counter-clockwise of the nose = turn left.
    assert wrap_deg(env.target_deg - env.yaw_deg()) == pytest.approx(err, abs=1e-6)
    assert ch(obs, "cmd_yaw") * 180 == pytest.approx(err, abs=1e-4)


def test_randomisation_stays_in_range_and_restores_nominal() -> None:
    env = _env()
    nominal = env.model.body_mass.copy()
    lo, hi = env.cfg.randomization.mass_scale
    seen = set()
    for seed in range(5):
        env.reset(seed=seed)
        scale = env.model.body_mass[env._torso] / nominal[env._torso]
        assert lo - 1e-9 <= scale <= hi + 1e-9
        seen.add(round(scale, 6))
    assert len(seen) > 1
    env.reset(seed=0, options={"nominal": True})
    np.testing.assert_allclose(env.model.body_mass, nominal)


def test_tipping_over_ends_the_episode_with_the_fall_penalty() -> None:
    env = _env()
    env.reset(seed=3, options={"nominal": True})
    env.data.qpos[3:7] = [math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]  # 90 deg roll
    _, reward, term, _, info = env.step(np.zeros(2))
    assert term and info["fell"]
    assert info["terms"]["fall"] == -env.cfg.reward.fall
    assert reward < -env.cfg.reward.fall + 3


def test_progress_is_capped_and_overspeed_penalised() -> None:
    env = _env()
    env.reset(seed=4, options={"nominal": True, "target_error_deg": 0.0, "retarget": False})
    cap = env.cfg.reward.speed_cap_m_s
    t = math.radians(env.target_deg)
    # Float the robot (no gravity, no contacts) and coast it along the target at three
    # times the cap, then check the books.
    import mujoco

    env.model.opt.gravity[:] = 0
    env.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    for _ in range(60):
        env.data.qvel[:] = 0
        env.data.qvel[0:2] = [3 * cap * math.cos(t), 3 * cap * math.sin(t)]
        _, _, _, _, info = env.step(np.zeros(2))
    assert info["terms"]["progress"] == pytest.approx(env.cfg.reward.progress, rel=1e-6)
    assert info["terms"]["overspeed"] < -1.0


def test_levelling_is_a_pure_tilt() -> None:
    from growbot_train.env import _rot_onto_z

    for v in ([5.59, 0.0, 8.06], [0.1, -3.0, 9.0], [0.0, 0.0, -9.8]):
        R = _rot_onto_z(np.array(v))
        np.testing.assert_allclose(R @ np.array(v), [0, 0, np.linalg.norm(v)], atol=1e-9)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    # A nose-up pitch is undone by pitching, so the nose (x) stays in the x-z plane.
    R = _rot_onto_z(np.array([5.59, 0.0, 8.06]))
    assert abs((R @ [1.0, 0, 0])[1]) < 1e-12


def test_camera_facing_measures_heading_from_the_tail() -> None:
    nose, cam = _env(), _env(facing="camera")
    for env in (nose, cam):
        env.reset(seed=3, options={"nominal": True})
    # Same seed, same pose: the camera end points exactly the other way.
    assert wrap_deg(cam.yaw_deg() - nose.yaw_deg() - 180.0) == pytest.approx(0.0, abs=1e-6)
    obs, _ = cam.reset(seed=3, options={"nominal": True, "target_error_deg": 0.0})
    assert ch(obs, "cmd_yaw") == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        _env(facing="sideways")


def test_folded_start_lies_flatter_and_holds_its_legs() -> None:
    env = _env(lying_start_prob=1.0)
    env.reset(seed=0, options={"nominal": True})
    rest = env._rest_pitch
    _, _, _, _, info = env.step(np.arctanh(env.prev_action * 0.999))
    assert info["lift_deg"] < -10.0  # folded: well below its standing pitch
    assert env.prev_action.max() < -0.7  # the policy starts from where the legs are
    # Levelling still happened on neutral legs, so "level" is the standing stance.
    env2 = _env()
    env2.reset(seed=0, options={"nominal": True})
    assert rest == pytest.approx(env2._rest_pitch, abs=1e-6)


def test_stance_bonus_pays_for_standing_taller() -> None:
    cfg = Config()
    cfg.task.stand_lift_deg = 6.0
    cfg.reward.stance = 0.5
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    _, _, _, _, rest = env.step(np.zeros(2))
    env.reset(seed=0, options={"nominal": True})
    for _ in range(40):
        _, _, _, _, tall = env.step(np.full(2, np.arctanh(0.9)))
    assert tall["lift_deg"] > 4.0
    assert tall["terms"]["stance"] > rest["terms"]["stance"]
    plain = _env()
    plain.reset(seed=0)
    assert "stance" not in plain.step(np.zeros(2))[4]["terms"]  # off by default


def test_going_over_backwards_is_a_fall_and_leaning_back_costs() -> None:
    import mujoco

    cfg = Config()
    cfg.task.back_fall_pitch_deg = 75.0
    cfg.reward.back_lean = 2.0
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    _, _, fell, _, info = env.step(np.zeros(2))
    assert not fell and info["terms"]["back_lean"] == 0.0
    # Tip it to 80 degrees nose-up (nose-up is about -y) and hold it there for a tick.
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0.0, -math.radians(80.0), 0.0]), "xyz")
    env.data.qpos[3:7] = q
    env.data.qpos[2] = 0.09
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    _, reward, fell, _, info = env.step(np.zeros(2))
    assert fell and info["pitch_deg"] > 75.0
    assert info["terms"]["back_lean"] < -1.0


def test_shoves_spin_the_body_only_on_randomised_episodes() -> None:
    cfg = Config()
    cfg.randomization.push_every_s = (0.04, 0.04)
    cfg.randomization.push_pitch_rad_s = 10.0
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=1)
    spins = []
    for _ in range(5):
        env.step(np.zeros(2))
        spins.append(abs(env.data.qvel[4]))
    assert max(spins) > 1.0
    env.reset(seed=1, options={"nominal": True})
    env.step(np.zeros(2))
    assert abs(env.data.qvel[4]) < 1.0


def test_back_lean_is_measured_from_the_ordinary_stance_not_a_flat_settle() -> None:
    # A nose-heavy body can settle flat in reset; sitting back up in the ordinary stance
    # must not cost it anything, and a tail-heavy one resting steep is not a fall.
    cfg = Config()
    cfg.task.back_fall_pitch_deg = 75.0
    cfg.reward.back_lean = 2.0
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    env._rest_pitch = 0.0
    _, _, fell, _, info = env.step(np.zeros(2))
    assert not fell and info["terms"]["back_lean"] == 0.0
    assert 25.0 < env._nominal_rest_pitch < 45.0
    env._rest_pitch = 80.0
    import mujoco

    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([0.0, -math.radians(80.0), 0.0]), "xyz")
    env.data.qpos[3:7] = q
    env.data.qpos[2] = 0.09
    env.data.qvel[:] = 0.0
    mujoco.mj_forward(env.model, env.data)
    _, _, fell, _, info = env.step(np.zeros(2))
    tipped = info["pitch_deg"]
    assert tipped > 75.0 and not fell


def test_speed_command_rides_cmd_forward_and_redraws() -> None:
    from growbot_train.env import channels_for

    cfg = Config()
    cfg.task.speed = "command"
    cfg.task.speed_command_every_s = (0.2, 0.2)
    chans = channels_for(cfg)
    assert chans == (*CHANNELS[:9], "cmd_forward", *CHANNELS[9:])
    env = GrowBotHeadingEnv(cfg)
    obs, _ = env.reset(seed=3)
    assert obs.shape == (len(chans) * cfg.task.history_frames,)
    seen = set()
    for _ in range(400):
        obs, *_ = env.step(np.zeros(2))
        assert obs[chans.index("cmd_forward")] == pytest.approx(env.speed_cmd)
        seen.add(round(env.speed_cmd, 3))
    assert 0.0 in seen and min(seen) >= cfg.task.speed_back_min and max(seen) <= 1.0
    assert any(v < 0 for v in seen) and any(v > 0.5 for v in seen)
    env.reset(seed=3, options={"speed_cmd": 0.4})
    for _ in range(50):
        env.step(np.zeros(2))
    assert env.speed_cmd == 0.4


@pytest.mark.parametrize("speed", ["command", "still"])
def test_standing_still_never_pays(speed: str) -> None:
    cfg = Config()
    cfg.task.speed = speed
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True, "speed_cmd": 0.0})
    assert "cmd_forward" not in env.channels or speed == "command"
    for _ in range(25):
        _, _, _, _, info = env.step(np.zeros(2))
    assert info["speed_cmd"] == 0.0
    # Free pay for standing still is what froze the first camera walk and look runs.
    assert -0.2 * cfg.reward.progress < info["terms"]["progress"] <= 0.0
    if speed == "command":
        env.reset(seed=0, options={"nominal": True, "speed_cmd": 1.0})
        for _ in range(25):
            _, _, _, _, info = env.step(np.zeros(2))
        assert abs(info["terms"]["progress"]) < 0.2 * cfg.reward.progress


def test_a_stop_lets_it_drift_a_little_for_free() -> None:
    cfg = Config()
    cfg.task.speed = "still"
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    cap = cfg.reward.speed_cap_m_s
    terms = []
    for drift in (0.3 * cap, 2.0 * cap):
        env.vel_avg = np.array([drift, 0.0])
        env.data.qvel[0:2] = env.vel_avg
        _, _, _, _, info = env.step(np.zeros(2))
        terms.append(info["terms"]["progress"])
    assert terms[0] == 0.0
    assert terms[1] < -0.5 * cfg.reward.progress


def test_the_free_drift_is_only_for_an_exact_stop() -> None:
    cfg = Config()
    cfg.task.speed = "command"
    env = GrowBotHeadingEnv(cfg)
    cap = cfg.reward.speed_cap_m_s
    terms = {}
    for cmd in (0.0, 0.3):
        env.reset(seed=0, options={"nominal": True, "speed_cmd": cmd})
        tr = math.radians(env.target_deg + 90)  # sideways: no part of it is wanted
        env.vel_avg = 0.3 * cap * np.array([math.cos(tr), math.sin(tr)])
        env.data.qvel[0:2] = env.vel_avg
        _, _, _, _, info = env.step(np.zeros(2))
        terms[cmd] = info["terms"]["progress"]
    assert terms[0.0] == 0.0
    assert terms[0.3] < -0.05 * cfg.reward.progress


def test_turn_pays_for_closing_the_error_but_not_for_a_retarget() -> None:
    cfg = Config()
    cfg.reward.turn = 1.0
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    env.target_deg = wrap_deg(env.yaw_deg() + 90.0)
    env.data.qvel[5] = 2.0  # spinning CCW, toward the target
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["terms"]["turn"] > 0.5
    env.data.qvel[5] = -2.0
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["terms"]["turn"] < -0.5
    env.data.qvel[5] = 0.0
    env.target_deg = wrap_deg(env.target_deg + 120.0)  # a jump in the target alone pays nothing
    _, _, _, _, info = env.step(np.zeros(2))
    assert abs(info["terms"]["turn"]) < 0.5


def test_old_runs_with_a_retired_setting_still_load(tmp_path) -> None:
    from growbot_train.config import load

    old = Config().to_dict()
    old["reward"]["speed_track_tol"] = 1.5
    path = tmp_path / "config.json"
    path.write_text(json.dumps(old))
    assert load(path).reward.progress == Config().reward.progress


def test_walking_as_commanded_pays_in_full() -> None:
    cfg = Config()
    cfg.task.speed = "command"
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True, "speed_cmd": -1.0})
    tr = math.radians(env.target_deg)
    cap = cfg.reward.speed_cap_m_s
    env.vel_avg = -cap * np.array([math.cos(tr), math.sin(tr)])
    env.data.qvel[0:2] = env.vel_avg  # keep the average there through one step
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["terms"]["progress"] > 0.8 * cfg.reward.progress


def test_wobble_costs_rocking_and_bad_speed_mode_fails() -> None:
    cfg = Config()
    cfg.reward.wobble = 0.1
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    _, _, _, _, info = env.step(np.zeros(2))
    still = info["terms"]["wobble"]
    env.data.qvel[3:5] = [4.0, -4.0]  # a hard rock, mid-air so contact can't stop it
    env.data.qpos[2] += 0.05
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["terms"]["wobble"] < still - 0.5
    cfg.task.speed = "fast"
    with pytest.raises(ValueError):
        GrowBotHeadingEnv(cfg)


def test_camera_height_command_moves_the_stance_it_pays_for() -> None:
    from growbot_train.env import channels_for

    cfg = Config()
    cfg.task.speed = "command"
    cfg.task.tilt_command = True
    cfg.reward.stance = 0.5
    chans = channels_for(cfg)
    assert chans[8:11] == ("cmd_yaw", "cmd_forward", "cmd_pitch")
    env = GrowBotHeadingEnv(cfg)
    obs, _ = env.reset(seed=0, options={"nominal": True, "tilt_cmd": 1.0})
    assert obs[chans.index("cmd_pitch")] == 1.0
    _, _, _, _, tall = env.step(np.zeros(2))
    assert tall["want_lift_deg"] == cfg.task.tilt_lift_deg[1]
    env.reset(seed=0, options={"nominal": True, "tilt_cmd": -1.0})
    _, _, _, _, low = env.step(np.zeros(2))
    assert low["want_lift_deg"] == cfg.task.tilt_lift_deg[0]
    # Resting on neutral legs is near the tall stance and far above the low one.
    assert tall["terms"]["stance"] > 0.1 > low["terms"]["stance"]
    cfg.task.tilt_command_every_s = (0.2, 0.2)
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=4)
    seen = set()
    for _ in range(300):
        env.step(np.zeros(2))
        seen.add(round(env.tilt_cmd, 3))
    assert 1.0 in seen and min(seen) >= -1.0 and len(seen) > 5


def test_a_config_without_the_new_inputs_draws_no_extra_random_numbers() -> None:
    # Same seed, same episode as before the speed and height inputs existed.
    cfg = Config()
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=7)
    after_reset = env.np_random.random()
    cfg2 = Config()
    cfg2.task.speed = "command"
    env2 = GrowBotHeadingEnv(cfg2)
    env2.reset(seed=7)
    assert env2.np_random.random() != after_reset
