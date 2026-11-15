from __future__ import annotations

import json
import math

import mujoco
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


@pytest.mark.parametrize("body", ["rollout", "olie", "build"])
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


def _coast(env: GrowBotHeadingEnv, along: float, side: float) -> dict:
    """Float the robot and coast it at (along, side) m/s relative to the target."""
    import mujoco

    t = math.radians(env.target_deg)
    env.model.opt.gravity[:] = 0
    env.model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    for _ in range(200):  # long enough for the 1 s speed average to settle
        env.data.qvel[:] = 0
        env.data.qvel[0:2] = [
            along * math.cos(t) - side * math.sin(t),
            along * math.sin(t) + side * math.cos(t),
        ]
        _, _, _, _, info = env.step(np.zeros(2))
    return info


@pytest.mark.parametrize("cmd", [1.0, 0.5])
def test_progress_scored_speed_commands_pay_like_the_cap_walk(cmd: float) -> None:
    cfg = Config()
    cfg.task.speed = "command"
    cfg.task.speed_reward = "progress"
    cfg.task.speed_back_min = 0.0
    cap = cfg.reward.speed_cap_m_s
    opts = {"nominal": True, "target_error_deg": 0.0, "retarget": False, "speed_cmd": cmd}

    def run(along: float, side: float) -> dict:
        env = GrowBotHeadingEnv(cfg)
        env.reset(seed=4, options=opts)
        return _coast(env, along, side)

    # At the commanded speed it earns that share of the full progress pay, no overspeed.
    info = run(cmd * cap, 0.0)
    assert info["terms"]["progress"] == pytest.approx(cmd * cfg.reward.progress, rel=1e-2)
    assert info["terms"]["overspeed"] == pytest.approx(0.0, abs=1e-6)
    # Sideways flailing costs no progress (the "match" scoring charges it), and going
    # faster than asked is charged.
    assert run(0.0, cmd * cap)["terms"]["progress"] == pytest.approx(0.0, abs=1e-4)
    if cmd < 1.0:
        assert run(cap, 0.0)["terms"]["overspeed"] < 0.0


def test_progress_scoring_is_checked() -> None:
    cfg = Config()
    cfg.task.speed = "command"
    cfg.task.speed_reward = "progres"
    with pytest.raises(ValueError, match="speed_reward"):
        GrowBotHeadingEnv(cfg)
    cfg.task.speed_reward = "progress"
    cfg.task.speed_back_min = -1.0  # a back-away command it would be charged for obeying
    with pytest.raises(ValueError, match="speed_back_min"):
        GrowBotHeadingEnv(cfg)


def test_a_progress_scored_stop_lets_it_drift_back_a_little_for_free() -> None:
    cfg = Config()
    cfg.task.speed = "command"
    cfg.task.speed_reward = "progress"
    cfg.task.speed_back_min = 0.0
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=4, options={"nominal": True, "target_error_deg": 0.0, "speed_cmd": 0.0})
    free = cfg.reward.stop_drift_free * cfg.reward.speed_cap_m_s
    assert _coast(env, -0.5 * free, 0.0)["terms"]["progress"] == pytest.approx(0.0, abs=1e-4)
    env.reset(seed=4, options={"nominal": True, "target_error_deg": 0.0, "speed_cmd": 0.0})
    assert _coast(env, -3 * free, 0.0)["terms"]["progress"] < 0.0


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
    # Upstream's body: it can barely rise, so 0.9 of full swing lands near the 6 degrees
    # asked for. (The printed build stands far taller and would overshoot it.)
    cfg = Config()
    cfg.task.body, cfg.task.surfaces = "rollout", ()
    cfg.task.start_pitch_deg, cfg.task.servo_torque_nm = 0.0, 0.0
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
    # A retarget that fires inside the step, between the two yaw readings, pays nothing.
    env.data.qvel[5] = 0.0
    cfg.task.retarget_max_deg = 180.0
    for _ in range(20):
        old = env.target_deg
        env._next_retarget = 0.0
        _, _, _, _, info = env.step(np.zeros(2))
        if abs(wrap_deg(env.target_deg - old)) > 30.0:
            break
    else:
        pytest.fail("no big retarget drawn")
    assert abs(info["terms"]["turn"]) < 0.5
    # Spinning through the back of the target (error crossing +-180) pays nothing big.
    env.target_deg = wrap_deg(env.yaw_deg() + 179.0)
    env._next_retarget = 1e9
    env.data.qvel[5] = -2.0
    _, _, _, _, info = env.step(np.zeros(2))
    assert abs(info["terms"]["turn"]) < 3.0


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


def test_servo_torque_and_foot_width_draws_land_and_restore() -> None:
    env = GrowBotHeadingEnv(Config())
    env.reset(seed=0, options={"nominal": True})
    torque0 = env.model.actuator_forcerange.copy()
    width0 = {g: env.model.geom_size[g, 0] for leg in env._leg_all for g in leg}
    pos0 = {g: env.model.geom_pos[g, 0] for leg in env._leg_all for g in leg}
    env.reset(seed=0, options={"overrides": {"torque_scale": 0.5, "foot_width": 0.6}})
    np.testing.assert_allclose(env.model.actuator_forcerange, torque0 * 0.5)
    for g, w in width0.items():
        assert env.model.geom_size[g, 0] == pytest.approx(w * 0.6)
        # The tail-side and nose-side halves stay side by side, inside the narrower foot.
        assert env.model.geom_pos[g, 0] == pytest.approx(pos0[g] * 0.6)
    env.reset(seed=0, options={"nominal": True})
    np.testing.assert_allclose(env.model.actuator_forcerange, torque0)


def test_a_mis_trimmed_leg_holds_off_its_command() -> None:
    env = GrowBotHeadingEnv(Config())
    env.reset(seed=0, options={"overrides": {"trim": np.array([0.08, -0.05])}})
    env.step(np.zeros(2))
    np.testing.assert_allclose(env.data.ctrl, env.motor + [0.08, -0.05])


def test_late_sensors_show_the_reading_from_ticks_ago() -> None:
    cfg = Config()
    cfg.randomization.enabled = False
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"overrides": {"sensor_delay_s": 2 * CONTROL_DT}})
    raw = []
    for _ in range(6):
        env.step(np.array([0.9, -0.9]))  # swing the legs so the gyro moves
        raw.append(env._imu_raw()[1].copy())
        seen = env.hist[0, CHANNELS.index("gyro_x") : CHANNELS.index("gyro_x") + 3]
    np.testing.assert_allclose(seen, raw[-3], atol=1e-9)
    assert not np.allclose(raw[-3], raw[-1])


def test_a_run_saved_before_the_new_draws_keeps_its_old_randomisation(tmp_path) -> None:
    from growbot_train import config as config_mod

    data = Config().to_dict()
    for k in ("servo_torque_scale", "trim_err_rad", "sensor_delay_s", "foot_width_scale"):
        del data["randomization"][k]
    (tmp_path / "config.json").write_text(json.dumps(data))
    old = config_mod.load(tmp_path / "config.json").randomization
    assert old.servo_torque_scale == (1.0, 1.0) and old.trim_err_rad == 0.0
    assert old.sensor_delay_s == (0.0, 0.0) and old.foot_width_scale == (1.0, 1.0)


def test_a_run_saved_before_the_new_draws_replays_the_same_episodes() -> None:
    from growbot_train import config as config_mod

    data = Config().to_dict()
    for k in ("servo_torque_scale", "trim_err_rad", "sensor_delay_s", "foot_width_scale"):
        del data["randomization"][k]
    data["randomization"].update(
        friction=[0.6, 1.4], relay_delay_s=[0.02, 0.09], floor_friction_scale=[0.7, 1.3]
    )
    env = GrowBotHeadingEnv(config_mod.from_dict(data, saved=True))
    env.reset(seed=1000)
    # What the code before these draws gave this seed (b5a17c6).
    assert env.ep.delay_s == pytest.approx(0.08887969073245204, abs=1e-12)
    assert env.ep.imu_rot[0, 1] == pytest.approx(0.08692864920849568, abs=1e-12)


def test_a_mis_trimmed_robot_settles_and_levels_on_its_trimmed_legs() -> None:
    env = GrowBotHeadingEnv(Config())
    trim = np.array([0.08, -0.08])
    env.reset(seed=0, options={"nominal": True})
    straight = env.data.qpos[7:9].copy()
    env.reset(seed=0, options={"nominal": True, "overrides": {"trim": trim}})
    # The legs bear the body, so they don't reach the trim, but they lean its way.
    assert np.all(np.sign(env.data.qpos[7:9] - straight) == np.sign(trim))
    gyro0 = env._imu_raw()[1]
    env.step(np.zeros(2))
    # No kick on the first step: it was already standing on the trimmed legs.
    assert np.abs(env._imu_raw()[1]).max() < max(0.05, 2 * np.abs(gyro0).max())


def test_the_lens_heading_follows_a_roll_the_body_heading_ignores() -> None:
    cfg = Config()
    cfg.task.facing = "camera"
    body = GrowBotHeadingEnv(cfg)
    cfg_lens = Config()
    cfg_lens.task.facing = "camera"
    cfg_lens.task.heading_from = "lens"
    lens = GrowBotHeadingEnv(cfg_lens)
    for env in (body, lens):
        env.reset(seed=0, options={"nominal": True})
    # At rest the lens looks out over the tail end.
    assert abs(wrap_deg(lens.yaw_deg() - body.yaw_deg())) < 2.0
    turned = {}
    for name, env in (("body", body), ("lens", lens)):
        before = env.yaw_deg()
        q = env.data.qpos[3:7].copy()
        roll = np.array([math.cos(0.15), math.sin(0.15), 0.0, 0.0])  # 17 degrees, body x
        mujoco.mju_mulQuat(env.data.qpos[3:7], q, roll)
        mujoco.mj_forward(env.model, env.data)
        turned[name] = abs(wrap_deg(env.yaw_deg() - before))
    assert turned["body"] < 1.0, turned
    assert turned["lens"] > 10.0, turned


def test_heading_from_must_be_body_or_lens() -> None:
    cfg = Config()
    cfg.task.heading_from = "nose"
    with pytest.raises(ValueError):
        GrowBotHeadingEnv(cfg)


def _lens_env() -> GrowBotHeadingEnv:
    cfg = Config()
    cfg.task.facing = "camera"
    cfg.task.heading_from = "lens"
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    return env


def _xm(lx: float, ly: float) -> np.ndarray:
    # Only the third column (the torso's up axis, the lens) is read.
    xm = np.zeros(9)
    xm[2], xm[5] = lx, ly
    return xm


def test_the_lens_heading_blends_into_the_tail_near_straight_up() -> None:
    from growbot_train.env import LENS_BLEND

    tail = 0.0
    # A lens 30 degrees off the tail: far from vertical it reads the lens alone...
    far = GrowBotHeadingEnv._lens_deg(_xm(0.8 * math.cos(0.52), 0.8 * math.sin(0.52)), tail)
    assert abs(far - math.degrees(0.52)) < 1e-6
    # ...and across the blend edge it changes smoothly, with no snap.
    eps = 1e-4
    at = [
        GrowBotHeadingEnv._lens_deg(_xm(r * math.cos(0.52), r * math.sin(0.52)), tail)
        for r in (LENS_BLEND + eps, LENS_BLEND - eps)
    ]
    assert abs(at[0] - at[1]) < 0.1
    # Pointing almost straight up, the compass direction is the tail's.
    up = GrowBotHeadingEnv._lens_deg(_xm(1e-4 * math.cos(1.5), 1e-4 * math.sin(1.5)), tail)
    assert abs(up - tail) < 0.1


def test_a_lens_tipped_past_vertical_reads_the_tail() -> None:
    # Pointing back over the nose, the lens would read half a turn off; the tail stands in.
    assert GrowBotHeadingEnv._lens_deg(_xm(-0.6, 0.1), 0.0) == 0.0


def test_a_roll_toward_the_target_earns_no_heading_reward() -> None:
    env = _lens_env()
    body = env.body_yaw_deg()
    # Target 20 degrees off; roll both ways, the body staying put.
    env.target_deg = wrap_deg(body + 20.0)
    still = env.scored_err_deg()
    q = env.data.qpos[3:7].copy()
    lens_errs = []
    for sign in (1.0, -1.0):
        env.data.qpos[3:7] = q
        roll = np.array([math.cos(0.15), sign * math.sin(0.15), 0.0, 0.0])
        mujoco.mju_mulQuat(env.data.qpos[3:7], q, roll)
        mujoco.mj_forward(env.model, env.data)
        assert abs(wrap_deg(env.body_yaw_deg() - body)) < 1.0
        lens_errs.append(abs(wrap_deg(env.target_deg - env.yaw_deg())))
        # Whichever way the lens swings, the scored error never drops below the body's.
        assert env.scored_err_deg() >= still - 1.0
    # One of the rolls really did swing the lens onto the target, so the max mattered.
    assert min(lens_errs) < still - 10.0, (lens_errs, still)


def test_the_lens_heading_never_snaps_as_it_swings_round() -> None:
    # Sweep the lens from over the tail, through sideways, to over the nose, at a flat
    # length well past the fade: the heading must move in small steps all the way.
    prev = None
    for i in range(721):
        a = math.radians(i * 0.25)
        h = GrowBotHeadingEnv._lens_deg(_xm(0.6 * math.cos(a), 0.6 * math.sin(a)), 0.0)
        if prev is not None:
            assert abs(wrap_deg(h - prev)) < 2.0, (i * 0.25, prev, h)
        prev = h


def test_lens_heading_needs_camera_facing() -> None:
    cfg = Config()
    cfg.task.heading_from = "lens"
    with pytest.raises(ValueError):
        GrowBotHeadingEnv(cfg)


def test_the_shipped_look_config_scores_the_lens() -> None:
    from pathlib import Path

    from growbot_train import config as config_mod

    cfg = config_mod.load(Path(__file__).parent.parent / "configs" / "look.toml")
    assert cfg.task.heading_from == "lens"
    env = GrowBotHeadingEnv(cfg)
    assert env._lens


def test_the_asked_camera_aim_faces_the_target_and_follows_the_height_command() -> None:
    from pathlib import Path

    from growbot_train import config as config_mod

    cfg = config_mod.load(Path(__file__).parent.parent / "configs" / "look.toml")
    env = GrowBotHeadingEnv(cfg)
    ups = []
    for tilt in (-1.0, 1.0):
        env.reset(seed=0, options={"nominal": True, "tilt_cmd": tilt, "target_error_deg": 40.0})
        want = env.want_lens_dir()
        assert abs(np.linalg.norm(want) - 1.0) < 1e-9
        az = math.degrees(math.atan2(want[1], want[0]))
        assert abs(wrap_deg(az - env.target_deg)) < 1e-6
        ups.append(want[2])
        assert abs(np.linalg.norm(env.lens_dir()) - 1.0) < 1e-6
    # Low (-1) leans back less, so the camera looks further up than standing tall.
    assert ups[0] > ups[1]


def test_a_nose_first_lens_leans_away_from_the_target() -> None:
    env = GrowBotHeadingEnv(Config())
    env.reset(seed=0, options={"nominal": True, "target_error_deg": 0.0})
    want = env.want_lens_dir()
    assert want[2] < 0.99  # the default body rests nose-up, so the lens is tipped
    az = math.degrees(math.atan2(want[1], want[0]))
    assert abs(wrap_deg(az - env.target_deg - 180.0)) < 1e-6


def test_the_curriculum_narrows_every_draw_toward_the_built_robot() -> None:
    env = GrowBotHeadingEnv(Config())
    env.dr_scale = 0.1
    r = env.cfg.randomization
    for seed in range(8):
        env.reset(seed=seed)
        assert abs(env._body_params["mass_scale"] - 1.0) <= 0.1 * 0.25 + 1e-9
        assert abs(env.ep.delay_s - 0.04) <= 0.1 * 0.12 + 1e-9
        assert np.all(np.abs(env.ep.trim) <= 0.1 * r.trim_err_rad + 1e-9)
        # The first target starts close to straight ahead, too.
        assert abs(wrap_deg(env.target_deg - env.yaw_deg())) <= 0.1 * 180.0 + 1e-6


def test_the_curriculum_ramps_to_the_full_ranges() -> None:
    from growbot_train.train import ramp_scale

    cfg = Config()
    assert ramp_scale(cfg, 0) == 1.0  # off by default
    cfg.randomization.ramp_steps = 1000
    cfg.randomization.ramp_from = 0.2
    assert ramp_scale(cfg, 0) == pytest.approx(0.2)
    assert ramp_scale(cfg, 500) == pytest.approx(0.6)
    assert ramp_scale(cfg, 5000) == 1.0


def test_ramp_to_holds_the_bodies_narrow_while_the_targets_widen() -> None:
    cfg = Config()
    cfg.randomization.ramp_to = 0.35
    env = GrowBotHeadingEnv(cfg)
    env.dr_scale = 1.0  # end of the ramp, and evaluation
    assert env._body_k() == pytest.approx(0.35)
    assert env._near((0.5, 1.5), 1.0) == pytest.approx((0.825, 1.175))
    assert env._target_k() == 1.0
    env.dr_scale = 0.2  # below the cap the ramp is unchanged
    assert env._body_k() == pytest.approx(0.2)
    for bad in (0.0, 0.1):  # out of range, and under ramp_from
        cfg.randomization.ramp_to = bad
        with pytest.raises(ValueError, match="ramp_to"):
            GrowBotHeadingEnv(cfg)


def test_the_sweep_tests_the_edges_training_saw() -> None:
    from growbot_train.evaluate import sweep_cases

    cfg = Config()
    full = sweep_cases(cfg)
    cfg.randomization.ramp_to = 0.5
    half = sweep_cases(cfg)
    lo = cfg.randomization.mass_scale[0]
    assert full["light"]["mass_scale"] == pytest.approx(lo)
    assert half["light"]["mass_scale"] == pytest.approx(1.0 + (lo - 1.0) * 0.5)


def test_the_speed_curriculum_starts_every_command_at_full_speed() -> None:
    from growbot_train.train import cmd_scale

    cfg = Config()
    cfg.task.speed = "command"
    cfg.task.speed_stop_prob = 0.3
    cfg.task.speed_back_min = 0.0
    assert cmd_scale(cfg, 10**9) == 1.0  # off by default
    cfg.task.speed_ramp_steps = 1000
    assert cmd_scale(cfg, 0) == 0.0
    assert cmd_scale(cfg, 500) == pytest.approx(0.5)
    assert cmd_scale(cfg, 5000) == 1.0
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0)
    env.cmd_scale = 0.0
    assert {env._draw_speed() for _ in range(50)} == {1.0}
    env.cmd_scale = 0.5
    half = [env._draw_speed() for _ in range(400)]
    assert all(v == 0.0 or 0.5 <= v <= 1.0 for v in half)
    assert 0.0 in half  # stops come back, at half their rate
    env.cmd_scale = 1.0
    full = [env._draw_speed() for _ in range(400)]
    assert min(v for v in full if v > 0) < 0.2


def test_wandering_while_stopped_costs_past_the_free_radius() -> None:
    cfg = Config()
    cfg.task.speed = "still"
    cfg.reward.anchor = 0.5
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["terms"]["anchor"] == 0.0  # still inside the free 2 cm
    env.data.qpos[0] += 0.10
    mujoco.mj_forward(env.model, env.data)
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["drift_m"] == pytest.approx(0.10, abs=0.01)
    assert info["terms"]["anchor"] == pytest.approx(-0.5 * (info["drift_m"] - 0.02) / 0.15)
    # Past 15 cm beyond the free radius it saturates at the weight, so it never outbids
    # facing the target.
    env.data.qpos[0] += 0.20
    mujoco.mj_forward(env.model, env.data)
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["terms"]["anchor"] == pytest.approx(-0.5)
    env.data.qpos[0] -= 0.25
    mujoco.mj_forward(env.model, env.data)
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["terms"]["anchor"] == pytest.approx(-0.5 * (info["drift_m"] - 0.02) / 0.15)


def test_a_new_target_starts_a_new_anchor() -> None:
    cfg = Config()
    cfg.task.speed = "still"
    cfg.reward.anchor = 0.5
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    env.data.qpos[0] += 0.10
    mujoco.mj_forward(env.model, env.data)
    env._next_retarget = 0.0  # the target moves this tick
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["drift_m"] < 0.01


def test_a_narrow_heading_reward_pays_only_near_the_target() -> None:
    cfg = Config()
    cfg.reward.heading = 1.0
    cfg.reward.heading_width_deg = 40.0
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True, "retarget": False, "target_error_deg": 0.0})
    _, _, _, _, info = env.step(np.zeros(2))
    near = info["terms"]["heading"]
    env.target_deg = wrap_deg(env.yaw_deg() + 60.0)
    _, _, _, _, info = env.step(np.zeros(2))
    assert near > 0.9
    assert info["terms"]["heading"] < 0.15


def test_the_anchor_follows_a_walk_and_holds_from_a_stop() -> None:
    cfg = Config()
    cfg.task.speed = "command"
    cfg.reward.anchor = 0.5
    env = GrowBotHeadingEnv(cfg)
    env.reset(seed=0, options={"nominal": True})
    env.speed_cmd = 1.0
    env.data.qpos[0] += 0.10
    mujoco.mj_forward(env.model, env.data)
    _, _, _, _, info = env.step(np.zeros(2))
    assert "anchor" not in info["terms"] and info["drift_m"] == 0.0  # walking: no anchor
    env.speed_cmd = 0.0
    _, _, _, _, info = env.step(np.zeros(2))
    assert info["drift_m"] < 0.01  # the stop anchors where the walk left it


def test_the_stop_drift_band_narrows_with_the_curriculum() -> None:
    cfg = Config()
    cfg.reward.stop_drift_free = 0.5
    cfg.randomization.ramp_from = 0.2
    env = GrowBotHeadingEnv(cfg)
    env.dr_scale = 0.2
    assert env._stop_drift_free() == 0.5  # no final set: fixed
    cfg.reward.stop_drift_free_final = 0.25
    assert env._stop_drift_free() == pytest.approx(0.5)
    env.dr_scale = 0.6
    assert env._stop_drift_free() == pytest.approx(0.375)
    env.dr_scale = 1.0  # full ramp, and evaluation
    assert env._stop_drift_free() == pytest.approx(0.25)
