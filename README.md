# GrowBot Training

Teaches the two-leg GrowBot a **slow walk that follows a heading**. The policy gets one
steering number, the **heading error**: how many degrees the target direction is off the
robot's nose, from -180 to 180. It turns toward the target and walks that way at a
capped, gentle speed. The policy doesn't care where the number comes from. On the robot,
the app can work it out from the phone's gyro (plus the compass when it's steady) to hold
a fixed direction, or from where a face or object sits in the camera frame to walk toward
it.

Everything here runs on a normal Linux machine with plain MuJoCo and Stable-Baselines3
PPO on the CPU, one simulator per core. A finished run comes out as one JSON file in
the format the JBrain2 app's policy runner loads.

There are two walks, each a config file:

| Config | Walks | Stance | Starts |
| --- | --- | --- | --- |
| [`heading_walk.toml`](configs/heading_walk.toml) | nose first | its resting stance (about 34 degrees nose-up) | standing |
| [`camera_walk.toml`](configs/camera_walk.toml) | **tail first, toward what the phone's camera sees** | a few degrees taller, to lift the camera's view | standing (folded starts and shoves come back once it can scoot) |

See [The camera walk](#the-camera-walk) for why the second one exists and what it can and
can't do with this body.

> Status: sim-only. The pipeline is tested end to end, but no policy from it has been
> tried on a real robot yet.

## Quick start

```bash
./install.sh            # once: makes .venv with everything (CPU-only PyTorch)
./run.sh                # train, export, evaluate, replay, pack: runs/<date-time>/ and .tar.gz
```

`./run.sh` takes a run name and any trainer options after it:

```bash
./run.sh first-try --steps 5000000                     # a shorter run
./run.sh fast-cap  --set reward.speed_cap_m_s=0.04     # change one setting for this run
```

Each run folder ends up with `policy_heading_walk.json` (the file the phone runs),
`replay.html` (watch it in 3D, below), `eval.txt` (its scores), `config.json` (the
settings that made it), checkpoints and TensorBoard logs.

## Watching it walk

There are two ways to see how a policy actually behaves on a simulated robot. Both run
the exported JSON, the same file the phone would run.

**In any browser, phone included: `replay.html`.** Every run writes one. It's a single
file: open it and it plays six walks in 3D (target ahead, to the side, behind, a target
that keeps turning, and two on randomly varied robots), with the target direction as a
red arrow, the path drawn behind the robot, and the live heading error and speed. Drag to
orbit, pinch or scroll to zoom, scrub the timeline, change playback speed. It needs no
install or graphics drivers, because the walks are simulated when the page is made; the
page only plays them back. To make one for any policy:

```bash
.venv/bin/python -m growbot_train.replay runs/first-try/policy_heading_walk.json
```

**Live and steerable, on a machine with a screen: the MuJoCo viewer.**

```bash
.venv/bin/python -m growbot_train.view runs/first-try/policy_heading_walk.json
```

**Left/Right arrows** turn the target 30 degrees, **Up** points it at the robot's nose,
**Backspace** restarts on a new random robot, and **N** switches to the as-designed
robot. Here you can poke at it: swing the target behind it mid-stride and see how it
reacts. `--video walk.mp4` records an mp4 instead of opening a window (on a machine
without a screen, put `MUJOCO_GL=egl` in front).

## Scoring it

```bash
.venv/bin/python -m growbot_train.evaluate runs/first-try/policy_heading_walk.json --sweep
```

prints, for each starting heading error, the average speed toward the target, overall
speed, how often it broke the speed cap, the heading error over the second half, how
long it took to turn onto the target, and falls. `--sweep` repeats it at the edges of the
robot variations (lightest, heaviest, slipperiest and so on).

Training progress: `.venv/bin/tensorboard --logdir runs` and open the page it prints.
The `walk/` graphs are the useful ones: `v_along` (speed toward the target, should climb
to near the cap), `abs_err_deg` (should fall), `speed` (should stay under the cap) and
`falls_per_1k_steps` (should stay near zero), plus one graph per reward term.

### How long it takes

The default run is 30 million steps, about 330 hours of simulated walking. In the
sandbox this was built in, 4 cores did about 1,500 steps a second. A 16-core Strix
Halo running 32 simulators should manage several times that, so expect roughly one to
two hours. A run saves a checkpoint every 2 million steps, so you can stop early and
export any checkpoint:

```bash
.venv/bin/python -m growbot_train.export runs/first-try --model checkpoints/model_10000000.zip
```

A stopped run can be continued, with the settings it started with (its `config.json`)
and `--steps` more steps:
`python -m growbot_train.train --out runs/first-try --resume runs/first-try/checkpoints/model_10000000.zip --steps 20000000`.

## What the robot sees and does

Every 40 ms (25 times a second) the policy gets the last 5 ticks of:

| Input | What it is |
| --- | --- |
| `accel_x/y/z` | accelerometer, gravity included, m/s², levelled at the starting stance so gravity reads straight up (+z). x toward the nose, y to the robot's left |
| `gyro_x/y/z` | turn rate, rad/s, same frame |
| `prev_action_0/1` | the leg commands it sent last tick |
| `cmd_yaw` | the heading error divided by 180, so -1 to 1. **Positive means the target is to the left** (counter-clockwise seen from above) |
| `cmd_forward` | camera walk only: the speed command, -1 to 1 (see "The camera walk") |
| `cmd_pitch` | camera walk only: the camera-height command, -1 to 1 (see "The camera walk") |
| `cycle:gait_sin/cos` | a steady 1 Hz clock that starts at zero when the walk starts, to help it keep a rhythm |

It answers with two numbers between -1 and 1, the right leg and the left leg, which
become servo angles of up to 1 radian (about 57 degrees) either side of straight down.

These are all things the phone on the robot can measure. There are no camera images and
no orientation angles. For the first walk the heading error is the only steering
input; the camera walk adds a speed and a camera height.

## The reward

Reinforcement learning works by trial and error: the policy tries things in the
simulator and gets a score (the reward) every tick. Over millions of ticks it learns to
do more of whatever scored well. So **the reward is the description of the behaviour you
want**, and changing it is how you change what the robot learns.

Every tick the reward is the sum of these terms. All the weights live in
[`configs/heading_walk.toml`](configs/heading_walk.toml) under `[reward]`.

| Term | Weight | What it pays for or punishes |
| --- | --- | --- |
| progress | `progress = 2.0` | Moving in the target direction. Full value at the speed cap, a fraction below it, nothing extra above it, negative when moving away. |
| heading | `heading = 0.5` | Pointing the nose at the target. Full value dead on, half at 90 degrees off, zero facing away. |
| overspeed | `overspeed = 4.0` | Going faster than `speed_cap_m_s` (3 cm/s). Grows with the square of the excess, so a little over is mild and a lot over is harsh. |
| tilt | `tilt = 1.0` | Leaning away from how it stood at the start. (This body naturally rests nose-up at about 34 degrees, so "upright" means its own starting stance, the same reference the app's tip check uses.) |
| fall | `fall = 10.0` | A one-off penalty when it tips more than `fall_tilt_deg` (55 degrees) from its stance. The episode ends there, so it also loses all the reward it would have earned. |
| action_rate | `action_rate = 0.05` | Jerky leg commands: the change since last tick, squared. |
| effort | `effort = 0.01` | Big leg angles, a little. |
| back_lean | `back_lean = 0.0` (camera walk: 2.0) | Leaning back more than `back_lean_margin_deg` (12 degrees) past the stance it should hold: weight times (degrees over, divided by 10) squared. |
| wobble | `wobble = 0.0` (look: 0.05) | Rocking: weight times (roll rate squared plus pitch rate squared), rad/s. A steadier camera. |
| turn | `turn = 0.0` (camera walk: 0.5, look: 1.0) | Each degree the heading error shrank this tick, against the current target (negative when it grew). It sums to weight times (starting error minus final error), so it can't be farmed, but it gives a clear tick-by-tick signal: turning on the spot moves the heading term too little for training to notice, and the first look run never learned to turn. |
| stance | `stance = 0.0` (camera walk: 0.5) | Standing `stand_lift_deg` taller than its resting stance: full value on target, fading out a `stance_tol_deg` (6 degrees) or so either side. Off for the first walk. When it's on, `tilt` measures from the taller stance. |

With a speed command (`task.speed = "command"`, the camera walk) or none at all
(`"still"`, the look policy), `progress` pays instead for matching the wanted velocity,
direction included: the wanted speed minus how far off it is, over the cap. Standing
still scores exactly zero whatever the command, just as it does for the first walk:
walking at the commanded +1 or -1 earns the full value (a slower command earns its share
of it). During a stop, drifting up to `stop_drift_free` (half the cap) is free, so
turning on the spot to face the target costs nothing; drifting further costs. The first
version paid a third of the weight for standing still when told to walk, and charged for
every bit of drift during a stop, which outbid the heading reward: the first camera walk
and look runs both froze in place rather than turn.

Speeds are averaged over `speed_window_s` (1 second) before the cap is checked, because a
paddling gait lurches within each step.

Each target lasts 6 to 14 seconds and then moves by up to 120 degrees, so the policy
practises changing course mid-walk, not only setting off.

### Changing it

Edit the number in the TOML file, or try a change for one run without editing anything:

```bash
./run.sh slower --set reward.speed_cap_m_s=0.02
```

What to try when the result isn't right:

| You see | Try |
| --- | --- |
| Walks too fast | lower `speed_cap_m_s`, or raise `overspeed` |
| Hardly moves | raise `progress`; check `walk/term_overspeed` isn't scaring it off |
| Walks but wanders off course | raise `heading` |
| Won't turn around when the target is behind | raise `heading`; give it longer episodes (`task.episode_seconds`) |
| Twitchy, buzzing legs | raise `action_rate` |
| Falls over | raise `tilt` or `fall` |

Change one thing at a time and compare runs in TensorBoard and in their replays. A weight
that's much larger than the others drowns them out: if `overspeed` is 50 the safest move
is to stand still, and it will learn exactly that.

## The camera walk

The phone lies on top of the body with its rear camera facing up, at the high (nose) end. The body rests nose-up,
so that camera actually looks up and back over the **tail**. To walk toward a face the
camera sees, the robot has to walk tail first. That's the second config,
`configs/camera_walk.toml`, and it changes these things (the lines marked CHANGED):

- `task.facing = "camera"`: the heading error is measured from the tail end. 0 means the
  thing the camera is looking at is dead ahead. The app puts the face's left-right angle
  in the frame there.
- `reward.stance = 0.5` with `task.stand_lift_deg = 6`: a bonus for holding the body 6
  degrees more nose-up than it rests, which tilts the camera's view down from the sky
  toward the horizon behind it, where faces are.
- **It scoots, it doesn't walk.** The paddle legs only push well one way. Across 468
  fixed open-loop gaits in the sim, nose first reaches 3 cm/s easily, but the best
  ordinary tail-first gait managed 0.6 cm/s and most about 0.1. A gait search that
  allowed any rhythm and rocking found a fast rock-and-shuffle, about two beats a
  second, that scoots tail first at about 0.9 cm/s on the standard body. So the camera
  walk's speed cap is 1 cm/s (`reward.speed_cap_m_s = 0.01`) and the policy's clock
  runs at 2 Hz (`task.gait_hz = 2`). The first two camera walk runs, at 3 cm/s and 1 Hz,
  never moved.
- Folded starts and shoves (`task.lying_start_prob`, `randomization.push_every_s`) are
  off until a run shows it can scoot; they made the first runs harder still.
- **Not going over backwards.** Standing taller moves it toward the one fall this body
  can't undo, so three things push against it. `reward.back_lean = 2` charges for leaning
  back more than 12 degrees past the stance it should hold, growing fast. Past 75 degrees
  nose-up counts as a fall (`task.back_fall_pitch_deg`), ending the episode with the fall
  penalty. Shoves that spin the body nose-up or nose-down
  (`randomization.push_every_s`, `push_pitch_rad_s`) can teach it to catch itself; they
  are off for now (above).

**How tall can it stand?** Not much, with this body. In the sim, the nominal build rests
at about 35 degrees nose-up. With both legs pushed all the way forward it reaches about 41
degrees, and past that it rolls onto its back. Across the randomised bodies the most it
gets is about 8 degrees above rest on average. So the aim is 6 degrees above its own
resting stance, not "upright". (About 1 in 5 randomised bodies carry their weight far
enough forward to rest nearly flat; for those, "taller" and "folded" are measured from
that flatter rest, so the same goal still makes sense.) If
you want the camera much closer to level, the bigger lever is the mount: a wedge under
the phone. The policy needs no retraining for a wedge as far as the sensors go, because
the IMU is levelled at the stance anyway (the extra weight up high is a separate
question).

**Why not a separate stand-up policy?** This robot never really falls over: switched off
it sits on its tail and feet, and folded it's a leg push away from standing. Getting
up is part of walking here, so one policy learns both. Episodes that start folded are
meant to be mixed in from the beginning, which is simpler for a skill this small; for
now they're off while it learns to scoot.

**Why not train it to get back up from its back?** In the sim it can't. Past about 95
degrees nose-up (for the standard build; a tail-heavy randomised body tips a little
later) it rolls over onto the phone, and then both legs point at the sky: the
servos only swing them 90 degrees each way, so they never reach the floor. No policy can
fix that, so training aims to never get there. If it happens on the real robot, a
hardware stop would be the answer: a short strut or bumper behind the tail that props it
before it goes past vertical.

**Three inputs, one policy.** The camera walk is meant to be the one policy for following
faces, so it is steered by three numbers the app works out from the face it tracks:

- **Heading** (`cmd_yaw`): where the face is, from the camera end.
- **Speed** (`cmd_forward`, `task.speed = "command"`): +1 scoots toward it at the cap (1
  cm/s), 0 stands still and keeps turning to face it, -1 backs away at the cap. From how
  big the face looks: small means go closer, right-sized means stop, too big means back
  off. Each episode the speed changes every 3 to 8 seconds; 3 in 10 are stops.
- **Camera height** (`cmd_pitch`, `task.tilt_command`): +1 stands 6 degrees taller than
  rest (the camera about 49 degrees above the horizon), -1 lowers 30 degrees, nearly flat
  (the camera looking almost straight up), linear in between. It only goes that way: the
  body can't lean far enough to aim the camera level. Half the time it's "stand tall".

Because there's more to learn, episodes run 30 seconds, the target moves more often
(every 4 to 10 seconds), and the run is 80 million steps. The exported file's
`obs_spec.cmd_forward` and `obs_spec.cmd_pitch` blocks describe the two new inputs; the
app's runner reads them from `cmd.fwd` and `cmd.pitch`.

Run it with `./run.sh camera-try --config configs/camera_walk.toml`, or from JBrain's Jobs
launcher ("GrowBot camera walk").

## The look policy

The camera walk at speed 0 already stands still and turns to face the heading. If that
turns out wobbly, `configs/look.toml` trains a specialist that never walks
(`task.speed = "still"`, so it has no speed input): the target moves every 5 to 12
seconds (turning on the spot manages about 8 degrees a second), facing it pays three
times as much, turning toward it pays double (`reward.turn`), and rocking costs
(`reward.wobble`). It keeps the camera height input. It's the camera walk otherwise, and
only the lines marked CHANGED differ from that file. From JBrain it's the "GrowBot look"
job.

## Training on a different robot every time (randomisation)

No two GrowBots are the same, and the simulator is not the real world. So every episode
gets a slightly different robot, using the ranges upstream's DR/RMA walk was trained
with: weight (80 to 125%), where the battery sits, leg length and left/right mismatch,
servo strength, speed and lag, floor friction, a 20 to 90 ms delay on the relay to the
legs with 8% of commands dropped, a phone mounted up to 0.15 rad crooked, and sensor
noise and bias. The heading error gets noise, a slow drift (like a gyro-integrated
heading) and a random refresh rate (like a camera target). All of it is under
`[randomization]` in the config, and `enabled = false` turns it off for experiments.

## Bodies

`growbot_train/bodies/` holds two MuJoCo models from upstream
[britcruise9/GrowBot](https://github.com/britcruise9/GrowBot) (`policy/Harsh_policies`):
`rollout.xml` (the public rollout build, upstream's walk body, the default) and
`olie.xml` (the enclosure build with the battery under the tail). Pick one with
`task.body`. Both are under upstream's PolyForm Noncommercial licence
([`LICENSE-upstream`](growbot_train/bodies/LICENSE-upstream)); the files are unchanged
([`UPSTREAM.md`](growbot_train/bodies/UPSTREAM.md) has the commit), and the trainer adds
the phone's IMU site and sensors when it loads them.

## Getting a policy onto the robot

`policy_heading_walk.json` is in the JBrain2 app's `growbot-policy` format, the
`segments` dialect that `growbot/site/growbot_runner.js`'s `Walker` loads (like
`policy_drrma.json`). It needs no change to the runner: the heading error rides the
runner's existing `cmd_yaw` input, and the file's `obs_spec.cmd_yaw` block says so.
(The camera walk's camera height needs a runner that knows `cmd_pitch`, which JBrain2's
`main` does.)
The exporter checks every file it writes against the trained network (the result is in
`export_verify.log`), and `tests/test_runner_parity.py` runs it through the app's own
runner (CI does this against JBrain2's `main`):

```bash
GROWBOT_SITE=../JBrain2/growbot/site .venv/bin/python -m pytest tests/test_runner_parity.py
```

What the app still needs before it can drive with one (app-side, not done here):

1. **A heading source.** Each tick, set `cmd.yaw = heading_error_degrees / 180`, positive
   for a target to the left. Today the walk passes a constant (`-1` for left, `+1` for
   right), which is the opposite sign and not an error, so this walk needs its own path.
   For a fixed direction, integrate the gyro's yaw rate from the start of the walk (and
   blend in the compass when it's steady). For a target, use its horizontal position in
   the camera frame and the camera's field of view.
2. **A gait entry** that loads the file (for example under `policies/`), runs the usual
   1.2 s legs-neutral stance first (the simulator starts each episode the same way), and
   ticks at the file's `control_hz` (25).
3. **Levelling and axes.** The policy sees the IMU *levelled at its stance*: after the
   legs-neutral settle, the readings are rotated by the smallest rotation that makes the
   gravity measured right then read straight up (+z). That way the body's own nose-up
   rest (about 34 degrees) and a crooked phone mount both disappear, and x stays toward
   the nose, y to the robot's left. The runner already has this rotation (`rotZto`), but
   today's walk path levels only about the phone's x axis (`imuRot`), so this walk needs
   the full version. The runner's `IMU_SIGN` calibration must be `[1, 1, 1]` for it (the
   default flips y). Check on the bench that tipping the nose up and turning left read
   the same signs as in the simulator before walking it.

## Keeping results

`runs/` is scratch and isn't committed. To keep a result, copy it into `policies/`:

```bash
scripts/keep_policy.sh runs/first-try                          # from a local run
scripts/keep_policy.sh ~/Downloads/growbot_heading_walk_artifacts.tar.gz my-walk  # from JBrain
```

That copies the policy JSON, the config that made it, its evaluation and its replay page,
ready to commit.

## Running it from JBrain

JBrain's Launcher (the Jobs tile, once called Math) has **GrowBot heading walk**, **GrowBot camera walk**
and **GrowBot look** jobs. Each clones this repo on the box, runs `./install.sh` and the tests,
trains with `./run.sh` (the others with `--config configs/camera_walk.toml` or
`configs/look.toml`), and
hands back the run's
`.tar.gz` (policy, replay page, checkpoints, evaluation, TensorBoard logs) as the job's
download. Its results page shows the export check and the evaluation summary.

### Watching while it trains

Every run keeps a `live/` folder up to date, and the Launcher shows it on the job's page
(and a debug token can read it):

| File | What | Updated |
| --- | --- | --- |
| `progress.json` | percent done, time left, speed toward the target, heading error, falls | after every batch of experience (a few seconds) |
| `replay.html` | 3D replay of the network as it is now: three 24-second walks, played in a loop; the open page reloads itself between walks, so it picks up the newest one on its own | every half million steps (`train.live_replay_every_steps`; a few minutes apart on the box) |
| `policy_latest.json` | that network as an app policy | with the replay |

Its `state` goes from "training" to "finishing" while the run evaluates and draws the
final replay. Then `replay.html` becomes the full six-walk replay of the final policy,
`policy_final.json` and `eval.txt` appear, and `progress.json` says "done". If a step
fails, it says "failed" instead of sitting at "training". A resumed run keeps its history.

## Optional: JAX on the Radeon iGPU

Upstream trains with MJX (MuJoCo on JAX) and Brax PPO on a GPU, which can run thousands
of simulators at once. On the Strix Halo's Radeon 8060S (gfx1151) that needs a ROCm build
of JAX, which is experimental for this chip. `scripts/try_jax_rocm.sh` sets up a separate
`.venv-jax` and reports whether JAX can see the GPU:

```bash
scripts/try_jax_rocm.sh
```

If it lists only a CPU device, stay with the CPU path above, which is the supported one.
If it does find the GPU, upstream's notebooks in `policy/Harsh_policies` show the MJX
version of this kind of training; porting this task's reward to it is a follow-up worth
doing only if the CPU runs prove too slow.

## Files

| Path | What |
| --- | --- |
| `configs/heading_walk.toml` | Every setting, with comments |
| `configs/camera_walk.toml`, `configs/look.toml` | The camera walk and the look policy, changes marked |
| `growbot_train/env.py` | The simulator task: inputs, randomisation, reward |
| `growbot_train/train.py` | PPO training, checkpoints, export at the end |
| `growbot_train/export.py` | Trained network to the app's JSON, with a match check |
| `growbot_train/evaluate.py` | Scores an exported JSON, optional robustness sweep |
| `growbot_train/replay.py` | Writes the browser 3D replay page (`replay_template.html`) |
| `growbot_train/view.py` | Live MuJoCo viewer with a steerable target, or an mp4 |
| `growbot_train/policy.py` | Runs an exported JSON with the app runner's arithmetic |
| `growbot_train/live.py` | The run's `live/` folder: progress, latest replay and policy |
| `install.sh`, `run.sh` | Set up; train, export, evaluate, replay and pack in one go |
| `scripts/keep_policy.sh` | Copy a result into `policies/` |
| `scripts/try_jax_rocm.sh` | Optional: does JAX see the GPU? |
| `policies/` | Kept results |
| `tests/` | `pytest`; the parity test needs node and a JBrain2 checkout |
