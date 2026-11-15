# GrowBot Training

Teaches the two-leg GrowBot a **slow walk that follows a heading**. The policy gets one
steering number, the **heading error**: how many degrees the target direction is off the
robot's nose, from -180 to 180. It turns toward the target and walks that way at a
capped, gentle speed. The policy doesn't care where the number comes from. On the robot,
the app can work it out from the phone's gyro (plus the compass when it's steady) to hold
a fixed direction, or from where a face or object sits in the camera frame to walk toward
it.

Where this is heading, phase by phase: [docs/ROADMAP.md](docs/ROADMAP.md).

Everything here runs on a normal Linux machine with plain MuJoCo and Stable-Baselines3
PPO on the CPU, one or more simulators per core. A finished run comes out as one JSON file in
the format the JBrain2 app's policy runner loads.

There are five walks, each a config file:

| Config | Walks | Stance | Starts |
| --- | --- | --- | --- |
| [`heading_walk.toml`](configs/heading_walk.toml) | nose first | its resting stance (about 34 degrees nose-up) | standing |
| [`nose_walk.toml`](configs/nose_walk.toml) | nose first, low (about 12 degrees nose-up), at a commanded speed (from a stop up to 3 cm/s), steering within 60 degrees; on top of a bench waddle that always runs (`task.residual_gait`, `configs/gaits/nose_walk.json`): the speed sets its step rate, a built-in rule steers it (`task.residual_steer_deg`), and the policy adds corrections | about 12 degrees nose-up, lower than at rest | standing |
| [`camera_walk.toml`](configs/camera_walk.toml) | **tail first, toward what the phone's camera sees** | a few degrees taller, to lift the camera's view | standing (folded starts and shoves come back once it can scoot) |
| [`lift_fall.toml`](configs/lift_fall.toml) | tail first by lifting the body and letting it fall, steering within 30 degrees (wider turns are a stop and a spin, the look policy) | tall, bouncing (about 55 to 64 degrees as built) | standing |
| [`lift_fall_speed.toml`](configs/lift_fall_speed.toml) | as lift and fall, at a commanded speed (from a stop up to 4 cm/s, its honest top), steering within 60 degrees, on the lift-and-fall gait as a residual (`task.residual_gait`, the speed setting its step rate). Experimental: it fell 13 in 20 by 5M; the master's forward walk is `lift_fall.toml` within 30 degrees | as lift and fall | standing |

The nose walk is the heading walk with a speed input: the master policy's way of backing
away from the camera. Scored on matching the command, a policy that can't walk yet earns
most by keeping still, so asked for slow walks and stops from the start it never learned
to walk at all (its first run stood still for 3M steps). Two settings fix that.
`task.speed_reward = "progress"` pays like the heading walk, for speed along the target
up to the command, and charges for going faster, so early flailing costs no more than
standing still. `task.speed_ramp_steps` starts every command at full speed with no stops
and widens the spread to the full range over the first 8M steps. Its evaluation adds a
speed-tracking table, the speed it walks at for each commanded share of the cap. The first
run walked 1.7 cm/s but held 34-50 degrees off and rarely came round from behind, so it now
uses a turning reward: lift and fall's `turn` and `arrive`, plus progress and heading
that count only near the target (`progress_width_deg`, `heading_width_deg`). With that it
turned round from 180 degrees with no falls, but walked 1.0-1.3 cm/s whatever speed it was
asked for: the free band (`stop_drift_free`, 1.5 cm/s) set the overspeed line for stops
and every command under half the cap. So the band is 0.6 cm/s and wandering from a stop
costs (`anchor`); charging overspeed harder as well (20) froze the next run. It no longer turns much during
stops; the master hands stops to the look policy. See
[The camera walk](#the-camera-walk) for why the camera walk exists and what it can and
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
file: open it and it plays up to nine walks in 3D (target ahead, to the side, behind, a
target that keeps turning, two on randomly varied robots, the speed stepped full, half,
stop, full for a job with a speed input, and two drawn as training draws them at the end of its
curriculum: a random robot, first target, target moves and speeds, so early in a run
they are harder than what it is training on yet). Each walk keeps to the job's own
range: a job trained within 60 degrees gets 60-degree starts and turns, and "behind"
becomes 60 degrees right. Every command input is drawn as an arrow: the target direction in red (a full 3D arrow, tipped up to the camera aim
asked for, when the job has a camera-height input), and on the floor in green the speed
command (its length is the speed, as a share of the cap; it points away from the target
when the command backs away). Beside each, faint, is what the robot did: where the camera
actually points (camera-first jobs) and the averaged velocity. The path is drawn behind the
robot, and the live heading error, speed and commands read out at the top. Drag to
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

### Following a person

```bash
.venv/bin/python -m growbot_train.follow policies/look-v6/policy_look.json --seconds 120
```

runs the follow-me loop of [docs/follow_me.md](docs/follow_me.md) on one skill: a
person walks to random spots around the robot and sometimes sits, a phone camera model
sees their face (field of view, jitter, dropouts that grow with distance, a drifting
distance bias, 100-200 ms of latency), a tracker smooths it, and the follower turns it
into the heading, speed and camera-height inputs, with the voice script ("follow me",
"come closer", "go further back", "stay", "come") changing the set distance. The policy
only ever gets what the camera saw; the person's true position only scores the run: how
long they were in view, aim and pitch error, distance error, speed while told to stay,
and falls. One skill at a time, the follower held to what that skill can do (look only
turns and tilts, the nose walk only backs away, with its camera looking out of its tail
so its heading is the person's turned half a turn), as the baseline the master policy
has to beat. The set distance is along the floor; the camera's eye-spacing distance is
the line to the face, so the follower converts it with the face's elevation. The field
of view is tested where the face lands in the image, which matters because the lens
looks 30-85 degrees above the horizon. Distance error is scored only while the distance
is being held (not on "stay", not while searching). First baseline (nominal body, 3
seeds of 120 s, no falls): look-v6 keeps the person in view 68% of the time with a mean
aim error of 45 degrees; the nose walk 34%, mostly losing people off the sides of the
image and, with no camera-height input, faces that come too close above it. A person walking at
0.3-0.7 m/s a metre or two away crosses the view faster than either skill turns.
Distractor people and the replay page come next.

**Hold.** Look is a gait with a clock, so even on target its legs keep twitching and it
creeps. So once the camera has been within 6 degrees of the person for a second and the
robot is asked to stand, the follower stops calling the policy and the servos hold the
average of the last second of commands, which is the stance the policy was holding
(Jeff, 2026-11-15). It lets go past 15 degrees, when asked to walk or change camera
height by more than 0.15, or when the person is lost. `--hold-check` measures it, with
a perfect compass so the policy isn't chasing gyro drift: look-v7 settled on 9 of 12
random bodies (one fell turning in, two never got within 6 degrees), and on those it
creeps 0.64 cm/s on target with the policy running and 0.01 with Hold, its heading
moves 1.9 degrees in 20 s instead of 4.3, and a 90 degree turn from the held pose
still ends 5 degrees off, as it does without the hold. In the follow runs it rarely engages,
since the person seldom stands still long enough for look to settle on them.

Training progress: `.venv/bin/tensorboard --logdir runs` and open the page it prints.
The `walk/` graphs are the useful ones: `v_along` (speed toward the target, should climb
to near the cap), `abs_err_deg` (should fall), `speed` (should stay under the cap) and
`falls_per_1k_steps` (should stay near zero), plus one graph per reward term.

### How long it takes

The default run is 30 million steps, about 330 hours of simulated walking. In the
sandbox this was built in, 4 cores did about 1,500 steps a second. A 16-core Strix
Halo running 32 simulators should manage several times that, so expect roughly one to
two hours. On the box, one simulator per worker process kept the cores only 14% busy: every tick
the learner sends each simulator its action over a pipe and waits for the reply, and
that round trip cost more than the physics. `train.envs_per_worker` (look uses 4) has
each worker process step that many simulators per round trip, with `n_steps` cut by
the same factor so a rollout holds the same number of samples (`growbot_train/vec.py`).
The launcher can run up to three jobs side by side, each pinned to its own third of the
cores. A pinned run keeps the whole machine's simulator count, so its rollouts and
results match a run on every core, but starts only one worker process per core it may
use (each stepping several simulators) and sizes the learner's threads to its share.

A run saves a checkpoint every 2 million steps, so you can stop early and
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
| `cycle:gait_sin/cos` | a steady clock (`task.gait_hz`: 1 Hz, camera walk 2 Hz) that starts at zero when the walk starts, to help it keep a rhythm; under a residual gait (nose walk) it runs at the speed command's size times that, a stop holding it |

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
| heading | `heading = 0.5` | Pointing the nose at the target. Full value dead on, half at 90 degrees off, zero facing away. With `heading_width_deg` > 0 (look: 40) it is exp(-(error/width)^2) instead, paying only near the target. |
| progress_width_deg | `progress_width_deg = 0.0` | With > 0, progress (paid or charged) is scaled by exp(-(error/width)^2), so it counts only while pointing near the target. Speed along the target alone pays 77% walking 40 degrees off. Tried on lift and fall and taken back off after 16M steps, where that gait, which turns by arcing, had stopped turning (from a 90-degree start, 58 degrees off, against 26-30 without it); the same run recovered by 50M and is kept as `policies/lift-fall-2`, so whether the gates go back on is open. |
| arrive | `arrive = 0.0` (lift and fall: 30) | A one-off bonus each target that starts at least `arrive_within_deg` (15 degrees) off, once the error has stayed under it for `arrive_hold_s` (1 s): weight times (1 minus the time since the target moved over `arrive_by_s`, 20 s), so 95% for turning at once and nothing after 20 s. Discounting already favours a quick turn a little; this pays for arriving and settling. |
| overspeed | `overspeed = 4.0` | Going faster than `speed_cap_m_s` (3 cm/s; camera walk 1 cm/s with `overspeed = 1.0`). Grows with the square of the excess, so a little over is mild and a lot over is harsh. |
| tilt | `tilt = 1.0` | Leaning away from how it stood at the start. (This body naturally rests nose-up at about 34 degrees, so "upright" means its own starting stance, the same reference the app's tip check uses.) |
| fall | `fall = 10.0` | A one-off penalty when it tips more than `fall_tilt_deg` (55 degrees) from its stance. The episode ends there, so it also loses all the reward it would have earned. |
| action_rate | `action_rate = 0.05` | Jerky leg commands: the change since last tick, squared. |
| effort | `effort = 0.01` | Big leg angles, a little. |
| back_lean | `back_lean = 0.0` (camera walk: 2.0) | Leaning back more than `back_lean_margin_deg` (12 degrees) past the stance it should hold: weight times (degrees over, divided by 10) squared. |
| wobble | `wobble = 0.0` (look: 0.05) | Rocking: weight times (roll rate squared plus pitch rate squared), rad/s. A steadier camera. |
| turn | `turn = 0.0` (camera walk: 0.5, look: 1.0) | Each degree the heading error shrank this tick, against the current target (negative when it grew). It sums to weight times (starting error minus final error), so it can't be farmed, but it gives a clear tick-by-tick signal: turning on the spot moves the heading term too little for training to notice, and the first look run never learned to turn. |
| stance | `stance = 0.0` (camera walk: 0.5) | Standing `stand_lift_deg` taller than its resting stance: full value on target, fading out a `stance_tol_deg` (6 degrees) or so either side. Off for the first walk. When it's on, `tilt` measures from the taller stance. |
| frame | `frame = 0.0` (look: 2.0) | Keeping the person's face in the phone camera's picture (Jeff, 2026-11-15). Each target gets a virtual face out in the person's direction (0.7-2.5 m away, 1.1-1.65 m high; for a nose-first skill the person is behind the nose). It pays full weight while the face is inside the 60 by 75 degree picture and nothing outside it, using the same image geometry as the follow tracker. Standing (speed 0, as look always is) it pays weight times (1 - (degrees across / 30) squared), so a stop has to centre the face; the degrees across are averaged with a time constant of `frame_avg_s` (1 s, a few waddle steps), so a gait's periodic sway is fine while the average points at the face (Jeff, 2026-11-15). Being in the picture is judged every frame, since the tracker has to see the face. `evaluate` reports `frame%` (time in the picture) and `img_x` (the averaged aim, degrees off centre). |

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

The phone lies screen up on top of the body, so its front (screen-side) camera, the one
that finds faces, points out of the top of the body. The body rests nose-up, so that
camera actually looks up and back over the **tail**, which is the robot's front for
following: standing taller tilts it down toward faces. To walk toward a face the
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
  runs at 2 Hz (`task.gait_hz = 2`). Turning on the spot still drifts more than that
  cap, so a stop keeps the old 1.5 cm/s of free drift (`reward.stop_drift_free = 1.5`)
  and overspeed is gentler (`reward.overspeed = 1`). The look policy, which never walks,
  keeps the 3 cm/s cap and 1 Hz clock. The first two camera walk runs, at 3 cm/s and 1 Hz,
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

## Lift and fall

A clip of someone else's GrowBot (2026-11-14) moves camera first by lifting its body high
on its tail with both legs and letting it fall back a little further along. Upstream's
live-app walk (`app/policies/forward-0930`: both legs about 55 degrees forward, swinging 14
either way at 1.8 Hz) only bobs in place on this body, and PPO started from it froze, so
that one was dropped. The bench (`research/locomotion/search.py`, scenario `w1`: stand
tall, stock bare legs) found a gait that does what the clip does: the legs snap across
most of their range about 2.3 times a second, hold the body up around 60 degrees on its
feet alone two-thirds of the time, and each time it drops back onto its tail it lands a
little further tail first: about 5 cm/s on the as-built robot on both floors. Open loop it only works on robots close
to the as-built one; `configs/lift_fall.toml` clones it (`configs/gaits/lift_fall.json`)
and lets PPO add the feedback that keeps it going on the rest, steered by the heading.
Two earlier runs from it froze, and the reason was measurable: played open loop under
the full randomisation, the gait earns less than standing still, so PPO unlearns it in
the first million steps. So this one starts on robots close to the as-built one
(`randomization.ramp_steps`: every body, servo, friction and sensor-offset range, and
how far off the target starts and jumps, begins at 35% of its width and widens over the
first 25M steps, half of a full run). The body ranges stop at 60%
(`randomization.ramp_to`) while the targets keep widening. The legs are pinned to the STL
(Jeff's robot matches it) and only the weight and its balance point vary a little: the
earlier runs randomised leg length too, and their falls (about 1 minute-long walk in 10)
were mostly on bodies a few percent shorter in the leg. With the geometry pinned the old
walk fell 2 times in 100 at 35% of the remaining ranges and 35 in 100 at full. It walks at the cap with no speed input (the gait has one speed), and explores with
less noise (`train.log_std_init = -1.5`) so its timing survives the first updates. The
speed cap is 5 cm/s, the stance bonus and lean-back penalty are off as the body bounces,
and only going over onto the screen (past 88 degrees) or forwards ends an episode.
From JBrain it's the "GrowBot lift and fall" job.

The run with the 35% body cap (run ca607df3) is kept as `policies/lift-fall-1`: on
random bodies it ends about 17 degrees off the heading, walks 3.9 cm/s toward it, and
turns a quarter in about 5 s, with 4 falls in 36 evaluations (half of them on half
turns). Its 16M checkpoint (`policies/lift-fall-1-16m`, 27 degrees, 5 falls) is kept
beside it to show the rest of the run was worth it. A run with the heading gates on
(`reward.heading_width_deg = 40`, `progress_width_deg = 45`; run 527daab9, the second
slot) looked worse at 16M, so main took the gates back off, but at 50M it finished
better than lift-fall-1 on random bodies and is kept as `policies/lift-fall-2`, the
forward teacher now: about 10 degrees off, 4.1 cm/s toward the target, half a turn in
about 14 s (a quarter in about 6 s, a little slower), 3 falls in 36. Its config is in
its folder, since main's no longer has the gates. In the sweep it falls as often as
lift-fall-1 (50 of 153), and short-legged and nose-heavy bodies fall on every big turn,
so falls are its main gap.

## The look policy

The camera walk at speed 0 already stands still and turns to face the heading. If that
turns out wobbly, `configs/look.toml` trains a specialist that never walks
(`task.speed = "still"`, so it has no speed input): the target moves every 15 to 30
seconds by up to half a turn, so it learns to spin round to what is behind it (turning
on the spot manages about 8 degrees a second). Facing it pays three times as much,
turning toward it pays double (`reward.turn`), and rocking costs (`reward.wobble`).
Facing is scored on the camera (`task.heading_from = "lens"`): the heading is where
the phone's lens points, projected flat, and the reward takes the worse of that and the
body's own heading, so swinging the lens over with a roll earns nothing and only turning
the whole body does. Leaning costs four times as much (`reward.tilt = 4`), and the
lowest camera height is 20 degrees under rest rather than 30, so the lens never looks
straight up and loses its heading. An earlier run leaned toward the target instead of
turning, because leaning was cheap. The next never found turning on its own (its heading
error sat at 85 to 90 degrees, no better than chance, for 8 million steps), so look now
starts from a spin: before PPO it clones `configs/gaits/look_spin.json`, an
anti-phase leg swing that spins the nominal body about 16 degrees a second, steered
(`train.warm_gait_steer`): the gait as saved toward a target counter-clockwise, legs
swapped toward one clockwise, and standing still within 10 degrees of it. It steers on
the heading error the policy observes, and with targets all the way round even while the
curriculum keeps the bodies narrow. The clone alone, before any PPO, faces targets
dead behind within 13 seconds on the nominal body (2 degrees off after), and averages
26 degrees of error on fully random bodies with 3 falls in 24. Its legs swing up to a radian, so `task.action_to_rad` is 1.3, and the body
randomisation ramps up over the first 15M steps, as for lift and fall. That run turned, but slid about 1 cm/s
(mostly backward) while it did (the spin pivots near the tail), and its heading error
stalled near 56 degrees. Two changes: wandering from where each turn began now costs
(`reward.anchor`, past the first 2 cm, rising over the next 15 cm to a cap per tick), and facing pays only near the
target (`reward.heading_width_deg = 40`: 60 degrees off earns a tenth, where the cosine
paid three quarters). The policy isn't told how far it has wandered, as the phone can't
measure that. The weights were checked against the cloned spin in the simulator:
charging drift harder (a tighter speed band, or an uncapped anchor) made standing still
score better than turning, which is how earlier runs froze. The run with those (Jeff,
2026-11-14) turned without falling but still crept backward at about 1.7 cm/s, most
of it inside the free speed band, and it held the camera about 37 degrees up whatever
the camera height asked (42 to 68). The clone had held one posture whatever the input,
so the policy learned to ignore it. So the clone now lowers or raises both legs with the
camera height input (`train.warm_gait_tilt_offset`), and holding the asked height pays
twice as much (`reward.stance = 1`). That run (v5) followed the camera height (65
degrees up when asked 68, 38 when asked 42), but a free speed band narrowing to 0.75
cm/s (`reward.stop_drift_free_final`) only made it turn less: it still slid about 1.4
cm/s, 28 cm over a 30 s episode. The slide came from the spin it cloned, which pivots
near the tail and travels about 2 cm/s on its own. So look now clones
`configs/gaits/look_spin_still.json`, from a CMA-ES search on the bench scored on turn
rate minus net travel: on the nominal body it spins about 24 degrees a second
travelling 0.4 cm/s (the first spin: 12 and 2.0), and on held-out random bodies at half
the randomisation 17 and 1.1 (the first: 11 and 1.8). Its legs swing up to 1.25 rad,
so `task.action_to_rad` is 1.45, and the band is back at 1.5 cm/s. A rubber tail pad
cut the first spin's travel by a third in the simulator, but the new spin needs none. It keeps the camera height input. At 17 degrees a second (less on the most varied
bodies) half a turn takes 10 s or more, and the trained policy turns slower than the
spin it clones, so the old 6 to 14 s between targets often moved the target before the
robot got there. The target now moves every 15 to 30 s, the camera height every 8 to 20 s, and
an episode lasts a minute so it holds two to four targets. The gyro heading drift
(`randomization.heading_drift_deg_per_s`) builds for the whole episode, so look's error
at 30-60 s isn't comparable with the 30 s runs before.
v6 (run 36d29750, kept as `policies/look-v6`) evaluated at 16 degrees of heading error
averaged over its starts (7 to 10 from most starts, 27 and 34 from +90 and 180 degrees),
but took 20 to 28 s for half a turn, against about 11 for the spin it cloned, and crept
1 to 2 cm/s. v7 (run 200aa414, kept as `policies/look-v7`) ends 11.5 degrees off on
average from every start, 180 included, takes about 20 s for half a turn and 10 s for a
quarter, creeps 0.7 cm/s, and fell in none of its 36 randomised evaluations and none of
the 153 in the sweep, though short-legged, nose-heavy and mis-trimmed bodies end 35-47
degrees off from a quarter turn. In the follow-me test it keeps the person in view less
than v6 (56% vs 68% on the nominal body), so the follow baseline uses both. v8 (run
e71638c2) added an arrive bonus (`reward.arrive = 30`) and a firmer anchor (1.0), and
came out worse: 15 degrees off, worst from +90 on odd bodies, and no faster. So both
went back to v7's, and the next run goes after turn speed instead. This body drifts more
the faster it spins, and at a free band of 1.5 cm/s (`stop_drift_free = 0.5`) the squared
overspeed charge held v7 to about 9 degrees a second, where the open-loop spin it clones
does 15 on random bodies (24 on the nominal one). The band is now 2.4 cm/s (0.8), and
creep on target is left to the follower's Hold, which stops it outright (see
[Following a person](#following-a-person)).

v9 (run 32fae7c5, kept as `policies/look-v9`) is that run. It is the look teacher now:

- **Aim:** 10.6 degrees off on random bodies, against v7's 11.5.
- **Odd bodies:** the bodies v7 left 35-47 degrees off from a quarter turn
  (short-legged, nose-heavy, mis-trimmed) now end 11-15 off.
- **Quarter turns:** mostly 7-9 s in the sweep (short legs from +90 still takes 18 s,
  v7 25), where v7 ranged 7-25.
- **Half turns:** about 20 s on random bodies, the same as v7; 13-21 s in the sweep
  (v7 18-29), but 17.6 s on the nominal body (v7 15.8). Measured another way (24
  held-out random bodies, 9 s spins from a standstill), the open-loop spin averages
  only 11 degrees a second, and a CMA-ES search for a faster spin that holds up there
  found none. The servos stay stock, so about 20 s is the half turn these legs allow
  and is the goal now; an open-loop spin could reach the earlier 12 s goal only near
  the as-built body.
  The rest of the look goal: within 15 degrees, under 1 cm/s of creep, and no sweep
  case falling more than 1 run in 10.
- **Creep and falls:** it creeps 0.8 cm/s on target (v7 0.7), which Hold stops. 0 of 36
  randomised evaluations fell. One sweep case did: long legs from 180 degrees, 1 run in
  3, over that 1-in-10 limit. That case also ends 64 degrees off.

The next run (v10) turns on the camera-picture reward (`reward.frame = 2.0`, see
[The reward](#the-reward)): look's stop is where the follower needs the face centred, and
v9 lands it about 7 degrees off centre on random bodies (`img_x`; 1-3 on the nominal one).

Unlike the camera walk it trains on the printed build with bare-print surfaces (see
[Bodies](#bodies)), like the heading walk.
It's the camera walk otherwise, and only the lines marked CHANGED differ from that file.
From JBrain it's the "GrowBot look" job.

## The master policy

One file for the app, whatever the follower asks: `growbot_train/master.py` distils the
three skills into one student, as [follow_me.md](docs/follow_me.md#1-one-master-policy)
plans. The student takes the follower's `cmd_yaw` (from where the camera points),
`cmd_forward` (signed: the sign alone picks the gait) and `cmd_pitch`. It also sees one
gait clock per teacher at that teacher's rate (`cycle:look_sin` and so on; the app runner
steps any declared clock). Each tick one teacher labels what the legs should do:

- speed below 0 and within 45 degrees (let go at 55): the nose walk, backing away nose
  first (wider, it steers too weakly to its left, so look turns it first);
- speed above 0 and within 30 degrees (let go at 40): lift and fall (Jeff, 2026-11-15:
  wider than that, stop, spin with look, walk again);
- otherwise: look's spin, which also handles walking toward or backing from someone
  well off to the side or behind, until the heading comes inside the limit.

The teachers' commands are matched in radians, so the student swings the servos' full
1.57 radians. Each teacher keeps its own sensor history every tick, so it takes over a
hand-over with a full history. Training is DAgger: in the first round the teachers
drive, then the student drives more and more while the teachers keep labelling, so it
learns to recover from its own mistakes. Minute-long episodes switch mode every 3-12 s,
move the target and change the camera height. Bodies vary only as far as every teacher
trained on (lift-fall-2 saw 35% of the ranges). Each mode falls by its own teacher's
rule: lift and fall's gait rears up past look's back-fall line by design.

```bash
.venv/bin/python -m growbot_train.master distill --out runs/master    # about 10 minutes on 4 cores
.venv/bin/python -m growbot_train.master evaluate runs/master/policy_master.json
```

`evaluate` drives the same robots once with the teachers (through the same mode rule)
and once with the master, per mode and start. It then counts falls against mode
switches in 20 minute-long runs of random commands, driven each way on the same scripts.
The first master (`policies/master-1`, from look-v7, lift-fall-2 and nose-walk-1, with
the walk handing over at 60 degrees then) is
within 5% of its teachers on speed while walking and backing. Standing to turn, it drifts
less than look does (0.95 against 1.13 cm/s from +180). Its heading error is within 1.3
degrees of theirs everywhere: the worst is backing from -45, 8.0 against 6.7. Its turn
times are within 1.3 s: it is slower backing from -45 (5.5 s against 4.3) and quicker
on a half turn (14.4 s against 15.6). It fell 3 times in 114 mode switches, against a
bar of under 1 in 50. Its teachers, driving the same scripts, fell 3 times in 100. In
the per-mode runs only lift-fall-2's walk falls, master or teacher, so the master gets
better as the teachers do.

Its limits, until the walks it learns from are retrained:

- **Bodies:** it only ever saw bodies as far from the as-built one as lift-fall-2 trained
  on (35% of the ranges), in every mode. Look and the nose walk handle the full ranges,
  but the master is untested there.
- **Forward speed:** lift-fall-2 has one speed, so forward is stop-or-go. Anything above
  0 means full speed, as the follower already treats it.
- **Residual-gait teachers** (the nose walk since it walks on a gait, with the policy
  correcting): the student is labelled with the teacher's whole leg angles (gait,
  steering rule and correction). It still outputs plain angles, so the phone runs it
  with nothing new in the app runner. The app's clocks only run at a fixed rate, while
  such a teacher's clock runs at its speed. So the mode it teaches is stop-or-go at full
  speed, where the two clocks agree. Slower backing waits for a speed-scaled clock in
  the runner. A hand-over into such a mode picks the gait up at full swing,
  wherever its clock is, which the teacher never trained on. The switch test counts the
  falls that causes.

It isn't listed in `policies/index.toml` yet: the Body tab's runner takes it, but
nothing in the app sends a signed speed until the follower does.

## Training on a different robot every time (randomisation)

No two GrowBots are the same, and the simulator is not the real world. So every episode
gets a slightly different robot, using the ranges upstream's DR/RMA walk was trained
with: weight (80 to 125%), where the battery sits, leg length and left/right mismatch,
servo strength, speed and lag, floor friction, a 20 to 160 ms delay on the relay to the
legs with 8% of commands dropped, a phone mounted up to 0.15 rad crooked, and sensor
noise and bias. On top of that, from the ranges upstream's shipped walks
(`app/policies/forward-0930`, `diy-v1`) were trained with: servo torque from half to
1.2x an MG90S's, each leg's zero off by up to 5 degrees (trim), the phone's readings
arriving up to 120 ms late, the foot's contact width down to 57% of the printed 21 mm,
and friction from 0.2 to 2.2 (0.5 to 1.6x a floor's own with per-surface materials). The heading error gets noise, a slow drift (like a gyro-integrated
heading) and a random refresh rate (like a camera target). `ramp_steps` turns on a
curriculum: every body, servo, friction and sensor-offset range starts at `ramp_from` of
its width around the as-built robot and widens to the full range over that many steps.
`ramp_to` stops the body ranges at that share of their width, in training and evaluation
alike (the `--sweep` edges too), while the target ranges still widen to full. A run
resumed from before `ramp_to` existed keeps its saved settings, so it has no cap.
The sensor noise, the heading's refresh rate and the choice of floor stay at full
strength throughout. The ramp counts absolute steps, so a run cut shorter than
`ramp_steps` ends before reaching the full ranges. All of it is under
`[randomization]` in the config, and `enabled = false` turns it off for experiments.

## Bodies

`growbot_train/bodies/` holds three MuJoCo models; pick one with `task.body`.

- **`build.xml`, the default: the robot as printed**, with a phone on top. Generated by
  [`research/locomotion/make_body.py`](research/locomotion/make_body.py) from upstream's
  print files (`hardware/print/`): a 163 x 75 mm tray closed by a backplate, about 30 mm
  deep, its ends cut back at the bottom (141 mm along the underside; about 45 degrees
  and 11 mm high at the tail, 30 degrees and 16 mm at the nose), both MG90S servos in a
  cradle at mid-length (hip about 18 mm above the underside), 74 mm legs, 4 AA cells at
  the tail end, the board (its power switch is the side wall's one opening) at the nose
  end and a Pixel 10 in an OtterBox Defender Pro XT case (254 g) on the backplate, with
  the IMU in the phone. Torso 472 g, centre of mass 8 mm toward the tail. It rests at
  about 43 degrees nose-up, on the tail's 45-degree cut face and its feet. Its weight
  sits high and nearly over its feet: set down level it can balance on its soles, so
  episodes set it down tail first (`task.start_pitch_deg`).
- `rollout.xml` (upstream's public rollout build, its walk body) and `olie.xml` (the
  enclosure build with the battery under the tail), from upstream
  [britcruise9/GrowBot](https://github.com/britcruise9/GrowBot) (`policy/Harsh_policies`),
  under upstream's PolyForm Noncommercial licence
  ([`LICENSE-upstream`](growbot_train/bodies/LICENSE-upstream)) and unchanged
  ([`UPSTREAM.md`](growbot_train/bodies/UPSTREAM.md) has the commit). Both are flat 4.5
  mm plates with no phone. The camera walk still uses `rollout` until the tail-first foot
  work settles.

The trainer adds the IMU site and sensors when it loads a body (the build body places its
own site, in the phone).

**What touches the floor** (`task.surfaces`). MuJoCo takes the larger of two touching
geoms' friction, so the original setup, one friction on every geom, lets the floor set
the grip of every contact. With `task.surfaces` set (the default), the floor is
frictionless and four surfaces carry their own material's friction against it: the tail
edge, the rest of the underside, and the tail-side and nose-side halves of each foot.
Materials and floors are in [`growbot_train/surfaces.py`](growbot_train/surfaces.py):
`"pla"` bare print, `"rubber"` liquid dip or a rubber band, and a few others, each
against a `"hard"` floor (wood, laminate, a table) and `"carpet"` (softer contact).
Each episode draws a floor from `task.floors` and scales each surface's friction and the
floor's by `randomization.surface_friction_scale` and `floor_friction_scale`; the
evaluation sweep runs each floor on its own. `task.servo_torque_nm = 0.25` limits the
servos to an MG90S's real torque (upstream's model allows 1.5).

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
ready to commit, and prints the entry that offers it in JBrain.

### Driving a kept policy from JBrain

[`policies/index.toml`](policies/index.toml) lists the kept walks JBrain's Body tab
offers. Paste the entry `keep_policy.sh` printed, fix its title, and merge to `main`.
Then tap **Refresh** under "From the training repo" in the Body tab's Trained walks sheet
(or a debug token calls `POST /api/debug/growbot/policies/refresh`). New entries are
added to the box's library. An entry whose file changed is updated in place, and the name
and choices you gave it stay. An entry you take out is removed. Only runs worth driving
go in (owner, 2026-11-14: "the ones we pick"). `tests/test_policies_index.py` checks the
file the way the box does, and that every listed policy loads.

## Running it from JBrain

JBrain's Launcher (the Jobs tile, once called Math) lists the jobs in
[`jobs.toml`](jobs.toml): today **GrowBot heading walk**, **GrowBot camera walk** and
**GrowBot look**. Each clones this repo on the box, runs `./install.sh` and the tests,
trains with `./run.sh jlaunch --config <its config>`, and hands back the run's `.tar.gz`
(policy, replay page, checkpoints, evaluation, TensorBoard logs) as the job's download.
Its results page shows the export check and the evaluation summary.

### Adding a job

A new job needs no JBrain update. Add a config under `configs/`, add a `[[job]]` entry
for it to `jobs.toml` (the file's header lists the fields), and merge to `main`. Then tap
the refresh button on the Jobs screen (or a debug token calls
`POST /api/debug/jlaunch/specs/refresh`) and the job appears, ready to start. The box also
re-reads the file when it starts and just before each Start, so a config fix merged a
minute ago is what trains. Each run clones the exact commit it read and records it.

An entry can only pick a config: the box runs the same steps for every job, so this file
can't make it run anything else. `tests/test_jobs_manifest.py` checks the file the way
the box does. Removing an entry takes the job off the list; its runs and results stay.

### Watching while it trains

Every run keeps a `live/` folder up to date, and the Launcher shows it on the job's page
(and a debug token can read it):

| File | What | Updated |
| --- | --- | --- |
| `progress.json` | percent done, time left, speed toward the target, heading error, falls | after every batch of experience (a few seconds) |
| `replay.html` | 3D replay of the network as it is now: six or seven 24-second walks (seven with a speed input; all but the duplicate random-robot ones; a few seconds' pause, logged), played in a loop; the open page reloads itself between walks, so it picks up the newest one on its own | every half million steps (`train.live_replay_every_steps`; a few minutes apart on the box) |
| `policy_latest.json` | that network as an app policy | with the replay |

Its `state` goes from "training" to "finishing" while the run evaluates and draws the
final replay. Then `replay.html` becomes the full replay of the final policy,
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
| `configs/nose_walk.toml`, `configs/camera_walk.toml`, `configs/look.toml`, `configs/lift_fall.toml`, `configs/lift_fall_speed.toml` | The nose walk, the camera walk, the look policy, the lift and fall and its speed-input version, changes marked |
| `configs/gaits/` | Gaits a run clones before PPO (`train.warm_gait`), or walks on with the policy adding corrections (`task.residual_gait`) |
| `growbot_train/bodies/build.xml` | The printed build with a phone (generated by `research/locomotion/make_body.py`) |
| `growbot_train/env.py` | The simulator task: inputs, randomisation, reward |
| `growbot_train/surfaces.py` | Contact materials and floors |
| `growbot_train/gait.py` | Open-loop gaits, and cloning one into a policy before PPO (`train.warm_gait`) |
| `research/locomotion/` | Gait and body-change search on a fast open-loop bench (its README) |
| `hardware/cam_foot_leg/` | Printable left and right legs with the bench's winning cam sole, on upstream's horn mount (its README) |
| `growbot_train/train.py` | PPO training, checkpoints, export at the end |
| `growbot_train/vec.py` | Several simulators per worker process (`train.envs_per_worker`) |
| `growbot_train/export.py` | Trained network to the app's JSON, with a match check |
| `growbot_train/evaluate.py` | Scores an exported JSON, optional robustness sweep |
| `growbot_train/follow.py` | The follow-me harness: person, camera model, tracker, follower, voice script |
| `growbot_train/master.py` | Distils the skills into one master policy and scores it against them |
| `growbot_train/replay.py` | Writes the browser 3D replay page (`replay_template.html`) |
| `growbot_train/view.py` | Live MuJoCo viewer with a steerable target, or an mp4 |
| `growbot_train/policy.py` | Runs an exported JSON with the app runner's arithmetic |
| `growbot_train/live.py` | The run's `live/` folder: progress, latest replay and policy |
| `docs/ROADMAP.md` | The order of work, phase by phase, with what ends each phase |
| `docs/follow_me.md` | The plan: one master policy following a person, with voice commands, and its final simulator test |
| `jobs.toml` | The jobs JBrain's Launcher offers (see [Adding a job](#adding-a-job)) |
| `install.sh`, `run.sh` | Set up; train, export, evaluate, replay and pack in one go |
| `scripts/keep_policy.sh` | Copy a result into `policies/` |
| `scripts/try_jax_rocm.sh` | Optional: does JAX see the GPU? |
| `policies/` | Kept results; `policies/index.toml` lists the ones JBrain's Body tab offers |
| `tests/` | `pytest`; the parity test needs node and a JBrain2 checkout |
