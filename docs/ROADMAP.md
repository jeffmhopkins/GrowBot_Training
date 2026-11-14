# GrowBot roadmap

Status: living plan. Last verified 2026-11-14.

Where the GrowBot policies are going: one master policy on the printed robot that
follows a person by camera and answers voice commands ("follow me", "come", "come
closer", "go further back", "stay"). The design is [follow_me.md](follow_me.md); this
page is the order of work and what has to be true before each phase ends.

Everything trains on the **stock legs** (the printed build with bare-print feet) until
phase 7. Foot geometry is parked.

## How work reaches the box

- **One job at a time.** JBrain's launcher refuses a second start until the first ends.
- **An Ops → Update kills a running job.** Seen on 2026-11-14: the job is gone after
  the update, not paused, so updates wait for a gap between runs. Every thread that owns a run says when
  the box is free.
- **No update needed for a new run.** A job clones this repo's `main` when it starts, and
  the Jobs list refreshes from `jobs.toml`, so a config change goes live on merge.
- **Ownership:** the locomotion thread owns the tall and low gaits (lift and fall, the
  backing waddle) and foot geometry (parked). The look and spin thread owns look, the follow
  test, the distillation and the final test. Runs are queued between them in the order
  below.

## Phase 1: the three skills, on stock legs

Each skill trains on its own on the printed build.

| Skill | Config | Owner | State |
| --- | --- | --- | --- |
| Look: spin in place to a heading, tilt the camera | `configs/look.toml` | look thread | training (steered-spin warm start, #22) |
| Tall walk: camera first, lift and fall, forward | `configs/lift_fall.toml` | locomotion thread | curriculum fix merged (#21), next run queued |
| Backing waddle: low, nose first, away from the camera (Jeff, 2026-11-14) | `configs/heading_walk.toml` | locomotion thread | trained on the build (run 00624683); needs a variable speed |

The tall walk also needs two things it lacks today, as follow-up runs:

- **Variable speed:** its cloned gait has one speed (5 cm/s). It has to learn slower
  speeds down to a stop, or the follower is stop-or-go.
- **Camera tilt while walking:** a camera-height input, so the camera can stay on a face
  while it walks.

Phase 1 also adds eval cases: `evaluate.py` today always commands full speed, standing
tall, so it can't yet score a commanded speed, backing, or a camera height.

**Exit** (falls counted per evaluation episode):

- **Look:** on random bodies, it faces a target anywhere around it, including dead
  behind. The heading error settles under 15 degrees, and falls are under 1 in 50 episodes.
- **Tall walk:** forward at its commanded speed within 20% across the speed range. It
  steers to a heading within 15 degrees, with under 1 fall in 50 episodes.
- **Backing waddle:** backs away at its commanded speed and holds its heading, with
  under 1 fall in 50 episodes.
- **Every skill:** passes `evaluate.py --sweep` with no edge case falling more than
  1 in 10.

**Box order:** look → tall walk → backing waddle with variable speed → tall walk with
variable speed and tilt.

## Phase 2: the follow test, single-skill baseline

The simulated follow-me harness from [follow_me.md](follow_me.md), run first on each
skill alone:

- a wandering person with a billboarded face;
- a camera model with phone-like noise, lag and dropouts;
- the tracker and follower;
- scripted voice commands;
- a 3D replay showing the people, the commands, the policy's inputs as arrows, and the
  tracker's estimate beside the truth.

**Exit:**

- The harness runs end to end on each skill: look (speed 0), the tall walk (forward)
  and the backing waddle (backing).
- It reports distance error, aim error, settle times and falls.
- Those numbers become the baseline phase 4 must beat.

No box run: this runs locally.

## Phase 3: the master policy

One student with the shared inputs (heading, signed speed, camera height), distilled
from the three skills, then fine-tuned with PPO:

- each skill teaches only in its own mode;
- the teachers are matched in radians and on the same heading definition, with each
  one's gait clock handled;
- episodes switch modes partway through, so it learns the hand-overs.

The gaps it has to bridge are listed in [follow_me.md](follow_me.md#1-one-master-policy).

**Exit:**

- In each mode, the master matches its teacher's phase 1 eval within 10%.
- It changes mode on command without falling: forward to stop to backing and back,
  under 1 fall in 50 switches.

**Box order:** distillation, then the PPO fine-tune (one or two runs).

## Phase 4: the final follow-me test

The full simulator acceptance test on the master policy:

- a person who wanders, turns their back and sits;
- one to three distractor people who cross between and stand nearer;
- realistic phone noise, never ground truth;
- the whole voice script, so the set distance changes live and "stay" holds;
- nominal and random bodies, on both floors.

**Exit:**

- It keeps the person in frame through the script, and settles to each new distance.
- It holds still on "stay" and never locks onto a distractor.
- No falls on the nominal body, and under 1 in 20 runs on random bodies. These numbers
  are provisional: they are revisited against the phase 2 baseline.
- A replay to watch.

## Phase 5: the phone app

In the JBrain2 PWA:

- **The master policy** through the app's runner (`cmd_yaw`, `cmd_forward`,
  `cmd_pitch`).
- **The person tracker:** MediaPipe face landmarker with pose as the fallback, giving
  direction, pitch and distance. It has multi-person locking: it chooses whoever said
  "follow me" and keeps identity through crossings.
- **The follower and voice commands.**
- **Field of view and distance calibration:** one "stand here" step per phone and
  person.

**Exit:** with the robot held still on a stand, the app's commands track a person
walking around it the way the simulator's tracker does: right signs, plausible distances,
and the right person kept through a crossing.

## Phase 6: real-robot trials

- **On the bench first:** the IMU signs and levelling match the simulator. The stance
  settles, and each skill runs on its own.
- **Then on the floor:** spin and look, walk tall, back away, and the full follow with
  voice commands, on hard floor and carpet.
- **Compare:** the robot against the simulator's numbers, and feed back what differs
  (friction, servo lag, the camera's real noise) into the randomisation.

**Exit:** it follows Jeff around a room on "follow me", keeps its distance, answers
"come closer", "go further back" and "stay", and doesn't fall over in a 5-minute session.

## Phase 7, later: feet and legs (parked)

Parked on 2026-11-14 to get everything working on stock legs first:

- grippy feet (liquid rubber);
- the cam-foot leg (`hardware/cam_foot_leg/`);
- the foot-geometry search (`research/locomotion/`).

Picked up once phase 6 works. Any change of legs means retraining the phase 1 skills
on the new body, then repeating phases 3 and 4.
