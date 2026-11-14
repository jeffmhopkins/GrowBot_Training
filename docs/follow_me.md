# Follow me: the plan

Status: plan, not built. Last verified 2026-11-14.

The end use for the GrowBot policies (Jeff, 2026-11-14): say "follow me" and the robot
follows the person, keeping a set distance and keeping the phone's camera on them, and
answers "come", "come closer", "go further back" and "stay". This page is how the pieces
fit, and the simulator test that has to pass before it goes on the robot.

## The pieces

```
camera frame ─► person tracker ─► follower ─► master policy ─► two servos
  (phone)        face / body        3 commands     (one JSON)
                     ▲
voice ───────────────┘ (sets the follower's distance, or holds it)
```

### 1. One master policy

One exported policy with one input set, so the app never swaps files mid-walk:

| Input | Meaning |
| --- | --- |
| `cmd_yaw` | heading error to the person, degrees / 180, positive = turn left |
| `cmd_forward` | signed speed, -1..+1 of the cap |
| `cmd_pitch` | camera height / tilt, -1 low (looking up) .. +1 tall |

What it does with them:

- **Speed above 0:** walk tall, camera first, toward the heading (lift and fall,
  `configs/lift_fall.toml`).
- **Speed below 0:** back away in the low waddle (camera walk, `configs/camera_walk.toml`).
- **Speed 0:** stand, spin in place to face the heading, and tilt the camera to the
  commanded pitch (look, `configs/look.toml`).

Each skill trains on its own first (their inputs differ: lift and fall has no speed
input, the camera walk and look do). The master is then distilled from them into one
student with the shared inputs above: each teacher is queried only in its own mode,
matched in radians (their `action_to_rad` differ), and the training episodes switch
modes partway through so the student also learns the hand-overs. PPO fine-tunes the
student after.

### 2. Person tracker (on the phone, in the browser)

MediaPipe's face landmarker, with its pose landmarker as the fallback, both running on
the phone in the PWA:

- **Face, when they face the robot.** Direction comes from the face box's centre: its
  horizontal offset times the camera's horizontal field of view (calibrated once per
  phone) gives the heading error; its vertical offset gives the pitch error. Distance
  comes from the eye spacing in pixels against an assumed 63 mm interpupillary distance.
- **Body, when they turn away or are too far for a face.** The same, from the pose
  landmarks: direction from the shoulders' midpoint, distance from the shoulder width
  against about 0.4 m. A per-person calibration (one "stand here" at a known distance)
  fixes both assumptions.
- **Lost.** No face or body for a moment: hold the last heading, speed 0, and spin
  slowly toward where they were last seen. After a few seconds, stop and say so.

Readings are smoothed (a short moving average) and stamped with their age; the policy
already trains with heading readings that are noisy and up to 6 ticks stale.

### 3. Follower

A small controller between the tracker and the policy:

- **Heading:** the tracker's heading error, straight into `cmd_yaw`.
- **Speed:** from the distance error against the set distance, with a dead band. Too
  far walks forward, too close backs away, and within the band it's 0, so it stops and
  just turns. The speed is capped and ramps, so a mode change is never a jump.
- **Pitch:** the tracker's vertical error, integrated into `cmd_pitch`, so the camera
  keeps the face mid-frame.

### 4. Voice

The voice commands only change the follower's settings, never the policy:

| Say | Effect |
| --- | --- |
| "follow me" | start following at the set distance (default about 1 m) |
| "come" | follow, and shorten the set distance to the near limit once |
| "come closer" | set distance minus a step (about 0.25 m), down to a near limit |
| "go further back" | set distance plus a step, up to a far limit |
| "stay" | speed 0 until released: stop walking, but keep turning and tilting to watch |

"follow me" or "come" releases a stay.

## The final test, in the simulator

The acceptance test for the master policy, before it goes on the robot. A scripted
person walks around the robot, and the whole loop runs as the app would, with no
access to the simulator's truth except through what the camera would see:

- **The person:** a figure with a face billboarded toward the robot, wandering on a
  random path: sometimes walking away, sometimes coming close, sometimes circling, and
  the face moving up and down (sitting, standing). Now and then they turn their back,
  so the tracker has to fall back to the body.
- **The camera model:** project the face and shoulders through the phone camera's pose
  and field of view, add pixel noise and dropouts, and hand the tracker the same
  readings MediaPipe would.
- **The commands:** a script of voice commands through the run: "follow me", "come
  closer", "go further back", "stay" then "follow me", and "come". The set distance
  therefore changes live, and the stays hold.
- **Robots:** the nominal body, then randomised bodies and both floors.

It reports, over time and in summary:

- **Distance error:** actual distance minus the set distance.
- **Aim error:** where the camera points, against the face, horizontal and vertical.
- **Time to settle** after each command or turn-away, plus falls.
- **A replay** in the browser 3D viewer with the person in it, so you can watch it
  follow.

It passes when the robot keeps the person in frame through the whole script, settles
to the set distance after each command, holds still on "stay", and doesn't fall. The
thresholds get set once there's a baseline. The same harness first runs on each single
skill (with the follower limited to that skill's mode) for that baseline.

## Order of work

1. Each skill trains and passes its own eval: look (spin and aim), lift and fall (tall,
   forward), camera walk (low, backward).
2. The person, camera model, tracker and follower in the simulator, with the baseline
   on the single skills.
3. The master distilled and fine-tuned; the final test above.
4. On the phone: MediaPipe tracker, follower and voice in the PWA, with the master
   policy through the app's runner.
