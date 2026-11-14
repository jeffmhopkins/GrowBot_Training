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
| `cmd_yaw` | heading error to the person, degrees / 180, positive = target to the left (turn left), measured from where the camera points |
| `cmd_forward` | signed speed, -1..+1 of one shared cap |
| `cmd_pitch` | camera height: -1 low (the camera looks almost straight up) .. +1 tall (toward the horizon) |

What it does with them:

- **Speed above 0:** walk tall, camera first, toward the heading. Teacher: lift and
  fall (`configs/lift_fall.toml`).
- **Speed below 0:** back away low. Teacher: the camera walk's backing half
  (`configs/camera_walk.toml`). Which gait Jeff means by "the original low waddle" is
  still to settle: the camera walk, or the first heading walk, which is lower but walks
  nose first.
- **Speed 0:** stand, spin in place to face the heading, and tilt the camera to the
  commanded pitch. Teacher: look (`configs/look.toml`).

Each skill trains on its own first. The master is then distilled from them into one
student with the shared inputs above. Each teacher is queried only in its own mode, with
its own inputs rebuilt from the student's. The training episodes switch modes partway
through, so the student also learns the hand-overs. PPO fine-tunes the student after.

The teachers don't line up yet, and the distillation has to bridge these gaps:

- **Inputs:**
  - lift and fall sees only the heading, with no speed and no camera height;
  - look sees heading and camera height, with no speed;
  - only the camera walk sees all three.
- **Speed:**
  - lift and fall walks only at its 5 cm/s cap: its cloned gait has one speed, and PPO
    unlearned it when asked for others;
  - the camera walk's cap is 1 cm/s.
  
  So forward speeds between 0 and full have no teacher yet. The roadmap trains them
  (phase 1); until then the follower treats forward as stop-or-go. The master's cap
  per direction has to be set.
- **Camera height:** -1 is 20 degrees under rest in look and 30 in the camera walk.
  Lift and fall stands about 18 degrees taller than rest whatever `cmd_pitch` says, so
  `cmd_pitch` only steers the camera at speed 0 and when backing, until the tall walk
  learns a camera-height input (roadmap phase 1).
- **Clock:** each teacher's gait clock ticks at its own rate (look 1 Hz, camera walk
  2 Hz, lift and fall 2.347 Hz). The student needs the clock per mode, or its own clock
  with the teachers' phases mapped onto it.
- **Heading:** look measures the error from where the camera points (`heading_from =
  "lens"`), which is what the tracker reports. The other two measure it from the tail
  end's direction; the difference is small when the robot isn't rolling.
- **Body:** look and lift and fall train on the printed build. The camera walk trains on
  upstream's sim body, so it needs retraining on the build before it can teach.
- **Leg range:** each teacher's full action means a different swing (`action_to_rad`), so
  they are matched in radians, not raw actions.

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
already trains with heading readings that are noisy (3 degrees), drifting (0.5 degrees a
second) and held for up to 5 ticks (0.2 s), with no added delay. A real camera pipeline
lags more, so the final test adds the lag (below) and the policy may need to train with
it.

**More than one person.** The tracker follows one chosen person, not whoever is in frame:

- **Choosing:** at "follow me" it locks onto the person who said it. That is the face
  whose mouth is moving as the phrase is heard; failing that, the nearest, then the most
  central. "Follow me" said again re-chooses.
- **Keeping:** each frame, every detection is matched to the locked track by predicted
  position (a constant-velocity filter on bearing and distance), size, and an appearance
  signature (face embedding when the face shows; clothing colour across the torso when
  only the body does). A detection far from the prediction is never taken just because
  it is the only one.
- **Crossings and short occlusions:** while someone walks between, the track coasts on
  its prediction for up to about 2 s, and re-attaches only to a detection that matches
  both position and appearance.
- **Lost for longer:** stop, turn toward the last bearing, and re-acquire only on a strong
  appearance match. Otherwise stay put and say "I lost you" rather than follow a
  stranger.

### 3. Follower

A small controller between the tracker and the policy:

- **Heading:** the tracker's heading error, straight into `cmd_yaw`.
- **Speed:** from the distance error against the set distance, with a dead band. Too
  far walks forward, too close backs away, and within the band it's 0, so it stops and
  just turns. Backing ramps with the error. Forward is stop-or-go at full speed until a
forward teacher learns slower speeds (see the gaps above).
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
- **Other people:** one to three distractors walk their own paths, cross between the
  robot and the person, stand near them, and sometimes stand nearer or more central than
  them, so the tracker has to keep the right one.
- **The camera model, never ground truth:** the tracker only gets what a phone would.
  Every face and shoulder line is projected through the phone camera's actual pose and
  field of view, so the robot's own rocking moves the image. On top of that:
  - box jitter of a few pixels;
  - a distance error of 10 to 20% that drifts slowly (the eye-spacing and
    shoulder-width assumptions are off);
  - detection dropouts, more when a face is small or turned;
  - 100 to 200 ms of frame latency;
  - a bearing error on top: noise and slow drift, like the heading noise the policy
    already trains with.
- **The commands:** a script of voice commands through the run: "follow me", "come
  closer", "go further back", "stay" then "follow me", and "come". The set distance
  therefore changes live, and the stays hold.
- **Robots:** the nominal body, then randomised bodies and both floors.

It reports, over time and in summary:

- **Distance error:** actual distance minus the set distance.
- **Aim error:** where the camera points, against the face, horizontal and vertical.
- **Time to settle** after each command or turn-away, plus falls.
- **Identity:** the share of time it is locked on the right person, and any switches to
  a distractor.
- **A replay** in the browser 3D viewer with everyone in it:
  - the voice commands shown as they fire;
  - the policy's three inputs drawn as arrows (heading, signed speed, camera pitch), in
    the viewer's 3D input-arrow style;
  - where the tracker believes the person is, drawn next to where they really are, so
    noise and lag are visible.

It passes when the robot keeps the person in frame through the whole script, settles
to the set distance after each command, holds still on "stay", never switches to a
distractor, and doesn't fall. The thresholds get set once there's a baseline. The same harness first runs on each single
skill (with the follower limited to that skill's mode) for that baseline.

## Order of work

1. Each skill trains and passes its own eval: look (spin and aim), lift and fall (tall,
   forward), the low backing gait (retrained on the build).
2. The person, camera model, tracker and follower in the simulator, with the baseline
   on the single skills.
3. The master distilled and fine-tuned; the final test above.
4. On the phone: MediaPipe tracker, follower and voice in the PWA, with the master
   policy through the app's runner.
