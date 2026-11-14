# Locomotion research bench

Which body changes and gaits get GrowBot moving tail first, toward what the phone's
camera sees, fastest and most reliably? PPO takes hours per try; an open-loop gait on
this bench takes a twentieth of a second. So body changes (what to dip in rubber, foot
shapes) and gait families are compared here, and only the winners go to training.

| File | What |
| --- | --- |
| `lab.py` | The bench: the robot with chosen materials and feet, a gait played through the trainer's relay and servo model, scored on tail-first speed |
| `search.py` | Every scenario: CMA-ES over a free-form two-leg gait (8 keyframes per leg, 0.5-5 Hz), tuned on a few random robots on both floors, then tested on 32 it never saw |
| `codesign.py` | Tunes a curved ("cam") foot's shape and its gait together |
| `report.py` | The results tables, plus each gait across slick to grippy floors |
| `make_body.py` | Writes `growbot_train/bodies/build.xml`, the printed body with a phone |

```bash
python research/locomotion/search.py --stock          # bare print and liquid-dip setups
python research/locomotion/search.py --feet           # reprinted foot shapes
python research/locomotion/search.py --only d2 f13    # some scenarios
python research/locomotion/codesign.py --from f13 f14
python research/locomotion/report.py results
```

Only `mujoco` and `numpy` are needed (`.venv` has both). Each scenario takes about five
minutes on four cores. Results land in `results/` as JSON: the gait, its speed on the
as-designed robot on each floor, and its spread over the held-out robots.

What the bench models that the training env doesn't (yet): ratchet (directional)
friction such as angled bristles, and reshaped feet (round, bevelled, an L toe, a curved
cam sole, a chamfered rubber edge). Both share `growbot_train/surfaces.py`, so materials
and floors mean the same thing in each.
