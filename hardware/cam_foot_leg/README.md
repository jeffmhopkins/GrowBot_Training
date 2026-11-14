# Cam-foot leg

A printable pair of legs with the locomotion bench's winning curved ("cam") sole
(`research/locomotion/codesign.py`), mounted exactly like upstream's stock leg.

```bash
pip install -e ".[cad]"                       # manifold3d and shapely
python hardware/cam_foot_leg/make_leg.py      # out/cam_foot_leg_left.stl and _right.stl
python hardware/cam_foot_leg/make_leg.py --dip 1.0   # for a thicker rubber coat
```

The first run fetches upstream's `growbot_diy_leg_v1_r01.stl` at a pinned commit (checked
by hash) into `.upstream/`.

## What it is

- **Sole:** the bench's foot to the hundredth of a millimetre. It spans 20.85° either side
  of the leg's line, 48 mm wide. It sits 73.46 mm from the axle at the tail edge, 67.41 mm
  on the leg's line and 61.37 mm at the nose edge. It's 13 mm thick, like the stock leg.
- **Rubber dip:** the sole is dipped from 2.6 mm in from the tail edge to the nose edge.
  That stretch is printed `--dip` mm (default 0.75) inside the simulated curve, so the
  coat brings it back to the shape the bench tested. The square step where it starts is
  the line to dip to. The bare tail strip stays plain PLA.
- **Mount:** the horn slot, the spline bore and the screw hole aren't redrawn. They're
  cut from upstream's leg, so the MG90S horn fits and screws in exactly as it does on the
  stock leg.
- **Organic cutouts:** twelve bone-like cells, a Voronoi pattern relaxed to even
  sizes, with every corner rounded so the ribs taper and fillet into each other. The ribs
  are 3.4 mm beside the hub and the sole, which take the most load, and 2.2 mm across the
  middle. The rim, flanks and a teardrop boss round the horn slot stay solid. The fan has
  about 1.4 times the stock leg's side area but the same 21.1 cm³ of plastic. Every edge
  on both faces has a 0.8 mm chamfer. `CELL_SEED` and `CELL_COUNT` give other patterns.

## Printing and fitting

- The STLs lie flat, outer (screw-head) face down. Every void widens upward, so print in
  PLA with no supports, like upstream's legs. The 0.8 mm chamfers are modelled as 0.2 mm
  steps, so print at 0.2 mm layers or finer.
- **L** and **R** are engraved on the servo face. They're mirror images: fit each leg so
  its long (tail) corner points to the tail, the battery end. Only one leg does that on
  each side.
- Dip to the step, sole down, and let the coat cure before fitting.

Upstream's leg is PolyForm Noncommercial 1.0.0
(`growbot_train/bodies/LICENSE-upstream`), and so is this part derived from it.
