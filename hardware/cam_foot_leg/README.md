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
- **Femur shape:** a bone shaft runs from a teardrop head round the horn slot and flares
  into the sole, filleted where it meets both. An S-shaped marrow slot runs down the shaft.
- **Sculpted servo face:** the shaft's top rounds over toward every edge and the slot,
  and dishes a little mid-shaft like a long bone's waist. It fades back to the full
  13 mm round the hub and along the sole, so the horn capture and the sole's contact
  width are untouched. It's a heightfield on top of a flat outer face, so it still prints
  without supports. Each leg uses 22.2 cm³ of plastic (the stock leg uses 21.1).
- Every edge of the flat outer face, and of the full-thickness hub and sole, has a 0.8 mm
  chamfer.

## Printing and fitting

- The STLs lie flat, outer (screw-head) face down. Every void widens upward, so print in
  PLA with no supports, like upstream's legs. The 0.8 mm chamfers are modelled as 0.2 mm
  steps, so print at 0.2 mm layers or finer; the sculpted face prints as ordinary top
  layers.
- **L** and **R** are engraved on the servo face. They're mirror images: fit each leg so
  its long (tail) corner points to the tail, the battery end. Only one leg does that on
  each side.
- Dip to the step, sole down, and let the coat cure before fitting.

Upstream's leg is PolyForm Noncommercial 1.0.0
(`growbot_train/bodies/LICENSE-upstream`), and so is this part derived from it.
