# RAS 598 – Assignment 3: 3D Discrete Bayes Filter (Histogram Filter)

## Overview

This package implements a **3D Histogram Filter** to localise a robot in a
cave environment using noisy odometry and landmark (fiducial) sightings.

---

## Quick Start

```bash
# 1. Copy the package into your workspace
cp -r ras598_assignment_3 ~/ros2_ws/src/

# 2. Build
cd ~/ros2_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select ras598_assignment_3 --symlink-install
source install/setup.bash

# 3. Launch everything
ros2 launch ras598_assignment_3 bayes_launch.py
```

---

## Architecture

### State Space

| Axis | World range | Resolution | Grid size |
|------|------------|------------|-----------|
| X    | −20 m → +20 m | 0.5 m/cell | 80 cells  |
| Y    | −20 m → +20 m | 0.5 m/cell | 80 cells  |
| θ    | 0° → 360°     | 10°/bin    | 36 bins   |

Total belief array: `(80, 80, 36)` float64 — ≈ 1.8 M cells.

---

### Coordinate Mapping (`real_to_grid` / `grid_to_real`)

```
World x ∈ [−20, +20]  →  ix = floor((x − (−20)) / 0.5)
World y ∈ [−20, +20]  →  iy = floor((+20 − y)  / 0.5)   ← Y is FLIPPED
θ (radians)           →  it = floor(degrees(θ) / 10) % 36
```

The Y-axis is flipped because RViz/OccupancyGrid row-0 is at the **top**
(highest world-y), matching the standard image convention.

---

### Prediction Step (Motion Model)

1. **Orientation shift** — `delta_theta` in radians → integer bin shift →
   `np.roll(belief, bins, axis=2)`.
2. **Spatial shift** — `delta_x / 0.5` → `np.roll(belief, cells, axis=0)`;
   `-delta_y / 0.5` → `np.roll(belief, cells, axis=1)` (Y flip!).
3. **Diffusion** — `scipy.ndimage.gaussian_filter(sigma=[0.8, 0.8, 0.6])`
   spreads the cloud to model accumulated odometry uncertainty.
4. **Normalise** — re-scale so the array sums to 1.

---

### Update Step (Measurement Model)

For every detected fiducial `(id, range_meas, bearing_meas)`:

1. Look up the landmark's true world coordinates `(lx, ly)`.
2. For **every** grid cell `(ix, iy, it)`:
   - Compute `expected_range = √((lx−x)² + (ly−y)²)`
   - Compute `expected_bearing = atan2(ly−y, lx−x) − θ`
3. Gaussian likelihood:

```
L = exp(−½·((r_exp − r_meas)/σ_r)²) × exp(−½·((β_exp − β_meas)/σ_β)²)
```

4. Multiply likelihood into `belief` (Bayes rule).
5. Normalise.

All of steps 2–3 are computed **fully vectorised** using pre-built meshgrids
`_X, _Y, _TH` of shape `(80, 80, 36)`, so no Python loops over cells.

---

### Why Is the Belief Ring-Shaped After One Range Measurement?

A single range measurement `r` says: *"I am at distance r from landmark k."*
This is consistent with **any** cell on the circle of radius r centred on
the landmark — regardless of the robot's heading — creating a ring
(torus in 3D because all orientations share it).

Only when a **bearing** measurement is added does the ring collapse along
the angular dimension, leaving an arc.  After observing **two or more**
landmarks the arcs intersect, converging to a tight peak near the true pose.

---

## Published Topics

| Topic | Type | Description |
|-------|------|-------------|
| `viz/belief_costmap` | `OccupancyGrid` | Belief marginalised over θ |
| `viz/landmarks` | `MarkerArray` | Orange cylinders at landmark positions |
| `viz/gt_path` | `Path` | Green – ground truth trajectory |
| `viz/odom_path` | `Path` | Red – raw odometry trajectory |

---

## Tuning Parameters (top of `bayes_filter.py`)

| Parameter | Default | Effect |
|-----------|---------|--------|
| `SIGMA_RANGE` | 1.2 m | Range measurement noise |
| `SIGMA_BEARING` | 15° | Bearing measurement noise |
| `DIFFUSE_XY` | 0.8 cells | Spatial diffusion per step |
| `DIFFUSE_THETA` | 0.6 bins | Angular diffusion per step |

Increase diffusion if the belief collapses too quickly and loses the true pose.
Decrease if the cloud never sharpens.

---

## Viva Voce Preparation

**Bayes filter logic → code:**

- `predict()` implements `bel̄(x) = ∫ p(x|u,x') bel(x') dx'` via grid shifts + blur.
- `update()` implements `bel(x) = η · p(z|x) · bel̄(x)` via cell-wise likelihood multiplication.

**Scenarios:**
- *Kidnapped robot*: belief suddenly becomes uniform; recovery happens when a landmark is seen and the ring/arc pattern re-emerges.
- *Uniform initial belief*: every cell starts equally likely; converges after 2-3 landmark sightings.
- *Grid resolution trade-off*: finer grid → better accuracy but O(N³) memory/time; coarser → faster but coarser estimates.
