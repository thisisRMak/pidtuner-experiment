# LQG Design Track — Preset Catalog Review

Generated 2026-08-18. One pass of `LQR` (each plant's own suggested `Q`/`R`, i.e. the same weights your `.m` files use) over all 12 plants currently in the preset catalog (`pidtuner/lqg_examples_m/*.m` → `pidtuner/lqg_examples_json/*.json`), with the full pre-/post-design correctness check suite (see `docs/lqg_testing.md` for what each check verifies).

**Summary: 12/12 plants pass every check.**

**Open question — sensitivity/complementary sensitivity (Ms/Mt) loop-breaking point:** all Ms/Mt figures below are evaluated at the plant input (`loop_point = "plant_input"`). For this batch every plant is full-state-feedback LQR, where that's the standard place to judge robustness and the classical LQR guarantees (Ms<=2, etc.) do apply. If output-feedback (LQG/Kalman) plants are added to this report later, plant-input Ms/Mt does **not** carry the same guarantee there (see `lqg_frequency.py`'s module docstring) — flagging this now as something to verify with you rather than presenting it as settled.

## Source files that needed a fix before they'd run

These were excluded from the catalog until now because the `.m` file as originally provided had a bug unrelated to the plant data itself. Flagging so you can confirm the fix matches what you intended:

- **`pidtuner/lqg_examples_m/AIFurnaceModel.m`** (`furnace_model`): source had a stray trailing ')' after `lqr(A,B,Q,R)`, a plain syntax typo — fixed by removing it.
- **`pidtuner/lqg_examples_m/AIF100Engine.m`** (`f100_engine`): source's `R` was undefined at the `lqr(A,B,Q,R)` call; now `R=eye(5)`, matching nu=5 from B/D's column count.
- **`pidtuner/lqg_examples_m/AIExample2RTP.m`** (`example2_rtp`): the lqr() call referenced undefined uppercase A,B,C,D while only lowercase a,b,c,d were assigned, and the dimensions didn't match even correcting the case — fixed by using consistent, correctly-sized A,B,Q,R.

## Results

| Plant | nx/nu/ny | Q kind | Stable | Checks |
|---|---|---|---|---|
| `airc` | 5/3/3 | output_weighted | True | all pass |
| `aircraft_hall` | 5/2/2 | custom | True | all pass |
| `autm` | 12/2/2 | output_weighted | True | all pass |
| `chemical_reactor` | 4/2/4 | identity | True | all pass |
| `distillation_column` | 11/2/3 | identity | True | all pass |
| `drone` | 6/2/2 | output_weighted | True | all pass |
| `example2_rtp` 🔧 fixed | 3/3/3 | custom | True | all pass |
| `f100_engine` 🔧 fixed | 4/5/5 | output_weighted | True | all pass |
| `furnace_model` 🔧 fixed | 8/4/4 | identity | True | all pass |
| `generic_rtp` | 15/5/5 | output_weighted | True | all pass |
| `rpv` | 6/2/2 | output_weighted | True | all pass |
| `tgen` | 6/2/2 | output_weighted | True | all pass |

## Per-plant detail

### `airc` — Aircraft (Maciejowski)

Source: `pidtuner/lqg_examples_m/AIAIRC.m` (Maciejowski (via AIAIRC.m))

nx=5, nu=3, ny=3, Q=output_weighted, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 0
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 9.00e-14 (tol 1e-06 × scale 1.15e+01)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 0.0329
- [PASS] closed-loop poles stable — max Re(pole) = -0.656

Gain matrix K (3×5):

```
[[-0.1316,  0.3586,  1.0376,  0.4805, -0.1446],
 [ 0.3834,  0.9042,  0.5311,  0.1647, -0.1655],
 [-0.9142, -0.2621, -2.5291, -0.667 ,  1.3503]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 7.16 s,  ISU (control energy) = 6.21,  |u|_peak = 3.02,  final ||x|| = 1.84e-06

Closed-loop step response (r = ones), per output channel:
  y0: Overshoot = 1.6%,  Rise (10-90%) = 2.52 s,  Settling (2%) = 5.56 s
  y1: Overshoot = 13.6%,  Rise (10-90%) = 0.859 s,  Settling (2%) = 7.56 s
  y2: Overshoot = 27%,  Rise (10-90%) = 0.573 s,  Settling (2%) = 4.01 s

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 1.31

### `aircraft_hall` — Aircraft (Hall)

Source: `pidtuner/lqg_examples_m/AIAircraftHall.m` (Hall 1971, = AILQG.pdf Example 1 (via AIAircraftHall.m))

nx=5, nu=2, ny=2, Q=custom, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 0
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 8.68e-14 (tol 1e-06 × scale 4.16e+01)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 0.0677
- [PASS] closed-loop poles stable — max Re(pole) = -0.0501

Gain matrix K (2×5):

```
[[0.3354, 1.0294, 0.0913, 0.1059, 0.5844],
 [0.1751, 0.3608, 0.7282, 1.0261, 0.8115]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 80.3 s,  ISU (control energy) = 18.5,  |u|_peak = 3.1,  final ||x|| = 7.85e-07

Closed-loop step response (r = ones), per output channel:
  y0: Overshoot = 207%,  Rise (10-90%) = 0.75 s,  Settling (2%) = 94.5 s
  y1: Overshoot = 0%,  Rise (10-90%) = 45.8 s,  Settling (2%) = 87 s

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 1.54

### `autm` — AUTM

Source: `pidtuner/lqg_examples_m/AIAUTM.m` ((via AIAUTM.m))

nx=12, nu=2, ny=2, Q=output_weighted, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = -3.92e-14
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 1.98e-11 (tol 1e-06 × scale 8.67e+02)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 3.47e-09
- [PASS] closed-loop poles stable — max Re(pole) = -0.112

Gain matrix K (2×12):

```
[[  0.1996,   0.5743,  -0.772 , -10.8369,  -0.7529,  -0.0459,   0.0922,
    0.0973,   0.6347,  -0.9279,  -0.1177,  -0.025 ],
 [  0.0216,   0.0673,  -0.131 ,  -1.2584,   0.0214,   4.5264,   1.8633,
    0.9305,   1.6294,  17.9675,   5.338 ,   0.172 ]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 25.1 s,  ISU (control energy) = 414,  |u|_peak = 31.1,  final ||x|| = 2.77e-08

Closed-loop step response (r = ones), per output channel:
  y0: Overshoot = 8.49%,  Rise (10-90%) = 0.335 s,  Settling (2%) = 15.1 s
  y1: Overshoot = 0.924%,  Rise (10-90%) = 1.67 s,  Settling (2%) = 8.37 s

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 0.966

### `chemical_reactor` — Chemical reactor (Munro)

Source: `pidtuner/lqg_examples_m/AIChemicalReactor1.m` (Munro (via AIChemicalReactor1.m))

nx=4, nu=2, ny=4, Q=identity, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 1
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 4.68e-14 (tol 1e-06 × scale 2.25e+00)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 0.0533
- [PASS] closed-loop poles stable — max Re(pole) = -3.12

Gain matrix K (2×4):

```
[[ 0.0532,  0.9421,  0.3513,  0.8411],
 [-2.5325, -0.0716, -1.7794,  1.1645]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 1.32 s,  ISU (control energy) = 1.74,  |u|_peak = 3.22,  final ||x|| = 4.03e-07

Closed-loop step response (r = ones), per output channel:
- not available: plant is non-square (nu=2, ny=4) — reference tracking (N̄) requires nu == ny

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 1.64

### `distillation_column` — Distillation column (Davison)

Source: `pidtuner/lqg_examples_m/AIDistillationColumn.m` (Davison 2011 (via AIDistillationColumn.m))

nx=11, nu=2, ny=3, Q=identity, R=scaled_identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 1
- [PASS] R positive definite — min eig(R) = 0.1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 1.38e-14 (tol 1e-06 × scale 2.32e+02)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 5.1
- [PASS] closed-loop poles stable — max Re(pole) = -0.00325

Gain matrix K (2×11):

```
[[ 0.1107,  0.032 ,  0.0152,  0.0047, -0.0051, -0.0108, -0.0142, -0.0138,
  -0.0041,  0.0384,  0.122 ],
 [ 0.3517,  0.5633,  1.0519,  1.2559,  1.2407,  1.3247,  1.2797,  1.1202,
   0.8239,  0.6827,  0.2254]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 695 s,  ISU (control energy) = 1.34e+03,  |u|_peak = 9.92,  final ||x|| = 2.04e-07

Closed-loop step response (r = ones), per output channel:
- not available: plant is non-square (nu=2, ny=3) — reference tracking (N̄) requires nu == ny

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 0.997

### `drone` — Drone

Source: `pidtuner/lqg_examples_m/AIDrone.m` ((via AIDrone.m))

nx=6, nu=2, ny=2, Q=output_weighted, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 0
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 3.99e-13 (tol 1e-06 × scale 2.84e+01)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 0.0122
- [PASS] closed-loop poles stable — max Re(pole) = -0.0414

Gain matrix K (2×6):

```
[[ 7.1942, -0.506 , -2.5174,  0.063 ,  3.8659, -0.6998],
 [ 2.9509,  0.4007, -3.1548,  0.1377, -0.6998,  5.1549]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 77.2 s,  ISU (control energy) = 37.7,  |u|_peak = 7.4,  final ||x|| = 3.7e-07

Closed-loop step response (r = ones), per output channel:
- not available: reference tracking failed: [[A,B],[C,D]] is singular — the plant has a transmission zero at the origin

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 1.52

### `example2_rtp` — RTP (Franklin/Powell/Emami-Naeini 8e)

Source: `pidtuner/lqg_examples_m/AIExample2RTP.m` (Franklin/Powell/Emami-Naeini, Feedback Control of Dynamic Systems, 8th ed. (via AIExample2RTP.m))

**Fixed for this review:** the lqr() call referenced undefined uppercase A,B,C,D while only lowercase a,b,c,d were assigned, and the dimensions didn't match even correcting the case — fixed by using consistent, correctly-sized A,B,Q,R.

nx=3, nu=3, ny=3, Q=custom, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 1
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 4.11e-14 (tol 1e-06 × scale 5.15e+01)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 2.19
- [PASS] closed-loop poles stable — max Re(pole) = -0.334

Gain matrix K (3×3):

```
[[ 0.8809,  0.9486, -0.2091],
 [ 0.3691,  2.3973, -0.5508],
 [ 0.0067, -1.0922,  4.0978]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 16 s,  ISU (control energy) = 13.2,  |u|_peak = 3.01,  final ||x|| = 5.88e-07

Closed-loop step response (r = ones), per output channel:
  y0: Overshoot = 39.4%,  Rise (10-90%) = 1.24 s,  Settling (2%) = 17.7 s
  y1: Overshoot = 0.000144%,  Rise (10-90%) = 8.21 s,  Settling (2%) = 13.5 s
  y2: Overshoot = 3.72e-05%,  Rise (10-90%) = 4.5 s,  Settling (2%) = 10.1 s

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 0.975

### `f100_engine` — F-100 Engine

Source: `pidtuner/lqg_examples_m/AIF100Engine.m` ((via AIF100Engine.m))

**Fixed for this review:** source's `R` was undefined at the `lqr(A,B,Q,R)` call; now `R=eye(5)`, matching nu=5 from B/D's column count.

nx=4, nu=5, ny=5, Q=output_weighted, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 0.00121
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 2.90e-12 (tol 1e-06 × scale 1.09e+03)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 0.00249
- [PASS] closed-loop poles stable — max Re(pole) = -0.801

Gain matrix K (5×4):

```
[[ 6.5682e+00,  2.3469e+01,  4.6092e-01, -4.0718e-03],
 [ 1.9633e+01,  4.6834e-01, -1.4132e-01,  6.9045e-02],
 [ 3.6922e+00, -8.3887e-01, -4.3673e-02,  1.4063e-02],
 [-3.4932e-01, -7.2209e-01, -8.5574e-03, -4.1932e-04],
 [-2.4116e-01, -4.2678e-01, -6.4373e-03, -8.1862e-05]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 2.72 s,  ISU (control energy) = 45,  |u|_peak = 30.5,  final ||x|| = 1.01e-07

Closed-loop step response (r = ones), per output channel:
  y0: Overshoot = 5.8%,  Rise (10-90%) = 0.0469 s,  Settling (2%) = 0.281 s
  y1: Overshoot = 0%,  Rise (10-90%) = 0.188 s,  Settling (2%) = 0.422 s
  y2: Overshoot = 22.4%,  Rise (10-90%) = 0.0469 s,  Settling (2%) = 0.844 s
  y3: Overshoot = 62.6%,  Rise (10-90%) = nan s,  Settling (2%) = 1.97 s
  y4: Overshoot = 943%,  Rise (10-90%) = nan s,  Settling (2%) = 2.58 s

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 0.925

### `furnace_model` — Furnace (Davison 2011 / Rosenbrock)

Source: `pidtuner/lqg_examples_m/AIFurnaceModel.m` (Davison 2011 / Rosenbrock (via AIFurnaceModel.m))

**Fixed for this review:** source had a stray trailing ')' after `lqr(A,B,Q,R)`, a plain syntax typo — fixed by removing it.

nx=8, nu=4, ny=4, Q=identity, R=scaled_identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 1
- [PASS] R positive definite — min eig(R) = 0.1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 6.10e-15 (tol 1e-06 × scale 5.05e+00)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 0.297
- [PASS] closed-loop poles stable — max Re(pole) = -0.195

Gain matrix K (4×8):

```
[[-1.7604, -0.8922,  0.3162,  1.2776, -0.7672,  0.8696,  0.0809, -0.137 ],
 [-0.5919, -1.8683, -1.1784, -0.8362,  1.0883, -0.4679,  0.4791,  0.2977],
 [-1.7951,  0.557 ,  1.0403, -1.3485, -0.2636, -0.5024,  0.3879, -0.0788],
 [-1.1643,  1.5341, -1.6505, -0.2164,  0.8964,  0.2342, -0.025 ,  0.1476]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 18.3 s,  ISU (control energy) = 4.33,  |u|_peak = 3.08,  final ||x|| = 6.56e-07

Closed-loop step response (r = ones), per output channel:
  y0: Overshoot = 1.83%,  Rise (10-90%) = 0.963 s,  Settling (2%) = 1.73 s
  y1: Overshoot = 3.26%,  Rise (10-90%) = 0.77 s,  Settling (2%) = 5.59 s
  y2: Overshoot = 3.7%,  Rise (10-90%) = 0.578 s,  Settling (2%) = 4.24 s
  y3: Overshoot = 0.2%,  Rise (10-90%) = 0.578 s,  Settling (2%) = 1.16 s

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 0.929

### `generic_rtp` — Generic RTP (Rapid Thermal Processing)

Source: `pidtuner/lqg_examples_m/AIGeneric_RTP.m` ((via AIGeneric_RTP.m, matrices from rtpsystem.dat))

nx=15, nu=5, ny=5, Q=output_weighted, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = -3.75e-17
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 2.16e-14 (tol 1e-06 × scale 1.00e+00)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 3.99e-09
- [PASS] closed-loop poles stable — max Re(pole) = -0.00829

Gain matrix K (5×15):

```
[[-0.1137, -0.2869,  0.0227, -0.2829, -0.1523,  0.1704, -0.0272,  0.1279,
   0.0321,  0.0258,  0.0074, -0.0079, -0.0069,  0.0018,  0.0201],
 [-0.3029, -0.1789,  0.1443, -0.052 ,  0.0438, -0.0681,  0.0076, -0.222 ,
   0.064 , -0.0259,  0.0238,  0.0406, -0.0128, -0.0088,  0.0028],
 [-0.2274,  0.0765, -0.1902, -0.0045, -0.1321, -0.1759,  0.0668,  0.1149,
   0.1438,  0.0151,  0.0009,  0.029 ,  0.0391,  0.0068,  0.0036],
 [-0.0937,  0.0291, -0.1924, -0.0968,  0.2261,  0.1525,  0.0804,  0.0351,
   0.046 , -0.1091, -0.0283, -0.0048,  0.0167, -0.0243,  0.0016],
 [-0.0533,  0.0111, -0.1145, -0.0597,  0.2316,  0.0922,  0.0536, -0.042 ,
  -0.0529,  0.2124, -0.0434,  0.0318, -0.0107,  0.008 ,  0.0014]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 331 s,  ISU (control energy) = 1.46,  |u|_peak = 0.545,  final ||x|| = 8.3e-08

Closed-loop step response (r = ones), per output channel:
  y0: Overshoot = 1.16e-05%,  Rise (10-90%) = nan s,  Settling (2%) = 0 s
  y1: Overshoot = 0.000239%,  Rise (10-90%) = nan s,  Settling (2%) = 0 s
  y2: Overshoot = 0.00187%,  Rise (10-90%) = nan s,  Settling (2%) = 0 s
  y3: Overshoot = 0%,  Rise (10-90%) = nan s,  Settling (2%) = 0 s
  y4: Overshoot = 0.00185%,  Rise (10-90%) = nan s,  Settling (2%) = 0 s

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 1.25

### `rpv` — RPV (Maciejowski)

Source: `pidtuner/lqg_examples_m/AIRPV.m` (Maciejowski (via AIRPV.m))

nx=6, nu=2, ny=2, Q=output_weighted, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 0
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 4.47e-15 (tol 1e-06 × scale 1.41e+00)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 9.97e-08
- [PASS] closed-loop poles stable — max Re(pole) = -0.0276

Gain matrix K (2×6):

```
[[-2.7711e-04, -5.1897e-01, -2.0582e-01, -8.0728e-01,  1.9225e-01,
  -1.3430e-01],
 [ 1.9506e-04,  3.3602e-01,  1.4388e-01,  5.8935e-01, -1.3430e-01,
   9.3924e-02]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 188 s,  ISU (control energy) = 2.2,  |u|_peak = 1.47,  final ||x|| = 2.69e-06

Closed-loop step response (r = ones), per output channel:
  y0: Overshoot = 17%,  Rise (10-90%) = nan s,  Settling (2%) = 55.8 s
  y1: Overshoot = 22.5%,  Rise (10-90%) = nan s,  Settling (2%) = 89.8 s

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 1.23

### `tgen` — Turbo-generator (Maciejowski)

Source: `pidtuner/lqg_examples_m/AITGEN.m` (Maciejowski (via AITGEN.m))

nx=6, nu=2, ny=2, Q=output_weighted, R=identity, stable=True

Checks:
- [PASS] Q symmetric — max|Q-Qᵀ| = 0.00e+00
- [PASS] R symmetric — max|R-Rᵀ| = 0.00e+00
- [PASS] Q positive semi-definite — min eig(Q) = 2.46e-15
- [PASS] R positive definite — min eig(R) = 1
- [PASS] (A,B) stabilizable — all unstable/marginal modes are controllable
- [PASS] (A,√Q) detectable — all unstable/marginal modes are detectable through Q
- [PASS] S solves the Riccati equation — ‖residual‖ = 1.45e-11 (tol 1e-06 × scale 4.04e+02)
- [PASS] S symmetric — max|S-Sᵀ| = 0.00e+00
- [PASS] S positive semi-definite — min eig(S) = 0.00738
- [PASS] closed-loop poles stable — max Re(pole) = -1.21

Gain matrix K (2×6):

```
[[-0.0142, -0.2345, -1.3021, -1.0754, -1.0543, -0.6483],
 [ 0.447 ,  4.5257, -3.4562,  6.4824,  5.1872, 14.2096]]
```

Regulator step (x0 = ones): control effort and settling —
- settling (2%) = 2.52 s,  ISU (control energy) = 15.5,  |u|_peak = 27.4,  final ||x|| = 3.32e-07

Closed-loop step response (r = ones), per output channel:
  y0: Overshoot = 0%,  Rise (10-90%) = 1.68 s,  Settling (2%) = 3.52 s
  y1: Overshoot = 0%,  Rise (10-90%) = 1.65 s,  Settling (2%) = 3.27 s

Sensitivity/complementary sensitivity at the plant input (loop_point=plant_input): Ms = 1, Mt = 1.27

