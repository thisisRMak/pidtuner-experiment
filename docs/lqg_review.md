# LQG Design Track — Preset Catalog Review

Generated 2026-08-02. One pass of `LQR` (each plant's own suggested `Q`/`R`, i.e. the same weights your `.m` files use) over all 11 plants currently in the preset catalog (`pidtuner/lqg_examples_m/*.m` → `pidtuner/lqg_examples_json/*.json`), with the full pre-/post-design correctness check suite (see `docs/lqg_testing.md` for what each check verifies).

**Summary: 11/11 plants pass every check.**

## Source files that needed a fix before they'd run

These were excluded from the catalog until now because the `.m` file as originally provided had a bug unrelated to the plant data itself. Flagging so you can confirm the fix matches what you intended:

- **`pidtuner/lqg_examples_m/AIFurnaceModel.m`** (`furnace_model`): source had a stray trailing ')' after `lqr(A,B,Q,R)`, a plain syntax typo — fixed by removing it.
- **`pidtuner/lqg_examples_m/AIF100Engine.m`** (`f100_engine`): source's `R` was undefined at the `lqr(A,B,Q,R)` call; now `R=eye(5)`, matching nu=5 from B/D's column count.

## Still excluded, pending your input

- **`AIExample2RTP.m`** (`example2_rtp`): the lqr() call references undefined uppercase A,B,C,D (only lowercase a,b,c,d are assigned); even mapping case, Q (4x4) doesn't match a's 3 states and R=eye(2) doesn't match b's 3 inputs.

## Results

| Plant | nx/nu/ny | Q kind | Stable | Checks |
|---|---|---|---|---|
| `airc` | 5/3/3 | output_weighted | True | all pass |
| `aircraft_hall` | 5/2/2 | custom | True | all pass |
| `autm` | 12/2/2 | output_weighted | True | all pass |
| `chemical_reactor` | 4/2/4 | identity | True | all pass |
| `distillation_column` | 11/2/3 | identity | True | all pass |
| `drone` | 6/2/2 | output_weighted | True | all pass |
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

