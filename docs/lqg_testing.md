# LQG Design Track — Testing Guide

What the tests in `test_lqg.py`/`test_lqg_checks.py` verify, what the
pre-/post-design checks in `lqg_checks.py` mean, and how to run all of it —
companion to `docs/lqg_plan.md` (the design/scope doc) and
`docs/lqg_review.md` (a generated report over the preset catalog).

## Running things

```bash
# Automated test suite (64 tests across the two files below)
python3 -m pytest test_lqg.py test_lqg_checks.py -v

# Interactive: any plant, any method, with checks printed
python3 cli_lqg.py --plant-preset aircraft_hall --method lqg --sim output_feedback

# List all preset plants
python3 cli_lqg.py --list-plants

# One pass over every preset plant + a professor-facing report
python3 lqg_review.py    # writes docs/lqg_review.md
./examples/run_lqg_demo.sh          # broader demo, ends by calling the above
```

## `test_lqg.py`: what's validated and how

| Class under test | How it's validated |
|---|---|
| `StateSpacePlant` | Shape validation, `is_controllable`/`is_observable` (PBH test), `tf_to_state_space` round-trips a known transfer function's open-loop response to 1e-4 relative tolerance against `TransferFunction.simulate`. |
| `LQR` | **Golden value**: `aircraft_hall` = `AILQG.pdf` Example 1 (Hall-71 aircraft). The PDF prints `S`, `K`, and closed-loop poles to 4-5 digits — checked to `atol=2e-3`. |
| `OutputWeightedLQR`, `BrysonLQR` | No PDF/`.m` answer key exists for these on any preset plant, so validated structurally (Q has the expected form, closed loop is stable) plus the catalog-wide smoke tests below. |
| `LQG` | No professor `.m` file includes a Kalman filter at all (every one of the 13 files is full-state-feedback), so validated against the PDF's own algebraic identity instead (eq. 108): the poles of the full plant+estimator system equal the union of the LQR poles and the estimator poles. Checked on the double integrator and on 3 preset plants (`rpv`, `airc`, `drone`). |
| `ImplicitModelFollowing` | **Golden value**: `AILQG.pdf` Example 3 (an electro-mechanical system). `S`, `K`, and closed-loop poles match the PDF's printed values to `atol=2e-3` — same exact-match standard as `aircraft_hall`. |
| `ExplicitModelFollowing` | **Not** golden-value tested against `AILQG.pdf` Example 4 — see "The Example 4 discrepancy" below. Validated structurally instead: `S` satisfies the ARE it's supposed to solve (residual ~1e-12), is symmetric PSD, `K1`/`K2` have the right shapes, and the closed loop is stable. |
| `add_reference_tracking` | Steady-state output converges to the commanded reference (`aircraft_hall`, a square plant) to `atol=1e-3`; raises on non-square plants (`distillation_column`, 3 outputs / 2 inputs). |
| `simulate_state_feedback`, `simulate_output_feedback` | Regulator response decays to ~0, saturation clips `u` to `[u_min, u_max]`, the Kalman-filtered estimator converges from a deliberately wrong initial guess, `output_feedback` raises without a Kalman filter. |
| Preset catalog | All 11 plants: present/absent as expected, every preset's `LQR` design is stable with the right `K` shape, every preset plant is controllable (stronger than the `stabilizable` property `lqg_checks.py` actually requires — every professor-provided plant happens to be fully controllable). |

### The Example 4 discrepancy

`AILQG.pdf` Example 4 (explicit model-following) reuses Example 3's plant,
`Am`, `Q1`, and `R`. Its printed `S` (eq. 67) does **not** satisfy the
continuous-time algebraic Riccati equation it's supposed to solve:

```
residual = AᵀS + SA - SB·R⁻¹·BᵀS + Q̂
‖residual‖ for this implementation's S:  ~1e-12   (i.e. essentially exact)
‖residual‖ for the PDF's printed S:      ~14      (i.e. not a solution)
```

Both matrices are symmetric, and the entries are in the same ballpark
(differences up to ~0.05 on values around 200, and ~0.005 on the gain `K`) —
consistent with a transcription or rounding artifact in the PDF's own
printed table, not a sign bug or wrong formula on this end. Corroborating
evidence: `ImplicitModelFollowing`'s general formula (eq. 58-60) was
validated exactly against Example 3 using the *same* plant/`Am`/`Q1`/`R`,
and separately matches the general-case code in the professor's own
`AIKreindlerRothschildModelFollowingN.m` (`Qhat=(C*A-Am*C)'*Qi*(C*A-Am*C)`,
etc. — the same construction, just on a bigger augmented F-4 system). So
rather than force a match to numbers that don't satisfy their own defining
equation, `ExplicitModelFollowing` is checked against the ARE residual
directly — see `test_S_solves_the_riccati_equation` in `test_lqg.py`, and
`postcheck_are_residual` in `lqg_checks.py`, which is exactly this same
computation. **Worth flagging to the professor** — see
`docs/lqg_review.md`.

## `lqg_checks.py`: the pre-/post-design checks

These run automatically in `cli_lqg.py` (suppress with `--no-checks`) and
are exercised in isolation (good and bad inputs) plus against real design
results in `test_lqg_checks.py`. "Pre-design" and "post-design" describe
what each check expresses — an assumption the Riccati solve depends on, vs.
a property the solve should have produced — not literal call order (the
design classes compute `Q`/`R`/`N` internally before solving, so there's no
separate "propose, then solve" step to hook into; both check sets run
right after `design()` returns, using the actual `Q`/`R`/`N` it solved
with).

### Pre-design

| Check | What it catches |
|---|---|
| `Q`/`R` symmetric | `solve_continuous_are` doesn't itself verify this — an asymmetric "cost" isn't a valid quadratic form, and the "optimal" `S` returned silently loses its meaning. |
| `R` positive definite | `R⁻¹` is used directly in the gain formula (`K = R⁻¹(Nᵀ+BᵀS)`); a singular or indefinite `R` makes that inversion meaningless or blow up. |
| `Q` positive semi-definite | An indefinite `Q` can make the "cost" unbounded below, so there's no optimum to find — the ARE solver may still return *a* solution, just not one that means anything. |
| `(A,B)` stabilizable | PBH test, restricted to eigenvalues with `Re(λ) ≥ 0`. Necessary for a stabilizing solution to exist at all. Deliberately weaker than full controllability (`StateSpacePlant.is_controllable()`, which checks every eigenvalue) — LQR doesn't need to be able to move a mode that's already stable. |
| `(A,√Q)` detectable | Dual PBH test. Necessary for the ARE's stabilizing solution to also make the *closed loop* stable — an unstable mode `Q` doesn't penalize can pass through unpunished and stay unstable even though the ARE "solved." Uses a symmetric square root of `Q` (via `eigh`) since output-weighted `Q=CᵀQyC` is rank-deficient whenever `ny < nx`, so a Cholesky factor doesn't exist. |

### Post-design

| Check | What it catches |
|---|---|
| `S` solves the Riccati equation | The strongest available check: recomputes the ARE residual directly, rather than inferring correctness from downstream properties a subtly-wrong `S` could still happen to satisfy. This is the check that caught the `AILQG.pdf` Example 4 discrepancy above. |
| `S` symmetric, PSD | Any correct Riccati solution has both properties by construction; a solver numerical issue or bad conditioning shows up here first. |
| Closed-loop poles stable | `eig(A-BK)` (or `eig(A-BK1)` for explicit model-following) all have `Re(λ) < 0`, and none are `NaN`/`Inf`. |
| *(LQG only)* `P` symmetric, PSD; estimator poles stable | Same two checks, applied to the dual (Kalman) problem — `P` is a covariance, so it must be PSD; `eig(A-KfC)` must be stable for the estimate to actually converge. |

## Model-following classes (`ImplicitModelFollowing`, `ExplicitModelFollowing`)

Not currently wired into `cli_lqg.py --method` or the preset catalog sweep:
unlike `LQR`/`OutputWeightedLQR`/`BrysonLQR`/`LQG`, these need a *target
model* (`Am`, `Q1`) as an explicit design choice — there's no "suggested
Am" the way there's a `suggested_Q`/`suggested_R` per preset plant, so
running them over the catalog wouldn't have an obvious per-plant Am to use.
They're validated directly in `test_lqg.py`/`test_lqg_checks.py` against
`AILQG.pdf` Examples 3-4 instead. Wiring them into the CLI would need
`--model-A`/`--model-Q1`-style flags per invocation; flag if that's wanted.

## LLM supervisor (`cli_supervisor_lqg.py`)

A conversational layer over the same four methods, mirroring
`cli_supervisor.py`'s PID supervisor but as a separate script/session — see
`docs/cli_guide.md` "Design notes" for why it's not a mode flag on the PID
one. `supervisor_tools_lqg.run_lqg_benchmark(plant_preset, x_max=None,
u_max=None)` is the one benchmark tool: it always runs `LQR` (suggested
Q/R), `OutputWeightedLQR` (default Qy=R=I), `BrysonLQR` (x_max/u_max default
to all-ones if not given), and `LQG` (suggested Q/R + Qw=0.01I, Rv=0.1I),
against one named preset plant, and returns each row's `stable`,
`all_checks_passed` (from this same `lqg_checks.py` suite), regulator
metrics (`settling_2pct`, `ISU`, `u_peak`, `final_state_norm`), a
`pole_margin` robustness proxy (`-max(Re(closed_loop_poles))`, standing in
for the Ms/Mt this track doesn't compute for MIMO plants), and the rounded
`K` gain matrix. Model-following isn't included, same reason as above — no
auto-derivable `Am`/`Q1` per preset. Tested in `test_supervisor_lqg.py`
(scripted fake-LLM session tests, no live Ollama needed, same pattern as
`test_supervisor.py`).

## `AIExample2RTP.m` — still excluded

Held pending clarification (see `docs/lqg_plan.md` "Known issues in the
source material" and `docs/lqg_review.md`): the `lqr()` call
references undefined uppercase `A,B,C,D` (only lowercase `a,b,c,d` are
assigned, 3 states / 3 inputs), and even correcting the case mismatch, the
file's `Q` (4×4) and `R=eye(2)` don't match `a`/`b`'s dimensions.
