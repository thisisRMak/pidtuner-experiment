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
| Preset catalog | All 12 plants: present/absent as expected, every preset's `LQR` design is stable with the right `K` shape, every preset plant is controllable (stronger than the `stabilizable` property `lqg_checks.py` actually requires — every professor-provided plant happens to be fully controllable). |

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

**Wired into `cli_lqg.py --method implicit`/`explicit`** (2026-08-02) via
`--am-diag` (desired model pole magnitudes, positive numbers, one per
output — `Am = diag(-am_diag)`) and `--Q1-scale` (default 1.0). Unlike
`LQR`/`OutputWeightedLQR`/`BrysonLQR`/`LQG`, these need a *target model*
(`Am`, `Q1`) as an explicit design choice — there's no "suggested Am" the
way there's a `suggested_Q`/`suggested_R` per preset plant, so `--am-diag`
is required (no default) rather than guessed, and the preset-catalog sweep
(`lqg_review.py`) still only covers the four methods that do have a
suggested Q/R. `ExplicitModelFollowing` additionally rejects a non-Hurwitz
`Am` at construction (`xm` is uncontrollable inside the augmented system —
eq. 51 has no `B` for the `xm` block — so an unstable target model makes
the augmented Riccati problem literally unsolvable; caught with a clear
message instead of letting `scipy` raise an opaque `LinAlgError`).

`--sim model_following` (or any `--sim` value other than `none`, for
`--method explicit`) runs `lqg_simulate.simulate_explicit_model_following`:
the augmented `[x; xm]` closed loop, `xm` evolving autonomously under `Am`
(no exogenous input, eq. 48) while the plant chases it via `u = -K1·x -
K2·xm`. `--plot` adds a third panel for `xm(t)` when present. Golden-tested
qualitatively in `test_lqg.py` (`TestSimulateExplicitModelFollowing`): both
`y` and `xm` converge to the same trajectory by the end of the simulated
window.

Both are validated directly in `test_lqg.py`/`test_lqg_checks.py` against
`AILQG.pdf` Examples 3-4 (see "The Example 4 discrepancy" above).

## Cross-method comparisons (`lqg_compare.py`)

Two separate comparisons, not one six-method table — they answer different
questions, so merging them would compare apples to oranges:

- **`compare_regulator_methods(ex, ...)`** — `LQR` (suggested Q/R) /
  `OutputWeightedLQR` / `BrysonLQR` / `LQG`, same plant, same objective
  (regulate `x` to `0` efficiently). Directly comparable on
  `settling_2pct`/`ISU`/`pole_margin` — the LQG-track analog of
  `compare.compare_all_methods` bundling all 9 PID methods. All four are
  simulated on a **shared time axis**, sized to the slowest of the four, so
  their `||x(t)||`/`||u(t)||` trajectories can be overlaid on one plot.
  `cli_lqg.py --method all` prints a text/JSON table and (`--plot`) the
  overlay; `supervisor_tools_lqg.run_lqg_benchmark`'s first four rows come
  from the same function.

- **`compare_model_following(plant, Am, Q1, R, ...)`** — `ImplicitModelFollowing`
  vs. `ExplicitModelFollowing`, same target model. A *different* objective
  ("match this target model," not "regulate efficiently"), so comparing its
  `settling_2pct` against the regulator family's would be apples-to-oranges
  — compared instead against the target model's own free response `xm_ref(t)`
  (`ẋm = Am·xm`, `xm0 = ones(nxm)`), which is literally what both designs are
  trying to resemble. Returns `(rows, (t, xm_ref))`.

  The two rows aren't simulated the same way, and that asymmetry is
  inherent, not a bug: **implicit** has no live model signal to compare in
  real time (its mechanism is algebraic — shaping the closed-loop poles
  toward `Am`'s, not tracking an `xm(t)` trajectory), so it's simulated as
  a plain regulator response (`x0 = ones(nx)`, same default as
  `compare_regulator_methods`) and its `y(t)` is what's overlaid against
  `xm_ref`. **Explicit** comes from its own tracking simulation (`x0 = 0`,
  `xm0 = ones(nxm)`), which by construction chases the identical
  `xm_ref` trajectory. Concretely (`aircraft_hall`, `am_diag=[0.1, 0.07]`):
  explicit's final `||y - xm_ref||` is ~2e-9 vs. implicit's ~1.6e-6 — three
  orders of magnitude tighter, which is the whole point of having a live
  feedforward term. `test_lqg_compare.py`'s
  `test_explicit_tracks_the_model_much_more_tightly_than_implicit` checks
  this holds, not just that both "run."

  `cli_lqg.py --method model_following_all` (`--am-diag` required, same as
  `--method implicit`/`explicit`) prints a text/JSON table and (`--plot`)
  one panel per output channel: the target model (dashed) plus both
  designs' `y(t)` overlaid. `supervisor_tools_lqg.run_lqg_benchmark`'s
  optional `am_diag` rows come from the same function.

Both `cli_lqg.py` and `supervisor_tools_lqg.py` call into `lqg_compare.py`
as the shared core (mirroring `compare.py`'s role for `cli.py`/
`supervisor_tools_whitebox.py`) and do their own presentation on top (CLI
text/plot vs. the supervisor's rounded/JSON-safe rows) — `lqg_compare.py`
itself has no plotting code, matching `compare.py`'s convention.

### Custom weights and reference-tracking on the regulator comparison

`compare_regulator_methods` also takes two independent, opt-in extensions
(2026-08-02), both needed for the LLM supervisor to actually *improve* a
design rather than only picking among four fixed weight-selection
strategies:

- **`Q_diag`/`R_diag`** (must be given together) — a 5th "Custom LQR" row,
  `LQR(plant, Q=diag(Q_diag), R=diag(R_diag))`, using caller-supplied
  per-state/per-input weights instead of the preset's suggested `Q`/`R`.
  This is the actual lever for iterating: propose weights, look at the
  resulting metrics, propose different weights based on what changed, call
  again. `cli_lqg.py --Q-diag`/`--R-diag` work on both `--method lqr`
  (replaces the suggested-Q design) and `--method all` (adds the 5th row);
  `supervisor_tools_lqg.run_lqg_benchmark`'s `Q_diag`/`R_diag` parameters
  do the same.

- **`reference`** (length `ny`, requires a square plant `nu == ny`, else a
  clear `ValueError` naming the plant) — every row is passed through
  `add_reference_tracking` and simulated tracking that constant reference
  instead of the plain zero-reference regulator, gaining
  `lqg_simulate.compute_tracking_metrics`'s sign-aware per-channel
  Overshoot/Rise/Settling (`sim.tracking_metrics`, `None` when `reference`
  isn't given). `cli_lqg.py --method all --reference-tracking --reference
  ...` prints these underneath the comparison table and switches
  `plot_regulator_comparison` from the `||x(t)||`/`||u(t)||` norm plot to
  one panel per output channel (each method's `y(t)` vs. a dashed
  reference line — see the example plot referenced in `docs/cli_guide.md`).

Combined, these directly answer "can it compare overshoot/rise/settling
across solutions and then modify Q/R accordingly": yes — e.g. on
`aircraft_hall` with `R_diag=[0.1, 0.1]` vs. the suggested `R`, channel 0's
overshoot dropped from 96% to 37% while both settling times stayed
comparable, a concrete, empirically-checked example
(`test_lqg_compare.py::test_custom_weights_change_overshoot`,
`test_supervisor_lqg.py::test_iteration_workflow_different_weights_change_metrics`).
The supervisor's system prompt (`supervisor_prompts_lqg.py`) explicitly
tells the model to iterate empirically rather than reason abstractly about
which direction to move a weight — Q/R's effect on overshoot isn't simple
or monotonic in a coupled MIMO system, as that same example shows (lower
`R` produced *less* overshoot here, not more, which is the opposite of the
naive "lower control-effort penalty → more aggressive → more overshoot"
intuition for a decoupled SISO system).

## Reference-tracking step metrics (Overshoot/Rise/Settling)

`lqg_simulate.compute_tracking_metrics(t, y, r)` — the MIMO/LQG-track
analog of `simulate.py`'s step-response metrics, computed per output
channel when `simulate_state_feedback` is given a nonzero constant `r`
(i.e. an actual reference-tracking simulation, not the plain regulator
case). **Sign-aware**, unlike `simulate.py`'s original (which silently
assumes a positive-going step via `np.max(y)`): for a negative reference,
overshoot is measured against `np.min(y)` (how far below the negative
target the response dips), not `np.max(y)`. This matters concretely here —
the LQG track's own reference-tracking test uses a mixed-sign command
(`r=[1.0, -0.5]`), exactly the case the unsigned formula gets wrong. See
`test_lqg.py::TestComputeTrackingMetrics` for a worked example of the
difference.

There's no equivalent for the plain regulator case (driving `x` to `0` has
no "final value" to overshoot past in the same sense) — `settling_2pct` in
`compute_regulator_metrics` is the metric that already covers that case,
and `sim.tracking_metrics` is `None` when `r` wasn't given.

`cli_lqg.py --reference-tracking` now actually simulates tracking (it
previously only added the `N̄` feedforward gain and printed it, without
ever driving a `--sim state_feedback` run toward a nonzero reference — a
gap closed alongside these metrics). `--reference <floats>` sets the
commanded value (default all-ones); the tracking metrics print
automatically whenever they're present.

## LLM supervisor (`cli_supervisor_lqg.py`)

A conversational layer over the same techniques, mirroring
`cli_supervisor.py`'s PID supervisor but as a separate script/session — see
`docs/cli_guide.md` "Design notes" for why it's not a mode flag on the PID
one. `supervisor_tools_lqg.run_lqg_benchmark(plant_preset, x_max=None,
u_max=None, Q_diag=None, R_diag=None, reference=None, am_diag=None,
q1_scale=1.0)` is the one benchmark tool: it always runs `LQR` (suggested
Q/R), `OutputWeightedLQR` (default Qy=R=I), `BrysonLQR` (x_max/u_max
default to all-ones if not given), and `LQG` (suggested Q/R + Qw=0.01I,
Rv=0.1I) against one named preset plant, and returns each row's `stable`,
`all_checks_passed` (from this same `lqg_checks.py` suite), regulator
metrics (`settling_2pct`, `ISU`, `u_peak`, `final_state_norm`), a
`pole_margin` robustness proxy (`-max(Re(closed_loop_poles))`, standing in
for the Ms/Mt this track doesn't compute for MIMO plants), and the rounded
gain matrix/matrices (`K`, or `K1`/`K2` for model-following rows).
Implicit and explicit model-following are included too, but only when the
caller (the LLM, on the user's behalf) supplies `am_diag` — the system
prompt instructs it to ask the user for target pole magnitudes rather than
invent them, and to only bring up model-following when the user's goal is
"match this dynamic behavior," not as a default option alongside the
regulator-family four. `Q_diag`/`R_diag` (a 5th "Custom LQR" row) and
`reference` (per-channel Overshoot/Rise/Settling on every row) are the
iteration levers — see "Custom weights and reference-tracking" above; the
system prompt instructs the model to propose weights, call the tool, look
at what changed, and propose again, empirically, rather than reasoning
abstractly about which direction to move a weight. Tested in
`test_supervisor_lqg.py` (scripted fake-LLM session tests,
no live Ollama needed, same pattern as
`test_supervisor.py`).

## `AIExample2RTP.m` — fixed, no longer excluded

Fixed 2026-08-02 (see `docs/lqg_plan.md` "Known issues in the source
material" and `docs/lqg_review.md`): the source's `lqr()` call had
referenced undefined uppercase `A,B,C,D` while only lowercase `a,b,c,d`
were assigned (3 states / 3 inputs), and even correcting the case
mismatch, the file's `Q` (4×4) and `R=eye(2)` didn't match `a`/`b`'s
dimensions. The source now uses consistent, correctly-sized `A,B,Q,R`
(custom hand-picked `Q` with off-diagonal cross terms, `R=I(3)`) and is
ported into the catalog as `example2_rtp` — the preset catalog has no
remaining exclusions.
