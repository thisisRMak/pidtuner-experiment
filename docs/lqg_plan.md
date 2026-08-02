# LQG Design Track — Plan

Status: **Phase 1 and Phase 2 implemented.** `plant.py`'s `StateSpacePlant`,
`lqg_design_methods.py` (`LQR`, `OutputWeightedLQR`, `LQG`,
`add_reference_tracking`, shared infra), `lqg_bryson.py` (`BrysonLQR`),
`lqg_implicit.py` (`ImplicitModelFollowing`), `lqg_explicit.py`
(`ExplicitModelFollowing`) — those three split into their own files
(2026-08-02) specifically so they're easy to read/diff side by side,
`lqg_examples.py`/`lqg_examples_gen.py`, `lqg_simulate.py`, `cli_lqg.py`,
`lqg_checks.py` (pre-/post-design correctness checks — see
`docs/lqg_testing.md`), `lqg_review.py`,
`examples/run_lqg_demo.sh`, `test_lqg.py`, `test_lqg_checks.py`. An LLM
supervisor also ships for this track (2026-08-02): `cli_supervisor_lqg.py`,
`supervisor_session_lqg.py`, `supervisor_common_lqg.py`,
`supervisor_prompts_lqg.py`, `supervisor_tools_lqg.py`,
`test_supervisor_lqg.py`, `examples/run_supervisor_lqg_demo.sh` — a
deliberately separate script/session from the PID supervisor
(`cli_supervisor.py`), not a mode flag on it; see `docs/cli_guide.md`
"Design notes" for why, and for a unified how-to guide across both tracks'
CLIs (one-off, conversational, and batch runs).
`ImplicitModelFollowing` is golden-tested against `AILQG.pdf` Example 3
(exact match); `ExplicitModelFollowing` is validated structurally against
Example 4 instead, since the PDF's own printed numbers for that example
don't satisfy their own Riccati equation — see `docs/lqg_testing.md` "The
Example 4 discrepancy." Neither model-following class is wired into
`cli_lqg.py`'s `--method`/preset sweep yet, since they need a per-design
target model (`Am`, `Q1`) with no natural "suggested" value per preset
plant — see `docs/lqg_testing.md`. Companion to `docs/refactor.md` (the
completed PID modularization) — this document originally planned the
structure below before implementation started; it's kept as the design
record rather than rewritten past-tense throughout.

## What the professor's material actually is

`lqg_examples_m/*.m` (13 files, `rtpsystem.dat` alongside — moved from the
repo-root `121lqrexamples/` into `pidtuner/lqg_examples_m/` on 2026-08-02)
is **not** 13
parallel "tuning methods" the way the 9 PID methods are. On inspection:

- **12 of the 13 files are the same technique — plain LQR
  (`[K,S,E]=lqr(A,B,Q,R)`) — applied to 12 different real-world MIMO plants**,
  differing mainly in how `Q`/`R` are chosen.
- **1 file is a genuinely distinct technique**: `AIKreindlerRothschildModelFollowingN.m`
  (170 lines) builds implicit *and* explicit model-following controllers for
  an F-4 aircraft, with the augmented-state/`Qhat`/`Nhat`/`Rhat` algebra
  matching `AILQG.pdf` §4.1–4.2 (eq. 50–60) exactly, plus closed-loop
  simulation via `lsim`.
- **One of the 13 files is the PDF's own worked example**, which makes it a
  free regression test: `AIAircraftHall.m` = PDF **Example 1** (§3.2, the
  Hall-71 aircraft, hand-picked `Q` with cross-terms). PDF prints the
  expected `S` and `K` to 5 digits and the closed-loop poles — a
  golden-value check for the `LQR` class.
- `AIKreindlerRothschildModelFollowingN.m` builds an augmented F-4
  aircraft + actuator dynamics + command-generator system, not directly
  reproducible as a simple golden-value test — but its own code
  (`Qhat=(C*A-Am*C)'*Qi*(C*A-Am*C); N=B'*C'*Qi*(C*A-Am*C); Rhat=R+B'*C'*Qi*C*B`)
  is exactly the eq. 58-60 formula `ImplicitModelFollowing` implements,
  strong corroborating evidence beyond the PDF text alone. The actual
  golden-value validation for both model-following classes used the PDF's
  **Examples 3 and 4** instead (§4.2/§4.1, an electro-mechanical system,
  eq. 61-71) — `ImplicitModelFollowing` matches Example 3 exactly;
  `ExplicitModelFollowing` doesn't match Example 4's printed numbers (see
  `docs/lqg_testing.md` "The Example 4 discrepancy") and is validated
  structurally instead.

So the real inventory is: **one core algorithm (LQR) + a handful of
weight-selection strategies for building `Q`/`R`, one model-following
technique, and a 12-plant benchmark library** — plus everything in the PDF
that has *zero* worked example from the professor: frequency shaping (§3.3),
and the Kalman filter / output-feedback LQG (§6–7) and reference-input
augmentation (§8). See "Phasing" below for how that gap shapes scope.

### The 12 plants (citations preserved from file comments)

| File | Plant | states / inputs / outputs | Q / R pattern | Notes |
|---|---|---|---|---|
| `AIChemicalReactor1.m` | Chemical reactor (Munro) | 4 / 2 / – | `Q=I`, `R=I` | clean |
| `AIGeneric_RTP.m` | Generic RTP (loads `rtpsystem.dat`) | 15 / 5 / 5 | `Q=CᵀC`, `R=I` | needs the external data file (present, plain ASCII, `np.loadtxt`-able) |
| `AIAIRC.m` | Aircraft (Maciejowski) | 5 / 3 / 3 | `Q=CᵀC`, `R=I` | clean |
| `AIDrone.m` | Drone | 6 / 2 / 2 | `Q=CᵀC`, `R=I` | clean |
| `AIRPV.m` | RPV (Maciejowski) | 6 / 2 / 2 | `Q=CᵀC`, `R=I` | clean |
| `AITGEN.m` | Turbo-generator (Maciejowski) | 6 / 2 / 2 | `Q=CᵀC`, `R=I` | clean |
| `AIAircraftHall.m` | Aircraft (Hall) | 5 / 2 / 2 | custom hand-picked `Q`, `R=I` | **= PDF Example 1**, golden test |
| `AIF100Engine.m` | F-100 engine | 4 / 5 / 5 | `Q=CᵀC`, `R=I` | **fixed** (2026-08-02): `R=eye(5)` now matches `nu=5` from `B`/`D`'s column count — ported as `f100_engine` |
| `AIFurnaceModel.m` | Furnace (Davison 2011 / Rosenbrock) | 8 / 4 / 4 | `Q=I`, `R=0.1·I` | **fixed** (2026-08-02): stray trailing `)` removed, shapes consistent, LQR stabilizes it — ported as `furnace_model` |
| `AIAUTM.m` | AUTM | 12 / 2 / 2 | `Q=CᵀC`, `R=I` | clean |
| `AIDistillationColumn.m` | Distillation column (Davison 2011) | 11 / 2 / 3 | `Q=I`, `R=0.1·I` | clean, non-square C |
| `AIExample2RTP.m` | RTP (FPE 8e) | — | — | **still broken**: `lqr()` call references undefined uppercase `A`,`B`,`C`,`D` (only lowercase `a,b,c,d` are assigned); even mapping case, `Q` (4×4) doesn't match `a`'s 3 states and `R=eye(2)` doesn't match `b`'s 3 inputs — holding off pending clarification, see "Known issues" |

1 of the 12 is still not directly runnable as-is (`AIFurnaceModel.m` and
`AIF100Engine.m` were both fixed and folded in). **Decision: the preset
catalog ships the 11 clean plants** — see "Known issues" and "Decisions"
below.

## Design-method menu, phased by evidence

Mirrors the PID README's numbered-method table, but phased so each phase is
backed by validation data wherever validation data exists. Per project
decision, full state-feedback LQR *and* the Kalman filter/output-feedback
LQG compensator ship together in Phase 1 rather than gating the filter
behind model-following — see "Decisions" below.

**Phase 1 — LQR + LQG core**

*Full state feedback, static gain `K`:*
1. `LQR` — raw/expert path: caller supplies `Q`, `R` (and optionally the
   cross-term `N`) directly. Matches `AIChemicalReactor1`, `AIDistillationColumn`,
   `AIFurnaceModel`, `AIAircraftHall`. **Validated against PDF Example 1.**
2. `OutputWeightedLQR` — `Q = CᵀQyC`, PDF §3.4 (eq. 32–34). Matches 7 of the
   11 clean plants (`AIGeneric_RTP`, `AIAIRC`, `AIDrone`, `AIRPV`, `AITGEN`,
   `AIAUTM`, `AIF100Engine`).
3. `BrysonLQR` — `Qii = 1/x_max²`, `Rii = 1/u_max²`, PDF §3.1 eq. 27. No
   direct `.m` example, but it's the PDF's headline "give me sane defaults"
   heuristic — the LQG analog of AMIGO/SIMC's role on the PID side.

Validated with the clean plants in the preset library (see "Known issues"
below for the 1 excluded pending professor confirmation); `AIAircraftHall.m`
gives an exact numeric answer key for `LQR`.

*Output feedback (Kalman filter, separation principle), PDF §6–7 (eq. 93–111):*
4. `LQG` — Kalman filter gain `Kf`/`L` (steady-state algebraic Riccati
   solution) combined with any Phase-1 state-feedback design via the
   separation principle. **No professor `.m` file touches this** — every one
   of the 13 files assumes full state feedback — so this class is built
   straight from the PDF's equations and validated synthetically: inject
   known process/measurement noise (`Qw`, `Rv`), confirm `Kf` converges to
   the steady-state ARE solution, and confirm the closed-loop poles equal
   the union of the LQR poles and the estimator poles (eq. 108) — that
   identity is a strong, PDF-derived correctness check even without a
   numeric answer key from an external source.

**Phase 2 — Model-following (static gain on an augmented state) — implemented**
5. `ImplicitModelFollowing` (PDF §4.2, eq. 55–60) — golden-tested against
   PDF Example 3 (exact match: `S`, `K`, closed-loop poles).
6. `ExplicitModelFollowing` (PDF §4.1, eq. 50–54) — Example 4 reuses
   Example 3's plant/`Am`/`Q1`/`R`, but its printed `S` doesn't satisfy its
   own Riccati equation (see `docs/lqg_testing.md` "The Example 4
   discrepancy"), so this class is validated structurally (ARE residual,
   symmetry/PSD-ness, closed-loop stability) instead of against those
   specific printed digits.
7. `add_reference_tracking(result)` — the `N̄` feedforward for zero
   steady-state error to a constant reference (PDF §8, eq. 112–117). A
   post-processor applicable to any Phase-1 or Phase-2 result, same role as
   `halve_gains` on the PID side — not a competing method. (Shipped
   alongside Phase 1, ahead of the rest of Phase 2 — see the status note at
   the top of this doc.)

Both classes' general formula also matches
`AIKreindlerRothschildModelFollowingN.m`'s own code exactly
(`Qhat=(C*A-Am*C)'*Qi*(C*A-Am*C)`, etc., on a bigger augmented F-4 +
actuator + command-generator system) — corroborating evidence, though not
reproduced as a full golden-value test here (that system's `lsim`
step-response reproduction was judged out of scope relative to the
Examples 3/4 validation above). Was kept as Phase 2 (after the LQR/LQG
core) because it's structurally the most complex technique here (state
augmentation) and fewer examples back it, vs. the dozen backing Phase 1.

## Architecture

Same flat-file, domain-prefixed convention as `cli_blackbox.py`,
`supervisor_tools_whitebox.py`, etc. — no subpackages, matching the existing
repo style.

| File | Role | Phase |
|---|---|---|
| `plant.py` (extended) | Add `StateSpacePlant` (A,B,C,D; MIMO, `nx`/`nu`/`ny` independent). `TransferFunction.to_state_space()` for the SISO case | 1 |
| `lqg_design_methods.py` | `BaseControlDesignMethod` (new, independent hierarchy — not force-unified with `BaseTuningMethod`) → `LQR`, `OutputWeightedLQR`, `LQG` (Kalman filter + separation principle), plus `add_reference_tracking`. Also holds the result types (`StateFeedbackGains`, `LQGDesignResult`, `KalmanFilterResult`) and the shared Riccati solvers (`_lqr_core`, `_kalman_core`) — a separate `lqg_gains.py` was originally planned here but the split wasn't worth a second file in practice; they live alongside the design classes instead | 1 |
| `lqg_examples.py` + `lqg_examples_json/*.json` | The plant preset catalog (11 clean plants; 1 excluded pending professor confirmation — see "Known issues"), loadable by name (`--plant-preset aircraft_hall`), each with citation + quirk notes in metadata; source `.m` files live in `lqg_examples_m/` | 1 |
| `lqg_simulate.py` | State-feedback and output-feedback (Kalman-filtered) closed-loop sim: `ẋ = (A-BK)x`, MIMO trajectories, actuator saturation (plain clip — no back-calc-style anti-windup yet, that's PID-specific machinery) | 1 |
| `cli_lqg.py` | argparse CLI, same shape as `cli.py` (flat script, matching current per-mode convention — no unified dispatcher yet) | 1 |
| `lqg_checks.py` | Pre-/post-design correctness checks (Q/R well-posedness, stabilizability, detectability, ARE residual, S/P symmetric-PSD, closed-loop stability) — printed by `cli_lqg.py`, see `docs/lqg_testing.md` | 1 |
| `lqg_bryson.py` — **done** | `BrysonLQR` — split out of `lqg_design_methods.py` (2026-08-02) into its own file, alongside `lqg_implicit.py`/`lqg_explicit.py` below, specifically so the three "how do I pick Q/R" methods are easy to read/diff side by side; all three still subclass `BaseControlDesignMethod` from `lqg_design_methods.py` | 1 |
| `lqg_implicit.py` — **done** | `ImplicitModelFollowing` | 2 |
| `lqg_explicit.py` — **done** | `ExplicitModelFollowing` + `ExplicitModelFollowingResult` (a distinct result type, not `LQGDesignResult`, since its control law is `u=-K1x-K2xm`, not a single `u=-Kx`) | 2 |
| `lqg_simulate.py` (cont.) — **not done** | Command-generator / augmented-state simulation (`lsim`-equivalent) for `ExplicitModelFollowing` — the design/gain computation is implemented and tested, but there's no `simulate_explicit_model_following()` yet | 2 |
| `lqg_review.py` — **done** | One-pass LQR sweep + full check suite over the preset catalog, written to `docs/lqg_review.md`/JSON for external review | — |
| `compare.py` (extended) | Generalize `Ms`/`Mt` to MIMO via singular values (SISO formulas are the scalar special case); `ISU=∫u²dt` already matches the LQ cost's control term | whenever cross-method comparison is wanted, likely alongside or after Phase 2 |

`identify.py`/`blackbox.py`-equivalent work for LQG (subspace identification
— N4SID/ERA — to get `(A,B,C,D)` from I/O data instead of a known model) is
explicitly **out of scope for this plan**, same status as the web GUI:
noted as a future direction, not designed here.

## CLI vs. GUI — recommendation: CLI only, for now

Everything in Phases 1–2 is expressible as CLI args/config + saved plots +
JSON output, with nothing lost relative to a GUI:

- **The PID GUI's value was low-dimensional widgets** (a slider for `τc`, a
  couple of pole entries) over a **small, fixed menu of named recipes**.
  LQG's actual parameter space is `Q`/`R`/`N` **matrices**, sized by however
  many states the plant has (up to 15 in `rtpsystem.dat`) — a form full of
  spinboxes for a 15×15 diagonal doesn't reduce cognitive load the way the
  PID panels did; a config file or Python call is a more natural interface
  for matrix input than Tkinter widgets.
- **The actual workflow the professor's examples demonstrate** is "pick a
  plant, pick a weight-selection strategy, get `K` + a response plot" — that
  is exactly `cli_lqg.py --plant-preset aircraft_hall --method output_weighted
  --plot out.png`, with matplotlib already forced to the non-interactive
  `Agg` backend in `cli.py` precedent (save-to-file works fine headless).
- **Comparison/heatmap value survives without a GUI shell** — `compare.py`
  already produces JSON rows and PNG figures from `cli.py --method all
  --json`; the same pattern extends to LQG once its `compare.py` extension
  lands.
- The **LLM supervisor is CLI-only** on both tracks (`cli_supervisor.py`
  for PID, `cli_supervisor_lqg.py` for this one) and specifically proves the
  "which design is better for my priorities" conversational workflow
  doesn't need a GUI at all.
- Web GUI + stack are explicitly undecided; sinking Tkinter effort into a
  parameter shape (matrices) that's a poor fit for form widgets, right
  before a possible web rewrite, is the wrong moment to build one.

**Revisit trigger:** build a GUI (Tkinter or web, TBD) only once real usage
shows a recurring interactive need the CLI can't serve — e.g., live
slider-driven `Q`/`R` sensitivity exploration for a teaching demo. Until
then, `app.py` stays PID-only; no LQG tab is planned in this phase.

## Validation strategy

- `AIAircraftHall.m` → golden test for `LQR`: same `A,B,C,D,Q,R` as PDF
  Example 1, compare `K`/`S`/closed-loop poles to the PDF's printed 5-digit
  values.
- PDF Examples 3/4 (electro-mechanical system, eq. 61-71) → golden test for
  `ImplicitModelFollowing` (exact match) and structural validation (ARE
  residual, not printed-digit match) for `ExplicitModelFollowing` — see
  "The Example 4 discrepancy" in `docs/lqg_testing.md`.
  `AIKreindlerRothschildModelFollowingN.m`'s own code corroborates the same
  general formula on a bigger augmented F-4 system, but wasn't reproduced
  as a full golden-value test (out of scope — see `docs/lqg_testing.md`
  "Model-following classes").
- The other 10 clean plants: no independent answer key (nobody has run these
  in MATLAB to record expected `K`), so they serve as smoke-test/regression
  fixtures (LQR converges, closed-loop stable, `K` has the right shape) and
  as the demo catalog for the CLI, not as numeric golden values.
- The `LQG` Kalman-filter class has no source data at all — validate
  synthetically against the PDF's own algebraic identities (separation
  principle, eq. 108).
- `lqg_checks.py`'s pre-/post-design checks (Q/R well-posedness,
  stabilizability, detectability, ARE residual, symmetric-PSD, closed-loop
  stability — see `docs/lqg_testing.md`) run over every design in
  `cli_lqg.py` by default, and pass for all 11 preset plants (see
  `docs/lqg_review.md`).

## Known issues in the source material

Originally 3 of the 12 plant files didn't run as-is; 2 have since been
corrected and ported. 1 remains broken:
- `AIExample2RTP.m` — the `lqr()` call references undefined uppercase
  `A`,`B`,`C`,`D`; only lowercase `a,b,c,d` are assigned (3 states, 3
  inputs). Even correcting the case mismatch, the file's `Q` is 4×4 (needs
  3×3) and `R=eye(2)` (needs 3×3) — still effectively a stub, not a
  transcription-ready example. Holding off — needs the professor's/user's
  input on what `Q`/`R` were actually intended, not a guess.
- ~~`AIF100Engine.m` — `R` undefined~~ **fixed 2026-08-02**: `R=eye(5)` now
  matches `nu=5` (from `B`'s and `D`'s column counts); LQR stabilizes it
  (it's already open-loop stable) and it's fully controllable. Ported as
  `f100_engine` (4 states, 5 inputs — over-actuated).
- ~~`AIFurnaceModel.m` — stray trailing `)`~~ **fixed 2026-08-02**: parses
  and stabilizes cleanly, ported as `furnace_model` (Davison 2011 /
  Rosenbrock, 8 states / 4 inputs / 4 outputs).

**Decision: check with the professor before porting the remaining one.**
The preset catalog now ships 11 clean, directly-runnable plants
(`AIChemicalReactor1`, `AIGeneric_RTP`, `AIAIRC`, `AIDrone`, `AIRPV`,
`AITGEN`, `AIAircraftHall`, `AIAUTM`, `AIDistillationColumn`,
`AIFurnaceModel`, `AIF100Engine`). `AIExample2RTP.m` is added once a
corrected original is available, rather than guessing at the
undefined/mismatched `Q`/`R`/`A`/`B` values — a reconstructed version
(closer to a stub than a plant right now) would misrepresent what's
actually a professor-provided example.

## Decisions

Resolved before implementation starts:

1. **Branding**: keep "pidtuner" as the umbrella package/repo name. LQG
   ships as new files inside the existing project — no renaming, no
   duplication of the reusable infra (`supervisor_*`, `compare.py`,
   `widgets.py`, CI/PyInstaller scaffolding). Confirmed via `grep` that no
   code imports anything as `pidtuner.X` (flat sibling-module imports
   throughout), so this costs nothing now and a rename stays cheap later if
   the tool's identity shifts.
2. **CLI shape**: flat per-mode scripts, matching `cli.py`/`cli_blackbox.py`/
   `cli_supervisor.py`/`cli_astrom_batch.py`. `cli_lqg.py` joins them as its
   own script. A unified `pidtuner` subcommand dispatcher is explicitly
   parked, not ruled out — revisit once more modes exist.
3. **Broken example files**: excluded from the initial preset catalog
   pending professor confirmation (see "Known issues" above).
4. **LQG phasing**: the Kalman filter / output-feedback LQG compensator
   ships in Phase 1 alongside full-state-feedback LQR, not deferred behind
   model-following — see the Phase 1 description above for how it's
   validated without a professor-provided answer key.
