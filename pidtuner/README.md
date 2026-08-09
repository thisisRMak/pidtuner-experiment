# PID Tuner — ENGR105

A Tkinter desktop app for tuning PID controllers using nine methods taught
in the course, with overlay comparison of multiple tunings against the
same plant.

## Running

### Option A — conda (recommended)

```bash
conda env create -f environment.yml
conda activate pidtuner
python app.py
```

Or without the `environment.yml`, in one line:

```bash
conda create -n pidtuner python numpy scipy matplotlib -y
conda activate pidtuner
python app.py
```

Tkinter is bundled with conda's Python on all platforms — no extra step
needed (this is the easiest cross-platform path, especially on Linux
where system Python normally needs `apt install python3-tk`).

To remove the environment when you're done:

```bash
conda deactivate
conda env remove -n pidtuner
```

### Option B — pip

```bash
pip install numpy scipy matplotlib
python app.py
```

On Linux you'll also need `sudo apt install python3-tk` (Debian/Ubuntu)
or `sudo dnf install python3-tkinter` (Fedora). On Windows/macOS the
official python.org installer includes Tk.

### Run tests

No Tkinter needed — pure stdlib `unittest`:

```bash
python test_pid_tuner.py
# or:  python -m unittest test_pid_tuner -v
```

Python 3.10+ is required for either path.

## What's included

Plant input — two tabs:
- **Symbolic**: `1000/((s+1)(10s+1))`, `(s+2)/(s^2+3s+1)`, etc. Implicit
  multiplication is supported (`10s` = `10*s`, `(s+1)(s+2)` = `(s+1)*(s+2)`).
- **MATLAB form**: gain · num / den with descending-powers coefficient
  lists, the same convention as MATLAB's `tf([1],[10 11 1])`.

Both forms accept an optional dead time `L` for `G(s) · exp(-Ls)`.

Tuning methods (nine):

1. **Stable pole cancellation** — the "quick and dirty" method from
   class (HO PS4). The PID's two zeros are placed on top of two
   chosen stable plant poles; Kd is the free parameter. UI offers
   auto-pick (two slowest stable poles) or manual entry. RHP-pole
   plants are refused, per Prof. Emami: "never, ever do RHP
   cancellations".
2. **Ziegler–Nichols Method I** — reaction-curve rules. A simulated
   step test on the plant gives a fitted FOPDT (K, τ, L), and the 1942
   ZN table gives Kp, Ti, Td.
3. **Ziegler–Nichols Method II** — ultimate-gain rules. Ku and Pu can
   be obtained two ways (radio buttons):
   - *Analytical*: solve `phase(G(jω)) = −180°` from the Bode plot.
     Exact when the model is known. Refuses with a clear message if
     no finite Ku exists (e.g., pure first-order plants).
   - *Relay test*: Aström-Hägglund describing-function method.
     Simulates a hysteretic relay in closed loop, reads Ku from the
     limit-cycle amplitude and Pu from the period.
4. **AMIGO** — Aström-Hägglund robust rules from handout HO13.
   FOPDT-based, targets ~20% overshoot with a sensitivity-function
   robustness constraint. Includes the integrating-plant variant.
5. **SIMC** — Skogestad rules from handout HO14. Single tuning
   parameter τc (default τc = L is "fast and robust"). Optional τ₂
   input enables the second-order PID form.
6. **Boyd convex-concave** — ECC 2013 paper. Maximize Ki subject to
   robustness constraints |S|∞ ≤ Ms, |T|∞ ≤ Mt, formulated as a
   linearized LP that iterates to convergence. Operates on the true
   plant frequency response (not an FOPDT proxy), so it handles
   higher-order plants naturally.

7. **Cohen–Coon** — reaction-curve rules (1953) from the FOPDT fit.
   Like ZN-I but uses τ and L separately, with a dead-time correction
   that improves on ZN-I for delay-dominant plants (large L/τ). Targets
   a quarter-decay load response.

8. **Chien–Hrones–Reswick (CHR)** — 1952 refinement of ZN-I from the
   FOPDT fit, with a selector for setpoint-tracking (servo) vs.
   load-rejection (regulator) and a 0% vs. 20% overshoot target — four
   PID variants in all.

9. **Tyreus–Luyben** — the conservative ultimate-gain rule from Luyben
   & Luyben, *Essentials of Process Control* (1997). Same Ku, Pu inputs
   as ZN-II (Bode crossover or relay test) but far more damped; a PI
   variant is available. Already detuned, so no "halve gains" needed.

**"Halve gains" toggle** (between method args and the Tune button):
applies the PEI9e recommendation — divide Kp, Ki, Kd by 2 after
tuning. Intended for ZN-I/II when used for setpoint tracking (the
original rules were tuned for disturbance rejection and overshoot
heavily on steps). Exposed for *all* methods so you can experiment;
each variant gets a separate ½-labelled entry in the overlay so you
can compare full vs halved side-by-side.

Setpoint waveforms (Closed-loop simulation panel): **step**, **ramp**,
**pulse**. The three are useful for different things:
- *Step*: classic tracking — overshoot, rise time, settling time.
- *Ramp*: tests integral action — a type-1 loop has finite ss
  tracking error; type-2+ goes to zero.
- *Pulse*: tests disturbance-style rejection — amplitude during a
  finite window, residual after it ends.

Metrics displayed adapt to the chosen waveform (overshoot/rise/settle
for step, ss-error/max-error for ramp, peak-error/residual for pulse).

Closed-loop simulation:
- Setpoint, duration, actuator saturation (`u_min`, `u_max`).
- Anti-windup: **conditional-integration** (default, unchanged from
  before — integral only accumulates when the unsaturated command is
  in range) or **back-calculation** (opt-in — integral keeps
  accumulating but is corrected by `Ka*(u_sat - u_unsat)`). See
  "Anti-windup" below; neither mode has any effect unless `u_min`/`u_max`
  actually saturate the actuator.
- Optional derivative low-pass filter with `N = 10`, applied to `-pv`
  rather than to error (avoids derivative kick on setpoint steps).
  Strongly recommended; toggle off to see the unfiltered behavior.

### Anti-windup

Both modes solve the same problem: when the actuator saturates
(the unsaturated command falls outside `[u_min, u_max]`), a plain
integrator keeps accumulating error it can't act on ("integrator
windup"), producing overshoot and a slow recovery once the error
reverses sign. They differ in how they prevent it:

- **Conditional-integration** (default): freeze the integral outright
  while saturated. Simple, no extra parameter, and this project's
  original anti-windup behavior — nothing changes unless you opt into
  back-calculation.
- **Back-calculation** ("tracking" anti-windup, Åström & Hägglund):
  keep integrating, but continuously correct with the saturation error
  itself:

  ```
  dI/dt = e + Ka*(u_sat - u_unsat)
  ```

  When not saturated, `u_sat == u_unsat` and this reduces to ordinary
  integration. When saturated, the correction term actively pulls the
  integral back toward what the actuator can deliver, at a rate set by
  `Ka`, instead of merely pausing.

  `Ka` is auto-derived from whatever gains a tuning method already
  produced, via `Ka = 1/Tt`. For a full PID, `Tt = sqrt(Ti·Td)` — Åström
  & Hägglund's own stated rule of thumb (*Advanced PID Control*, Ch. 3,
  "Integrator Windup" / "Back-Calculation and Tracking"). For PI-only
  gains (`Td = 0`) that formula degenerates to `Tt = 0` (`Ka = ∞`), so
  this project falls back to `Tt = Ti` instead — that fallback is *not*
  from the textbook (the book only mentions `Tt = Ti` in an unrelated
  context, where it's explicitly called "often too large"); it's this
  codebase's own stopgap for the degenerate case. `Ti`/`Td` come from the
  gains themselves (`Ti = Kp/Ki`, `Td = Kd/Kp`). An explicit `Ka` override
  is available in both the GUI (blank = auto) and CLI (`--Ka`).

  The auto-derived `Ka` is a reasonable default, not a guarantee of
  better performance — for some plants/saturation levels back-calculation
  with the auto `Ka` can actually overshoot *more* than conditional
  integration; `Ka` is itself a design knob. Compare both modes on your
  plant rather than assuming back-calculation always wins.

This is orthogonal to which of the 9 tuning methods produced the gains —
none of them know about actuator limits, so anti-windup is purely a
property of the simulation step that follows tuning, not of tuning
itself. See `simulate.py`'s module docstring for the full derivation.

CLI: `--u-min`/`--u-max` set the saturation bounds (default: unbounded,
i.e. no saturation — unchanged from before); `--antiwindup
{conditional,back_calc}` selects the mode (default: `conditional`);
`--Ka` overrides the auto-derived gain. When saturation is requested,
the CLI prints an extra "Saturated-actuator simulation" block (and adds
a `saturated_sim` key in `--json` mode) alongside the normal
(always-unsaturated) comparison metrics, so you can see both the
idealized cross-method comparison and the actual saturated response
side by side.

Overlay plotting:
- Every successful tune is added to the session list with a unique
  color. Three stacked panels show **PV vs setpoint**, **control
  effort u(t)**, and **error e(t)** — all overlays share the same
  time axis.
- Each entry has a checkbox: untick to hide without losing it.
  "Remove unchecked" prunes the list permanently.
- Closed-loop stability is checked via the characteristic equation
  (with 2nd-order Padé for dead time). Unstable tunings are labeled
  `[UNSTABLE]` in the legend rather than silently producing a
  diverging trace.

Compare all methods:
- The **Compare all methods** button (above the single-method box) tunes
  every applicable method on the current plant at once, overlays them all
  on the response plots, and lists each in the session overlay — untick
  any to declutter.
- Output appears in the right pane as closable tabs (each with an "×",
  plus a **Close all tabs** button); the pane is empty until the first
  tune or comparison:
  - **Response** — the three stacked plots (PV/SP, u(t), e(t)) overlaying
    every shown method.
  - **Heatmap** — one row per method; columns for setpoint overshoot,
    settling time, tracking IAE, load-rejection IAE, maximum sensitivity
    Ms, complementary sensitivity Mt, and control-signal total variation
    TV(u). Every metric is lower-is-better, so each column is colored
    green (best) → red (worst).
  - **Radar** — each method as a polygon over six normalized axes
    (tracking, load rejection, robustness, low overshoot, speed,
    smoothness), scaled so the outer edge is best.
- A single **Tune & simulate** also populates all three tabs (heatmap and
  radar then reflect whatever is in the session). The metrics live in
  `compare.py`; Ms/Mt come from the loop frequency response L = C·P, and
  the load-rejection IAE from a unit step injected at the plant input.

## File layout

| File | Role |
|------|------|
| `plant.py` | Transfer-function parser (symbolic + MATLAB), simulation, frequency response, stability |
| `identify.py` | FOPDT step-fit, relay test, Bode-crossover ultimate gain |
| `tune.py` | Nine tuning methods + `halve_gains` post-processor |
| `simulate.py` | Closed-loop PID simulation (anti-windup + D-filter), setpoint waveforms (step/ramp/pulse), load-disturbance injection, performance metrics |
| `compare.py` | Cross-method comparison: robustness metrics (Ms, Mt, gain/phase margin), load-rejection IAE, and the `compare_all_methods` driver |
| `app.py` | Tkinter UI, method dispatch, and the Compare-all-methods window (heatmap table + radar) |
| `test_pid_tuner.py` | unittest suite — 90 tests, no Tkinter dependency, includes a benchmark summary table |
| `pidtuner.spec` | PyInstaller spec for building a standalone executable |
| `.github/workflows/build.yml` | GitHub Actions: builds Windows/macOS/Linux executables on every tag |

Each module is independent and can be imported and used from a script
or notebook without the UI:

```python
from plant import TransferFunction
from identify import run_step_test
from tune import tune_simc
from simulate import simulate_closed_loop, format_metrics

plant = TransferFunction.parse("1000/((s+1)(10s+1))", L=0.5)
_, _, _, _, fopdt = run_step_test(plant)
result = tune_simc(fopdt)
sim = simulate_closed_loop(plant, result.gains, setpoint=1.0)
print(result.gains.pretty())
print(format_metrics(sim.metrics))
```

## Notes on conventions

- Controllers are kept internally in **parallel form** `Kp + Ki/s + Kd·s`.
  Each tuning rule that uses the textbook `(Kp, Ti, Td)` form converts
  via `Ki = Kp/Ti`, `Kd = Kp·Td`. Both are shown in the result panel.
- Polynomials are stored in **descending powers of s**, matching scipy
  and MATLAB.
- Dead time `L` is applied as `exp(-Ls)` to the output, simulated by
  pure sample-shift after ZOH discretization (i.e. `delay = round(L/dt)`).
  For Boyd this is handled exactly in the frequency-response evaluation;
  for the closed-loop stability check it's approximated with a 2nd-order
  Padé.
