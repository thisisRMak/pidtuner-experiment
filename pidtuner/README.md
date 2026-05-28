# PID Tuner — ENGR105

A Tkinter desktop app for tuning PID controllers using six methods taught
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

Tuning methods (six):

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
- Anti-windup via conditional integration (integral only accumulates
  when the unsaturated command is in range).
- Optional derivative low-pass filter with `N = 10`, applied to `-pv`
  rather than to error (avoids derivative kick on setpoint steps).
  Strongly recommended; toggle off to see the unfiltered behavior.

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

## File layout

| File | Role |
|------|------|
| `plant.py` | Transfer-function parser (symbolic + MATLAB), simulation, frequency response, stability |
| `identify.py` | FOPDT step-fit, relay test, Bode-crossover ultimate gain |
| `tune.py` | Six tuning methods + `halve_gains` post-processor |
| `simulate.py` | Closed-loop PID simulation (anti-windup + D-filter), setpoint waveforms (step/ramp/pulse), performance metrics |
| `app.py` | Tkinter UI and method dispatch |
| `test_pid_tuner.py` | unittest suite — 72 tests, no Tkinter dependency, includes a benchmark summary table |

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
