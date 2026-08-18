# Derivative Filtering: When It's Applied and When It Isn't

Short reference for how PIDTuner handles the `Kd` term. See also the
module docstring in `pidtuner/pid_simulate.py` and the "Derivative filter"
section of `pidtuner/README.md`.

## The short version

The derivative is filtered **only in the time-domain simulator**
(`pidtuner/pid_simulate.py`). Every gain-selection method — Boyd loop-shaping,
pole placement/cancellation, Ziegler-Nichols/Tyreus-Luyben, and the
closed-loop stability check — works with the **unfiltered** ideal term
`Kd·s`.

| Where | Filtered? | Code |
|---|---|---|
| Closed-loop step/ramp/pulse simulation | **Yes** | `pid_simulate.py: pid_step()`, `simulate_closed_loop()` |
| Boyd loop-shaping (`L(jω)` frequency sweep) | No | `pid_tuning_methods.py:284` |
| Pole placement / pole cancellation | No | `pid_tune.py:57-63` |
| Ziegler-Nichols / Tyreus-Luyben rules | No | `pid_tuning_methods.py` (classical formulas) |
| Frequency-response comparison `C(jω)` (`compare.robustness_metrics`) | **Yes** (default `N=10`, matching sim; pass `N=0` for the ideal comparison) | `pid_compare.py: _controller_response()` |
| Closed-loop stability check (`closed_loop_poles`) | **Yes** (default `N=10`; `simulate_closed_loop` passes the run's actual `N_eff`) | `pid_simulate.py: closed_loop_poles()`, `is_closed_loop_stable()` |

Only the *gain-selection* methods (Boyd, pole placement, ZN/TL/AMIGO/SIMC/...)
still solve for `Kp, Ki, Kd` against the ideal `Kd·s` term — see "Why the
split exists" below for why that one is a harder fix than the other two.
The *verification* layer (stability check, robustness margins) now checks
what's actually simulated, filter pole included.

## Why the split exists

- **Simulation needs it.** A step setpoint is discontinuous; fed into an
  ideal differentiator `Kd·s` it produces an (numerically) infinite
  derivative on the first sample, which saturates the plotted `u(t)` and
  hides the actual design behavior. Filtering is a physical/numerical
  necessity for stepping a step function through a D-term.
- **Frequency-domain tuning doesn't hit that problem.** `C(jω) = Kp +
  Ki/(jω) + Kd·jω` is finite at every evaluated frequency point — there's
  no discontinuity to guard against, so nothing forces filtering in.
- **Adding it would break the tuning algebra.** The filtered form is
  `Kd·s/(1 + τ_d·s)` with `τ_d = Kd/(N·Kp)` — `Kd` appears in both the
  numerator and (via `τ_d`) the denominator, so it's no longer affine in
  the gains. Boyd's method specifically relies on `L(jω)` being affine in
  `(Kp, Ki, Kd)` to pose gain selection as a tractable per-frequency
  constraint problem; pole placement relies on a 2nd-order characteristic
  polynomial for its closed-form solve. Either breaks if a filter pole is
  folded in, and `τ_d` can't even be computed until `Kd`/`Kp` are already
  known — a chicken-and-egg problem for a solver that's trying to find
  them.
- **The verification layer doesn't have that excuse, and no longer takes
  it.** `closed_loop_poles()` and `compare.robustness_metrics()` aren't
  solving for `Kp/Ki/Kd` — they're checking gains that already exist, so
  there's no chicken-and-egg problem: `τ_d = Kd/(N·|Kp|)` is just a number
  once `Kp`/`Kd` are known. Both now fold `Kd·s/(1+τ_d·s)` into `C(s)`
  (`N=10` default; `simulate_closed_loop` passes its actual `N_eff` so the
  reported `stable` flag matches what was simulated). Pass `N=0` to either
  function to get the old ideal-`Kd·s` comparison. With the default
  `N=10` the filter pole sits roughly a decade past the derivative zero
  (`ω = N·Kp/Kd`), so the correction is usually small — but it stops being
  small for aggressive filtering (low `N`) or gains with small `Kp`/large
  `Kd`, which is exactly when it's worth having gotten right.

## The filter algorithm (simulation only)

Continuous form (ISA/standard filtered derivative), applied to `-pv`
rather than `-e` to avoid "derivative kick" on setpoint steps:

```
D(s) = -Kd·s / (1 + τ_d·s) · PV(s)      τ_d = Td/N = Kd/(N·Kp)
```

`N` is the filter bandwidth ratio (default `N = 10`). Discretized with
backward-Euler in `pid_step()` (`pid_simulate.py:105-147`):

```python
tau_d = gains.Kd / (N * abs(gains.Kp))     # if Kp, Kd, N all nonzero
alpha = dt / (tau_d + dt)                  # backward-Euler LPF coefficient
dpv = (pv - state.prev_pv) / dt            # raw derivative of measurement
state.d_filt += alpha * (dpv - state.d_filt)   # exponential smoothing
d_term = -gains.Kd * state.d_filt
```

`alpha = dt/(τ_d+dt)` is the exact backward-Euler discretization of the
lag `τ_d·ẏ + y = u`, so `d_filt` tracks a low-pass-filtered estimate of
`d(pv)/dt`. If `Kp≈0`, `Kd≈0`, or `N=0` (filter explicitly disabled —
`use_d_filter=False` in `simulate_closed_loop()`), `pid_step()` falls
back to a raw (unfiltered, kick-prone) derivative of error instead.

## Toggling it

- CLI/library: `simulate_closed_loop(..., N=10.0, use_d_filter=True)`.
- GUI: "Derivative filter (N=10) — recommended" checkbox, on by default
  (`pid_app.py:349-351`).
