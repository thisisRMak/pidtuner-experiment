"""Entity A: the signal generator. Owns the true transfer function.

`SignalGenerator` wraps a `plant.TransferFunction` (which may include dead
time) and runs experiments against it, publishing the result as a
`signal_format.Signal` — arrays plus an explicit sample time and safe
metadata, nothing plant-shaped. This is the only place a continuous-time
ground-truth model gets discretized (via `TransferFunction.discretize`,
zero-order hold) into the sampled artifact the rest of the pipeline works
with; there is no reverse (discrete-to-continuous) step anywhere.

The private `_simulate_*` helpers are also used by `identify.run_step_test`/
`run_relay_test` so the generation logic exists in exactly one place.
"""

from __future__ import annotations

import numpy as np

from plant import TransferFunction
from signal_format import Signal


class SignalGenerator:
    def __init__(self, plant: TransferFunction):
        self.plant = plant

    # ── step test ────────────────────────────────────────────────────────
    def _simulate_step_response(self, step_amp=1.0, t_max=None, noise_sigma=0.0,
                                 seed=None, dt=None):
        plant = self.plant
        if dt is None:
            dt = plant.auto_dt()
        if t_max is None:
            poles = plant.poles()
            if len(poles):
                real_parts = np.real(poles)
                slow_decay = (1.0 / max(np.min(np.abs(real_parts[real_parts < -1e-9])), 1e-3)
                              if np.any(real_parts < -1e-9) else 10.0)
            else:
                slow_decay = 10.0
            t_max = max(10.0 * slow_decay + 5.0 * plant.L, 20.0)

        t = np.arange(0.0, t_max + dt, dt)
        # u is held at step_amp for every sample from t=0 onward — the "before
        # the step" value is what y[0]=y0 already represents in simulate(), so
        # zeroing u[0] here would double up on that and shift the effective
        # step edge one full dt late. plant.simulate()'s own `delay` handling
        # is what accounts for plant.L; this array should just be the step.
        u = np.full_like(t, step_amp)
        y_true = plant.simulate(t, u)
        if noise_sigma > 0:
            rng = np.random.default_rng(seed)
            y_meas = y_true + rng.normal(0.0, noise_sigma, size=y_true.shape)
        else:
            y_meas = y_true.copy()
        return t, u, y_true, y_meas, dt

    def step_test(self, step_amp=1.0, t_max=None, noise_sigma=0.0, seed=None,
                  dt=None) -> Signal:
        """Publish a step-test signal. Only the measured (possibly noisy)
        output is published — a real sensor never hands over the noise-free
        signal either, and shipping it would itself be a ground-truth leak.

        `dt` overrides the auto-picked sample time (plant.auto_dt(), which
        gives only L/5 samples across the dead time). Denser sampling
        (smaller dt) meaningfully improves delay-sensitive identification —
        see cli_pid.py --gen-signal --dt."""
        t, u, y_true, y_meas, dt = self._simulate_step_response(
            step_amp=step_amp, t_max=t_max, noise_sigma=noise_sigma, seed=seed, dt=dt)
        return Signal(
            t=t, u=u, y=y_meas, dt=dt, experiment="step",
            meta={"step_amp": float(step_amp), "noise_sigma": float(noise_sigma),
                  "seed": seed, "t_max": float(t[-1])},
        )

    # ── relay test ───────────────────────────────────────────────────────
    def _simulate_relay_response(self, h=1.0, setpoint=0.0, hysteresis=0.0, t_max=80.0,
                                  dt=None):
        plant = self.plant
        if dt is None:
            dt = plant.auto_dt()
        n = int(t_max / dt) + 1
        t = np.arange(n) * dt
        Ad, Bd, C, D = plant.discretize(dt)
        nx = Ad.shape[0]
        Bd_flat = Bd.flatten() if nx else np.zeros(0)
        d_scalar = float(np.atleast_2d(D).flatten()[0])
        delay = int(round(plant.L / dt))
        x = np.zeros(nx)
        y = np.zeros(n)
        u = np.zeros(n)
        relay_state = h

        for i in range(1, n):
            e = setpoint - y[i - 1]
            if e > hysteresis:
                relay_state = h
            elif e < -hysteresis:
                relay_state = -h
            u[i] = relay_state
            j = i - 1 - delay
            u_eff = u[j] if j >= 0 else 0.0
            if nx:
                x = Ad @ x + Bd_flat * u_eff
                y[i] = float((C @ x).item()) + d_scalar * u_eff
            else:
                y[i] = d_scalar * u_eff
        return t, u, y, dt

    def relay_test(self, h=1.0, setpoint=0.0, hysteresis=0.0, t_max=80.0,
                   dt=None) -> Signal:
        """Publish a closed-loop relay-feedback experiment signal."""
        t, u, y, dt = self._simulate_relay_response(
            h=h, setpoint=setpoint, hysteresis=hysteresis, t_max=t_max, dt=dt)
        return Signal(
            t=t, u=u, y=y, dt=dt, experiment="relay",
            meta={"h": float(h), "setpoint": float(setpoint),
                  "hysteresis": float(hysteresis), "t_max": float(t_max)},
        )
