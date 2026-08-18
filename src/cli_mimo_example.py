#!/usr/bin/env python3
"""Runnable example for the MIMO-PI track (mimo_pi.py) — Windup_AEN 7.pdf
§9.3 multivariable integral anti-windup, Phase 1.

Not a full CLI (no argparse, no --json/--plot flags like cli_pid.py/cli_lqg.py
have) — this is a quick, self-contained demonstration you can point at
directly:

    python3 cli_mimo_example.py

"PI" = Proportional-Integral: the controller is Uc = KP·E + KI·∫E dt (E =
r - y), the matrix generalization of a plain PI loop — not the LQR/LQG
track's full-state-feedback u = -Kx. See mimo_pi.py's module docstring for
the three anti-windup modes this compares.

Prints a metrics table (all three antiwindup modes under actuator
saturation) and saves examples/out/mimo_pi_antiwindup.png — per-channel
y(t) for all three modes overlaid against the reference, and u(t) with the
saturation limits marked, the MIMO analog of the SISO track's
antiwindup_conditional.png / antiwindup_backcalc.png demo plots.
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plant import StateSpacePlant
from mimo_pi import MIMOPIGains, simulate_mimo_pi, format_mimo_pi_metrics, saturation_mask


MODES = ("conditional", "resettable", "hanus")
COLORS = {"conditional": "#d55e00", "resettable": "#0072b2", "hanus": "#009e73"}


def build_example():
    """A stable 2-state, 2-input, 2-output plant with input coupling (B has
    off-diagonal terms — actuator 0 also pushes on state 1 and vice versa),
    same fixture as test_mimo_pi.py's coupled_plant(). Actuator limits
    (±0.6) are tight enough to force real, sustained saturation but loose
    enough that the loop actually desaturates and recovers within the sim
    horizon — that recovery phase is where the three modes visibly diverge
    (see test_mimo_pi.py's test_modes_diverge_when_saturated)."""
    plant = StateSpacePlant(
        A=[[-1.0, 0.0], [0.0, -2.0]],
        B=[[1.0, 0.5], [0.3, 1.0]],
        C=[[1.0, 0.0], [0.0, 1.0]],
        D=[[0.0, 0.0], [0.0, 0.0]],
    )
    gains = MIMOPIGains(KP=2.0 * np.eye(2), KI=1.0 * np.eye(2))
    u_min, u_max = [-0.6, -0.6], [0.6, 0.6]
    return plant, gains, u_min, u_max


def run_all_modes(plant, gains, u_min, u_max):
    sims = {}
    for mode in MODES:
        sims[mode] = simulate_mimo_pi(plant, gains, u_min=u_min, u_max=u_max,
                                      antiwindup=mode)
    return sims


def print_summary(sims):
    for mode, sim in sims.items():
        print(f"--- {mode} ---")
        print(f"stable={sim.stable}  saturated_samples={saturation_mask(sim).sum()}  "
              f"final y={sim.y[-1]}")
        print(format_mimo_pi_metrics(sim.metrics))
        print()


def plot_comparison(sims, u_min, u_max, path):
    ny = next(iter(sims.values())).y.shape[1]
    fig, axes = plt.subplots(ny + 1, 1, figsize=(9, 3.2 * (ny + 1)), sharex=True)

    for j in range(ny):
        ax = axes[j]
        example = next(iter(sims.values()))
        ax.plot(example.t, example.r[:, j], "k--", linewidth=1.2, label="reference")
        for mode, sim in sims.items():
            ax.plot(sim.t, sim.y[:, j], color=COLORS[mode], linewidth=1.5, label=mode)
        ax.set_ylabel(f"y{j}(t)")
        ax.legend(fontsize="small")
        ax.grid(True, alpha=0.3)
    axes[0].set_title("MIMO-PI anti-windup comparison (Windup_AEN 7.pdf §9.3)")

    ax_u = axes[-1]
    for mode, sim in sims.items():
        for k in range(sim.u.shape[1]):
            ax_u.plot(sim.t, sim.u[:, k], color=COLORS[mode], linewidth=1.0,
                     alpha=0.8 if k == 0 else 0.5,
                     label=f"{mode} (u{k})" if k == 0 else None)
    for bound in (u_min[0], u_max[0]):
        ax_u.axhline(bound, color="#666", linestyle=":", linewidth=1.0)
    ax_u.set_ylabel("u(t)")
    ax_u.set_xlabel("Time (s)")
    ax_u.legend(fontsize="small")
    ax_u.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path)
    plt.close()
    print(f"saved {path}")


if __name__ == "__main__":
    plant, gains, u_min, u_max = build_example()
    sims = run_all_modes(plant, gains, u_min, u_max)
    print_summary(sims)
    plot_comparison(sims, u_min, u_max, "examples/out/mimo_pi_antiwindup.png")
