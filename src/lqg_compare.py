"""Cross-method comparison for the LQR/LQG design track — the LQG-track
analog of pid_compare.py. Pure data/logic, no plotting (same convention
pid_compare.py follows — cli_pid.py owns all matplotlib code; cli_lqg.py does too).

Two comparisons, not one, because they answer different questions:
  - compare_regulator_methods: LQR / OutputWeightedLQR / BrysonLQR / LQG,
    same plant, same objective (regulate x to 0 efficiently) — directly
    comparable on ISU/settling_2pct/pole_margin, the regulator-family
    metrics lqg_simulate.py already computes. Mirrors
    compare.compare_all_methods bundling all 9 PID methods behind one call.
  - compare_model_following: ImplicitModelFollowing vs
    ExplicitModelFollowing, same target model (Am, Q1) — a different
    objective ("match this target model", not "regulate efficiently"), so
    comparing its settling_2pct against the regulator family's would be
    apples-to-oranges. Compared instead against the model's own free
    response xm_ref(t), which is what each design is actually trying to
    resemble — that comparison is the point, not a shared metric table.

Used by both cli_lqg.py (--method all / model_following_all) and
supervisor_tools_lqg.py (run_lqg_benchmark) — the shared core, same role
pid_compare.py plays for cli_pid.py/supervisor_tools_whitebox_pid.py. Each caller does
its own presentation on top (CLI text/plot, or LLM-safe rounded JSON).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import StateSpace, lsim

from lqg_design_methods import LQR, OutputWeightedLQR, LQG, add_reference_tracking
from lqg_bryson import BrysonLQR
from lqg_implicit import ImplicitModelFollowing
from lqg_explicit import ExplicitModelFollowing
from lqg_checks import checks_for_result
from lqg_simulate import simulate_state_feedback, simulate_explicit_model_following, auto_t_end


@dataclass
class ComparisonRow:
    name: str
    result: object   # LQGDesignResult or ExplicitModelFollowingResult
    sim: object       # LQGSimResult
    checks: dict      # checks_for_result(result) — {"pre": [...], "post": [...], ...}


def compare_regulator_methods(ex, x_max=None, u_max=None, Qy_scale=1.0, R_scale=1.0,
                              Qw_scale=0.01, Rv_scale=0.1, Q_diag=None, R_diag=None,
                              reference=None, t_end=None, dt=0.01) -> list:
    """LQR (suggested Q/R) / OutputWeightedLQR / BrysonLQR / LQG (suggested
    Q/R + Qw/Rv), all simulated as a regulator response (x0=ones(nx)) on a
    shared time axis sized to the slowest of the four — so their
    trajectories can be overlaid on one plot. Returns a list of 4
    ComparisonRow (5 if Q_diag/R_diag given), in that fixed order.

    Q_diag/R_diag: optional custom weights (one value per state/input) --
    given together, adds a 5th "Custom LQR" row using LQR(plant,
    Q=diag(Q_diag), R=diag(R_diag)) instead of the preset's suggested Q/R.
    The lever a caller (human or an LLM supervisor) needs to actually
    iterate on a design based on observed metrics, rather than only
    choosing among the four fixed weight-selection strategies.

    reference: optional constant reference command (length ny) -- when
    given, every row is passed through add_reference_tracking and
    simulated tracking it, adding sign-aware Overshoot/Rise/Settling
    per-channel metrics (lqg_simulate.compute_tracking_metrics) to each
    row's sim.tracking_metrics instead of the plain regulator response.
    Requires a square plant (nu == ny) -- add_reference_tracking's own
    constraint, checked here with a clearer message naming the plant."""
    plant = ex.plant
    Q, R = ex.build_suggested_Q(), ex.build_suggested_R()
    x_max_ = np.ones(plant.nx) if x_max is None else np.asarray(x_max, dtype=float)
    u_max_ = np.ones(plant.nu) if u_max is None else np.asarray(u_max, dtype=float)

    designs = [
        ("LQR (suggested Q/R)", LQR(plant, Q=Q, R=R).design()),
        ("Output-weighted LQR",
         OutputWeightedLQR(plant, Qy=Qy_scale * np.eye(plant.ny), R=R_scale * np.eye(plant.nu)).design()),
        ("Bryson's rule", BrysonLQR(plant, x_max=x_max_, u_max=u_max_).design()),
        ("LQG (Kalman filter)",
         LQG(plant, Q=Q, R=R, Qw=Qw_scale * np.eye(plant.nx), Rv=Rv_scale * np.eye(plant.ny)).design()),
    ]

    if Q_diag is not None or R_diag is not None:
        if Q_diag is None or R_diag is None:
            raise ValueError("Q_diag and R_diag must be given together")
        Q_diag_ = np.asarray(Q_diag, dtype=float)
        R_diag_ = np.asarray(R_diag, dtype=float)
        if Q_diag_.shape != (plant.nx,):
            raise ValueError(f"Q_diag must have {plant.nx} entries (one per state), "
                             f"got {len(Q_diag_)}")
        if R_diag_.shape != (plant.nu,):
            raise ValueError(f"R_diag must have {plant.nu} entries (one per input), "
                             f"got {len(R_diag_)}")
        designs.append(("Custom LQR (Q_diag/R_diag)",
                        LQR(plant, Q=np.diag(Q_diag_), R=np.diag(R_diag_)).design()))

    if reference is not None:
        if plant.nu != plant.ny:
            raise ValueError(
                f"reference-tracking requires a square plant (nu == ny); "
                f"{ex.name} has nu={plant.nu}, ny={plant.ny}"
            )
        reference_ = np.asarray(reference, dtype=float)
        if reference_.shape != (plant.ny,):
            raise ValueError(f"reference must have {plant.ny} entries (one per output), "
                             f"got {len(reference_)}")
        designs = [(name, add_reference_tracking(result)) for name, result in designs]

    if t_end is None:
        t_end = max(auto_t_end(result.closed_loop_poles) for _, result in designs)
    t = np.arange(0.0, t_end + dt, dt)

    rows = []
    for name, result in designs:
        if reference is not None:
            r_arr = np.tile(reference_, (len(t), 1))
            sim = simulate_state_feedback(result, t, r=r_arr)
        else:
            sim = simulate_state_feedback(result, t)
        rows.append(ComparisonRow(name=name, result=result, sim=sim, checks=checks_for_result(result)))
    return rows


def model_reference_response(Am, xm0, t):
    """The target model's own free response, ẋm = Am·xm — what both
    model-following designs are actually trying to make the plant's output
    resemble. Not a ComparisonRow (no plant/checks/gains involved), just
    the (t, xm) trajectory to overlay against each design's y(t)."""
    nxm = Am.shape[0]
    sys = StateSpace(Am, np.zeros((nxm, 1)), np.eye(nxm), np.zeros((nxm, 1)))
    _, _, xm = lsim(sys, U=np.zeros((len(t), 1)), T=t, X0=xm0)
    xm = np.atleast_2d(xm)
    if xm.shape[0] != len(t) and xm.shape[1] == len(t):
        xm = xm.T
    return xm


def compare_model_following(plant, Am, Q1, R, t_end=None, dt=0.01):
    """ImplicitModelFollowing vs ExplicitModelFollowing, same (Am, Q1, R) —
    a comparison of *how well each one's closed loop resembles the target
    model*, not a shared efficiency metric. Returns
    (rows, (t, xm_reference)): rows is [implicit_row, explicit_row], and
    xm_reference is Am's own free response (from xm0=ones(nxm)) for the
    caller to overlay against each row's sim.y.

    Implicit has no live model signal to compare in real time (its whole
    mechanism is algebraic — shaping the closed-loop poles toward Am's, not
    tracking an xm(t) trajectory) — so implicit is simulated as a plain
    regulator response (x0=ones(nx), same default as
    compare_regulator_methods) and its y(t) is what's overlaid against
    xm_reference. Explicit's y(t) comes from its own tracking simulation
    (x0=0, xm0=ones(nxm) — see simulate_explicit_model_following), which by
    construction chases the identical xm_reference trajectory (both use
    xm0=ones(nxm))."""
    implicit_result = ImplicitModelFollowing(plant, Am=Am, Q1=Q1, R=R).design()
    explicit_result = ExplicitModelFollowing(plant, Am=Am, Q1=Q1, R=R).design()

    if t_end is None:
        t_end = max(auto_t_end(implicit_result.closed_loop_poles),
                   auto_t_end(explicit_result.closed_loop_poles, extra_poles=np.linalg.eigvals(Am)))
    t = np.arange(0.0, t_end + dt, dt)

    implicit_sim = simulate_state_feedback(implicit_result, t)
    explicit_sim = simulate_explicit_model_following(explicit_result, t)

    rows = [
        ComparisonRow(name="Implicit model-following", result=implicit_result,
                      sim=implicit_sim, checks=checks_for_result(implicit_result)),
        ComparisonRow(name="Explicit model-following", result=explicit_result,
                      sim=explicit_sim, checks=checks_for_result(explicit_result)),
    ]
    nxm = Am.shape[0]
    xm_ref = model_reference_response(Am, np.ones(nxm), t)
    return rows, (t, xm_ref)


def default_model_am(plant, speedup=2.0):
    """Auto-derived target model Am for compare_bryson_output_modelfollowing
    below, when the caller hasn't supplied one — there's no textbook
    "suggested Am" the way there's a suggested Q/R (AILQG.pdf's own worked
    examples always hand-pick Am). Built as a diagonal (ny, ny) target:
    -speedup/tau_dominant on every channel, where tau_dominant is the time
    constant of the plant's slowest *stable* open-loop pole (1.0s if none
    are stable) — i.e. "aim for a decoupled, first-order response about
    `speedup`x faster than the plant's own slowest dynamics."

    This is a convenience default, not a validated design choice — flagged
    as a follow-up to confirm with Prof Emami-Naeini whether it's the right
    behavior before it's relied on for the paper."""
    poles = np.linalg.eigvals(plant.A)
    stable_rates = -np.real(poles[np.real(poles) < -1e-9])
    tau_dominant = 1.0 / np.min(stable_rates) if len(stable_rates) else 1.0
    return -(speedup / tau_dominant) * np.eye(plant.ny)


def compare_bryson_output_modelfollowing(ex, x_max=None, u_max=None, Qy_scale=1.0,
                                         R_scale=1.0, Am=None, Q1_scale=1.0,
                                         t_end=None, dt=0.01):
    """The LQG-UI "4-curve" comparison from the 2026-08-18 meeting notes
    (item 5: "step response ... 3 curves, Bryson, Implicit/Explicit, Output
    weights") — Bryson's rule / Output-weighted LQR / Implicit
    model-following / Explicit model-following, all simulated as a
    regulator response (x0=ones(nx)) on a shared time axis. A different
    bundle than compare_regulator_methods (LQR/OutputWeighted/Bryson/LQG)
    or compare_model_following (Implicit/Explicit only) above — this one
    exists specifically for that meeting-notes request, not as a
    general-purpose comparison.

    Am: target model dynamics for Implicit/Explicit, shape (ny, ny). If
    None, auto-derived via default_model_am(plant) — see its docstring;
    this is a convenience default, not a validated design choice.

    Returns (rows, Am_used, (t, xm_ref)): rows is a list of 4 ComparisonRow
    in fixed order [Bryson, Output-weighted, Implicit, Explicit]; Am_used is
    whichever Am ended up being used; (t, xm_ref) is the target model's own
    free response for plotting a dashed reference line, same shape as
    compare_model_following's second return value."""
    plant = ex.plant
    x_max_ = np.ones(plant.nx) if x_max is None else np.asarray(x_max, dtype=float)
    u_max_ = np.ones(plant.nu) if u_max is None else np.asarray(u_max, dtype=float)
    Am_used = default_model_am(plant) if Am is None else np.atleast_2d(np.asarray(Am, dtype=float))
    Q1 = Q1_scale * np.eye(plant.ny)
    R = R_scale * np.eye(plant.nu)

    designs = [
        ("Bryson's rule", BrysonLQR(plant, x_max=x_max_, u_max=u_max_).design()),
        ("Output-weighted LQR",
         OutputWeightedLQR(plant, Qy=Qy_scale * np.eye(plant.ny), R=R_scale * np.eye(plant.nu)).design()),
        ("Implicit model-following", ImplicitModelFollowing(plant, Am=Am_used, Q1=Q1, R=R).design()),
    ]
    explicit_result = ExplicitModelFollowing(plant, Am=Am_used, Q1=Q1, R=R).design()

    if t_end is None:
        t_end = max(
            *(auto_t_end(result.closed_loop_poles) for _, result in designs),
            auto_t_end(explicit_result.closed_loop_poles, extra_poles=np.linalg.eigvals(Am_used)),
        )
    t = np.arange(0.0, t_end + dt, dt)

    rows = []
    for name, result in designs:
        sim = simulate_state_feedback(result, t)
        rows.append(ComparisonRow(name=name, result=result, sim=sim, checks=checks_for_result(result)))
    explicit_sim = simulate_explicit_model_following(explicit_result, t)
    rows.append(ComparisonRow(name="Explicit model-following", result=explicit_result,
                              sim=explicit_sim, checks=checks_for_result(explicit_result)))

    nxm = Am_used.shape[0]
    xm_ref = model_reference_response(Am_used, np.ones(nxm), t)
    return rows, Am_used, (t, xm_ref)
