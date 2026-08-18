"""Pre- and post-design correctness checks for the LQR/LQG design track —
see docs/lqg_testing.md for the full explanation of what each check means
and why it's here.

"Pre-design" checks validate the assumptions the Riccati solve depends on
(Q/R well-posedness, stabilizability, detectability) using the *actual*
Q/R/N a design method built (e.g. OutputWeightedLQR's Q=CᵀQyC, or
ImplicitModelFollowing's derived Q̂/R̂) — not the raw plant. In practice
these run right after design(), same as the post-design checks, since the
design classes compute Q/R/N internally before solving and don't expose a
separate "propose Q/R, then solve" step; the pre/post split is about what
each check expresses (an assumption the solve depends on vs. a property the
solve should have produced), not literal call order.

Every check returns a `CheckResult(name, passed, detail)` so callers (the
CLI, tests, notebooks) can act on pass/fail programmatically instead of
re-parsing printed text.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str


def _symmetric_residual(M):
    M = np.atleast_2d(M)
    return float(np.max(np.abs(M - M.T))) if M.size else 0.0


def _pbh_unstable_modes(A, matrix_for_lambda, tol=1e-8):
    """Popov-Belevitch-Hautus test, restricted to eigenvalues with
    Re(λ) >= -tol. Stabilizability/detectability only require this —
    already-stable modes don't need to be controllable/observable for LQR
    to still produce a stabilizing, well-behaved solution. (Contrast with
    plant.py's is_controllable()/is_observable(), which check every
    eigenvalue — the stronger property needed for full pole placement.)
    Returns the list of failing eigenvalues (empty = check passed)."""
    nx = A.shape[0]
    bad = []
    for lam in np.linalg.eigvals(A):
        if np.real(lam) < -tol:
            continue
        M = matrix_for_lambda(lam)
        if np.linalg.matrix_rank(M, tol=1e-8) < nx:
            bad.append(lam)
    return bad


# ─────────────────────────────────────────────────────────────────────────────
# Pre-design checks
# ─────────────────────────────────────────────────────────────────────────────

def precheck_Q_R(Q, R, tol=1e-8, q_label="Q", r_label="R"):
    """Q, R symmetric; R positive definite; Q positive semi-definite — the
    minimum well-posedness conditions for xᵀQx + uᵀRu to be a valid
    quadratic cost. An indefinite Q/R still lets solve_continuous_are run
    (it doesn't itself check this), but the "optimal" S it returns loses
    its stability/optimality guarantees — this is the check that would have
    caught that silently-wrong-answer failure mode. q_label/r_label rename
    the checks (e.g. "Qw"/"Rv" for the Kalman/dual problem) so printed
    output isn't ambiguous about which matrix is being checked."""
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    results = []

    q_sym = _symmetric_residual(Q)
    results.append(CheckResult(f"{q_label} symmetric", q_sym < tol,
                               f"max|{q_label}-{q_label}ᵀ| = {q_sym:.2e}"))

    r_sym = _symmetric_residual(R)
    results.append(CheckResult(f"{r_label} symmetric", r_sym < tol,
                               f"max|{r_label}-{r_label}ᵀ| = {r_sym:.2e}"))

    q_eigs = np.linalg.eigvalsh((Q + Q.T) / 2)
    results.append(CheckResult(f"{q_label} positive semi-definite", bool(np.all(q_eigs >= -tol)),
                               f"min eig({q_label}) = {float(np.min(q_eigs)):.3g}"))

    r_eigs = np.linalg.eigvalsh((R + R.T) / 2)
    results.append(CheckResult(f"{r_label} positive definite", bool(np.all(r_eigs > tol)),
                               f"min eig({r_label}) = {float(np.min(r_eigs)):.3g}"))
    return results


def precheck_stabilizability(A, B, tol=1e-8, label="(A,B)"):
    """(A, B) stabilizable: PBH rank([A-λI, B]) == nx for every eigenvalue
    with Re(λ) >= 0. Necessary for a stabilizing LQR solution to exist at
    all; weaker than full controllability, which isn't required — LQR
    doesn't need to be able to move already-stable modes. `label` renames
    the pair in the printed check name (e.g. "(Aᵀ,Cᵀ)" for the Kalman dual
    problem, where this same test is applied to the estimator's own A/C)."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    bad = _pbh_unstable_modes(A, lambda lam: np.hstack([A - lam * np.eye(A.shape[0]), B]), tol)
    passed = len(bad) == 0
    detail = ("all unstable/marginal modes are controllable" if passed else
              f"uncontrollable unstable/marginal mode(s): {bad}")
    return CheckResult(f"{label} stabilizable", passed, detail)


def precheck_detectability(A, Q, tol=1e-8, label="(A,√Q)", q_label="Q"):
    """(A, √Q) detectable: PBH rank([A-λI; √Q]) == nx for every eigenvalue
    with Re(λ) >= 0. Necessary for the ARE's stabilizing solution to also
    make the *closed loop* stable — an unstable mode Q doesn't penalize can
    remain unstable in closed loop even though the ARE "solved." Uses a
    symmetric matrix square root of Q (via eigh) rather than a Cholesky
    factor since Q may be rank-deficient (only PSD, not PD) — e.g. every
    output-weighted Q=CᵀQyC here is rank <= ny < nx whenever ny < nx."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    eigvals, eigvecs = np.linalg.eigh((Q + Q.T) / 2)
    eigvals = np.clip(eigvals, 0, None)
    Qsqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
    bad = _pbh_unstable_modes(A, lambda lam: np.vstack([A - lam * np.eye(A.shape[0]), Qsqrt]), tol)
    passed = len(bad) == 0
    detail = (f"all unstable/marginal modes are detectable through {q_label}" if passed else
              f"undetectable unstable/marginal mode(s): {bad} ({q_label} doesn't "
              f"penalize them, so stability of the closed loop isn't guaranteed)")
    return CheckResult(f"{label} detectable", passed, detail)


def run_prechecks(A, B, Q, R, q_label="Q", r_label="R",
                  stabilizability_label="(A,B)", detectability_label="(A,√Q)"):
    checks = precheck_Q_R(Q, R, q_label=q_label, r_label=r_label)
    checks.append(precheck_stabilizability(A, B, label=stabilizability_label))
    checks.append(precheck_detectability(A, Q, label=detectability_label, q_label=q_label))
    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Post-design checks
# ─────────────────────────────────────────────────────────────────────────────

def postcheck_are_residual(A, B, Q, R, S, N=None, tol=1e-6):
    """The strongest available correctness check: does S actually satisfy
    the algebraic Riccati equation it's supposed to solve?
        AᵀS + SA - (SB+N)R⁻¹(BᵀS+Nᵀ) + Q ≈ 0
    This catches solver numerical failures directly, rather than inferring
    correctness from downstream properties (stability, PSD-ness) that a
    subtly-wrong S could still happen to satisfy. This is also the check
    that caught the discrepancy in AILQG.pdf's own Example 4 (see
    docs/lqg_testing.md) — residual ~1e-12 for this implementation's S vs.
    ~14 for the PDF's printed table."""
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    nx, nu = A.shape[0], B.shape[1]
    Nmat = np.zeros((nx, nu)) if N is None else np.atleast_2d(np.asarray(N, dtype=float))
    Rinv = np.linalg.inv(R)
    resid = A.T @ S + S @ A - (S @ B + Nmat) @ Rinv @ (B.T @ S + Nmat.T) + Q
    resid_norm = float(np.linalg.norm(resid))
    scale = max(float(np.linalg.norm(Q)), float(np.linalg.norm(S)), 1.0)
    passed = resid_norm < tol * scale
    return CheckResult("S solves the Riccati equation", passed,
                       f"‖residual‖ = {resid_norm:.2e} (tol {tol:.0e} × scale {scale:.2e})")


def postcheck_symmetric_psd(M, label, tol=1e-6):
    """Shared by S (state-feedback) and P (Kalman covariance) — both must
    be symmetric PSD by construction if the solve is correct; a solver bug
    or a badly-conditioned problem shows up here first."""
    scale = max(float(np.max(np.abs(M))), 1.0)
    sym_resid = _symmetric_residual(M)
    sym_ok = sym_resid < tol * scale
    eigs = np.linalg.eigvalsh((M + M.T) / 2)
    psd_ok = bool(np.all(eigs >= -tol * scale))
    return [
        CheckResult(f"{label} symmetric", sym_ok, f"max|{label}-{label}ᵀ| = {sym_resid:.2e}"),
        CheckResult(f"{label} positive semi-definite", psd_ok,
                   f"min eig({label}) = {float(np.min(eigs)):.3g}"),
    ]


def postcheck_poles_stable(poles, label, tol=1e-9):
    poles = np.asarray(poles)
    finite = bool(np.all(np.isfinite(poles)))
    real_parts = np.real(poles)
    passed = finite and bool(np.all(real_parts < -tol))
    worst = float(np.max(real_parts)) if len(real_parts) else float("nan")
    return CheckResult(f"{label} stable", passed, f"max Re(pole) = {worst:.3g}")


def run_postchecks_state_feedback(A, B, Q, R, S, closed_loop_poles, N=None):
    checks = [postcheck_are_residual(A, B, Q, R, S, N)]
    checks.extend(postcheck_symmetric_psd(S, "S"))
    checks.append(postcheck_poles_stable(closed_loop_poles, "closed-loop poles"))
    return checks


def run_postchecks_kalman(A, C, Kf, P, estimator_poles, Qw, Rv):
    """Kalman filter design is the dual LQR problem (Aᵀ, Cᵀ, Qw, Rv) — see
    lqg_design_methods._kalman_core — so the Riccati residual check is run
    on that dual pair; P is that problem's "S", hence checked the same way."""
    checks = [postcheck_are_residual(A.T, C.T, Qw, Rv, P)]
    checks.extend(postcheck_symmetric_psd(P, "P"))
    checks.append(postcheck_poles_stable(estimator_poles, "estimator poles"))
    return checks


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch over the result types in lqg_design_methods.py / lqg_simulate.py
# ─────────────────────────────────────────────────────────────────────────────

def checks_for_result(result):
    """Runs the full pre+post check suite for any LQGDesignResult (LQR,
    OutputWeightedLQR, BrysonLQR, LQG, ImplicitModelFollowing all return
    this type) or ExplicitModelFollowingResult, using the actual Q/R/N (or
    Q̂/R̂ for model-following) the design solved with. Returns
    {"pre": [...], "post": [...]}, with a "kalman" key added when the
    result carries a Kalman filter."""
    from lqg_design_methods import LQGDesignResult
    from lqg_explicit import ExplicitModelFollowingResult

    if isinstance(result, ExplicitModelFollowingResult):
        plant = result.plant
        nx, nu = plant.nx, plant.nu
        nxm = result.Am.shape[0]
        Aaug = np.block([[plant.A, np.zeros((nx, nxm))],
                         [np.zeros((nxm, nx)), result.Am]])
        Baug = np.vstack([plant.B, np.zeros((nxm, nu))])
        Qhat = np.block([[plant.C.T @ result.Q1 @ plant.C, -plant.C.T @ result.Q1],
                         [-result.Q1 @ plant.C, result.Q1]])
        out = {
            "pre": run_prechecks(Aaug, Baug, Qhat, result.R),
            "post": run_postchecks_state_feedback(
                Aaug, Baug, Qhat, result.R, result.S, result.closed_loop_poles),
        }
        return out

    if isinstance(result, LQGDesignResult):
        A, B = result.plant.A, result.plant.B
        out = {
            "pre": run_prechecks(A, B, result.Q, result.R),
            "post": run_postchecks_state_feedback(
                A, B, result.Q, result.R, result.S, result.closed_loop_poles, result.N),
        }
        if result.kalman is not None:
            out["kalman_pre"] = run_prechecks(
                A.T, result.plant.C.T, result.kalman.Qw, result.kalman.Rv,
                q_label="Qw", r_label="Rv",
                stabilizability_label="(Aᵀ,Cᵀ)", detectability_label="(Aᵀ,√Qw)")
            out["kalman_post"] = run_postchecks_kalman(
                A, result.plant.C, result.kalman.Kf, result.kalman.P,
                result.kalman.estimator_poles, result.kalman.Qw, result.kalman.Rv)
        return out

    raise TypeError(f"no checks defined for result type {type(result).__name__}")


def format_checks(checks, indent="  "):
    lines = []
    for c in checks:
        mark = "PASS" if c.passed else "FAIL"
        lines.append(f"{indent}[{mark}] {c.name} — {c.detail}")
    return "\n".join(lines)
