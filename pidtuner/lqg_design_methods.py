"""Core LQR/LQG design methods and shared infrastructure — the LQG-track
analog of tuning_methods.py, per docs/lqg_plan.md Phase 1-2.

Holds the Phase-1 core (`LQR`, `OutputWeightedLQR`, `LQG`) plus everything
they and the split-out methods share: the result types (`StateFeedbackGains`,
`KalmanFilterResult`, `LQGDesignResult`), the Riccati solvers (`_lqr_core`,
`_kalman_core`), and `BaseControlDesignMethod`. Three "how do I pick Q/R"
methods live in their own files instead, specifically so they're easy to
read/diff side by side: `lqg_bryson.BrysonLQR`, `lqg_implicit.
ImplicitModelFollowing`, `lqg_explicit.ExplicitModelFollowing` — all three
still subclass `BaseControlDesignMethod` and import `_lqr_core`/the result
types from here.

Every method here is full-state-feedback LQR (u = -Kx) built from a
StateSpacePlant plus cost weights, following AILQG.pdf §3. `LQG` additionally
combines that with a steady-state Kalman filter (§6-7) via the separation
principle, so the estimated state x̂ (not the true state) drives the control
law when not all states are measured.

Design methods differ only in how Q/R/N (and the Kalman filter's Qw/Rv) get
built; the underlying Riccati solve is the same for all of them (`_lqr_core`
below), matching MATLAB's `[K,S,E] = lqr(A,B,Q,R,N)` via
scipy.linalg.solve_continuous_are.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from scipy.linalg import solve_continuous_are

from plant import StateSpacePlant


# ─────────────────────────────────────────────────────────────────────────────
# Result types — analog of PIDGains / TuningResult
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StateFeedbackGains:
    """Full-state-feedback control law u = -Kx (+ optional reference
    feedforward, added later by add_reference_tracking)."""

    K: np.ndarray  # shape (nu, nx)

    def pretty(self):
        return f"K =\n{np.array2string(self.K, precision=4, separator=', ')}"


@dataclass
class KalmanFilterResult:
    """Steady-state Kalman filter: x̂̇ = Ax̂ + Bu + Kf(y - Cx̂)."""

    Kf: np.ndarray             # filter gain, shape (nx, ny)
    P: np.ndarray              # steady-state error covariance, shape (nx, nx)
    estimator_poles: np.ndarray  # eig(A - Kf @ C)
    Qw: np.ndarray = field(default=None)  # process noise covariance used
    Rv: np.ndarray = field(default=None)  # measurement noise covariance used

    def pretty(self):
        return (f"Kf =\n{np.array2string(self.Kf, precision=4, separator=', ')}\n"
                f"estimator poles: {np.array2string(self.estimator_poles, precision=4)}")


@dataclass
class LQGDesignResult:
    method: str
    plant: StateSpacePlant
    gains: StateFeedbackGains
    S: np.ndarray                    # Riccati solution
    closed_loop_poles: np.ndarray    # eig(A - B K)
    Q: np.ndarray
    R: np.ndarray
    N: Optional[np.ndarray] = None
    kalman: Optional[KalmanFilterResult] = None
    Nbar: Optional[np.ndarray] = None  # reference feedforward, set by add_reference_tracking
    notes: str = ""

    def is_stable(self):
        return bool(np.all(np.real(self.closed_loop_poles) < -1e-9))


# ─────────────────────────────────────────────────────────────────────────────
# Shared Riccati core — AILQG.pdf eq. 12-14 (K = R⁻¹(Nᵀ+BᵀS), ARE for S)
# ─────────────────────────────────────────────────────────────────────────────

def _lqr_core(A, B, Q, R, N=None):
    """Solve the continuous-time LQR problem. Returns (K, S, closed_loop_poles).

    Matches MATLAB's [K,S,E] = lqr(A,B,Q,R,N):
      S solves the algebraic Riccati equation (eq. 14),
      K = R⁻¹(Nᵀ + BᵀS)                                (eq. 13),
      closed-loop poles = eig(A - BK)                   (eq. 9).
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    S = solve_continuous_are(A, B, Q, R, s=N)
    Rinv = np.linalg.inv(R)
    Nt = np.zeros((B.shape[1], A.shape[0])) if N is None else np.asarray(N, dtype=float).T
    K = Rinv @ (Nt + B.T @ S)
    clp = np.linalg.eigvals(A - B @ K)
    return K, S, clp


def _kalman_core(A, C, Qw, Rv):
    """Solve the dual (estimation) Riccati problem for the steady-state
    Kalman filter. Returns (Kf, P, estimator_poles).

    Duality with LQR (AILQG.pdf §6.1, eq. 101-103): the estimation problem
    (A, C, Qw, Rv) is the LQR problem (Aᵀ, Cᵀ, Qw, Rv) transposed, so the
    same Riccati solver applies with A→Aᵀ, B→Cᵀ, Q→Qw, R→Rv.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    C = np.atleast_2d(np.asarray(C, dtype=float))
    Qw = np.atleast_2d(np.asarray(Qw, dtype=float))
    Rv = np.atleast_2d(np.asarray(Rv, dtype=float))
    P = solve_continuous_are(A.T, C.T, Qw, Rv)
    Kf = P @ C.T @ np.linalg.inv(Rv)
    est_poles = np.linalg.eigvals(A - Kf @ C)
    return Kf, P, est_poles


# ─────────────────────────────────────────────────────────────────────────────
# Design method classes
# ─────────────────────────────────────────────────────────────────────────────

class BaseControlDesignMethod:
    """Base class for all LQR/LQG design methods. Independent of
    tuning_methods.BaseTuningMethod — PID gains and state-feedback gains are
    different enough (scalars vs. matrices, no shared math) that force-fitting
    a common base class would only obscure both."""
    name: str

    def design(self) -> LQGDesignResult:
        raise NotImplementedError("Subclasses must implement design()")


class LQR(BaseControlDesignMethod):
    """Raw/expert LQR: caller supplies Q, R (and optionally the cross-term
    N) directly. AILQG.pdf §3.2, eq. 10-14.

    Matches AIChemicalReactor1.m, AIDistillationColumn.m, AIAircraftHall.m
    (the last is the PDF's own Example 1 — see test_lqg.py for the golden-
    value check)."""

    name = "LQR"

    def __init__(self, plant: StateSpacePlant, Q, R, N=None):
        self.plant = plant
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.N = None if N is None else np.atleast_2d(np.asarray(N, dtype=float))

    def design(self) -> LQGDesignResult:
        K, S, clp = _lqr_core(self.plant.A, self.plant.B, self.Q, self.R, self.N)
        return LQGDesignResult(
            method=self.name, plant=self.plant, gains=StateFeedbackGains(K=K),
            S=S, closed_loop_poles=clp, Q=self.Q, R=self.R, N=self.N,
            notes="Q/R supplied directly (raw/expert path).",
        )


class OutputWeightedLQR(BaseControlDesignMethod):
    """Output-weighted LQR: Q = Cᵀ·Qy·C, penalizing outputs rather than raw
    states. AILQG.pdf §3.4, eq. 32-34 (the N/D cross-terms from output
    weighting are dropped here since every ported example uses D=0).

    Matches AIGeneric_RTP.m, AIAIRC.m, AIDrone.m, AIRPV.m, AITGEN.m,
    AIAUTM.m — 6 of the 9 clean plants, the majority pattern in the
    professor's examples."""

    name = "Output-weighted LQR"

    def __init__(self, plant: StateSpacePlant, Qy=None, R=None):
        self.plant = plant
        self.Qy = np.eye(plant.ny) if Qy is None else np.atleast_2d(np.asarray(Qy, dtype=float))
        self.R = np.eye(plant.nu) if R is None else np.atleast_2d(np.asarray(R, dtype=float))

    def design(self) -> LQGDesignResult:
        Q = self.plant.C.T @ self.Qy @ self.plant.C
        K, S, clp = _lqr_core(self.plant.A, self.plant.B, Q, self.R)
        return LQGDesignResult(
            method=self.name, plant=self.plant, gains=StateFeedbackGains(K=K),
            S=S, closed_loop_poles=clp, Q=Q, R=self.R,
            notes="Q = Cᵀ·Qy·C (output-weighted).",
        )


class LQG(BaseControlDesignMethod):
    """Full LQG: an LQR state-feedback design (any Q/R, supplied directly —
    wrap OutputWeightedLQR/BrysonLQR's Q/R yourself if you want those instead)
    combined with a steady-state Kalman filter via the separation principle.
    AILQG.pdf §6-7, eq. 93-111.

    No professor .m example includes a Kalman filter — every one of the 13
    files assumes full state feedback — so this class is validated
    synthetically (see test_lqg.py): Kf converges to the steady-state ARE
    solution, and the closed-loop poles equal the union of the LQR poles and
    the estimator poles (eq. 108, the separation-principle identity)."""

    name = "LQG (LQR + Kalman filter)"

    def __init__(self, plant: StateSpacePlant, Q, R, Qw, Rv, N=None):
        self.plant = plant
        self.Q = np.atleast_2d(np.asarray(Q, dtype=float))
        self.R = np.atleast_2d(np.asarray(R, dtype=float))
        self.Qw = np.atleast_2d(np.asarray(Qw, dtype=float))
        self.Rv = np.atleast_2d(np.asarray(Rv, dtype=float))
        self.N = None if N is None else np.atleast_2d(np.asarray(N, dtype=float))

    def design(self) -> LQGDesignResult:
        K, S, clp = _lqr_core(self.plant.A, self.plant.B, self.Q, self.R, self.N)
        Kf, P, est_poles = _kalman_core(self.plant.A, self.plant.C, self.Qw, self.Rv)
        kalman = KalmanFilterResult(Kf=Kf, P=P, estimator_poles=est_poles,
                                    Qw=self.Qw, Rv=self.Rv)
        return LQGDesignResult(
            method=self.name, plant=self.plant, gains=StateFeedbackGains(K=K),
            S=S, closed_loop_poles=clp, Q=self.Q, R=self.R, N=self.N,
            kalman=kalman,
            notes="State feedback + steady-state Kalman filter (separation principle).",
        )


def lqg_full_closed_loop_poles(result: LQGDesignResult) -> np.ndarray:
    """Poles of the combined plant+estimator system (AILQG.pdf eq. 106-108):

        [ẋ]   [A-BK      BK   ] [x ]
        [x̃̇] = [ 0      A-KfC ] [x̃]

    where x̃ = x - x̂ is the estimation error. By construction this is
    block-triangular, so its eigenvalues are exactly the union of the LQR
    closed-loop poles and the estimator poles — the separation-principle
    identity used to validate `LQG` in test_lqg.py, computed independently
    of `result.closed_loop_poles`/`result.kalman.estimator_poles` as a
    cross-check (built from the full 2·nx system, not just concatenated).
    """
    if result.kalman is None:
        raise ValueError("result has no Kalman filter — not an LQG design")
    A, B = result.plant.A, result.plant.B
    C = result.plant.C
    K, Kf = result.gains.K, result.kalman.Kf
    nx = result.plant.nx
    top = np.hstack([A - B @ K, B @ K])
    bottom = np.hstack([np.zeros((nx, nx)), A - Kf @ C])
    full = np.vstack([top, bottom])
    return np.linalg.eigvals(full)


def add_reference_tracking(result: LQGDesignResult) -> LQGDesignResult:
    """Add the N̄ feedforward gain for zero steady-state error to a constant
    reference input, AILQG.pdf §8 eq. 112-117:

        [Nx]   [A B]⁻¹ [0]      N̄ = Nu + K·Nx
        [Nu] = [C D]    [I]     u = -Kx + N̄·r

    Requires the plant to have no transmission zero at the origin (the block
    matrix [[A,B],[C,D]] must be invertible). Same role as halve_gains() on
    the PID side: a post-processor applied to an existing design, not a
    competing method — returns a new LQGDesignResult with `Nbar` populated.
    """
    plant = result.plant
    nx, nu, ny = plant.nx, plant.nu, plant.ny
    block = np.block([[plant.A, plant.B], [plant.C, plant.D]])
    if block.shape[0] != block.shape[1]:
        raise ValueError(
            "reference tracking requires a square [[A,B],[C,D]] block "
            f"(nx+ny == nx+nu), got nx={nx}, nu={nu}, ny={ny} — supply a "
            "square-output plant or use N̄ manually"
        )
    rhs = np.vstack([np.zeros((nx, ny)), np.eye(ny)])
    try:
        sol = np.linalg.solve(block, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "reference tracking failed: [[A,B],[C,D]] is singular — the "
            "plant has a transmission zero at the origin"
        ) from exc
    Nx, Nu = sol[:nx, :], sol[nx:, :]
    Nbar = Nu + result.gains.K @ Nx
    return LQGDesignResult(
        method=result.method, plant=result.plant, gains=result.gains,
        S=result.S, closed_loop_poles=result.closed_loop_poles,
        Q=result.Q, R=result.R, N=result.N, kalman=result.kalman,
        Nbar=Nbar, notes=(result.notes + "\nN̄ reference feedforward added.").strip(),
    )
