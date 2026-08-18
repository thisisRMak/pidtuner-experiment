"""Object-oriented PID tuning methods.

Each method is encapsulated inside its own class inheriting from BaseTuningMethod.
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from plant import TransferFunction
from pid_identify import FOPDT
from dataclasses import dataclass, field

@dataclass
class PIDGains:
    """Parallel form: D(s) = Kp + Ki/s + Kd*s."""

    Kp: float = 0.0
    Ki: float = 0.0
    Kd: float = 0.0

    @classmethod
    def from_textbook(cls, Kp, Ti=None, Td=None):
        """Convert (Kp, Ti, Td) textbook form to (Kp, Ki, Kd) parallel form."""
        Ki = Kp / Ti if Ti and np.isfinite(Ti) and Ti > 0 else 0.0
        Kd = Kp * Td if Td else 0.0
        return cls(Kp=float(Kp), Ki=float(Ki), Kd=float(Kd))

    def to_textbook(self):
        Ti = self.Kp / self.Ki if abs(self.Ki) > 1e-12 else float("inf")
        Td = self.Kd / self.Kp if abs(self.Kp) > 1e-12 else 0.0
        return self.Kp, Ti, Td

    def pretty(self):
        Ti = self.Kp / self.Ki if abs(self.Ki) > 1e-12 else float("inf")
        Td = self.Kd / self.Kp if abs(self.Kp) > 1e-12 else 0.0
        Ti_str = "∞" if not np.isfinite(Ti) else f"{Ti:.4g} s"
        return (f"Kp = {self.Kp:.4g},  Ki = {self.Ki:.4g},  Kd = {self.Kd:.4g}\n"
                f"   (Ti = {Ti_str},  Td = {Td:.4g} s)")


@dataclass
class TuningResult:
    method: str
    gains: PIDGains
    # Optional context to display in the UI
    fopdt: Optional[FOPDT] = None
    Ku: Optional[float] = None
    Pu: Optional[float] = None
    omega_180: Optional[float] = None
    cancelled_poles: list = field(default_factory=list)
    free_param: dict = field(default_factory=dict)  # e.g. {'tau_c': 0.5}
    notes: str = ""
    black_box: bool = False  # set by blackbox.BlackBoxTuner, never by tune() itself


def halve_gains(result):
    """Return a new TuningResult with all gains halved.

    The Emami / Aström recommendation is to halve Kp specifically (ZN's
    rules were tuned for disturbance rejection and overshoot heavily on
    setpoint tracking). Some practitioners halve all three gains — same
    spirit, slightly different effect on derivative action. We halve all
    three for simplicity; the override is a free knob anyway.
    """
    new_gains = PIDGains(
        Kp=result.gains.Kp * 0.5,
        Ki=result.gains.Ki * 0.5,
        Kd=result.gains.Kd * 0.5,
    )
    return TuningResult(
        method=result.method + " ½",
        gains=new_gains,
        fopdt=result.fopdt,
        Ku=result.Ku,
        Pu=result.Pu,
        omega_180=result.omega_180,
        cancelled_poles=list(result.cancelled_poles),
        free_param=dict(result.free_param),
        notes=(result.notes + "\nAll gains halved post-tuning.").strip(),
        black_box=result.black_box,
    )


class BaseTuningMethod:
    """Base class for all PID tuning methods."""
    name: str

    def tune(self) -> TuningResult:
        """Run the tuning algorithm and return a TuningResult."""
        raise NotImplementedError("Subclasses must implement tune()")


class StablePoleCancellation(BaseTuningMethod):
    """Stable pole cancellation tuning method."""
    name = "Stable pole cancellation"

    def __init__(self, plant: TransferFunction, p1: float, p2: float, Kd: float = 1.0):
        self.plant = plant
        self.p1 = p1
        self.p2 = p2
        self.Kd = Kd

    def tune(self) -> TuningResult:
        # p1/p2 are real for two real poles, or a complex-conjugate pair
        # when cancelling a complex pole pair (see select_slowest_stable_poles).
        # np.real() handles both uniformly for the RHP check.
        for p, name in ((self.p1, "p1"), (self.p2, "p2")):
            if np.real(p) <= 0:
                raise ValueError(
                    f"Pole-cancellation requires a stable pole, but {name}={p} "
                    f"corresponds to an RHP or imaginary-axis pole. "
                    f"Refusing — see lecture: 'never, ever do RHP cancellations.'"
                )
        Kp_raw = (self.p1 + self.p2) * self.Kd
        Ki_raw = self.p1 * self.p2 * self.Kd
        if abs(np.imag(Kp_raw)) > 1e-9 or abs(np.imag(Ki_raw)) > 1e-9:
            raise ValueError(
                f"p1={self.p1!r} and p2={self.p2!r} are not real or a "
                f"complex-conjugate pair — the resulting gains would be complex."
            )
        Kp = float(np.real(Kp_raw))
        Ki = float(np.real(Ki_raw))
        return TuningResult(
            method=self.name,
            gains=PIDGains(Kp=Kp, Ki=Ki, Kd=self.Kd),
            cancelled_poles=[-self.p1, -self.p2],
            free_param={"Kd": float(self.Kd)},
            notes=(f"Controller zeros placed at s = -{self.p1:g}, s = -{self.p2:g} to "
                   f"cancel chosen plant poles. Kd is free and sets the "
                   f"closed-loop integrator gain (open loop reduces to "
                   f"K·Kd/s after cancellation)."),
        )


class ZieglerNicholsI(BaseTuningMethod):
    """Ziegler-Nichols Method I (reaction curve / FOPDT)."""
    name = "Ziegler–Nichols I"

    def __init__(self, fopdt: FOPDT):
        self.fopdt = fopdt

    def tune(self) -> TuningResult:
        L = max(self.fopdt.L, 1e-6)
        if self.fopdt.K == 0 or self.fopdt.tau == 0:
            raise ValueError("ZN-I needs a non-zero K and tau from the FOPDT fit.")
        Kp = 1.2 * self.fopdt.tau / (self.fopdt.K * L)
        Ti = 2.0 * L
        Td = 0.5 * L
        gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
        return TuningResult(
            method=self.name,
            gains=gains,
            fopdt=self.fopdt,
            notes=("ZN-I reaction-curve rules (1942). Designed for "
                   "disturbance rejection, tends to overshoot on setpoint "
                   "tracking. Apply the 'Halve gains' toggle for a more "
                   "conservative tracking response."),
        )


class ZieglerNicholsII(BaseTuningMethod):
    """Ziegler-Nichols Method II (ultimate-gain / Ku, Pu)."""
    name = "Ziegler–Nichols II"

    def __init__(self, Ku: float, Pu: float):
        self.Ku = Ku
        self.Pu = Pu

    def tune(self) -> TuningResult:
        Kp = 0.6 * self.Ku
        Ti = 0.5 * self.Pu
        Td = 0.125 * self.Pu
        gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
        return TuningResult(
            method=self.name,
            gains=gains,
            Ku=float(self.Ku),
            Pu=float(self.Pu),
            notes=(f"Ultimate-gain rules. Ku = {self.Ku:.4g}, Pu = {self.Pu:.4g} s. "
                   f"Like ZN-I, designed for disturbance rejection; apply "
                   f"the 'Halve gains' toggle for tracking."),
        )


class Amigo(BaseTuningMethod):
    """AMIGO tuning method (Aström–Hägglund 2006)."""
    name = "AMIGO"

    def __init__(self, fopdt: FOPDT, integrating: bool = False):
        self.fopdt = fopdt
        self.integrating = integrating

    def tune(self) -> TuningResult:
        L = max(self.fopdt.L, 1e-6)
        A = self.fopdt.K
        if abs(A) < 1e-12:
            raise ValueError("AMIGO needs non-zero gain A.")
        if self.integrating:
            Kp = 0.45 / A
            Ti = 8.0 * L
            Td = 0.5 * L
            notes = "AMIGO rules for integrating + dead-time process."
        else:
            tau = max(self.fopdt.tau, 1e-9)
            Kp = (1.0 / A) * (0.2 + 0.45 * tau / L)
            Ti = ((0.4 * L + 0.8 * tau) / (L + 0.1 * tau)) * L
            Td = (0.5 * tau / (0.3 * L + tau)) * L
            notes = ("AMIGO rules (Aström & Hägglund 2006). Targets ~20% "
                     "overshoot with a robustness constraint on the "
                     "sensitivity function — more conservative than ZN-I.")
        gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
        return TuningResult(
            method=self.name + (" (integrating)" if self.integrating else ""),
            gains=gains, fopdt=self.fopdt, notes=notes,
        )


class Simc(BaseTuningMethod):
    """SIMC (Skogestad) tuning method."""
    name = "SIMC"

    def __init__(self, fopdt: FOPDT, tau_c: Optional[float] = None, tau2: Optional[float] = None):
        self.fopdt = fopdt
        self.tau_c = tau_c
        self.tau2 = tau2

    def tune(self) -> TuningResult:
        L = max(self.fopdt.L, 1e-6)
        if abs(self.fopdt.K) < 1e-12:
            raise ValueError("SIMC needs non-zero K.")
        tau_c_val = self.tau_c if self.tau_c is not None else L

        tau = max(self.fopdt.tau, 1e-9)
        Kp = (1.0 / self.fopdt.K) * tau / (tau_c_val + L)
        Ti = min(tau, 4.0 * (tau_c_val + L))

        if self.tau2 is not None and self.tau2 > 0:
            Td = float(self.tau2)
            notes = (f"SIMC (Skogestad) for second-order plant, τ₁={tau:.3g} s, "
                     f"τ₂={self.tau2:.3g} s, τc={tau_c_val:.3g} s.")
        else:
            Td = 0.0
            notes = (f"SIMC (Skogestad) PI from FOPDT, τc={tau_c_val:.3g} s. "
                     f"For PID, supply a τ₂ from a second-order fit.")
        gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
        return TuningResult(
            method=self.name, gains=gains, fopdt=self.fopdt,
            free_param={"tau_c": float(tau_c_val)}, notes=notes,
        )


class Boyd(BaseTuningMethod):
    """Boyd convex-concave optimization tuning method."""
    name = "Boyd convex-concave"

    def __init__(self, plant: TransferFunction, Ms: float = 1.4, Mt: float = 1.4,
                 seed_gains: Optional[PIDGains] = None, use_derivative: bool = True,
                 n_freq: int = 300, max_iter: int = 40, tol: float = 1e-4):
        self.plant = plant
        self.Ms = Ms
        self.Mt = Mt
        self.seed_gains = seed_gains
        self.use_derivative = use_derivative
        self.n_freq = n_freq
        self.max_iter = max_iter
        self.tol = tol

    def _boyd_omega_grid(self, plant: TransferFunction) -> np.ndarray:
        poles = plant.poles()
        zeros = plant.zeros()
        feats = np.concatenate([np.abs(poles), np.abs(zeros)])
        feats = feats[feats > 1e-9]
        if len(feats) == 0:
            omega_lo, omega_hi = 1e-3, 1e3
        else:
            omega_lo = max(np.min(feats) * 1e-3, 1e-9)
            omega_hi = max(np.max(feats) * 1e3, omega_lo * 100)
        if plant.L > 0:
            omega_hi = max(omega_hi, 100.0 / plant.L)
        return np.logspace(np.log10(omega_lo), np.log10(omega_hi), self.n_freq)

    def tune(self) -> TuningResult:
        from scipy.optimize import linprog

        # Sign-flip for negative-DC plants so the loop has the right phase.
        sign = 1.0 if self.plant.dc_gain() >= 0 else -1.0
        plant_eff = TransferFunction(num=self.plant.num * sign, den=self.plant.den, L=self.plant.L)

        omega = self._boyd_omega_grid(plant_eff)
        P = plant_eff.freq_response(omega)

        # L(jω) = (Kp + Ki/(jω) + Kd*jω) * P(jω) is affine in (Kp, Ki, Kd)
        cols = [P, P / (1j * omega)]
        if self.use_derivative:
            cols.append(P * (1j * omega))
        basis = np.stack(cols, axis=1)
        n_var = basis.shape[1]

        Mt2 = self.Mt * self.Mt
        # Robustness circles in the Nyquist plane: |L(jω) - c| >= r
        #   Sensitivity:  c = -1, r = 1/Ms
        #   Comp. sens.:  c = -Mt²/(Mt²-1), r = Mt/(Mt²-1)
        circles = [(-1.0 + 0j, 1.0 / self.Ms),
                   (-Mt2 / (Mt2 - 1.0) + 0j, self.Mt / (Mt2 - 1.0))]

        # Seed: caller may supply seed (e.g. from SIMC). Otherwise crude default.
        if self.seed_gains is None:
            K_dc = max(abs(self.plant.dc_gain()), 1e-3)
            seed_vec = np.array([1.0 / K_dc, 0.1 / K_dc, 0.0])[:n_var]
        else:
            seed_vec = np.array([self.seed_gains.Kp, self.seed_gains.Ki, self.seed_gains.Kd])[:n_var]
        seed_vec = np.abs(seed_vec)
        seed_abs = np.maximum(seed_vec, 1e-6)
        bounds = [(0.0, float(b)) for b in seed_abs * 100.0]

        obj = np.zeros(n_var)
        obj[1] = -1.0  # maximize Ki

        alpha = np.maximum(seed_vec, 0.0)
        last_ki = alpha[1] if alpha[1] > 0 else -np.inf

        for _ in range(self.max_iter):
            L_k = basis @ alpha
            rows_A, rows_b = [], []
            for c, r in circles:
                d_k = L_k - c
                mag = np.maximum(np.abs(d_k), 1e-12)
                unit = np.conj(d_k) / mag
                # Linearize |L - c| at alpha: |L - c| ≈ Re(unit·(L - c)).
                # Constraint Re(unit·(L - c)) >= r  ⇒  -Re(unit·L) <= -r + Re(unit·c)
                coef = (unit[:, None] * basis).real
                rhs = r + (unit * c).real
                rows_A.append(-coef)
                rows_b.append(-rhs)

            res = linprog(obj, A_ub=np.vstack(rows_A),
                          b_ub=np.concatenate(rows_b),
                          bounds=bounds, method="highs")
            if not res.success:
                break
            alpha_new = res.x
            ki_new = alpha_new[1]
            if last_ki > 0 and abs(ki_new - last_ki) / max(last_ki, 1e-9) < self.tol:
                alpha = alpha_new
                break
            alpha, last_ki = alpha_new, ki_new

        Kp = float(alpha[0]) * sign
        Ki = float(alpha[1]) * sign
        Kd = float(alpha[2]) * sign if self.use_derivative else 0.0
        return TuningResult(
            method=self.name,
            gains=PIDGains(Kp=Kp, Ki=Ki, Kd=Kd),
            free_param={"Ms": float(self.Ms), "Mt": float(self.Mt)},
            notes=(f"Maximized Ki subject to |S|∞ ≤ {self.Ms:g} and "
                   f"|T|∞ ≤ {self.Mt:g} via linearized convex-concave iteration. "
                   f"Operates on the true plant frequency response."),
        )


class CohenCoon(BaseTuningMethod):
    """Cohen-Coon tuning method (1953)."""
    name = "Cohen–Coon"

    def __init__(self, fopdt: FOPDT):
        self.fopdt = fopdt

    def tune(self) -> TuningResult:
        L = max(self.fopdt.L, 1e-6)
        A = self.fopdt.K
        if abs(A) < 1e-12 or self.fopdt.tau == 0:
            raise ValueError("Cohen–Coon needs non-zero A and tau from the FOPDT fit.")
        tau = max(self.fopdt.tau, 1e-9)
        r = L / tau  # fractional dead time

        Kp = (1.0 / A) * (tau / L) * (4.0 / 3.0 + r / 4.0)
        Ti = L * (32.0 + 6.0 * r) / (13.0 + 8.0 * r)
        Td = 4.0 * L / (11.0 + 2.0 * r)
        gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
        return TuningResult(
            method=self.name,
            gains=gains,
            fopdt=self.fopdt,
            notes=("Cohen–Coon reaction-curve rules (1953). Like ZN-I but uses "
                   "tau and L separately, with a dead-time correction that helps "
                   "on delay-dominant plants (large L/tau). Targets quarter-decay "
                   "load response; can be aggressive on setpoint tracking."),
        )


_CHR_PID_TABLE = {
    ("setpoint", 0):  (0.60, ("tau", 1.00), 0.50),
    ("setpoint", 20): (0.95, ("tau", 1.40), 0.47),
    ("load", 0):      (0.95, ("L",   2.40), 0.42),
    ("load", 20):     (1.20, ("L",   2.00), 0.42),
}


class ChienHronesReswick(BaseTuningMethod):
    """Chien-Hrones-Reswick (CHR, 1952) tuning method."""
    name = "Chien–Hrones–Reswick"

    def __init__(self, fopdt: FOPDT, response: str = "setpoint", overshoot: int = 0):
        self.fopdt = fopdt
        self.response = response
        self.overshoot = overshoot

    def tune(self) -> TuningResult:
        resp = self.response.lower()
        ov = int(self.overshoot)
        key = (resp, ov)
        if key not in _CHR_PID_TABLE:
            raise ValueError(
                f"CHR: unknown variant {key}. response must be 'setpoint' or "
                f"'load'; overshoot must be 0 or 20.")
        L = max(self.fopdt.L, 1e-6)
        A = self.fopdt.K
        if abs(A) < 1e-12 or self.fopdt.tau == 0:
            raise ValueError("CHR needs non-zero A and tau from the FOPDT fit.")
        tau = max(self.fopdt.tau, 1e-9)

        Kp_coef, (Ti_base, Ti_coef), Td_coef = _CHR_PID_TABLE[key]
        Kp = Kp_coef * tau / (A * L)
        Ti = Ti_coef * (tau if Ti_base == "tau" else L)
        Td = Td_coef * L
        gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
        pretty_resp = "setpoint tracking" if resp == "setpoint" else "load rejection"
        return TuningResult(
            method=f"CHR ({resp} {ov}%)",
            gains=gains,
            fopdt=self.fopdt,
            free_param={"response": resp, "overshoot": ov},
            notes=(f"Chien–Hrones–Reswick (1952), tuned for {pretty_resp} with a "
                   f"{ov}% overshoot target. Unlike ZN-I, CHR uses tau and "
                   f"L separately and lets you pick servo vs. regulator behaviour."),
        )


class TyreusLuyben(BaseTuningMethod):
    """Tyreus-Luyben tuning method (1992 / 1997)."""
    name = "Tyreus–Luyben"

    def __init__(self, Ku: float, Pu: float, use_derivative: bool = True):
        self.Ku = Ku
        self.Pu = Pu
        self.use_derivative = use_derivative

    def tune(self) -> TuningResult:
        if self.use_derivative:
            Kp = self.Ku / 2.2
            Ti = 2.2 * self.Pu
            Td = self.Pu / 6.3
            struct = "PID"
        else:
            Kp = self.Ku / 3.2
            Ti = 2.2 * self.Pu
            Td = 0.0
            struct = "PI"
        gains = PIDGains.from_textbook(Kp, Ti=Ti, Td=Td)
        return TuningResult(
            method=self.name + ("" if self.use_derivative else " (PI)"),
            gains=gains,
            Ku=float(self.Ku),
            Pu=float(self.Pu),
            notes=(f"Tyreus–Luyben {struct} (Luyben & Luyben, 1997). "
                   f"Ku = {self.Ku:.4g}, Pu = {self.Pu:.4g} s. A conservative ultimate-gain "
                   f"rule: larger margins and far less overshoot than ZN-II, but "
                   f"slower. No 'halve gains' needed — it is already detuned."),
        )
