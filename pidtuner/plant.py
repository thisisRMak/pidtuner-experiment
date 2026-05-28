"""Plant: LTI transfer function G(s) = num(s)/den(s) * exp(-Ls).

Supports two user-facing input forms:
  1. Symbolic expression:   "1000 / ((s+1)*(10s+1))"
  2. MATLAB-style polys:    num=[1], den=[10, 11, 1], gain=1000
Both reduce to the same internal representation (num, den, L), where the
polynomials are stored in descending powers of s, matching scipy and MATLAB
conventions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from scipy.signal import tf2ss, cont2discrete


# ─────────────────────────────────────────────────────────────────────────────
# Polynomial helpers (descending powers of s — same convention as MATLAB)
# ─────────────────────────────────────────────────────────────────────────────

def poly_mul(a, b):
    return np.convolve(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def poly_add(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n = max(len(a), len(b))
    out = np.zeros(n)
    out[n - len(a):] += a
    out[n - len(b):] += b
    return out


def poly_trim(p):
    p = np.asarray(p, dtype=float)
    nz = np.flatnonzero(np.abs(p) > 1e-12)
    if len(nz) == 0:
        return np.array([0.0])
    return p[nz[0]:]


# ─────────────────────────────────────────────────────────────────────────────
# Symbolic TF parser
#   Recursive-descent parser for arithmetic on rational functions of s.
#   Implicit multiplication is supported: "10s", "(s+1)(s+2)", "2(s+1)".
#   Result is a (num, den) pair of polynomials in descending powers of s.
# ─────────────────────────────────────────────────────────────────────────────

class _Rat:
    """Rational function num/den, polynomials in descending powers of s."""

    __slots__ = ("num", "den")

    def __init__(self, num, den):
        self.num = np.asarray(num, dtype=float).ravel()
        self.den = np.asarray(den, dtype=float).ravel()

    @classmethod
    def const(cls, c): return cls([float(c)], [1.0])

    @classmethod
    def s(cls): return cls([1.0, 0.0], [1.0])

    def __neg__(self): return _Rat(-self.num, self.den)

    def __add__(self, o):
        return _Rat(
            poly_add(poly_mul(self.num, o.den), poly_mul(o.num, self.den)),
            poly_mul(self.den, o.den),
        )

    def __sub__(self, o): return self + (-o)

    def __mul__(self, o):
        return _Rat(poly_mul(self.num, o.num), poly_mul(self.den, o.den))

    def __truediv__(self, o):
        if not np.any(np.abs(o.num) > 1e-12):
            raise ValueError("division by zero in transfer-function expression")
        return _Rat(poly_mul(self.num, o.den), poly_mul(self.den, o.num))

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("exponent must be a non-negative integer")
        out = _Rat.const(1.0)
        for _ in range(n):
            out = out * self
        return out


def _tokenize(text):
    tokens = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i
            while j < n and (text[j].isdigit() or text[j] == "."):
                j += 1
            # scientific notation
            if j < n and text[j] in "eE":
                j += 1
                if j < n and text[j] in "+-":
                    j += 1
                while j < n and text[j].isdigit():
                    j += 1
            tokens.append(("NUM", float(text[i:j])))
            i = j
        elif c.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            sym = text[i:j].lower()
            if sym != "s":
                raise ValueError(
                    f"unknown symbol {text[i:j]!r}; only the Laplace variable 's' is allowed"
                )
            tokens.append(("S", None))
            i = j
        elif text[i:i + 2] == "**":
            tokens.append(("OP", "^"))
            i += 2
        elif c in "+-*/^":
            tokens.append(("OP", c))
            i += 1
        elif c == "(":
            tokens.append(("LP", None))
            i += 1
        elif c == ")":
            tokens.append(("RP", None))
            i += 1
        else:
            raise ValueError(f"unexpected character {c!r} in expression")
    return tokens


class _Parser:
    """Recursive-descent parser. Grammar:
        expr   : term (('+'|'-') term)*
        term   : factor (('*'|'/'|implicit) factor)*
        factor : ('+'|'-') factor | power
        power  : atom ('^' factor)?
        atom   : NUM | 's' | '(' expr ')'
    """

    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    def _peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def _eat(self):
        tok = self._peek()
        self.i += 1
        return tok

    def parse(self):
        e = self._expr()
        if self.i != len(self.t):
            raise ValueError(f"unexpected token at position {self.i}: {self._peek()!r}")
        return e

    def _expr(self):
        left = self._term()
        while True:
            tag, val = self._peek()
            if tag == "OP" and val in "+-":
                self._eat()
                right = self._term()
                left = left + right if val == "+" else left - right
            else:
                return left

    def _term(self):
        left = self._factor()
        while True:
            tag, val = self._peek()
            if tag == "OP" and val in "*/":
                self._eat()
                right = self._factor()
                left = left * right if val == "*" else left / right
            elif tag in ("NUM", "S", "LP"):
                # Implicit multiplication: "10s", "(s+1)(s+2)", "2(s+1)"
                right = self._factor()
                left = left * right
            else:
                return left

    def _factor(self):
        tag, val = self._peek()
        if tag == "OP" and val in "+-":
            self._eat()
            f = self._factor()
            return -f if val == "-" else f
        return self._power()

    def _power(self):
        base = self._atom()
        tag, val = self._peek()
        if tag == "OP" and val == "^":
            self._eat()
            exp = self._factor()
            if len(exp.den) != 1 or len(exp.num) != 1:
                raise ValueError("exponent must be an integer constant")
            n_float = float(exp.num[0]) / float(exp.den[0])
            n = int(round(n_float))
            if abs(n - n_float) > 1e-9:
                raise ValueError("exponent must be an integer constant")
            if n < 0:
                return _Rat.const(1.0) / (base ** abs(n))
            return base ** n
        return base

    def _atom(self):
        tag, val = self._eat()
        if tag == "NUM":
            return _Rat.const(val)
        if tag == "S":
            return _Rat.s()
        if tag == "LP":
            inner = self._expr()
            tag2, _ = self._eat()
            if tag2 != "RP":
                raise ValueError("missing ')'")
            return inner
        raise ValueError(f"unexpected token {tag!r}")


def parse_coeff_list(text):
    """Parse a MATLAB-style coefficient list: '10, 11, 1' or '10 11 1'."""
    text = text.strip()
    # strip MATLAB brackets if present
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        raise ValueError("empty coefficient list")
    parts = re.split(r"[,\s;]+", text)
    return np.array([float(p) for p in parts if p], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Plant: transfer function with optional dead time
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TransferFunction:
    num: np.ndarray
    den: np.ndarray
    L: float = 0.0

    def __post_init__(self):
        self.num = poly_trim(np.asarray(self.num, dtype=float))
        self.den = poly_trim(np.asarray(self.den, dtype=float))
        if not np.any(np.abs(self.den) > 0):
            raise ValueError("denominator polynomial is zero")
        if len(self.num) > len(self.den):
            raise ValueError(
                f"improper transfer function (num order {len(self.num) - 1} > "
                f"den order {len(self.den) - 1}). Plants must be proper."
            )
        self.L = max(float(self.L), 0.0)

    # ── constructors ────────────────────────────────────────────────────────
    @classmethod
    def fopdt(cls, K, tau, L):
        """First-order plus dead time: K * exp(-Ls) / (tau*s + 1)."""
        return cls(num=[K], den=[max(tau, 1e-9), 1.0], L=L)

    @classmethod
    def parse(cls, text, L=0.0):
        """Parse a symbolic expression like '1000/((s+1)(10s+1))'."""
        text = text.strip()
        if not text:
            raise ValueError("empty transfer-function expression")
        # accept optional 'G(s) =' or 'G =' prefix
        text = re.sub(r"^\s*G\s*\(\s*s\s*\)\s*=\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*G\s*=\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*=\s*", "", text)
        rat = _Parser(_tokenize(text)).parse()
        return cls(num=rat.num, den=rat.den, L=L)

    @classmethod
    def from_coeffs(cls, num, den, L=0.0, gain=1.0):
        """MATLAB-style: gain * num/den, polys in descending powers of s."""
        num_arr = np.asarray(num, dtype=float).ravel()
        den_arr = np.asarray(den, dtype=float).ravel()
        if num_arr.size == 0:
            num_arr = np.array([0.0])
        if den_arr.size == 0:
            raise ValueError("denominator must have at least one coefficient")
        return cls(num=num_arr * float(gain), den=den_arr, L=L)

    # ── analysis ────────────────────────────────────────────────────────────
    def order(self):
        return len(self.den) - 1

    def poles(self):
        return np.roots(self.den) if len(self.den) > 1 else np.array([])

    def zeros(self):
        return np.roots(self.num) if len(self.num) > 1 else np.array([])

    def dc_gain(self):
        """Steady-state gain. Returns inf if plant has an integrator."""
        if abs(self.den[-1]) < 1e-12:
            return float("inf")
        return float(self.num[-1]) / float(self.den[-1])

    def is_open_loop_stable(self):
        """All poles strictly in LHP (and no integrators)."""
        poles = self.poles()
        if len(poles) == 0:
            return True
        return bool(np.all(np.real(poles) < -1e-9))

    def has_rhp_poles(self):
        poles = self.poles()
        if len(poles) == 0:
            return False
        return bool(np.any(np.real(poles) > 1e-9))

    def auto_dt(self):
        """Pick a sim timestep small enough to resolve the fastest mode."""
        poles = self.poles()
        if len(poles):
            omega_max = max(np.max(np.abs(poles)), 1e-6)
        else:
            omega_max = 1.0
        base = 1.0 / (20.0 * omega_max)
        if self.L > 0:
            base = min(base, self.L / 5.0)
        return max(base, 1e-4)

    # ── simulation (discretize and step) ────────────────────────────────────
    def discretize(self, dt):
        A, B, C, D = tf2ss(self.num, self.den)
        if A.size == 0:
            return (np.zeros((0, 0)), np.zeros((0, 1)),
                    np.zeros((1, 0)), np.atleast_2d(D))
        Ad, Bd, Cd, Dd, _ = cont2discrete((A, B, C, D), dt, method="zoh")
        return Ad, Bd, Cd, Dd

    def simulate(self, t, u, y0=0.0):
        """Open-loop simulation: feed u(t), return y(t)."""
        dt = float(t[1] - t[0])
        Ad, Bd, C, D = self.discretize(dt)
        nx = Ad.shape[0]
        Bd_flat = Bd.flatten() if nx else np.zeros(0)
        d_scalar = float(np.atleast_2d(D).flatten()[0])
        delay = int(round(self.L / dt))
        n = len(t)
        x = np.zeros(nx)
        y = np.empty(n)
        y[0] = y0
        for i in range(1, n):
            j = i - 1 - delay
            u_eff = u[j] if j >= 0 else 0.0
            if nx:
                x = Ad @ x + Bd_flat * u_eff
                y[i] = float((C @ x).item()) + d_scalar * u_eff
            else:
                y[i] = d_scalar * u_eff
        return y

    def freq_response(self, omega):
        """G(jω) including dead time."""
        s = 1j * np.asarray(omega, dtype=float)
        H = np.polyval(self.num, s) / np.polyval(self.den, s)
        if self.L > 0:
            H = H * np.exp(-1j * omega * self.L)
        return H

    # ── pretty-printing ─────────────────────────────────────────────────────
    def pretty(self):
        return (f"num={np.array2string(self.num, precision=4, separator=', ')}  "
                f"den={np.array2string(self.den, precision=4, separator=', ')}  "
                f"L={self.L:g} s   (order {self.order()})")

    def latex_summary(self):
        """One-line description of poles for the UI."""
        poles = self.poles()
        if len(poles) == 0:
            return f"order 0, DC gain = {self.dc_gain():g}"
        stable = np.sum(np.real(poles) < -1e-9)
        unstable = np.sum(np.real(poles) > 1e-9)
        integ = np.sum(np.abs(np.real(poles)) <= 1e-9)
        parts = [f"order {self.order()}"]
        if stable:
            parts.append(f"{stable} stable pole(s)")
        if integ:
            parts.append(f"{integ} integrator(s)")
        if unstable:
            parts.append(f"{unstable} UNSTABLE pole(s)")
        return ", ".join(parts)
