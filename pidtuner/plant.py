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
    """Rational function num/den, polynomials in descending powers of s,
    plus an accumulated time-delay exponent (delay > 0 means a factor of
    exp(-delay*s) has been multiplied in — see the exp(...) grammar rule
    below). delay is tracked separately from num/den because it cannot be
    represented as a rational polynomial; it can only ever appear as a
    multiplicative factor of the whole expression (__add__/__sub__ reject
    it), matching TransferFunction's single lumped-delay representation.
    """

    __slots__ = ("num", "den", "delay")

    def __init__(self, num, den, delay=0.0):
        self.num = np.asarray(num, dtype=float).ravel()
        self.den = np.asarray(den, dtype=float).ravel()
        self.delay = float(delay)

    @classmethod
    def const(cls, c): return cls([float(c)], [1.0])

    @classmethod
    def s(cls): return cls([1.0, 0.0], [1.0])

    def __neg__(self): return _Rat(-self.num, self.den, self.delay)

    def __add__(self, o):
        if self.delay != 0.0 or o.delay != 0.0:
            raise ValueError(
                "a time-delay term (exp(...)) can only appear as a "
                "multiplicative factor of the whole expression, not added "
                "or subtracted"
            )
        return _Rat(
            poly_add(poly_mul(self.num, o.den), poly_mul(o.num, self.den)),
            poly_mul(self.den, o.den),
        )

    def __sub__(self, o): return self + (-o)

    def __mul__(self, o):
        return _Rat(poly_mul(self.num, o.num), poly_mul(self.den, o.den),
                     self.delay + o.delay)

    def __truediv__(self, o):
        if not np.any(np.abs(o.num) > 1e-12):
            raise ValueError("division by zero in transfer-function expression")
        return _Rat(poly_mul(self.num, o.den), poly_mul(self.den, o.num),
                     self.delay - o.delay)

    def __pow__(self, n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("exponent must be a non-negative integer")
        out = _Rat.const(1.0)
        for _ in range(n):
            out = out * self
        return out


def _exp_arg_to_delay(rat):
    """Validate that `rat` (the parsed argument of exp(...)) is a pure,
    causal delay exponent of the form -L*s (L >= 0), and return L.

    Only this restricted form is supported: exp(...) models a lumped input
    dead time, not a general exponential of a rational function of s.
    """
    if rat.delay != 0.0:
        raise ValueError("exp(...) argument cannot itself contain a time-delay term")
    if len(rat.den) != 1:
        raise ValueError(
            "exp(...) argument must be a simple expression like '-2.5*s', "
            "not a rational function"
        )
    num = rat.num / rat.den[0]
    if len(num) == 1:
        a, b = 0.0, float(num[0])
    elif len(num) == 2:
        a, b = float(num[0]), float(num[1])
    else:
        raise ValueError(
            "exp(...) argument must be linear in s (e.g. 'exp(-2.5*s)'), "
            "got a higher-order polynomial"
        )
    if abs(b) > 1e-9:
        raise ValueError(
            "exp(...) argument must have no constant term — only pure "
            "delay terms like 'exp(-2.5*s)' are supported"
        )
    if a > 1e-9:
        raise ValueError(
            "exp(...) argument must have a non-positive coefficient of s "
            "— a predictive/non-causal term like 'exp(2*s)' is not supported"
        )
    return -a


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
            # scientific notation — only commit to consuming 'e'/'E' if a
            # valid exponent (optional sign + at least one digit) follows;
            # otherwise leave it alone (e.g. "5exp(...)" must tokenize as
            # NUM(5) then FUNC(exp), not fail parsing "5e" as a bad float).
            if j < n and text[j] in "eE":
                k = j + 1
                if k < n and text[k] in "+-":
                    k += 1
                if k < n and text[k].isdigit():
                    while k < n and text[k].isdigit():
                        k += 1
                    j = k
            tokens.append(("NUM", float(text[i:j])))
            i = j
        elif c.isalpha():
            j = i
            while j < n and text[j].isalpha():
                j += 1
            sym = text[i:j].lower()
            if sym == "s":
                tokens.append(("S", None))
            elif sym == "exp":
                tokens.append(("FUNC", "exp"))
            else:
                raise ValueError(
                    f"unknown symbol {text[i:j]!r}; only the Laplace variable "
                    f"'s' and 'exp(...)' (for a time-delay term) are allowed"
                )
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
        atom   : NUM | 's' | '(' expr ')' | 'exp' '(' expr ')'
    'exp(...)' is restricted to a pure causal delay exponent -L*s (see
    _exp_arg_to_delay); the result carries no num/den contribution, only an
    accumulated _Rat.delay, and can only be combined multiplicatively.
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
            elif tag in ("NUM", "S", "LP", "FUNC"):
                # Implicit multiplication: "10s", "(s+1)(s+2)", "2(s+1)", "2exp(-s)"
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
        if tag == "FUNC":
            tag2, _ = self._eat()
            if tag2 != "LP":
                raise ValueError(f"expected '(' after {val}")
            inner = self._expr()
            tag3, _ = self._eat()
            if tag3 != "RP":
                raise ValueError("missing ')'")
            L = _exp_arg_to_delay(inner)
            return _Rat([1.0], [1.0], delay=L)
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
        """Parse a symbolic expression like '1000/((s+1)(10s+1))'.

        The expression may itself carry a dead-time factor, e.g.
        '5*exp(-3*s)/(s+1)' — the delay implied by any exp(...) term is
        combined with the separate `L` kwarg. Specifying delay in both
        places at once is rejected as ambiguous.
        """
        text = text.strip()
        if not text:
            raise ValueError("empty transfer-function expression")
        # accept optional 'G(s) =' or 'G =' prefix
        text = re.sub(r"^\s*G\s*\(\s*s\s*\)\s*=\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*G\s*=\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^\s*=\s*", "", text)
        rat = _Parser(_tokenize(text)).parse()
        expr_L = rat.delay
        if expr_L < -1e-9:
            raise ValueError(
                f"expression implies a negative (predictive/non-causal) time "
                f"delay L={expr_L:g}s — check for division by an exp(...) term"
            )
        expr_L = max(expr_L, 0.0)
        if expr_L > 1e-9 and L > 1e-9:
            raise ValueError(
                f"time delay specified twice: the expression implies "
                f"L={expr_L:g}s via exp(...), and L={L:g}s was also passed "
                f"separately — specify the delay in only one place"
            )
        total_L = expr_L if expr_L > 1e-9 else L
        return cls(num=rat.num, den=rat.den, L=total_L)

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
        if self.L > 0:
            parts.append(f"time delay L={self.L:g}s")
        return ", ".join(parts)
