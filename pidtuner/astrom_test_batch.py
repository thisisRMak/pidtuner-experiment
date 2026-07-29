"""Astrom & Hagglund's process test batch (PID Controllers: Theory, Design,
and Tuning, 2nd ed., p. 227) -- 9 parametrized plant families, expanded over
their listed parameter values, used in the book to validate tuning rules
across a broad sweep of dynamics rather than to check any single "textbook
answer." This module only builds the plant expressions; running/tuning them
is cli_astrom_batch.py's job.

Every family reduces to a symbolic expression consumable directly by
plant.TransferFunction.parse -- no separate delay/gain plumbing needed, the
parser's exp(...) grammar handles dead time inline.

Counts by family (P1..P9): 21, 21, 10, 6, 9, 9, 36, 11, 10 = 133 plants.
The book states 134; transcribing the listed ranges literally yields 133
(most likely P4's n is meant to start at 2, not 3 -- add 2 to P4_N below if
that 134th process should be included).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BatchPlant:
    family: str          # "P1".."P9"
    params: dict         # e.g. {"T": 0.3} or {"T": 5, "L1": 0.3}
    expr: str            # symbolic transfer function, ready for TransferFunction.parse
    id: str              # "P1_T=0.3" / "P7_T=5_L1=0.3" -- unique, filesystem-safe
    subpath: tuple        # ("P1", "T=0.3") -- path components under a family folder


FAMILY_DOCS = {
    "P1": "e^(-s) / (1+sT)  -- FOPDT, dead time fixed at 1, T swept",
    "P2": "e^(-s) / (1+sT)^2  -- second-order-plus-delay, repeated pole",
    "P3": "1 / ((s+1)(1+sT)^2)  -- third-order, one fixed + two swept poles",
    "P4": "1 / (s+1)^n  -- n-th order, no delay, repeated unit pole",
    "P5": "1 / ((1+s)(1+as)(1+a^2 s)(1+a^3 s))  -- 4 poles spread by powers of alpha",
    "P6": "e^(-sL1) / (s(1+sT1)), T1+L1=1  -- integrating plant with dead time",
    "P7": "T e^(-sL1) / ((1+sT)(1+sT1)), T1+L1=1  -- two time constants + delay",
    "P8": "e^(-a s) / (s+1)^3  -- fixed 3rd-order pole, delay swept",
    "P9": "1 / ((s+1)((sT)^2+1.4 sT+1))  -- one real pole + underdamped pair",
}

P1_T = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.3, 1.5, 2, 4, 6, 8, 10, 20, 50, 100, 200, 500, 1000]
P2_T = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 1.3, 1.5, 2, 4, 6, 8, 10, 20, 50, 100, 200, 500]
P3_T = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 2, 5, 10]
P4_N = [3, 4, 5, 6, 7, 8]
P5_ALPHA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
P6_L1 = [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
P7_T = [1, 2, 5, 10]
P7_L1 = [0.01, 0.02, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
P8_ALPHA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
P9_T = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def _fmt(v) -> str:
    """Compact, filesystem-safe param formatting matching the book's own
    notation (e.g. 0.02, 1.3, 1000, not 0.0200000 or 1.3000000000000003)."""
    f = float(v)
    return str(int(f)) if f == int(f) else f"{f:g}"


def _p1():
    for T in P1_T:
        yield BatchPlant("P1", {"T": T}, f"exp(-s)/(1+{T}*s)",
                          f"P1_T={_fmt(T)}", ("P1", f"T={_fmt(T)}"))


def _p2():
    for T in P2_T:
        yield BatchPlant("P2", {"T": T}, f"exp(-s)/(1+{T}*s)^2",
                          f"P2_T={_fmt(T)}", ("P2", f"T={_fmt(T)}"))


def _p3():
    for T in P3_T:
        yield BatchPlant("P3", {"T": T}, f"1/((s+1)*(1+{T}*s)^2)",
                          f"P3_T={_fmt(T)}", ("P3", f"T={_fmt(T)}"))


def _p4():
    for n in P4_N:
        yield BatchPlant("P4", {"n": n}, f"1/(s+1)^{n}",
                          f"P4_n={_fmt(n)}", ("P4", f"n={_fmt(n)}"))


def _p5():
    for a in P5_ALPHA:
        expr = f"1/((1+s)*(1+{a}*s)*(1+{a**2:g}*s)*(1+{a**3:g}*s))"
        yield BatchPlant("P5", {"alpha": a}, expr,
                          f"P5_alpha={_fmt(a)}", ("P5", f"alpha={_fmt(a)}"))


def _p6():
    for L1 in P6_L1:
        T1 = 1 - L1
        expr = f"exp(-{L1}*s)/(s*(1+{T1:g}*s))"
        yield BatchPlant("P6", {"L1": L1, "T1": T1}, expr,
                          f"P6_L1={_fmt(L1)}", ("P6", f"L1={_fmt(L1)}"))


def _p7():
    for T in P7_T:
        for L1 in P7_L1:
            T1 = 1 - L1
            expr = f"{T}*exp(-{L1}*s)/((1+{T}*s)*(1+{T1:g}*s))"
            yield BatchPlant("P7", {"T": T, "L1": L1, "T1": T1}, expr,
                              f"P7_T={_fmt(T)}_L1={_fmt(L1)}",
                              ("P7", f"T={_fmt(T)}", f"L1={_fmt(L1)}"))


def _p8():
    for a in P8_ALPHA:
        yield BatchPlant("P8", {"alpha": a}, f"exp(-{a}*s)/(s+1)^3",
                          f"P8_alpha={_fmt(a)}", ("P8", f"alpha={_fmt(a)}"))


def _p9():
    for T in P9_T:
        expr = f"1/((s+1)*(({T}*s)^2+1.4*{T}*s+1))"
        yield BatchPlant("P9", {"T": T}, expr,
                          f"P9_T={_fmt(T)}", ("P9", f"T={_fmt(T)}"))


_FAMILY_ITERS = {
    "P1": _p1, "P2": _p2, "P3": _p3, "P4": _p4, "P5": _p5,
    "P6": _p6, "P7": _p7, "P8": _p8, "P9": _p9,
}


def iter_batch(families=None):
    """Yield every BatchPlant in the batch, in family order (P1..P9). Pass
    `families` (an iterable of family ids) to restrict to a subset."""
    for fam in (families if families is not None else _FAMILY_ITERS):
        yield from _FAMILY_ITERS[fam]()


def batch_size(families=None) -> int:
    return sum(1 for _ in iter_batch(families))
