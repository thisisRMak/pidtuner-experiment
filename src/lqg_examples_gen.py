"""One-off generator for the lqg_examples_json/*.json preset catalog.

Transcribes all 12 directly-runnable plants from lqg_examples_m_revised/*2.m
(see docs/lqg_plan.md "Known issues in the source material" — none excluded
as of 2026-08-02, AIExample2RTP.m having been the last one fixed; the *2
revision, sent 2026-08-14/confirmed 2026-08-24, adds a Kalman filter/S/T/
PM-GM section on top of the same LQR-only content and is a byte-identical
carryover for most plants — see lqg_revisedexamples_plan.md). Kept in the
repo for provenance — rerun this if a transcription error is found here.

Run:  python lqg_examples_gen.py
"""

from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(__file__)
_OUT_DIR = os.path.join(_HERE, "lqg_examples_json")
_SOURCE_DIR = os.path.join(_HERE, "lqg_examples_m_revised")


def _dump(key, name, citation, source_file, A, B, C, D,
         suggested_Q_kind, suggested_R_kind, notes,
         suggested_Q=None, suggested_R_scale=1.0):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    C = np.asarray(C, dtype=float)
    D = np.asarray(D, dtype=float)
    nx = A.shape[0]
    nu = B.shape[1]
    ny = C.shape[0]
    assert A.shape == (nx, nx), f"{key}: A shape {A.shape}"
    assert B.shape == (nx, nu), f"{key}: B shape {B.shape}"
    assert C.shape == (ny, nx), f"{key}: C shape {C.shape}"
    assert D.shape == (ny, nu), f"{key}: D shape {D.shape}"
    data = {
        "key": key, "name": name, "citation": citation, "source_file": source_file,
        "A": A.tolist(), "B": B.tolist(), "C": C.tolist(), "D": D.tolist(),
        "suggested_Q_kind": suggested_Q_kind,
        "suggested_Q": (np.asarray(suggested_Q, dtype=float).tolist()
                       if suggested_Q is not None else None),
        "suggested_R_kind": suggested_R_kind,
        "suggested_R_scale": suggested_R_scale,
        "notes": notes,
    }
    os.makedirs(_OUT_DIR, exist_ok=True)
    path = os.path.join(_OUT_DIR, f"{key}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {path}  (nx={nx}, nu={nu}, ny={ny})")


def build_chemical_reactor():
    A = [[1.38, -0.2077, 6.715, -5.676],
         [-0.5814, -4.290, 0, 0.675],
         [1.067, 4.273, -6.654, 5.893],
         [0.0480, 4.273, 1.343, -2.104]]
    B = [[0, 0], [5.679, 0], [1.136, -3.146], [1.136, 0]]
    C = [[1, 0, 0, 0], [0, 1, 0, 0]]
    D = np.zeros((2, 2))
    _dump("chemical_reactor", "Chemical reactor (Munro)",
         "Munro (via AIChemicalReactor12.m)", "pidtuner/lqg_examples_m_revised/AIChemicalReactor12.m",
         A, B, C, D, "identity", "identity",
         "Q=I(4), R=I(2) in the source. The *2 revision adds an explicit "
         "output map, C=[[1,0,0,0],[0,1,0,0]] (2 outputs: states 1 and 2); "
         "the original .m had none, so this catalog previously defaulted "
         "C to I(4) — that default is now replaced by the source's own C.")


def build_airc():
    A = [[0, 0, 1.1320, 0, -1.0],
         [0, -0.0538, -0.1712, 0, 0.705],
         [0, 0, 0, 1, 0],
         [0, 0.0485, 0, -0.8556, -1.013],
         [0, -0.2909, 0, 1.0532, -0.6859]]
    B = [[0, 0, 0], [-0.12, 1, 0], [0, 0, 0], [4.419, 0, -1.665], [1.5750, 0, -0.0732]]
    C = [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0]]
    D = np.zeros((3, 3))
    _dump("airc", "Aircraft (Maciejowski)",
         "Maciejowski (via AIAIRC2.m)", "pidtuner/lqg_examples_m_revised/AIAIRC2.m",
         A, B, C, D, "output_weighted", "identity",
         "Q = CᵀC, R = I(3) in the source.")


def build_drone():
    A = [[-0.0827, -0.1423e-3, -0.9994, 0.0414, 0, 0.1862],
         [-46.86, -2.757, 0.3896, 0, -124.3, 128.6],
         [-0.4248, -0.06224, -0.0671, 0, -8.792, -20.46],
         [0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, -20, 0],
         [0, 0, 0, 0, 0, -20]]
    B = [[0, 0], [0, 0], [0, 0], [0, 0], [1, 0], [0, 1]]
    C = [[0, 1, 0, 0, 0, 0], [0, 0.07, 1, 0, 0, 0]]
    D = np.zeros((2, 2))
    _dump("drone", "Drone",
         "(via AIDrone2.m)", "pidtuner/lqg_examples_m_revised/AIDrone2.m",
         A, B, C, D, "output_weighted", "identity",
         "Q = CᵀC, R = I(2) in the source.")


def build_rpv():
    A = [[-0.02567, -36.617, -18.897, -32.090, 3.2509, -0.76257],
         [9.257e-5, -1.8977, 0.98312, -7.256e-4, -0.1708, -4.965e-3],
         [0.012338, 11.720, -2.6316, 8.758e-4, -31.604, 22.396],
         [0, 0, 1, 0, 0, 0],
         [0, 0, 0, 0, -30, 0],
         [0, 0, 0, 0, 0, -30]]
    B = [[0, 0], [0, 0], [0, 0], [0, 0], [30, 0], [0, 30]]
    C = [[0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]]
    D = np.zeros((2, 2))
    _dump("rpv", "RPV (Maciejowski)",
         "Maciejowski (via AIRPV2.m)", "pidtuner/lqg_examples_m_revised/AIRPV2.m",
         A, B, C, D, "output_weighted", "identity",
         "Q = CᵀC, R = I(2) in the source.")


def build_tgen():
    A = [[-18.4456, 4.2263, -2.2830, 0.2260, 0.4220, -0.0951],
         [-4.0977, -6.0706, 5.6825, -0.6966, -1.2246, 0.2873],
         [1.4449, 1.4336, -2.6477, 0.6092, 0.8979, -0.2300],
         [-0.00093, 0.232, -0.5002, -0.1764, -6.3152, 0.1350],
         [-0.0464, -0.3489, 0.7238, 6.3117, -0.6886, 0.3645],
         [-0.0602, -0.2361, 0.2300, 0.0915, -0.3214, -0.2087]]
    B = [[-0.2748, 3.1463], [-0.0501, -9.3737], [-0.1550, 7.4296],
         [-0.0716, -4.9176], [-0.0814, -10.2648], [0.0244, 13.7943]]
    C = [[0.5971, -0.7697, 4.8850, 4.8608, -9.8177, -8.8610],
         [3.1013, 9.3422, -5.6000, -0.7490, 2.9974, 10.5719]]
    D = np.zeros((2, 2))
    _dump("tgen", "Turbo-generator (Maciejowski)",
         "Maciejowski (via AITGEN2.m)", "pidtuner/lqg_examples_m_revised/AITGEN2.m",
         A, B, C, D, "output_weighted", "identity",
         "Q = CᵀC, R = I(2) in the source.")


def build_aircraft_hall():
    A = [[-2.97e-2, -1, 0, 4.38e-2, 0],
         [3.31e-1, -4.2e-3, -4.61e-2, 0, 0],
         [-1.13, 1.28e-1, -8.03e-1, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 1, 0, 0, 0]]
    B = [[0, 0], [3.81e-1, 4.0e-2], [6.7e-2, 1.59], [0, 0], [0, 0]]
    C = [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]
    D = np.zeros((2, 2))
    Q = [[1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0], [1, 0, 0, 0, 1]]
    _dump("aircraft_hall", "Aircraft (Hall)",
         "Hall 1971, = AILQG.pdf Example 1 (via AIAircraftHall2.m)",
         "pidtuner/lqg_examples_m_revised/AIAircraftHall2.m",
         A, B, C, D, "custom", "identity",
         "GOLDEN TEST CASE: matches AILQG.pdf Example 1 exactly. LQR(Q, "
         "R=I(2)) on this plant reproduces the PDF's printed S (eq. 25), K "
         "(eq. 26), and closed-loop poles to the digits shown — verified "
         "numerically while building this catalog, see test_lqg.py.",
         suggested_Q=Q)


def build_autm():
    A = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [-0.202, -1.15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, -2.36, -13.60, -12.90, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, -1.620, -9.40, -9.15, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
         [0, 0, 0, 0, 0, 0, 0, 0, -188, -111.6, -116.4, -20.8]]
    B = [[0, 0], [1.0439, 4.1486], [0, 0], [0, 0], [-1.749, 2.6775],
         [0, 0], [0, 0], [1.0439, 4.1486], [0, 0], [0, 0], [0, 0], [-1.749, 2.6775]]
    C = [[0.264, 0.806, -1.42, -15, 0, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0, 4.9, 2.12, 1.95, 9.35, 25.8, 7.14, 0]]
    D = np.zeros((2, 2))
    _dump("autm", "AUTM",
         "(via AIAUTM2.m)", "pidtuner/lqg_examples_m_revised/AIAUTM2.m",
         A, B, C, D, "output_weighted", "identity",
         "Q = CᵀC, R = I(2) in the source. 12 states — the largest of the "
         "hand-transcribed plants.")


def build_distillation_column():
    A = 1e-2 * np.array([
        [-1.4, 0.43, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0.95, -1.38, 0.46, 0, 0, 0, 0, 0, 0, 0, 0.05],
        [0, 0.95, -1.41, 0.63, 0, 0, 0, 0, 0, 0, 0.02],
        [0, 0, 0.95, -1.58, 1.1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0.95, -3.12, 1.50, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 2.02, -3.52, 2.20, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 2.02, -4.22, 2.80, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 2.02, -4.82, 3.70, 0, 0.02],
        [0, 0, 0, 0, 0, 0, 0, 2.02, -5.72, 4.20, 0.05],
        [0, 0, 0, 0, 0, 0, 0, 0, 2.02, -4.83, 0.05],
        [2.55, 0, 0, 0, 0, 0, 0, 0, 0, 2.55, -1.85],
    ])
    B = 1e-4 * np.array([
        [0, 0], [0.05, 25], [0.02, 50], [0.01, 50], [0, 50], [0, 50],
        [-0.05, 50], [-0.10, 50], [-0.4, 25], [-0.2, 25], [4.6, 0],
    ])
    C = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    D = np.zeros((2, 2))
    _dump("distillation_column", "Distillation column (Davison)",
         "Davison 2011 (via AIDistillationColumn2.m)",
         "pidtuner/lqg_examples_m_revised/AIDistillationColumn2.m",
         A, B, C, D, "identity", "scaled_identity",
         "Q = I(11), R = 0.1·I(2) in the source. The *2 revision drops C's "
         "third row (the original 3-output C=[[...,1,0],[1,...],[...,0,1]] "
         "is now square, 2 outputs matching 2 inputs) — the third row is "
         "left commented out in the source, not deleted.", suggested_R_scale=0.1)


def build_furnace_model():
    A = 1e-2 * np.array([
        [-23.226, 1.712, 1.1, 0.774, 0.093, 1.456, -1.060, 0.453],
        [0.232, -25.493, 0.318, -2.497, -2.235, -3.691, 0.133, -1.226],
        [0.666, 0.531, -25.651, 2.520, 2.862, 3.182, -0.386, 0.568],
        [2.531, -1.134, -2.152, -24.043, 2.580, -1.283, -0.503, 0.729],
        [0.546, -1.779, -3.258, 2.397, -20.030, 0.938, 0.350, -0.546],
        [-2.242, 3.054, 3.165, 1.670, -1.008, -18.773, -0.100, 0.204],
        [-5.724, -2.789, -1.336, -0.311, -0.211, 0.546, -22.403, -0.030],
        [-0.183, -1.271, -0.737, 0.607, -0.525, -0.145, 0.074, -20.380],
    ])
    B = [[-.334, -0.223, -0.4942, -0.416],
         [-0.161, -0.247, 0.1345, 0.330],
         [0.148, -0.329, 0.0593, -0.435],
         [0.199, -0.270, -0.2105, -0.258],
         [-0.157, 0.245, 0.0557, 0.281],
         [0.076, -0.048, -0.0740, -0.024],
         [-0.020, 0.050, 0.0288, 0.009],
         [-0.038, 0.058, 0.0084, 0.062]]
    C = [[-0.4177, -0.3583, -0.3344, 0.3396, -0.2580, -0.0917, 0.0403, -0.0567],
         [-0.2647, -0.4481, -0.1575, -0.1221, -0.0002, 0.0932, 0.0006, 0.0003],
         [-0.3225, 0.0137, 0.3904, -0.2739, 0.2818, 0.0959, -0.0427, 0.0692],
         [-0.2685, 0.2041, -0.2276, 0.0027, -0.0905, 0.0186, 0.0195, -0.0247]]
    D = np.zeros((4, 4))
    _dump("furnace_model", "Furnace (Davison 2011 / Rosenbrock)",
         "Davison 2011 / Rosenbrock (via AIFurnaceModel2.m)",
         "pidtuner/lqg_examples_m_revised/AIFurnaceModel2.m",
         A, B, C, D, "identity", "scaled_identity",
         "Q=I(8), R=0.1·I(4) in the source. Was excluded (stray trailing "
         "')' broke the .m file's lqr() call); re-added once the source "
         "was corrected — see docs/lqg_plan.md \"Known issues\".",
         suggested_R_scale=0.1)


def build_f100_engine():
    A = [[-3.91800e+00, 4.1886e+00, -4.1148e-02, 1.2279e-01],
         [-1.8061e-01, -2.1480e+00, 1.5853e-01, 6.6994e-04],
         [-1.3190e-01, -2.4056e-01, -6.6630e-01, 2.37700e-04],
         [-3.8191e-01, -1.0501e+00, -6.7400e-02, -2.0000e+00]]
    B = [[5.1991e-01, 1.1942e+00, 2.1974e-01, -2.4990e-02, -1.7226e-02],
         [3.6266e-01, 1.0836e-01, 7.2562e-03, -1.2133e-02, -7.2114e-03],
         [2.84270e-01, 3.3231e-02, 5.7770e-03, 5.7672e-03, 1.6319e-03],
         [9.3743e-01, 7.3072e-02, 1.7417e-02, 2.0418e-02, 1.0634e-01]]
    C = [[2.2043e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
         [0.0000e+00, 2.7339e+01, 0.0000e+00, 0.0000e+00],
         [3.7700e+00, 1.0341e+01, -7.6298e-03, -4.3237e-03],
         [8.0543e+00, 3.1436e-01, -6.6634e-02, -3.7135e-02],
         [-2.9070e+00, -7.9884e+00, -5.1265e-01, 2.6855e-03]]
    D = [[0, 0, 0, 0, 0],
         [0, 0, 0, 0, 0],
         [1.0036e+00, -8.2350e-01, -1.5200e-01, -5.6233e-02, -5.8600e-02],
         [9.7674e-01, -5.7450e+00, -3.8500e-01, 9.5762e-03, -2.2963e-02],
         [7.1316e+00, 5.5560e-01, 1.3247e-01, 1.5533e-01, 4.8290e-02]]
    _dump("f100_engine", "F-100 Engine",
         "(via AIF100Engine2.m)", "pidtuner/lqg_examples_m_revised/AIF100Engine2.m",
         A, B, C, D, "output_weighted", "identity",
         "Q = CᵀC, R = I(5) in the source. Was excluded (R was never "
         "defined); re-added once the source was corrected (R=eye(5), "
         "matching nu=5 from B/D's column count) — see docs/lqg_plan.md "
         "\"Known issues\". Over-actuated (4 states, 5 inputs); already "
         "open-loop stable, LQR just improves the response.")


def build_example2_rtp():
    A = [[-.0682, .0149, 0],
         [.0458, -.1181, .0218],
         [0, .04683, -.1008]]
    B = [[.3787, .1105, .0229],
         [0, .0449, .0735],
         [0, .0007, .4177]]
    C = np.eye(3)
    D = np.zeros((3, 3))
    Q = [[1, 0, 0], [0, 20, -10], [0, -10, 20]]
    _dump("example2_rtp", "RTP (Franklin/Powell/Emami-Naeini 8e)",
         "Franklin/Powell/Emami-Naeini, Feedback Control of Dynamic Systems, "
         "8th ed. (via AIExample2RTP2.m)",
         "pidtuner/lqg_examples_m_revised/AIExample2RTP2.m",
         A, B, C, D, "custom", "identity",
         "Q = [[1,0,0],[0,20,-10],[0,-10,20]] (hand-picked, off-diagonal "
         "cross terms), R = I(3) in the source. Was excluded (the lqr() "
         "call referenced undefined uppercase A,B,C,D while only lowercase "
         "a,b,c,d were assigned, and the dimensions didn't match even "
         "correcting the case); re-added once the source was corrected — "
         "see docs/lqg_plan.md \"Known issues\". Already open-loop stable "
         "(poles at -0.148, -0.053, -0.086); LQR just improves the response.",
         suggested_Q=Q)


def build_generic_rtp():
    dat_path = os.path.join(_SOURCE_DIR, "rtpsystem.dat")
    raw = np.loadtxt(dat_path)
    assert raw.shape == (20, 20), f"unexpected rtpsystem.dat shape {raw.shape}"
    nx, nu, ny = 15, 5, 5
    A = raw[:nx, :nx]
    B = raw[:nx, nx:nx + nu]
    C = raw[nx:nx + ny, :nx]
    D = raw[nx:nx + ny, nx:nx + nu]
    _dump("generic_rtp", "Generic RTP (Rapid Thermal Processing)",
         "(via AIGeneric_RTP2.m, matrices from rtpsystem.dat)",
         "pidtuner/lqg_examples_m_revised/AIGeneric_RTP2.m",
         A, B, C, D, "output_weighted", "identity",
         "Q = CᵀC, R = I(5) in the source. 15 states, 5 inputs, 5 outputs "
         "— the largest/highest-dimensional plant in the catalog; loaded "
         "from the committed rtpsystem.dat rather than hand-transcribed.")


if __name__ == "__main__":
    build_chemical_reactor()
    build_airc()
    build_drone()
    build_rpv()
    build_tgen()
    build_aircraft_hall()
    build_autm()
    build_distillation_column()
    build_furnace_model()
    build_f100_engine()
    build_example2_rtp()
    build_generic_rtp()
    print("done.")
