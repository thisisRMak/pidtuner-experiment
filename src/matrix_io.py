"""Shared matrix-entry/loading helpers for custom (non-preset) MIMO plants
— used by cli_lqg.py's --plant-file flag and streamlit_mimo_panel.py's
custom-entry mode, so both take the same file formats and the GUI's typed
matrices parse the same way the CLI's would.

Two independent entry points:
  - parse_matlab_literal: text -> np.ndarray, for hand-typed MATLAB-style
    matrix literals ("[0 1 0; 0 0 1; -6 -11 -6]"), used by the GUI's
    custom A/B/C/D text boxes.
  - load_plant_file: path -> StateSpacePlant, for file-based plant input
    (.json, matching the lqg_examples_json/ preset schema; .mat, via
    scipy.io.loadmat with A/B/C/D variables), used by the CLI's
    --plant-file flag.
"""

from __future__ import annotations

import json
import os

import numpy as np

from plant import StateSpacePlant


def parse_matlab_literal(text: str) -> np.ndarray:
    """Parse a MATLAB-style matrix literal into a 2-D array: rows separated
    by ';', entries within a row separated by whitespace and/or ','.
    Surrounding '[' ']' are optional. E.g. "[0 1 0; 0 0 1; -6 -11 -6]" or
    "1, 0; 0, 1"."""
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    rows = [r for r in text.split(";") if r.strip() != ""]
    if not rows:
        raise ValueError("empty matrix literal")
    data = []
    ncols = None
    for r in rows:
        vals = [float(v) for v in r.replace(",", " ").split()]
        if not vals:
            raise ValueError(f"empty row in matrix literal: {r!r}")
        if ncols is None:
            ncols = len(vals)
        elif len(vals) != ncols:
            raise ValueError(
                f"ragged matrix literal: row {r!r} has {len(vals)} entries, "
                f"expected {ncols}")
        data.append(vals)
    return np.array(data, dtype=float)


def load_plant_file(path: str) -> StateSpacePlant:
    """Load A/B/C/D from a file: .json (same schema as lqg_examples_json/
    presets — bare A/B/C/D keys, optional name) or .mat (scipy.io.loadmat,
    expects A/B/C/D variables — e.g. a plant exported straight from
    MATLAB)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path) as f:
            data = json.load(f)
        missing = [k for k in ("A", "B", "C", "D") if k not in data]
        if missing:
            raise ValueError(f"{path}: missing required key(s) {missing}")
        return StateSpacePlant(
            A=data["A"], B=data["B"], C=data["C"], D=data["D"],
            name=data.get("name", os.path.splitext(os.path.basename(path))[0]),
        )
    if ext == ".mat":
        from scipy.io import loadmat
        data = loadmat(path)
        missing = [k for k in ("A", "B", "C", "D") if k not in data]
        if missing:
            raise ValueError(f"{path}: missing required variable(s) {missing}")
        return StateSpacePlant(
            A=data["A"], B=data["B"], C=data["C"], D=data["D"],
            name=os.path.splitext(os.path.basename(path))[0],
        )
    raise ValueError(f"unsupported plant file extension {ext!r} (expected .json or .mat)")
