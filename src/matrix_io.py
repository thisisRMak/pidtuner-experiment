"""Shared matrix-entry/loading helpers for custom (non-preset) MIMO plants
— used by cli_lqg.py's --plant-file flag and streamlit_mimo_panel.py's
custom-entry mode, so both take the same file formats and the GUI's typed
matrices parse the same way the CLI's would.

Three entry points:
  - parse_matlab_literal: text -> np.ndarray, for hand-typed MATLAB-style
    matrix literals ("[0 1 0; 0 0 1; -6 -11 -6]"), used by the GUI's
    custom A/B/C/D text boxes.
  - format_matlab_literal: np.ndarray -> text, the reverse of
    parse_matlab_literal, for pre-filling those same text boxes from an
    existing plant's matrices.
  - load_plant_file: path -> StateSpacePlant, for file-based plant input
    (.json, matching the lqg_examples_json/ preset schema; .mat, via
    scipy.io.loadmat with A/B/C/D variables), used by the CLI's
    --plant-file flag.
"""

from __future__ import annotations

import ast
import json
import os

import numpy as np

from plant import StateSpacePlant


def parse_matlab_literal(text: str) -> np.ndarray:
    """Parse a matrix literal into a 2-D array. Accepts either MATLAB-style
    text (rows separated by ';', entries within a row separated by
    whitespace and/or ',', surrounding '[' ']' optional — e.g.
    "[0 1 0; 0 0 1; -6 -11 -6]" or "1, 0; 0, 1") or Python/numpy-style
    nested-list syntax (e.g. "[[1, 0], [0, 1]]"). A flat (1-D) Python list
    is treated as a single row, matching the MATLAB-style convention."""
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError):
        pass
    else:
        arr = np.asarray(parsed, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
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


def _format_number(x: float) -> str:
    x = float(x)
    if np.isfinite(x) and x == int(x):
        return str(int(x))
    return repr(x)


def format_matlab_literal(arr: np.ndarray) -> str:
    """Format a matrix as MATLAB-style literal text, the reverse of
    parse_matlab_literal — rows separated by '; ', entries within a row
    separated by ' ', wrapped in '[' ']' (e.g. "[0 1 0; 0 0 1; -6 -11 -6]").
    A 1-D array is treated as a single row, matching parse_matlab_literal's
    reverse convention."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"expected a 1-D or 2-D array, got ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError("empty matrix")
    rows = [" ".join(_format_number(v) for v in row) for row in arr]
    return "[" + "; ".join(rows) + "]"


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
