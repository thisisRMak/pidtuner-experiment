"""Preset catalog of real MIMO plants ported from the professor's
lqg_examples_m/*.m LQR examples — see docs/lqg_plan.md.

All 12 directly-runnable plants are included (AIExample2RTP.m, the last
holdout, was fixed 2026-08-02 — see docs/lqg_plan.md "Known issues in the
source material"). Regenerate the catalog with lqg_examples_gen.py if a
transcription error is found.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from plant import StateSpacePlant

_DIR = os.path.join(os.path.dirname(__file__), "lqg_examples_json")


@dataclass
class LQGExample:
    key: str
    name: str
    citation: str
    source_file: str
    plant: StateSpacePlant
    suggested_Q_kind: str   # "identity" | "output_weighted" | "custom"
    suggested_R_kind: str   # "identity" | "scaled_identity"
    suggested_R_scale: float
    notes: str
    suggested_Q_raw: Optional[list] = None  # only set when suggested_Q_kind == "custom"

    def build_suggested_Q(self) -> np.ndarray:
        if self.suggested_Q_kind == "identity":
            return np.eye(self.plant.nx)
        if self.suggested_Q_kind == "output_weighted":
            return self.plant.C.T @ self.plant.C
        if self.suggested_Q_kind == "custom":
            return np.array(self.suggested_Q_raw, dtype=float)
        raise ValueError(f"unknown suggested_Q_kind {self.suggested_Q_kind!r}")

    def build_suggested_R(self) -> np.ndarray:
        if self.suggested_R_kind in ("identity", "scaled_identity"):
            return self.suggested_R_scale * np.eye(self.plant.nu)
        raise ValueError(f"unknown suggested_R_kind {self.suggested_R_kind!r}")


def list_examples() -> list:
    if not os.path.isdir(_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(_DIR) if f.endswith(".json"))


def load_example(key: str) -> LQGExample:
    path = os.path.join(_DIR, f"{key}.json")
    if not os.path.isfile(path):
        raise ValueError(
            f"no such LQG example plant: {key!r}. Available: {list_examples()}"
        )
    with open(path) as f:
        data = json.load(f)
    plant = StateSpacePlant(
        A=np.array(data["A"], dtype=float),
        B=np.array(data["B"], dtype=float),
        C=np.array(data["C"], dtype=float),
        D=np.array(data["D"], dtype=float),
        name=data["name"],
    )
    return LQGExample(
        key=data["key"], name=data["name"], citation=data["citation"],
        source_file=data["source_file"], plant=plant,
        suggested_Q_kind=data["suggested_Q_kind"],
        suggested_Q_raw=data.get("suggested_Q"),
        suggested_R_kind=data["suggested_R_kind"],
        suggested_R_scale=data.get("suggested_R_scale", 1.0),
        notes=data.get("notes", ""),
    )
