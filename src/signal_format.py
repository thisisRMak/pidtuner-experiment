"""The interchange between the signal generator and the black-box tuner.

`Signal` carries only sampled input/output arrays, an explicit sample time,
and safe scalar metadata about how the experiment was run. It deliberately
has no notion of a transfer function, dead time, or any other ground-truth
model parameter — and this module does not import `plant`, so nothing here
could construct or smuggle one even by accident.

Transport comes in two forms that both hand entity C the same `Signal`
object: `save_signal`/`load_signal` for a file-based artifact, or simply
constructing/passing a `Signal` in-process for "live" consumption. Iterating
a `Signal` (`for t, u, y in signal`) is the one consumption interface that
works identically either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

_EXPERIMENTS = ("step", "relay")


@dataclass
class Signal:
    """A published input/output experiment record.

    t, u, y : sampled time, input, and measured-output arrays (uniform dt).
    dt      : the sample time, carried explicitly rather than re-derived.
    experiment : "step" or "relay" — which kind of test produced this signal.
    meta    : safe, JSON-serializable scalars describing the experiment
              (e.g. step_amp, noise_sigma, seed, h, setpoint, hysteresis).
              Never a model coefficient — there is nothing plant-shaped here.
    """

    t: np.ndarray
    u: np.ndarray
    y: np.ndarray
    dt: float
    experiment: str
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.experiment not in _EXPERIMENTS:
            raise ValueError(
                f"experiment must be one of {_EXPERIMENTS}, got {self.experiment!r}"
            )
        try:
            json.dumps(self.meta)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Signal.meta must contain only JSON-serializable scalars "
                "(no arrays, no model objects)"
            ) from e
        if not (len(self.t) == len(self.u) == len(self.y)):
            raise ValueError("t, u, y must have equal length")

    def __len__(self):
        return len(self.t)

    def __iter__(self):
        """Yield (t_k, u_k, y_k) in order — the transport-agnostic 'live' interface."""
        return zip(self.t, self.u, self.y)


def save_signal(signal: Signal, path: str) -> None:
    """Write a Signal to a self-describing .npz file.

    allow_pickle is never used on load — the on-disk format can only ever
    hold arrays and a JSON string, so it structurally cannot carry a
    pickled TransferFunction or any other non-scalar object.
    """
    np.savez(
        path,
        t=np.asarray(signal.t),
        u=np.asarray(signal.u),
        y=np.asarray(signal.y),
        dt=np.array(signal.dt),
        experiment=np.array(signal.experiment),
        meta_json=np.array(json.dumps(signal.meta)),
    )


def load_signal(path: str) -> Signal:
    """Read a Signal previously written by save_signal."""
    with np.load(path, allow_pickle=False) as data:
        return Signal(
            t=data["t"],
            u=data["u"],
            y=data["y"],
            dt=float(data["dt"]),
            experiment=str(data["experiment"]),
            meta=json.loads(str(data["meta_json"])),
        )
