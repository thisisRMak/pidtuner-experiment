"""blackbox.py — Entity C: the black-box PID tuner.

Isolation contract: this module never imports `plant.py` and never receives
a `plant.TransferFunction` built from ground-truth coefficients — it only
ever consumes `signal_format.Signal` objects (arrays + an explicit sample
time + safe metadata). It DOES legitimately construct `TransferFunction`
*surrogate* instances as outputs of its own identification (via
`identify.FOPDT.to_tf()` / `identify.SOPDT.to_tf()`), because
`TransferFunction` is just a rational-function container type, not "the
truth" by definition — those surrogates feed the same, unmodified `Boyd`,
`StablePoleCancellation`, and `find_ultimate_gain`/`select_slowest_stable_poles`
code the white-box pipeline uses. Isolation means "never reads a true
plant's num/den/L," not "never touches the type."
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Optional

from signal_format import Signal
from identify import (
    FOPDT,
    SOPDT,
    fit_fopdt_from_step,
    fit_sopdt_from_step,
    identify_ultimate_gain_from_relay,
    find_ultimate_gain,
)
from tune import select_slowest_stable_poles
from tuning_methods import (
    StablePoleCancellation,
    ZieglerNicholsI,
    ZieglerNicholsII,
    Amigo,
    Simc,
    Boyd,
    CohenCoon,
    ChienHronesReswick,
    TyreusLuyben,
    TuningResult,
)


def _safe(fn):
    try:
        return fn(), None
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        return None, str(exc)


@dataclass
class BlackBoxModel:
    """The process model identified purely from published signals."""

    fopdt: Optional[FOPDT] = None
    fopdt_reason: Optional[str] = None
    sopdt: Optional[SOPDT] = None
    sopdt_reason: Optional[str] = None
    Ku: Optional[float] = None
    Pu: Optional[float] = None
    ku_pu_source: Optional[str] = None  # "relay" | "surrogate-bode"
    ku_pu_reason: Optional[str] = None


def identify_from_signals(step_signal: Optional[Signal] = None,
                           relay_signal: Optional[Signal] = None) -> BlackBoxModel:
    """Identify a FOPDT model, a SOPDT surrogate (if the data supports one),
    and Ku/Pu, from published signals only. Ku/Pu prefers a direct relay-test
    measurement when available, falling back to the analytic Bode crossing
    of the fitted surrogate model otherwise."""
    model = BlackBoxModel()

    if step_signal is not None:
        if step_signal.experiment != "step":
            raise ValueError(f"expected a 'step' signal, got {step_signal.experiment!r}")
        step_amp = step_signal.meta.get("step_amp", 1.0)
        model.fopdt, model.fopdt_reason = _safe(
            lambda: fit_fopdt_from_step(step_signal.t, step_signal.y, step_amp))
        if model.fopdt is not None:
            model.sopdt, model.sopdt_reason = _safe(
                lambda: fit_sopdt_from_step(step_signal.t, step_signal.y, step_amp,
                                             fopdt_hint=model.fopdt))
        else:
            model.sopdt_reason = "no FOPDT fit available to seed a SOPDT fit"
    else:
        model.fopdt_reason = "no step signal supplied"
        model.sopdt_reason = "no step signal supplied"

    if relay_signal is not None:
        if relay_signal.experiment != "relay":
            raise ValueError(f"expected a 'relay' signal, got {relay_signal.experiment!r}")
        h = relay_signal.meta.get("h", 1.0)
        setpoint = relay_signal.meta.get("setpoint", 0.0)
        hysteresis = relay_signal.meta.get("hysteresis", 0.0)
        try:
            model.Ku, model.Pu = identify_ultimate_gain_from_relay(
                relay_signal.t, relay_signal.u, relay_signal.y, h,
                setpoint=setpoint, hysteresis=hysteresis)
            model.ku_pu_source = "relay"
        except Exception as exc:  # noqa: BLE001
            model.ku_pu_reason = f"relay identification failed: {exc}"

    if model.Ku is None:
        surrogate = model.sopdt.to_tf() if model.sopdt is not None else (
            model.fopdt.to_tf() if model.fopdt is not None else None)
        if surrogate is not None:
            try:
                Ku, Pu, _w180 = find_ultimate_gain(surrogate)
                model.Ku, model.Pu, model.ku_pu_source = Ku, Pu, "surrogate-bode"
            except Exception as exc:  # noqa: BLE001
                if model.ku_pu_reason is None:
                    model.ku_pu_reason = f"surrogate Bode crossing failed: {exc}"
        elif model.ku_pu_reason is None:
            model.ku_pu_reason = "no relay signal and no surrogate model available"

    return model


@dataclass
class BlackBoxTuningRow:
    name: str
    result: Optional[TuningResult]
    available: bool
    reason: Optional[str] = None


class BlackBoxTuner:
    """Entity C's orchestration: identify a model from published signals,
    then attempt all 9 tuning methods against it, reporting which ones
    were unavailable and why."""

    def __init__(self, step_signal: Optional[Signal] = None,
                 relay_signal: Optional[Signal] = None):
        self.step_signal = step_signal
        self.relay_signal = relay_signal
        self._model: Optional[BlackBoxModel] = None

    def identify(self) -> BlackBoxModel:
        self._model = identify_from_signals(self.step_signal, self.relay_signal)
        return self._model

    @staticmethod
    def _pole_cancel(surrogate):
        p1, p2 = select_slowest_stable_poles(surrogate)
        dc = abs(surrogate.dc_gain())
        Kd = 1.0 / dc if dc > 1e-9 else 1.0
        return StablePoleCancellation(surrogate, p1, p2, Kd=Kd).tune()

    #: CLI/API slug -> row name, for callers that want a single method instead
    #: of the full sweep. Mirrors cli.py's --method choices (minus "all").
    #: "chr" is intentionally absent — it fans out into 4 rows disambiguated
    #: by chr_response/chr_overshoot instead of a fixed name.
    METHOD_SLUGS = {
        "pole_cancellation": "Pole cancellation",
        "zn1": "ZN-I",
        "zn2": "ZN-II",
        "amigo": "AMIGO",
        "simc": "SIMC",
        "boyd": "Boyd",
        "cohen_coon": "Cohen–Coon",
        "tyreus_luyben": "Tyreus–Luyben",
    }

    def tune_all(self, method: Optional[str] = None,
                 chr_response: str = "setpoint", chr_overshoot: int = 0) -> list[BlackBoxTuningRow]:
        """Run the tuning methods against the identified model.

        `method` (default None == "all") restricts this to a single method,
        by the same slugs cli.py's --method uses ("zn1", "boyd", "chr", ...).
        Unknown slugs raise ValueError. Restricting also skips the *work* for
        the other methods, not just their reporting — e.g. Boyd's Ms/Mt
        search only runs when it was actually asked for.
        """
        if method not in (None, "all") and method not in self.METHOD_SLUGS and method != "chr":
            raise ValueError(
                f"Unknown method {method!r}; choose from "
                f"{sorted(list(self.METHOD_SLUGS) + ['chr'])}")

        model = self._model or self.identify()
        rows: list[BlackBoxTuningRow] = []

        def wanted(slug):
            return method in (None, "all") or method == slug

        def add(name, slug, fn, unavailable_reason):
            if not wanted(slug):
                return
            if unavailable_reason is not None:
                rows.append(BlackBoxTuningRow(name, None, False, unavailable_reason))
                return
            result, err = _safe(fn)
            if err is not None:
                rows.append(BlackBoxTuningRow(name, None, False, err))
            else:
                note = ("[black-box] Derived from a model identified purely from "
                        "published signals — no ground-truth plant was consulted. "
                        "Any reference in the note below to 'the true plant' "
                        "actually means this identified surrogate.")
                result = dataclasses.replace(
                    result, black_box=True, notes=(note + "\n" + result.notes).strip())
                rows.append(BlackBoxTuningRow(name, result, True, None))

        pole_surrogate = model.sopdt.to_tf() if model.sopdt is not None else None
        boyd_surrogate = pole_surrogate or (model.fopdt.to_tf() if model.fopdt is not None else None)

        seed = None
        if model.fopdt is not None and wanted("boyd"):
            seed, _ = _safe(lambda: Simc(model.fopdt).tune().gains)

        add("Pole cancellation", "pole_cancellation", lambda: self._pole_cancel(pole_surrogate),
            None if pole_surrogate is not None else
            (model.sopdt_reason or "no SOPDT surrogate available"))

        add("ZN-I", "zn1", lambda: ZieglerNicholsI(model.fopdt).tune(),
            None if model.fopdt is not None else model.fopdt_reason)

        add("ZN-II", "zn2", lambda: ZieglerNicholsII(model.Ku, model.Pu).tune(),
            None if model.Ku is not None else model.ku_pu_reason)

        add("AMIGO", "amigo", lambda: Amigo(model.fopdt).tune(),
            None if model.fopdt is not None else model.fopdt_reason)

        add("SIMC", "simc", lambda: Simc(model.fopdt, tau2=(model.sopdt.tau2 if model.sopdt else None)).tune(),
            None if model.fopdt is not None else model.fopdt_reason)

        add("Boyd", "boyd", lambda: Boyd(boyd_surrogate, seed_gains=seed).tune(),
            None if boyd_surrogate is not None else
            (model.fopdt_reason or "no surrogate model available"))

        add("Cohen–Coon", "cohen_coon", lambda: CohenCoon(model.fopdt).tune(),
            None if model.fopdt is not None else model.fopdt_reason)

        for resp, ov, label in [("setpoint", 0, "CHR set 0%"), ("setpoint", 20, "CHR set 20%"),
                                 ("load", 0, "CHR load 0%"), ("load", 20, "CHR load 20%")]:
            if method == "chr" and (resp, ov) != (chr_response, chr_overshoot):
                continue
            add(label, "chr", lambda resp=resp, ov=ov: ChienHronesReswick(model.fopdt, resp, ov).tune(),
                None if model.fopdt is not None else model.fopdt_reason)

        add("Tyreus–Luyben", "tyreus_luyben", lambda: TyreusLuyben(model.Ku, model.Pu).tune(),
            None if model.Ku is not None else model.ku_pu_reason)

        return rows
