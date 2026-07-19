# PIDTuner Modularization Refactor

Decomposed the monolithic `pidtuner/app.py` Tkinter application into focused
modules, refactored the 9 PID tuning methods into an object-oriented class
hierarchy, and added a CLI — while preserving all public signatures and
behavior exactly. Completed 2026-07-19 on branch `refactor/modularize`.

## Final module layout

| Module | Owns |
|---|---|
| `plant.py`, `identify.py`, `simulate.py` | Unchanged — plant modeling, FOPDT/relay identification, closed-loop simulation |
| `tuning_methods.py` | `BaseTuningMethod` + 9 subclasses (the real formulas/algorithms), plus `PIDGains`, `TuningResult`, `halve_gains` |
| `tune.py` | Thin `tune_*` wrapper functions, each instantiating the matching `tuning_methods` class and calling `.tune()`; also owns `select_slowest_stable_poles` (real pole-selection logic with no other home yet) |
| `compare.py` | Cross-method comparison metrics (heatmap/radar data layer); calls `tuning_methods` classes directly |
| `cli.py` | argparse CLI; calls `tuning_methods` classes directly |
| `widgets.py` | `ClosableNotebook` (Tkinter tab-close widget), extracted from `app.py` |
| `comparison_views.py` | Heatmap table + radar chart rendering, extracted from `app.py` |
| `response_plotting.py` | 3-axis response-plot figure creation/drawing, extracted from `app.py` |
| `parameter_panels.py` | Per-method argument panel builders, extracted from `app.py` |
| `app.py` | Tkinter controller/state only; imports the four modules above and calls `tuning_methods` classes directly for tune dispatch (same pattern as `cli.py`). Shrank from 1279 → ~900 lines. |

## Key design decisions

**`tune.py`'s wrapper functions are an intentional adapter layer, not vestigial.**
The real algorithm for every method now lives in its `tuning_methods` class;
`tune.py`'s functions are 1-line pass-throughs kept *only* because
`test_pid_tuner.py`'s ~72 assertions call them by name, and the refactor's
invariant was that existing callers/tests must keep working unchanged.
`app.py`, `cli.py`, and `compare.py` all bypass `tune.py` and call the
`tuning_methods` classes directly — proof the OO layer is real, not
decorative. See the `TODO` in `tune.py`'s module docstring for the deferred
option to delete these wrappers and rewrite the test suite against the
classes directly (parked; needs explicit sign-off since it touches every
test call site).

**Extraction preserved behavior by construction, not by re-derivation.**
Each new UI module (`widgets.py`, `comparison_views.py`, `response_plotting.py`,
`parameter_panels.py`) was created by moving the exact existing code out of
`app.py` verbatim (same widget calls, same strings, same layout), then
wiring `app.py` to call the moved code — never rewritten from scratch.

## Verification performed

- `pidtuner/test_pid_tuner.py`: **90/90 pass**, re-checked after every change.
- `cli.py` exercised for all 9 methods individually plus `--method all`,
  valid JSON output each time.
- `compare_all_methods()`: 12/12 methods stable, gains matching the
  pre-refactor baseline to 6+ significant figures.
- `app.py`/`cli.py` import cleanly; `pidtuner.spec` unchanged (still targets
  `app.py`, `main()` intact).
- Full interactive GUI click-through was not possible in the dev sandbox —
  the remote X display throws `BadValue`/`X_OpenFont` on a Unicode glyph
  (`⊞`) used in one button label. Confirmed this reproduces identically on
  the original, unmodified `app.py` from `main`, so it's a pre-existing
  environment limitation, not a regression. Partial GUI checks (widget
  construction, Matplotlib-in-Tk embedding, per-section frame builders up to
  that glyph) all passed.

## Known follow-ups

- `tune.py`'s 9 wrapper functions are now used only by the test suite —
  see the `TODO` there for the plan to eventually remove them.
- `select_slowest_stable_poles` is real, unique logic still living in
  `tune.py`; it conceptually belongs on `StablePoleCancellation` in
  `tuning_methods.py` but hasn't been moved (low priority, not blocking).
