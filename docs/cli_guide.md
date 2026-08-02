# CLI Guide — PID and LQR/LQG

One place covering both tracks' command-line tools: one-off runs,
conversational (LLM supervisor) runs, and batch runs that produce a log
file to review. Everything below assumes the `pidtuner` conda/pip
environment is active and you're running from the `pidtuner/` directory
(see `pidtuner/README.md` for environment setup).

There isn't a single unified `pidtuner` command — each mode is its own flat
script (`cli.py`, `cli_lqg.py`, ...), a deliberate choice (see
`docs/lqg_plan.md` "Decisions") kept consistent across both tracks. This
guide is the thing that ties them together narratively; `--help` on any
script is the source of truth for its exact flags.

## Which script do I want?

| I want to... | PID (transfer function) | LQR/LQG (state-space) |
|---|---|---|
| Tune one plant, one method, right now | `cli.py` | `cli_lqg.py` |
| Tune from signal data only (no known TF) | `cli_blackbox.py` | — (not built; see `docs/lqg_testing.md`) |
| Talk through tradeoffs conversationally | `cli_supervisor.py` | `cli_supervisor_lqg.py` |
| Run everything, save a report to review later | `cli_astrom_batch.py` | `lqg_review.py` |

## One-off runs

### PID — `cli.py`

```bash
# One method, human-readable text
python3 cli.py --plant "1000/((s+1)(10s+1))" --method simc

# One method, JSON, with a step-response plot
python3 cli.py --plant "1000/((s+1)(10s+1))" --method boyd --Ms 1.6 --Mt 1.6 --json --plot out.png

# All 9 methods at once (used as the basis for the batch runner below)
python3 cli.py --plant "1000/((s+1)(10s+1))" --method all --json > all_methods.json
```

Plant syntax, dead time (`--L`), post-processing (`--halve`), and the
black-box two-step pipeline (`cli.py --gen-signal` + `cli_blackbox.py`) are
covered with worked examples in `pidtuner/examples/run_whitebox_demo.sh` /
`run_blackbox_demo.sh`.

### LQR/LQG — `cli_lqg.py`

```bash
# List every preset plant (the professor-provided catalog)
python3 cli_lqg.py --list-plants

# One method, one preset, text output + the pre-/post-design checks
python3 cli_lqg.py --plant-preset aircraft_hall --method lqr

# Output-weighted LQR, JSON, with a state/control-effort plot
python3 cli_lqg.py --plant-preset drone --method output_weighted --Qy-scale 2.0 --json --plot out.png

# Full LQG (Kalman filter) + reference tracking, output-feedback simulation
python3 cli_lqg.py --plant-preset aircraft_hall --method lqg --sim output_feedback --reference-tracking
```

`--method` is one of `lqr`, `output_weighted`, `bryson`, `lqg`. Every run
prints pre-/post-design correctness checks by default (`--no-checks` to
suppress) — see `docs/lqg_testing.md` for what each one verifies.
`ImplicitModelFollowing`/`ExplicitModelFollowing` aren't wired into this
CLI yet (they need a target model `Am`/`Q1` with no natural per-preset
default); use them from Python directly (`lqg_implicit.py`/
`lqg_explicit.py`) or see `test_lqg.py` for worked examples.

## Conversational (LLM supervisor)

Both require a local [Ollama](https://ollama.com) daemon with a
tool-calling model pulled (`ollama pull qwen3-coder:30b`).

```bash
python3 cli_supervisor.py        # PID: describe your plant/priorities in chat
python3 cli_supervisor_lqg.py    # LQR/LQG: name a preset plant + priorities in chat
```

Both are thin conversational layers over the same benchmarks the one-off
CLIs run — see `pidtuner/examples/run_supervisor_demo.sh` /
`run_supervisor_lqg_demo.sh` for scripted (non-interactive) example
sessions piped via stdin. The two are separate scripts/sessions, not one
merged supervisor — see "Design notes" below for why.

## Batch runs (produce a log file to review)

### PID — `cli_astrom_batch.py`

Runs all 9 tuning methods against Åström & Hägglund's 133-process test
batch (*PID Controllers: Theory, Design, and Tuning*, 2nd ed., p. 227):

```bash
python3 cli_astrom_batch.py run --out-dir examples/out/astrom --plot
python3 cli_astrom_batch.py list --out-dir examples/out/astrom
python3 cli_astrom_batch.py show --out-dir examples/out/astrom P7/T=5/L1=0.3
```

Writes `manifest.json` (every plant → path, timestamp, git commit),
`summary.json`/`summary.txt` (per-method stability rate + Ms/Mt/IAE/OS%/ts
stats across the whole batch), and one `plant.json`/`result.json`
(+`step_response.png` with `--plot`) per plant. `show` pretty-prints one
saved result later without re-running anything.

### LQR/LQG — `lqg_review.py`

Runs `LQR` (each preset's own suggested `Q`/`R`) against every plant in the
preset catalog, with the full correctness-check suite:

```bash
python3 lqg_review.py
```

Writes `docs/lqg_review.md` (human-readable — the file to actually hand
someone for review, e.g. the professor) and
`examples/out/lqg_review.json` (machine-readable, same data). Currently
covers `LQR` only, not all four methods — extend `lqg_review.py`'s
`run_all()` if you want `output_weighted`/`bryson`/`lqg` rows too (the
`supervisor_tools_lqg.run_lqg_benchmark` function already runs all four
per plant and is a ready-made reference for how to build those rows).

`pidtuner/examples/run_lqg_demo.sh` exercises every `cli_lqg.py` method
individually first, then calls `lqg_review.py` as its last step — run that
instead of `lqg_review.py` directly if you want to see each method's
individual output along the way.

## Design notes (why things are split the way they are)

- **PID vs LQR/LQG stay separate tracks, not a unified CLI**: different
  plant representations (transfer function vs. state-space), different
  metric vocabularies, no shared math beyond both ultimately producing a
  gain to apply. See `docs/lqg_plan.md` "Decisions".
- **`cli_supervisor_lqg.py` is a separate script from `cli_supervisor.py`**,
  not a mode flag on the same one: the PID supervisor's `Session` class
  gates its one-of-two benchmark tools on `tf_known` (does the user know
  their transfer function?) — a concept with no LQG analog, since there's
  one LQG benchmark tool, always available, and no black-box LQG track to
  gate against. Forcing LQG through that two-tool-slot shape would mean
  either faking a boolean that means nothing or reworking tested PID
  session logic. See `supervisor_session_lqg.py`'s module docstring.
- **LQR and LQG are one benchmark tool, not two**, inside the LQG
  supervisor: LQG is literally LQR plus a steady-state Kalman filter,
  sharing the same Q/R machinery and `LQGDesignResult` type — splitting
  them would make "should I use LQR or LQG" (exactly the kind of tradeoff
  question a supervisor should answer) harder to ask in one turn. Mirrors
  how the PID supervisor already bundles all 9 tuning methods behind one
  `run_whitebox_benchmark` call rather than nine.
