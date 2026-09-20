# CLI Guide — PID and LQR/LQG

One place covering both tracks' command-line tools: one-off runs,
conversational (LLM supervisor) runs, and batch runs that produce a log
file to review. Everything below assumes the `pidtuner` conda/pip
environment is active and you're running from the `src/` directory
(see `src/README.md` for environment setup).

There isn't a single unified `pidtuner` command — each mode is its own flat
script (`cli_pid.py`, `cli_lqg.py`, ...), a deliberate choice (see
`docs/lqg_plan.md` "Decisions") kept consistent across both tracks. This
guide is the thing that ties them together narratively; `--help` on any
script is the source of truth for its exact flags.

## Which script do I want?

| I want to... | PID (transfer function) | LQR/LQG (state-space) |
|---|---|---|
| Tune one plant, one method, right now | `cli_pid.py` | `cli_lqg.py` |
| Tune from signal data only (no known TF) | `cli_pid_blackbox.py` | — (not built; see `docs/lqg_testing.md`) |
| Talk through tradeoffs conversationally | `cli_supervisor_pid.py` | `cli_supervisor_lqg.py` |
| Compare multiple models, one arbitrates | `cli_supervisor_judge_pid.py` | `cli_supervisor_judge_lqg.py` |
| Run everything, save a report to review later | `cli_pid_astrom_batch.py` | `lqg_review.py` |

## One-off runs

### PID — `cli_pid.py`

```bash
# One method, human-readable text
python3 cli_pid.py --plant "1000/((s+1)(10s+1))" --method simc

# One method, JSON, with a step-response plot
python3 cli_pid.py --plant "1000/((s+1)(10s+1))" --method boyd --Ms 1.6 --Mt 1.6 --json --plot out.png

# All 9 methods at once (used as the basis for the batch runner below)
python3 cli_pid.py --plant "1000/((s+1)(10s+1))" --method all --json > all_methods.json
```

Plant syntax, dead time (`--L`), post-processing (`--halve`), and the
black-box two-step pipeline (`cli_pid.py --gen-signal` + `cli_pid_blackbox.py`) are
covered with worked examples in `src/examples/run_whitebox_demo.sh` /
`run_blackbox_demo.sh`.

### LQR/LQG — `cli_lqg.py`

```bash
# List every preset plant (the professor-provided catalog)
python3 cli_lqg.py --list-plants

# One method, one preset, text output + the pre-/post-design checks
python3 cli_lqg.py --plant-preset aircraft_hall --method lqr

# Output-weighted LQR, JSON, with a state/control-effort plot
python3 cli_lqg.py --plant-preset drone --method output_weighted --Qy-scale 2.0 --json --plot out.png

# Full LQG (Kalman filter), output-feedback simulation
python3 cli_lqg.py --plant-preset aircraft_hall --method lqg --sim output_feedback

# Reference tracking (now actually simulates tracking --reference, not just
# computing/printing N̄ — a bug fixed alongside the metrics below) with the
# sign-aware Overshoot/Rise/Settling metrics. Only --sim state_feedback
# supports --reference-tracking's simulation (output_feedback doesn't take
# an r input at all).
python3 cli_lqg.py --plant-preset aircraft_hall --method lqr \
  --reference-tracking --reference 1.0 -0.5 --sim state_feedback

# Model-following: implicit (u=-Kx shaped toward a target model) and
# explicit (u=-K1x-K2xm, xm simulated alongside the plant)
python3 cli_lqg.py --plant-preset aircraft_hall --method implicit --am-diag 0.1 0.07
python3 cli_lqg.py --plant-preset aircraft_hall --method explicit --am-diag 0.1 0.07 \
  --sim model_following --plot out.png

# Comparison 1/2: regulator family (lqr/output_weighted/bryson/lqg), same
# plant/objective -- overlaid ||x(t)||/||u(t)|| plot
python3 cli_lqg.py --plant-preset aircraft_hall --method all --plot compare_regulator.png

# Comparison 2/2: implicit vs. explicit model-following, same target model
# -- overlaid per-output-channel plot against the target model itself
python3 cli_lqg.py --plant-preset aircraft_hall --method model_following_all \
  --am-diag 0.1 0.07 --plot compare_model_following.png

# Iterating on a design: custom Q/R weights (a 5th row alongside the other
# four) + reference tracking, to see how a weight change actually affects
# Overshoot/Rise/Settling -- not just the regulator metrics. Compare this
# run's numbers to the plain --method all --reference-tracking run above:
# lower R here measurably reduced overshoot on this plant, which is *not*
# the naive "less control-effort penalty -> more aggressive -> more
# overshoot" intuition -- check empirically, don't assume the direction.
python3 cli_lqg.py --plant-preset aircraft_hall --method all \
  --Q-diag 1 1 1 1 1 --R-diag 0.1 0.1 \
  --reference-tracking --reference 1.0 -0.5 --plot compare_custom.png
```

`--method` is one of `lqr`, `output_weighted`, `bryson`, `lqg`, `implicit`,
`explicit`, `all` (regulator-family comparison), `model_following_all`
(implicit vs. explicit comparison — see `docs/lqg_testing.md`
"Cross-method comparisons" for why these are two separate comparisons, not
one six-method table). Every run prints pre-/post-design correctness checks by
default (`--no-checks` to suppress) — see `docs/lqg_testing.md` for what
each one verifies. `implicit`/`explicit` require `--am-diag` (desired
model pole magnitudes, one per output — there's no "suggested" target
model the way there's a suggested Q/R for the other four methods, so it's
a required flag, never guessed); `explicit` additionally simulates the
model state `xm(t)` alongside the plant and rejects `--reference-tracking`
(it already tracks a target model, there's no separate constant-reference
mode for it). `--Q-diag`/`--R-diag` (given together) override the
suggested `Q`/`R` with custom per-state/per-input weights — on `lqr` this
replaces the design entirely, on `all` it adds a 5th "Custom LQR" row —
the way to actually iterate on a design rather than only choosing among
the fixed strategies; see `docs/lqg_testing.md` "Custom weights and
reference-tracking."

## Conversational (LLM supervisor)

Both scripts default to a local [Ollama](https://ollama.com) daemon with a
tool-calling model pulled (`ollama pull qwen3-coder:30b`):

```bash
python3 cli_supervisor_pid.py        # PID: describe your plant/priorities in chat
python3 cli_supervisor_lqg.py    # LQR/LQG: name a preset plant + priorities in chat
```

Pass `--provider anthropic` to talk to Claude instead of a local Ollama
daemon:

```bash
python3 cli_supervisor_pid.py --provider anthropic
python3 cli_supervisor_lqg.py --provider anthropic --model claude-sonnet-5
```

This needs an Anthropic API key, resolved in order: `--api-key` (if given,
nothing else is consulted) → `ANTHROPIC_API_KEY` from a `.env` file or the
process environment. With neither, the script exits with an error naming
all three ways to supply one — it never prompts interactively. `--model`
defaults to `claude-haiku-4-5` (cheapest) under `--provider anthropic`, or
`qwen3-coder:30b` under the default `--provider ollama`; `--host`/
`--num-ctx`/`--keep-alive` only apply to Ollama. See `--help` on either
script for the full flag list.

Both are thin conversational layers over the same benchmarks the one-off
CLIs run — see `src/examples/run_supervisor_demo.sh` /
`run_supervisor_lqg_demo.sh` for scripted (non-interactive) example
sessions piped via stdin (Ollama only; there's no Anthropic equivalent
demo script). The two are separate scripts/sessions, not one merged
supervisor — see "Design notes" below for why.

Every turn that produces a plottable result also saves a step-response
(PID) or Response-overlay (LQG) PNG, and the full conversation is
rewritten to an HTML report — both silently, into a per-session output
folder in the current directory, no flag needed. See "Output files"
below for exactly what gets written and why `--log-file` exists
alongside this.

### Judge mode (multi-model arbitration)

`cli_supervisor_judge_pid.py` (SISO/PID) and `cli_supervisor_judge_lqg.py`
(LQR/LQG) each run N candidate supervisors (the same conversation
`cli_supervisor_pid.py --provider ...` / `cli_supervisor_lqg.py --provider
...` would have) in lockstep, then ask a separate judge model to arbitrate
between their recommendations every round. Both are thin CLI wrappers over
the same `supervisor_session_judge_pid.JudgeSession` orchestration
`streamlit_judge_panel.py`'s GUI "LLM Judge" mode uses (SISO/PID track
only, so far) — the LQG one just swaps in `LQGSession` candidates and its
own judge system prompt (`supervisor_prompts_judge_lqg.
JUDGE_SYSTEM_PROMPT_LQG`), via a `judge_system_prompt` argument
`JudgeSession` takes for exactly this reuse.

```bash
python3 cli_supervisor_judge_pid.py \
  --candidate anthropic:claude-haiku-4-5 \
  --candidate openai:gpt-5.6-luna \
  --judge gemini:gemini-3.5-flash-lite \
  --verbose

python3 cli_supervisor_judge_lqg.py \
  --candidate anthropic:claude-haiku-4-5 \
  --candidate openai:gpt-5.6-luna \
  --judge gemini:gemini-3.5-flash-lite \
  --verbose
```

First message to try, PID: `1/(90s+1), delay 13, minimize overshoot`
(same syntax `cli_supervisor_pid.py` takes). LQG: `aircraft_hall, I care
most about control effort` (same syntax `cli_supervisor_lqg.py` takes) —
every candidate receives the identical text each round.

`--candidate provider[:model]` is repeatable (give at least 2; model defaults
to that provider's cheapest tier when omitted); `--judge provider[:model]` is
required — there's no default, and its exact (provider, model) pair must
differ from every candidate's (avoids a model favoring its own family's
answer). Provider is one of `anthropic`/`openai`/`gemini` — no Ollama option
here, since Judge mode is inherently a paid, multi-provider comparison. API
keys resolve the same three-source way as `--provider anthropic` above, just
one flag per provider: `--api-key-anthropic`/`--api-key-openai`/
`--api-key-gemini`.

`gemini-3.8-flash` (the pricier Gemini tier) has shown frequent `503
UNAVAILABLE` ("currently experiencing high demand") errors in live testing —
prefer `gemini-3.5-flash-lite` (the default, cheaper tier, used above) unless
you specifically need the bigger model. A candidate's own transient error (a
503, a rate limit) is caught per-candidate and doesn't sink the round; an
error from the judge call itself isn't (there's nothing left to arbitrate
without it) and surfaces as a REPL error message — just retry the same
message, which is safe since nothing about that failed round was recorded.

Only the judge's reply prints by default; `--verbose` (or `/candidates`
inside the REPL) also shows what each candidate actually did that round —
the same data `streamlit_judge_panel.py`'s transparency expander shows, as
plain text, and (when `--log-file` is given) lands in the log too. Same
`/reset`/`/quit` commands as the other supervisor CLIs. Plots/report save
the same way as the single-provider CLIs above — one file per candidate
per turn, since N candidates routinely give different answers worth
comparing side by side, unlike the GUI's session list (which dedupes
identical results across candidates for its own persistent, cross-turn
plot).

### Output files

All four conversational CLIs (`cli_supervisor_pid.py`,
`cli_supervisor_lqg.py`, and the two Judge scripts above) save into one
per-session folder of the current directory, created at startup and
named `controldesign-<timestamp>-<track/mode>/` — timestamp first so a
plain directory listing sorts chronologically. A `/reset` mid-session
keeps writing into the same folder with fresh content, it doesn't start
a new one. Example, two turns of `cli_supervisor_pid.py`:

```
controldesign-20260920-153042-supervisor-pid/
  report.html                       # the whole conversation so far, rewritten every turn
  turn001-01-1_90s_1.png            # one plot per plottable benchmark call
  turn002-01-1_90s_1.png
```

The Judge scripts add a candidate slug and, for LQG, a plot-kind suffix,
e.g. `turn001-01-anthropic_claude-haiku-4-5-aircraft_hall-response.png`
(+ `-fourcurve.png` only on a turn that supplied `am_diag`). Every write
prints a one-line confirmation (`saved plot: ...` / `saved report: ...`).
This is all silent/automatic — no flag needed, and there's no way to turn
it off.

A full session transcript — both what you type and every printed reply —
is available via `--log-file PATH` (all four scripts). This is *not* the
same as `python3 cli_supervisor_pid.py ... | tee session.log`: piping
stdout only captures what the script itself prints. When stdin is a real
interactive terminal (not piped/heredoc), the terminal's own line-editing
echo writes what you type directly to the tty, bypassing the pipe
entirely — `tee` never sees your side of the conversation, only the
replies. `--log-file` writes both sides explicitly instead, so it's the
only way to get a complete transcript of an interactive session.

## Batch runs (produce a log file to review)

### PID — `cli_pid_astrom_batch.py`

Runs all 9 tuning methods against Åström & Hägglund's 133-process test
batch (*PID Controllers: Theory, Design, and Tuning*, 2nd ed., p. 227):

```bash
python3 cli_pid_astrom_batch.py run --out-dir examples/out/astrom --plot
python3 cli_pid_astrom_batch.py list --out-dir examples/out/astrom
python3 cli_pid_astrom_batch.py show --out-dir examples/out/astrom P7/T=5/L1=0.3
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

`src/examples/run_lqg_demo.sh` exercises every `cli_lqg.py` method
individually first, then calls `lqg_review.py` as its last step — run that
instead of `lqg_review.py` directly if you want to see each method's
individual output along the way.

## Design notes (why things are split the way they are)

- **PID vs LQR/LQG stay separate tracks, not a unified CLI**: different
  plant representations (transfer function vs. state-space), different
  metric vocabularies, no shared math beyond both ultimately producing a
  gain to apply. See `docs/lqg_plan.md` "Decisions".
- **`cli_supervisor_lqg.py` is a separate script from `cli_supervisor_pid.py`**,
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
