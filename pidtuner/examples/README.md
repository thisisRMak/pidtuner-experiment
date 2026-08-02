# CLI demo scripts

Runnable, end-to-end examples of the PIDTuner CLIs. Run from anywhere; each
script `cd`s to `pidtuner/` itself and writes artifacts to `examples/out/`.
See `../../docs/cli_guide.md` for a narrative walkthrough of every script
(one-off runs, conversational runs, batch runs) across both the PID and
LQR/LQG tracks.

Requires the `pidtuner` conda/pip environment to be active (see
`../README.md`).

- **`run_whitebox_demo.sh`** — `cli.py`, the white-box tuner. Runs
  individual tuning methods (SIMC, Boyd, ZN-II, pole cancellation,
  Cohen-Coon...) against known transfer functions, in both text and JSON
  form, plus a `--method all` comparison with an overlaid step-response
  plot.

- **`run_blackbox_demo.sh`** — the two-process black-box pipeline.
  `cli.py --gen-signal` (entity A) publishes step/relay test signals to
  `.npz` files *without* exposing the plant's transfer function; then
  `cli_blackbox.py` (entity C) identifies a model and tunes all 9 methods
  from those signal files alone.

- **`run_supervisor_demo.sh`** — `cli_supervisor.py`, the conversational
  LLM supervisor. Requires a local [Ollama](https://ollama.com) daemon
  with `qwen3-coder:30b` pulled; the script checks for both and skips
  with instructions if unavailable. Pipes a scripted conversation into
  the REPL's stdin so it runs non-interactively.

- **`run_lqg_demo.sh`** — `cli_lqg.py`, the LQR/LQG design track (a
  separate paradigm from the PID tuners above — state-space plants, not
  transfer functions). Runs every method (`lqr`, `output_weighted`,
  `bryson`, `lqg`) plus reference tracking, then sweeps all 11 preset
  plants and writes a professor-facing review
  (`docs/lqg_review.md`) via `lqg_review.py`. See
  `docs/lqg_testing.md` for what the printed pre-/post-design checks mean.

- **`run_supervisor_lqg_demo.sh`** — `cli_supervisor_lqg.py`, the LQR/LQG
  conversational supervisor (a separate script/session from
  `cli_supervisor.py`, not a mode flag on it — see `docs/cli_guide.md`
  "Design notes" for why). Same Ollama requirement/skip behavior as
  `run_supervisor_demo.sh`. Its one benchmark tool bundles `lqr`,
  `output_weighted`, `bryson`, and `lqg` together (LQR and LQG are treated
  as one family here, not two).

Run them all:

```bash
for s in run_whitebox_demo.sh run_blackbox_demo.sh run_supervisor_demo.sh \
         run_lqg_demo.sh run_supervisor_lqg_demo.sh; do
    ./"$s"
done
```
