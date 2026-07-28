# CLI demo scripts

Runnable, end-to-end examples of the three PIDTuner CLIs. Run from anywhere;
each script `cd`s to `pidtuner/` itself and writes artifacts to `examples/out/`.

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

Run them all:

```bash
for s in run_whitebox_demo.sh run_blackbox_demo.sh run_supervisor_demo.sh; do
    ./"$s"
done
```
