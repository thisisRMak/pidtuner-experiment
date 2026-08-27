# Worked SISO PID example: course benchmark plant (paper's minimum SISO result)

One of two worked examples under `docs/worked_examples/` — see the
[top-level README](../README.md) for the other. Plant: the course benchmark
`G(s) = 1000/((s+1)(10s+1)) · exp(-0.5s)` (same plant as `benchmark_plant()`
in `src/test_pid_tuner.py`), tuned with Ziegler-Nichols II.

## CLI

Reproduce with:

```bash
cd src
python3 cli_pid.py --plant "1000 / ((s+1)*(10s+1))" --L 0.5 --method zn2 --json
python3 cli_pid.py --plant "1000 / ((s+1)*(10s+1))" --L 0.5 --method zn2 \
    --plot ../docs/worked_examples/course_benchmark/siso_zn2_cli_step.png
python3 cli_pid.py --plant "1000 / ((s+1)*(10s+1))" --L 0.5 --method all --json
```

Outputs in this directory:
- `siso_zn2_cli.json` — ZN-II gains + metrics (`--json`), full strength (the
  single-method path is unaffected by the compare-all default below)
- `siso_zn2_cli_step.png` — step-response plot
- `siso_all_methods_cli.json` — all 9 methods (12 rows incl. CHR/ZN variants)
  on the same plant; ZN-I/ZN-II appear as `"ZN-I ½"`/`"ZN-II ½"` since
  `compare_all_methods()` halves those two by default (see `pid_compare.py`)

## GUI (Streamlit)

Driven headlessly via Streamlit's `AppTest` framework (no display available
in this environment, so this is a scripted run of the real app code path,
not a manual browser click-through). Reproduce with:

```bash
cd docs/worked_examples/course_benchmark
python3 siso_gui_apptest.py
```

`siso_gui_apptest.py` boots `src/streamlit_app.py`, selects the SISO tab's
"Ziegler-Nichols II" method with `L=0.5`, clicks "Tune & simulate", and
prints the resulting gains plus the `N` filter-bandwidth session-state value
and the live `simulate_closed_loop` default `N`. Output: `siso_gui_apptest.log`.

The GUI run reproduces the same gains as the CLI (Kp=0.0143, Ki=0.00634,
Kd=0.00809), confirming both entry points hit the same tuning code
(`pid_tuning_methods.ZieglerNicholsII`) and simulate at the same default
`N=80`.
