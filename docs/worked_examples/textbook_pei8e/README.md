# Worked SISO PID example: PEI8e textbook plant

One of two worked examples under `docs/worked_examples/` — see the
[top-level README](../README.md) for the other. Plant: the FOPDT surrogate
`G(s) = 1/(90s+1)`, `L=13` that Franklin, Powell & Emami-Naeini, *Feedback
Control of Dynamic Systems* (8th ed., "PEI8e") use for the Example 2.18 heat
exchanger in Examples 4.9 (Ziegler-Nichols I, quarter-decay) and 4.10
(Ziegler-Nichols II, ultimate-gain) — this is the reference plant/reproduce
target for future comparisons against the textbook.

For a full numeric cross-check against the book's published Kp/Ti/Td (not
just PIDTuner's own internal consistency), see
`src/gen_zn_memo.py` → `docs/memos/2026-08-18-zn-textbook-validation-memo.html`,
and for a deeper worked run of all nine methods on this plant, see
`src/gen_pid_worked_example_v2_textbook_memo.py` →
`docs/memos/2026-08-25/2026-08-25-pid-worked-example-v2-textbook-memo.html`.
The files here are the same kind of raw, checked-in CLI snapshot as the
`course_benchmark/` example — for diffing behavior changes against a known
baseline, not for the narrative validation the memos provide.

## CLI

Reproduce with:

```bash
cd src
python3 cli_pid.py --plant "1/(90s+1)" --L 13 --method zn1 --json
python3 cli_pid.py --plant "1/(90s+1)" --L 13 --method zn1 \
    --plot ../docs/worked_examples/textbook_pei8e/siso_zn1_cli_step.png
python3 cli_pid.py --plant "1/(90s+1)" --L 13 --method zn2 --json
python3 cli_pid.py --plant "1/(90s+1)" --L 13 --method zn2 \
    --plot ../docs/worked_examples/textbook_pei8e/siso_zn2_cli_step.png
python3 cli_pid.py --plant "1/(90s+1)" --L 13 --method all --json
```

Outputs in this directory:
- `siso_zn1_cli.json` / `siso_zn2_cli.json` — ZN-I / ZN-II gains + metrics
  (`--json`), full strength (the single-method path is unaffected by the
  compare-all default below)
- `siso_zn1_cli_step.png` / `siso_zn2_cli_step.png` — step-response plots
- `siso_all_methods_cli.json` — all 9 methods (12 rows incl. CHR/ZN variants)
  on the same plant; pole cancellation errors out here (this plant only has
  one stable pole, needs two) rather than being skipped, matching how
  `compare_all_methods()` reports failures generally; ZN-I/ZN-II appear as
  `"ZN-I ½"`/`"ZN-II ½"` since `compare_all_methods()` halves those two by
  default (see `pid_compare.py`)

## GUI (Streamlit)

Driven headlessly via Streamlit's `AppTest` framework, same approach as
`course_benchmark/siso_gui_apptest.py`. Reproduce with:

```bash
cd docs/worked_examples/textbook_pei8e
python3 siso_gui_apptest.py
```

`siso_gui_apptest.py` boots `src/streamlit_app.py`, switches the SISO tab's
plant field to `1/(90s+1)` with `L=13` (the default field is the course
benchmark plant), then tunes ZN-I and ZN-II in turn and prints each result's
info block, plus the same `N` filter-bandwidth session-state check as
`course_benchmark/`'s script.

The GUI run reproduces the same gains as the CLI for both methods
(ZN-I: Kp=8.352, Ki=0.3228, Kd=54.02; ZN-II: Kp=6.912, Ki=0.2806, Kd=42.57),
confirming both entry points hit the same tuning code
(`pid_tuning_methods.ZieglerNicholsI`/`II`) on this plant too.
