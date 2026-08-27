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

No GUI (Streamlit `AppTest`) reproduction for this example yet — `course_benchmark/`
covers CLI/GUI parity; add one here the same way if that becomes useful for
this plant too.
