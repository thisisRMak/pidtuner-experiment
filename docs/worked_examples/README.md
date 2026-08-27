# Worked SISO PID examples

Two fixed reference plants PIDTuner is run against end-to-end via the real
CLI (and, for one, the real GUI), with checked-in actual output — not
hand-written numbers — so behavior changes can be diffed against a known
baseline instead of drifting unnoticed. Each subfolder is self-contained
with its own reproduce commands.

- [`course_benchmark/`](course_benchmark/) — the class benchmark plant
  `1000/((s+1)(10s+1))·exp(-0.5s)`, ZN-II. Covers both the CLI and the GUI
  (headless Streamlit `AppTest`), confirming both entry points hit the same
  tuning code and produce the same gains.
- [`textbook_pei8e/`](textbook_pei8e/) — the PEI8e (Franklin/Powell/
  Emami-Naeini, 8th ed.) Examples 4.9/4.10 heat-exchanger surrogate
  `1/(90s+1)`, `L=13`. ZN-I and ZN-II both highlighted, since the two
  textbook examples cover one method each. CLI only so far.

Both examples run PIDTuner's full 9-method comparison (`--method all`) in
addition to their headline single method, so `siso_all_methods_cli.json` in
each folder doubles as a reference for the comparison-table behavior more
generally, not just the one method each is named for.
