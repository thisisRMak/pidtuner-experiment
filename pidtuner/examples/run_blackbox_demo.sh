#!/usr/bin/env bash
# Demonstrates the black-box pipeline: cli.py exports a Signal (entity A,
# the "plant owner") to a .npz file; cli_blackbox.py (entity C, the
# "black-box tuner") consumes only that file and never sees --plant/--L.
# This mirrors the process-boundary isolation enforced by blackbox.py.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=examples/out
mkdir -p "$OUT"

PLANT="1000/((s+1)(10s+1))"
L=1.5

echo "### 1. Entity A: run a step test against the plant and publish the signal ###"
python3 cli.py --plant "$PLANT" --L "$L" \
    --gen-signal step --out-signal "$OUT/step.npz" \
    --step-amp 1.0 --noise-sigma 0.01 --seed 42

echo
echo "### 2. Entity A: run a relay test too, so ZN-II/Tyreus-Luyben become available ###"
python3 cli.py --plant "$PLANT" --L "$L" \
    --gen-signal relay --out-signal "$OUT/relay.npz" \
    --h 1.0 --setpoint 0.0 --hysteresis 0.02

echo
echo "### 3. Entity C: identify + tune from the step signal alone (text) ###"
python3 cli_blackbox.py --in-signal-step "$OUT/step.npz"

echo
echo "### 4. Entity C: add the relay signal too, JSON output, saved to file ###"
python3 cli_blackbox.py \
    --in-signal-step "$OUT/step.npz" \
    --in-signal-relay "$OUT/relay.npz" \
    --json > "$OUT/blackbox_result.json"
echo "wrote $OUT/blackbox_result.json"
python3 -c "
import json
r = json.load(open('$OUT/blackbox_result.json'))
print('Identified FOPDT:', r['model']['fopdt'])
print('Delay detected:', r['model']['delay_detected'], '-', r['model']['delay_reason'])
print('Ku, Pu:', r['model']['Ku'], r['model']['Pu'], '(source:', r['model']['ku_pu_source'] + ')')
"

echo
echo "### 5. Relay-only input (no step test at all) ###"
python3 cli_blackbox.py --in-signal-relay "$OUT/relay.npz"

echo
echo "Done. Signals + results in $OUT/"
