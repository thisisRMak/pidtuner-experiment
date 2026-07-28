#!/usr/bin/env bash
# Demonstrates cli.py: the white-box tuner, which knows the plant's true
# transfer function and can run any of the 9 tuning methods against it.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=examples/out
mkdir -p "$OUT"

PLANT="1000/((s+1)(10s+1))"

echo "### 1. Single method, human-readable text output (SIMC) ###"
python3 cli.py --plant "$PLANT" --method simc

echo
echo "### 2. Single method, JSON output (Boyd, custom Ms/Mt robustness bounds) ###"
python3 cli.py --plant "$PLANT" --method boyd --Ms 1.6 --Mt 1.6 --json

echo
echo "### 3. Plant with dead time, Ziegler-Nichols II (ultimate-gain) ###"
python3 cli.py --plant "1/(s+1)^3" --method zn2

echo
echo "### 4. Stable pole cancellation with explicit poles (p1/p2 are pole magnitudes, i.e. plant poles at s=-p1, s=-p2) ###"
python3 cli.py --plant "1/((s+1)(s+2)(s+3))" --method pole_cancellation --p1 1 --p2 2

echo
echo "### 5. Halve-gains post-processing (Prof.'s PEI9e recommendation) ###"
python3 cli.py --plant "$PLANT" --method zn1 --halve

echo
echo "### 6. Compare all 9 methods at once, JSON, saved to file ###"
python3 cli.py --plant "$PLANT" --method all --json > "$OUT/compare_all.json"
echo "wrote $OUT/compare_all.json ($(python3 -c "import json;print(len(json.load(open('$OUT/compare_all.json'))))") rows)"

echo
echo "### 7. Compare all methods + overlay step-response plot ###"
python3 cli.py --plant "$PLANT" --method all --plot "$OUT/compare_all.png" --json > /dev/null
echo "wrote $OUT/compare_all.png"

echo
echo "### 8. Plant with explicit dead time (delay-dominant, Cohen-Coon shines) ###"
python3 cli.py --plant "2/(5s+1)" --L 3 --method cohen_coon

echo
echo "Done. Artifacts in $OUT/"
