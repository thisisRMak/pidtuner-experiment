#!/usr/bin/env bash
# Demonstrates cli_lqg.py: the LQR/LQG design track. Exercises every method
# (lqr, output_weighted, bryson, lqg) and every plant in the preset catalog
# (--list-plants), plus reference tracking and the pre-/post-design checks
# (see docs/lqg_testing.md). Yes, this covers all professor-provided
# examples currently in the catalog — that's what section 6 below does.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=examples/out
mkdir -p "$OUT"

echo "### 1. List all preset plants ###"
python3 cli_lqg.py --list-plants

echo
echo "### 2. Single plant, LQR with its own suggested Q/R, text output + checks ###"
python3 cli_lqg.py --plant-preset aircraft_hall --method lqr

echo
echo "### 3. Output-weighted LQR (Q = CᵀQyC), JSON output ###"
python3 cli_lqg.py --plant-preset drone --method output_weighted --Qy-scale 2.0 --json > "$OUT/drone_output_weighted.json"
echo "wrote $OUT/drone_output_weighted.json"

echo
echo "### 4. Bryson's rule (Qii=1/x_max², Rii=1/u_max²) ###"
python3 cli_lqg.py --plant-preset chemical_reactor --method bryson --x-max 1 2 3 4 --u-max 5 5

echo
echo "### 5. Full LQG (Kalman filter, output-feedback simulation) + reference tracking ###"
python3 cli_lqg.py --plant-preset aircraft_hall --method lqg --sim output_feedback \
  --reference-tracking --plot "$OUT/aircraft_hall_lqg.png"
echo "wrote $OUT/aircraft_hall_lqg.png"

echo
echo "### 6. One LQR pass over every preset plant (the professor's own Q/R each time) ###"
for key in $(python3 cli_lqg.py --list-plants | awk '{print $1}'); do
  echo "--- $key ---"
  python3 cli_lqg.py --plant-preset "$key" --method lqr --sim none --json \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print('stable:', d['stable'], ' all_checks_passed:', d['checks_all_passed'])"
done

echo
echo "### 7. Professor-review report: same sweep as above, written to file ###"
python3 lqg_review.py

echo
echo "Done. Artifacts in $OUT/, professor report at ../docs/lqg_review.md"
