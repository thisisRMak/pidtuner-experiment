#!/usr/bin/env bash
# Demonstrates cli_lqg.py: the LQR/LQG design track. Exercises every method
# (lqr, output_weighted, bryson, lqg, implicit, explicit) and every plant in
# the preset catalog (--list-plants), plus reference tracking (with the
# sign-aware Overshoot/Rise/Settling metrics), the two cross-method
# comparisons (--method all / model_following_all), and the pre-/post-design
# checks (see docs/lqg_testing.md). Yes, section 6 below covers all
# professor-provided examples currently in the catalog.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=examples/lqg
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
echo "### 5. Full LQG (Kalman filter, output-feedback simulation) ###"
python3 cli_lqg.py --plant-preset aircraft_hall --method lqg --sim output_feedback \
  --plot "$OUT/aircraft_hall_lqg.png"
echo "wrote $OUT/aircraft_hall_lqg.png"

echo
echo "### 5b. Reference tracking (now actually simulates tracking --reference, not just N̄) ###"
echo "###     with the sign-aware Overshoot/Rise/Settling metrics ###"
python3 cli_lqg.py --plant-preset aircraft_hall --method lqr --sim state_feedback \
  --reference-tracking --reference 1.0 -0.5

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
echo "### 8. Model-following: implicit (u=-Kx) and explicit (u=-K1x-K2xm) ###"
python3 cli_lqg.py --plant-preset aircraft_hall --method implicit --am-diag 0.1 0.07
python3 cli_lqg.py --plant-preset aircraft_hall --method explicit --am-diag 0.1 0.07 \
  --sim model_following --plot "$OUT/aircraft_hall_explicit.png"
echo "wrote $OUT/aircraft_hall_explicit.png"

echo
echo "### 9. Comparison 1/2: regulator family (lqr/output_weighted/bryson/lqg), same plant/objective ###"
python3 cli_lqg.py --plant-preset aircraft_hall --method all --plot "$OUT/aircraft_hall_compare_regulator.png"
echo "wrote $OUT/aircraft_hall_compare_regulator.png"

echo
echo "### 10. Comparison 2/2: implicit vs. explicit model-following, same target model ###"
python3 cli_lqg.py --plant-preset aircraft_hall --method model_following_all --am-diag 0.1 0.07 \
  --plot "$OUT/aircraft_hall_compare_model_following.png"
echo "wrote $OUT/aircraft_hall_compare_model_following.png"

echo
echo "### 11. Iterating on a design: custom Q/R weights + reference tracking ###"
echo "###     (lower R measurably reduced overshoot here -- not the naive intuition, check empirically) ###"
python3 cli_lqg.py --plant-preset aircraft_hall --method all \
  --Q-diag 1 1 1 1 1 --R-diag 0.1 0.1 \
  --reference-tracking --reference 1.0 -0.5 --plot "$OUT/aircraft_hall_compare_custom.png"
echo "wrote $OUT/aircraft_hall_compare_custom.png"

echo
echo "Done. Artifacts in $OUT/, professor report at ../docs/lqg_review.md"
