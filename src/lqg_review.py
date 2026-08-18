#!/usr/bin/env python3
"""Generates a single pass of LQR (each preset's own suggested Q/R, i.e. the
same Q/R the professor's own .m file uses) over every plant in the preset
catalog, with the full pre-/post-design check suite (lqg_checks.py), and
writes the results to a JSON file (machine-readable) and a Markdown report
(for sharing with the professor).

Run:  python lqg_review.py
Writes:  examples/lqg/lqg_professor_review.json
         docs/lqg_review.md
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np

from lqg_examples import list_examples, load_example
from lqg_design_methods import LQR, add_reference_tracking
from lqg_checks import checks_for_result
from lqg_simulate import (simulate_state_feedback, auto_t_end,
                          compute_regulator_metrics,
                          format_regulator_metrics, format_tracking_metrics)
from lqg_frequency import compute_sensitivity

_HERE = os.path.dirname(__file__)
_JSON_OUT = os.path.join(_HERE, "examples", "lqg", "lqg_professor_review.json")
_MD_OUT = os.path.join(_HERE, "..", "docs", "lqg_review.md")

# Plants whose source .m file needed a fix before it would run at all —
# called out explicitly in the report so the professor can confirm the fix
# was the intended one, not just that "it runs now."
_RECENTLY_FIXED = {
    "furnace_model": "source had a stray trailing ')' after `lqr(A,B,Q,R)`, "
                     "a plain syntax typo — fixed by removing it.",
    "f100_engine": "source's `R` was undefined at the `lqr(A,B,Q,R)` call; "
                  "now `R=eye(5)`, matching nu=5 from B/D's column count.",
    "example2_rtp": "the lqr() call referenced undefined uppercase A,B,C,D "
                    "while only lowercase a,b,c,d were assigned, and the "
                    "dimensions didn't match even correcting the case — "
                    "fixed by using consistent, correctly-sized A,B,Q,R.",
}

# Plants still excluded from the catalog (see docs/lqg_plan.md "Known
# issues in the source material") — listed here so the professor sees what
# this report deliberately does *not* cover, and why. Empty as of
# 2026-08-02: all 12 source .m files now run cleanly.
_STILL_EXCLUDED = {}


def _array(a):
    return np.asarray(a, dtype=float).tolist()


def run_all():
    rows = []
    for key in list_examples():
        ex = load_example(key)
        plant = ex.plant
        Q, R = ex.build_suggested_Q(), ex.build_suggested_R()
        res = LQR(plant, Q=Q, R=R).design()
        checks = checks_for_result(res)
        all_checks = checks["pre"] + checks["post"]

        # Closed-loop step response, control effort, transient metrics —
        # simulate_state_feedback/compute_*_metrics already existed in
        # lqg_simulate.py but were never called from this report (2026-08-18).
        t_end = auto_t_end(res.closed_loop_poles)
        t = np.linspace(0.0, t_end, 400)
        reg_sim = simulate_state_feedback(res, t, x0=np.ones(plant.nx))
        regulator_metrics = compute_regulator_metrics(t, reg_sim.x, reg_sim.u)
        # Reference-tracking step response requires add_reference_tracking's
        # N-bar, which in turn requires a square [[A,B],[C,D]] block
        # (nu == ny and non-singular) — not guaranteed for every plant in
        # the catalog. Fall back to None + a note rather than failing the
        # whole report for a non-square plant.
        tracking_metrics = None
        tracking_unavailable = None
        if plant.nu == plant.ny:
            try:
                res_rt = add_reference_tracking(res)
                r_step = np.tile(np.ones(plant.ny), (len(t), 1))
                step_sim = simulate_state_feedback(res_rt, t, r=r_step)
                tracking_metrics = step_sim.tracking_metrics
            except ValueError as e:
                tracking_unavailable = str(e)
        else:
            tracking_unavailable = (f"plant is non-square (nu={plant.nu}, "
                                    f"ny={plant.ny}) — reference tracking (N̄) "
                                    f"requires nu == ny")

        # Sensitivity/complementary sensitivity, plant input only — see
        # lqg_frequency.py's module docstring for the loop-breaking-point
        # caveat (open question for output-feedback designs; this batch is
        # all full-state-feedback LQR so it doesn't arise here, but the
        # field is labeled with loop_point regardless for clarity).
        sens = compute_sensitivity(res)

        rows.append({
            "key": key,
            "name": ex.name,
            "citation": ex.citation,
            "source_file": ex.source_file,
            "nx": plant.nx, "nu": plant.nu, "ny": plant.ny,
            "Q_kind": ex.suggested_Q_kind, "R_kind": ex.suggested_R_kind,
            "K": _array(res.gains.K),
            "closed_loop_poles": [[float(p.real), float(p.imag)]
                                  for p in res.closed_loop_poles],
            "stable": res.is_stable(),
            "checks": [{"name": c.name, "passed": c.passed, "detail": c.detail}
                      for c in all_checks],
            "all_checks_passed": all(c.passed for c in all_checks),
            "recently_fixed": _RECENTLY_FIXED.get(key),
            "regulator_metrics": regulator_metrics,
            "tracking_metrics": tracking_metrics,
            "tracking_unavailable": tracking_unavailable,
            "Ms": sens.Ms,
            "Mt": sens.Mt,
            "loop_point": sens.loop_point,
        })
    return rows


def write_json(rows):
    os.makedirs(os.path.dirname(_JSON_OUT), exist_ok=True)
    payload = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "method": "LQR with each plant's own suggested Q/R (matching the "
                  "professor's own .m file's Q/R choice)",
        "n_plants": len(rows),
        "plants": rows,
    }
    with open(_JSON_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {_JSON_OUT}")


def write_markdown(rows):
    n_pass = sum(1 for r in rows if r["all_checks_passed"])
    lines = []
    lines.append("# LQG Design Track — Preset Catalog Review")
    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d')}. "
                 f"One pass of `LQR` (each plant's own suggested `Q`/`R`, i.e. "
                 f"the same weights your `.m` files use) over all "
                 f"{len(rows)} plants currently in the preset catalog "
                 f"(`pidtuner/lqg_examples_m/*.m` → `pidtuner/lqg_examples_json/*.json`), "
                 f"with the full pre-/post-design correctness check suite "
                 f"(see `docs/lqg_testing.md` for what each check verifies).")
    lines.append("")
    lines.append(f"**Summary: {n_pass}/{len(rows)} plants pass every check.**")
    lines.append("")
    lines.append("**Open question — sensitivity/complementary sensitivity (Ms/Mt) "
                 "loop-breaking point:** all Ms/Mt figures below are evaluated at "
                 "the plant input (`loop_point = \"plant_input\"`). For this batch "
                 "every plant is full-state-feedback LQR, where that's the "
                 "standard place to judge robustness and the classical LQR "
                 "guarantees (Ms<=2, etc.) do apply. If output-feedback (LQG/"
                 "Kalman) plants are added to this report later, plant-input Ms/Mt "
                 "does **not** carry the same guarantee there (see `lqg_frequency.py`'s "
                 "module docstring) — flagging this now as something to verify with "
                 "you rather than presenting it as settled.")
    lines.append("")

    if _RECENTLY_FIXED:
        lines.append("## Source files that needed a fix before they'd run")
        lines.append("")
        lines.append("These were excluded from the catalog until now because "
                     "the `.m` file as originally provided had a bug unrelated "
                     "to the plant data itself. Flagging so you can confirm "
                     "the fix matches what you intended:")
        lines.append("")
        for key, reason in _RECENTLY_FIXED.items():
            row = next(r for r in rows if r["key"] == key)
            lines.append(f"- **`{row['source_file']}`** (`{key}`): {reason}")
        lines.append("")

    if _STILL_EXCLUDED:
        lines.append("## Still excluded, pending your input")
        lines.append("")
        for key, reason in _STILL_EXCLUDED.items():
            lines.append(f"- **`AIExample2RTP.m`** (`{key}`): {reason}")
        lines.append("")

    lines.append("## Results")
    lines.append("")
    lines.append("| Plant | nx/nu/ny | Q kind | Stable | Checks |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        checks_str = "all pass" if r["all_checks_passed"] else (
            f"{sum(1 for c in r['checks'] if not c['passed'])} FAILED")
        fixed_note = " 🔧 fixed" if r["recently_fixed"] else ""
        lines.append(f"| `{r['key']}`{fixed_note} | {r['nx']}/{r['nu']}/{r['ny']} "
                     f"| {r['Q_kind']} | {r['stable']} | {checks_str} |")
    lines.append("")

    lines.append("## Per-plant detail")
    lines.append("")
    for r in rows:
        lines.append(f"### `{r['key']}` — {r['name']}")
        lines.append("")
        lines.append(f"Source: `{r['source_file']}` ({r['citation']})")
        if r["recently_fixed"]:
            lines.append("")
            lines.append(f"**Fixed for this review:** {r['recently_fixed']}")
        lines.append("")
        lines.append(f"nx={r['nx']}, nu={r['nu']}, ny={r['ny']}, "
                     f"Q={r['Q_kind']}, R={r['R_kind']}, stable={r['stable']}")
        lines.append("")
        lines.append("Checks:")
        for c in r["checks"]:
            mark = "PASS" if c["passed"] else "**FAIL**"
            lines.append(f"- [{mark}] {c['name']} — {c['detail']}")
        lines.append("")

        lines.append(f"Gain matrix K ({r['nu']}×{r['nx']}):")
        lines.append("")
        lines.append("```")
        lines.append(np.array2string(np.array(r["K"]), precision=4, separator=", "))
        lines.append("```")
        lines.append("")

        rm = r["regulator_metrics"]
        lines.append("Regulator step (x0 = ones): control effort and settling —")
        lines.append(f"- {format_regulator_metrics(rm)}")
        lines.append("")

        lines.append("Closed-loop step response (r = ones), per output channel:")
        if r["tracking_metrics"] is not None:
            lines.append(format_tracking_metrics(r["tracking_metrics"]))
        else:
            lines.append(f"- not available: {r['tracking_unavailable']}")
        lines.append("")

        lines.append(f"Sensitivity/complementary sensitivity at the plant input "
                     f"(loop_point={r['loop_point']}): Ms = {r['Ms']:.3g}, "
                     f"Mt = {r['Mt']:.3g}")
        lines.append("")

    os.makedirs(os.path.dirname(_MD_OUT), exist_ok=True)
    with open(_MD_OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {_MD_OUT}")


if __name__ == "__main__":
    rows = run_all()
    write_json(rows)
    write_markdown(rows)
    n_pass = sum(1 for r in rows if r["all_checks_passed"])
    print(f"{n_pass}/{len(rows)} plants passed every check.")
