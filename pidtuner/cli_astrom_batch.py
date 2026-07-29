#!/usr/bin/env python3
"""One-off batch runner over Astrom & Hagglund's 133-process test batch
(see astrom_test_batch.py; PID Controllers: Theory, Design, and Tuning,
2nd ed., p. 227).

Runs every one of the 9 tuning methods (compare_all_methods) against every
plant in the batch exactly once, and saves everything to disk so results can
be pulled up later without re-running anything. This is deliberately a
plain CLI, not a GUI feature -- see docs/refactor_prompt.md discussion #3
for why the LLM supervisor stayed conversational/GUI-adjacent while this
stays a batch job.

Commands
--------
    python cli_astrom_batch.py run  --out-dir <dir> [--plot] [--families P1,P7] [--overwrite]
    python cli_astrom_batch.py list --out-dir <dir> [--family P7]
    python cli_astrom_batch.py show --out-dir <dir> <path>

`run` executes the batch once and writes the folder hierarchy below.
`list` walks an existing run's manifest and prints what's available.
`show` pretty-prints one saved result (same text format as
`cli.py --method all`), given a path relative to --out-dir, e.g.
`P7/T=5/L1=0.3` or `P1/T=0.3`.

Folder hierarchy written by `run`
----------------------------------
    <out-dir>/
      manifest.json               every (family, params) -> relative dir, timestamp, git commit
      summary.json / summary.txt  per-method stability rate + Ms/Mt/IAE/OS%/ts stats across the batch
      P1/
        T=0.02/
          plant.json               {family, params, expr} -- reproducible via cli.py --plant
          result.json              compare_all_methods() rows, serialized (same shape as cli.py --json)
          step_response.png        only with --plot
        T=0.05/
          ...
      P7/
        T=5/
          L1=0.3/
            plant.json, result.json, step_response.png
      ...

Multi-parameter families (only P7: T x L1) nest one folder per parameter;
every other family is a single param level. `--plot` writes one PNG per
plant (133 total for the full batch) -- off by default since most of the
time you want result.json/summary.json, not 133 images.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astrom_test_batch import FAMILY_DOCS, BatchPlant, iter_batch
from compare import TABLE_METRICS, compare_all_methods
from plant import TransferFunction
from simulate import simulate_closed_loop

MANIFEST_NAME = "manifest.json"
SUMMARY_JSON_NAME = "summary.json"
SUMMARY_TXT_NAME = "summary.txt"
PLANT_NAME = "plant.json"
RESULT_NAME = "result.json"
PLOT_NAME = "step_response.png"


# ─────────────────────────────────────────────────────────────────────────────
# Row (de)serialization -- own copy, following the precedent already set by
# cli.py/cli_blackbox.py/supervisor_tools_whitebox.py: each entity keeps its
# own serializer rather than sharing one.
# ─────────────────────────────────────────────────────────────────────────────

def serialize_row_json(row: dict) -> dict:
    gains = row.get("gains")
    gains_dict = {"Kp": gains.Kp, "Ki": gains.Ki, "Kd": gains.Kd} if gains else None
    out = {}
    for k, v in row.items():
        if k == "gains":
            out[k] = gains_dict
        elif isinstance(v, (np.floating, float)):
            out[k] = float(v) if np.isfinite(v) else None
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        else:
            out[k] = v
    return out


def format_row_text(row: dict) -> str:
    if not row.get("stable", False):
        return f"Method: {row['name']} -> Unstable/Error: {row.get('error', 'Unknown')}"
    gains = row["gains"]
    return (
        f"Method: {row['name']}\n"
        f"  Gains: Kp={gains['Kp']:.6g}, Ki={gains['Ki']:.6g}, Kd={gains['Kd']:.6g}\n"
        f"  Metrics:\n"
        f"    Overshoot:          {row.get('OS%')}\n"
        f"    Settling Time (2%): {row.get('ts')} s\n"
        f"    IAE (Setpoint):     {row.get('IAE')}\n"
        f"    IAE (Load):         {row.get('IAE_load')}\n"
        f"    Peak Sensitivity Ms:{row.get('Ms')}\n"
        f"    Peak Comp Sens Mt:  {row.get('Mt')}\n"
        f"    Gain Margin GM:     {row.get('GM_dB')} dB\n"
        f"    Phase Margin PM:    {row.get('PM_deg')} deg\n"
        f"    Control Effort TV:  {row.get('u_tv')}\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────────────

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).parent, text=True
        ).strip()
    except Exception:
        return None


def _plot_plant(plant: TransferFunction, rows: list[dict], out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    any_stable = False
    for r in rows:
        if r.get("stable", False) and r.get("gains"):
            sim = simulate_closed_loop(plant, r["gains"], setpoint=1.0, setpoint_kind="step")
            ax1.plot(sim.t, sim.y, label=r["name"])
            ax2.plot(sim.t, sim.u)
            any_stable = True
    if any_stable:
        ax1.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Setpoint")
    ax1.set_ylabel("Output y(t)")
    ax1.set_title("Step Response comparison")
    ax1.legend(fontsize="small")
    ax1.grid(True)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Control Effort u(t)")
    ax2.grid(True)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _run_one(bp: BatchPlant, out_dir: Path, do_plot: bool) -> dict:
    """Run one plant through the full method comparison, write its files,
    and return a manifest entry (plus the raw rows, for summary stats)."""
    plant_dir = out_dir.joinpath(*bp.subpath)
    plant_dir.mkdir(parents=True, exist_ok=True)

    entry = {
        "id": bp.id, "family": bp.family, "params": bp.params,
        "expr": bp.expr, "path": str(Path(*bp.subpath).as_posix()),
    }
    (plant_dir / PLANT_NAME).write_text(json.dumps(entry, indent=2))

    try:
        plant = TransferFunction.parse(bp.expr)
        rows = compare_all_methods(plant, include_variants=True)
        error = None
    except Exception as exc:  # noqa: BLE001 - one bad plant must not kill the batch
        rows = []
        error = str(exc)

    serialized = [serialize_row_json(r) for r in rows]
    (plant_dir / RESULT_NAME).write_text(
        json.dumps({"error": error, "rows": serialized}, indent=2)
    )

    if do_plot and rows:
        try:
            _plot_plant(plant, rows, plant_dir / PLOT_NAME)
        except Exception as exc:  # noqa: BLE001 - plotting failure isn't fatal
            entry["plot_error"] = str(exc)

    entry["error"] = error
    entry["n_methods"] = len(rows)
    entry["n_stable"] = sum(1 for r in rows if r.get("stable"))
    return entry, rows


def _summarize(all_rows: list[tuple[str, dict]]) -> dict:
    """Per-method aggregate stats across the whole batch: stability rate and
    median/IQR of each metric in compare.TABLE_METRICS, over stable rows
    only. This -- not per-plant answer-checking -- is the meaningful
    regression signal for a 133-process sweep with no per-instance oracle."""
    by_method: dict[str, list[dict]] = {}
    for family, row in all_rows:
        by_method.setdefault(row["name"], []).append(row)

    summary = {}
    for name, rows in sorted(by_method.items()):
        n = len(rows)
        stable_rows = [r for r in rows if r.get("stable")]
        entry = {"n": n, "n_stable": len(stable_rows),
                 "stability_rate": len(stable_rows) / n if n else 0.0}
        for metric in TABLE_METRICS:
            vals = [r[metric] for r in stable_rows
                    if r.get(metric) is not None and np.isfinite(r[metric])]
            if vals:
                arr = np.array(vals)
                entry[metric] = {
                    "median": float(np.median(arr)),
                    "p25": float(np.percentile(arr, 25)),
                    "p75": float(np.percentile(arr, 75)),
                    "max": float(np.max(arr)),
                }
            else:
                entry[metric] = None
        summary[name] = entry
    return summary


def _summary_text(summary: dict) -> str:
    lines = ["=== Astrom batch summary (per method, across all plants) ===", ""]
    for name, s in summary.items():
        lines.append(f"{name}  (stable on {s['n_stable']}/{s['n']} = {s['stability_rate']:.0%})")
        for metric in TABLE_METRICS:
            m = s.get(metric)
            if m is None:
                continue
            lines.append(f"    {metric:8s} median={m['median']:.4g}  "
                         f"IQR=[{m['p25']:.4g}, {m['p75']:.4g}]  max={m['max']:.4g}")
        lines.append("")
    return "\n".join(lines)


def cmd_run(args) -> None:
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        print(f"Error: {out_dir} already exists and is non-empty. "
              f"Pass --overwrite to write into it anyway.", file=sys.stderr)
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    families = args.families.split(",") if args.families else None
    plants = list(iter_batch(families))
    print(f"Running {len(plants)} plants"
          f"{' (families: ' + ','.join(families) + ')' if families else ''}"
          f"{' with plots' if args.plot else ''} -> {out_dir}")

    manifest_entries = []
    all_rows = []
    for i, bp in enumerate(plants, 1):
        entry, rows = _run_one(bp, out_dir, args.plot)
        manifest_entries.append(entry)
        all_rows.extend((bp.family, r) for r in rows)
        print(f"  [{i}/{len(plants)}] {bp.id}: "
              f"{entry['n_stable']}/{entry['n_methods']} stable"
              f"{'  ERROR: ' + entry['error'] if entry['error'] else ''}")

    manifest = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "n_plants": len(plants),
        "families": sorted(FAMILY_DOCS.keys()),
        "family_docs": FAMILY_DOCS,
        "plants": manifest_entries,
    }
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))

    summary = _summarize(all_rows)
    (out_dir / SUMMARY_JSON_NAME).write_text(json.dumps(summary, indent=2))
    summary_text = _summary_text(summary)
    (out_dir / SUMMARY_TXT_NAME).write_text(summary_text)

    n_images = sum(1 for e in manifest_entries if (out_dir / e["path"] / PLOT_NAME).exists())
    print(f"\nDone. {len(plants)} plants, {n_images} plots written.")
    print(f"See {out_dir / SUMMARY_TXT_NAME} for the aggregate summary.")


# ─────────────────────────────────────────────────────────────────────────────
# list / show
# ─────────────────────────────────────────────────────────────────────────────

def _load_manifest(out_dir: Path) -> dict:
    path = out_dir / MANIFEST_NAME
    if not path.exists():
        print(f"Error: no {MANIFEST_NAME} in {out_dir} -- run `run` first.", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def cmd_list(args) -> None:
    manifest = _load_manifest(Path(args.out_dir))
    entries = manifest["plants"]
    if args.family:
        entries = [e for e in entries if e["family"] == args.family]
    for e in entries:
        flag = f"  ERROR: {e['error']}" if e["error"] else ""
        print(f"{e['path']:30s} {e['n_stable']}/{e['n_methods']} stable{flag}")
    print(f"\n{len(entries)} plant(s).")


def cmd_show(args) -> None:
    out_dir = Path(args.out_dir)
    result_path = out_dir / args.path / RESULT_NAME
    plant_path = out_dir / args.path / PLANT_NAME
    if not result_path.exists():
        print(f"Error: no result at {result_path} -- check `list` for valid paths.", file=sys.stderr)
        sys.exit(1)

    plant_info = json.loads(plant_path.read_text())
    result = json.loads(result_path.read_text())
    print(f"=== {plant_info['id']} ===")
    print(f"expr: {plant_info['expr']}")
    print(f"family: {plant_info['family']} ({FAMILY_DOCS.get(plant_info['family'], '')})")
    print(f"params: {plant_info['params']}")
    if result["error"]:
        print(f"ERROR: {result['error']}")
        return
    print()
    for row in result["rows"]:
        print(format_row_text(row))
        print("-" * 40)


# ─────────────────────────────────────────────────────────────────────────────
# argparse plumbing
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run/inspect Astrom & Hagglund's 133-process test batch "
                     "against all 9 PIDTuner tuning methods."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run the batch once and save results to --out-dir")
    p_run.add_argument("--out-dir", required=True, help="Output directory")
    p_run.add_argument("--plot", action="store_true", help="Also save a step-response PNG per plant")
    p_run.add_argument("--families", help="Comma-separated subset, e.g. P1,P7 (default: all 9)")
    p_run.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty --out-dir")
    p_run.set_defaults(func=cmd_run)

    p_list = sub.add_parser("list", help="List plants available in an existing --out-dir run")
    p_list.add_argument("--out-dir", required=True)
    p_list.add_argument("--family", help="Restrict to one family, e.g. P7")
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="Pretty-print one saved result")
    p_show.add_argument("--out-dir", required=True)
    p_show.add_argument("path", help="Plant path relative to --out-dir, e.g. P7/T=5/L1=0.3")
    p_show.set_defaults(func=cmd_show)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
