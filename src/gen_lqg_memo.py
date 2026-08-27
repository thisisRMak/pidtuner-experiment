#!/usr/bin/env python3
"""Generates docs/memos/<date>/<date>-lqg-validation-report.html from
examples/lqg/lqg_professor_review.json (written by lqg_review.py) — the
updated LQG validation report with the full metric set (checks, Riccati,
closed-loop poles, gain matrix K, closed-loop step response, control
effort, transient metrics, sensitivity/complementary sensitivity), styled
to match docs/memos/kd_filtering_memo.html's house CSS.

Originally a one-off re-run (2026-08-25) of the 2026-08-18 report against
the *2 revised catalog (lqg_examples_m_revised/, via
list_examples()/load_example()) — see the 2026-08-25 revised-examples
memo for that change. Rebuilt 2026-08-27 into a regularly re-runnable
one-or-all-plants tool (--plant KEY); the output directory is now dated
to the day it's actually run rather than pinned to 2026-08-25.

Run: python3 lqg_review.py && python3 gen_lqg_memo.py [--plant KEY]
"""
import argparse
import datetime
import html
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lqg_examples import load_example
from lqg_design_methods import LQR, add_reference_tracking
from lqg_simulate import simulate_state_feedback, auto_t_end, auto_plot_window

_HERE = os.path.dirname(__file__)
_JSON_IN = os.path.join(_HERE, "examples", "lqg", "lqg_professor_review.json")
_DATE = datetime.date.today().isoformat()
_OUT_DIR = os.path.join(_HERE, "..", "docs", "memos", _DATE)
_HTML_OUT = os.path.join(_OUT_DIR, f"{_DATE}-lqg-validation-report.html")

# Created here, before any plot is saved into it — gen_lqg_worked_examples_
# memos.py hit exactly this ordering trap once its own date became dynamic.
os.makedirs(_OUT_DIR, exist_ok=True)

_CSS = open(os.path.join(_HERE, "_memo_css.txt")).read()

# Same dt as cli_lqg.py's --dt default — used to resolve the step-response
# plot's simulation finely enough that a sub-second initial transient
# (e.g. generic_rtp's) isn't aliased away, independent of how long
# auto_t_end() says the run needs to be for its slowest pole.
_PLOT_DT = 0.01


def _e(s):
    return html.escape(str(s))


def _fmt_matrix(m):
    rows = []
    for row in m:
        rows.append("[" + ", ".join(f"{v: .4g}" for v in row) + "]")
    return "\n".join(rows)


def _plot_step_response(key, tracking_available):
    """Saves <key>_step.png alongside the report: the reference-tracking
    step response (r=ones) if the plant supports it, else the regulator
    response from x0=ones(nx) (same recipe as lqg_review.py's own sim, run
    again here since the JSON only carries scalar metrics, not the raw
    trajectory). Returns (filename, label)."""
    ex = load_example(key)
    plant = ex.plant
    Q, R = ex.build_suggested_Q(), ex.build_suggested_R()
    res = LQR(plant, Q=Q, R=R).design()
    t_end = auto_t_end(res.closed_loop_poles)
    t = np.arange(0.0, t_end + _PLOT_DT, _PLOT_DT)

    label = "regulator response (x0=ones, no reference tracking available)"
    if tracking_available:
        try:
            res_rt = add_reference_tracking(res)
            r_step = np.tile(np.ones(plant.ny), (len(t), 1))
            sim = simulate_state_feedback(res_rt, t, r=r_step)
            label = "reference-tracking step response (r=ones)"
        except ValueError:
            sim = simulate_state_feedback(res, t, x0=np.ones(plant.nx))
    else:
        sim = simulate_state_feedback(res, t, x0=np.ones(plant.nx))

    # Simulate the full auto_t_end() duration (needed for correctness on
    # slow poles), but only *display* up to auto_plot_window()'s crop --
    # otherwise a wide-pole-spread plant like generic_rtp (slowest pole
    # ~-0.0083 -> auto_t_end() ~1800s) flattens out over hundreds of
    # seconds of dead plot with its actual transient invisible at the left
    # edge. Same recipe as cli_lqg.py's --plot handling.
    plot_t_max = auto_plot_window(sim.t, sim.y, sim.u)
    idx = max(int(np.searchsorted(sim.t, plot_t_max, side="right")), 2)
    t_plot, y_plot, u_plot = sim.t[:idx], sim.y[:idx], sim.u[:idx]

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for i in range(y_plot.shape[1]):
        axes[0].plot(t_plot, y_plot[:, i], label=f"y{i}")
    axes[0].set_ylabel("Output y(t)")
    axes[0].legend(fontsize=8, ncol=min(y_plot.shape[1], 5))
    axes[0].grid(True)
    axes[0].set_title(f"{ex.name} ({key}) — {label}")
    for i in range(u_plot.shape[1]):
        axes[1].plot(t_plot, u_plot[:, i], label=f"u{i}")
    axes[1].set_ylabel("Control effort u(t)")
    axes[1].set_xlabel("Time (s)")
    axes[1].legend(fontsize=8, ncol=min(u_plot.shape[1], 5))
    axes[1].grid(True)
    fig.tight_layout()

    filename = f"{key}_step.png"
    fig.savefig(os.path.join(_OUT_DIR, filename))
    plt.close(fig)
    return filename, label


def build_html(payload):
    plants = payload["plants"]
    n = len(plants)
    n_pass = sum(1 for p in plants if p["all_checks_passed"])
    non_square = [p["key"] for p in plants if p["tracking_metrics"] is None
                 and p["tracking_unavailable"] and "non-square" in p["tracking_unavailable"]]
    singular_zero = [p["key"] for p in plants if p["tracking_metrics"] is None
                     and p["tracking_unavailable"] and "non-square" not in p["tracking_unavailable"]]

    parts = []
    parts.append(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>LQR/LQG Validation Report — Updated</title>
<style>{_CSS}</style>
</head>
<body>

<h1>LQR/LQG Design Track — Preset Catalog Validation Report (Updated)</h1>

<dl class="memo-header">
  <dt>Date</dt><dd>{_DATE}</dd>
  <dt>Re</dt><dd>Re-run of the 2026-08-18 validation report against the *2 revised plant catalog (lqg_examples_m_revised/) — full 12-plant catalog, closed-loop step response, control effort, transient response, and sensitivity/complementary sensitivity, same metric set as before</dd>
</dl>

<p>
This supersedes the 2026-08-02 report (archived alongside this one as
<code>2026-08-02-lqg-validation-report-archived.pdf</code>, unedited). That
version covered the correctness-check battery only (Q/R well-posedness,
stabilizability/detectability, Riccati residual, closed-loop pole
stability) plus gain-matrix shapes. This version adds, per plant: the full
gain matrix <b>K</b> (values, not just shape), closed-loop step response
(overshoot/rise/settling per channel), regulator control effort
(&int;u&#7502;u dt, peak |u|, settling time), and sensitivity/complementary
sensitivity peaks (Ms/Mt).
</p>

<p><b>Summary: {n_pass}/{n} plants pass every correctness check.</b></p>

<div class="callout">
<b class="pass">Note &mdash; non-square plants are out of scope (confirmed 2026-08-25).</b>
Per the 2026-08-25 conversation with Professor, non-square plants are confirmed out of scope
for this catalog. This also happens to already be moot for the current catalog: the *2 revision
of <code>AIDistillationColumn.m</code> (the one plant that was previously non-square, 3 outputs /
2 inputs) reduces its output map to 2&times;11, making it square &mdash; so there are currently
no non-square plants in this catalog at all. See P1 below, now marked resolved/obsolete.
</div>

<div class="callout">
<b class="pass">Note &mdash; PM/GM loop-breaking point confirmed as plant input (2026-08-25).</b>
Per the same 2026-08-25 conversation with Professor, all phase/gain-margin calculations are
with respect to the plant input. All Ms/Mt figures below are evaluated at the <b>plant input</b>
(<code>loop_point = "plant_input"</code>), so this report's methodology already matches that
confirmed convention. See P0 below, now marked resolved.
</div>

<h2>What we're asking</h2>

<p><b>P0 &mdash; mission critical:</b></p>
<ol>
<li class="pass">
<b>[RESOLVED] Ms/Mt loop-breaking point.</b> All Ms/Mt figures in this report are evaluated at
the <b>plant input</b> (<code>loop_point = "plant_input"</code>): L(s) =
K(sI&minus;A)<sup>-1</sup>B for full-state feedback. Per the 2026-08-25 conversation with
Professor, all phase/gain-margin calculations are with respect to the plant input &mdash; so
this report's methodology already matches that confirmed convention, and every plant below
reports Ms&asymp;1, consistent with the classical LQR return-difference guarantee (Ms&le;1,
&ge;60&deg; phase margin, &plusmn;&infin;/&minus;6dB gain margin) that holds at that point for
full-state feedback. One caveat this resolution doesn't erase, kept for the record: if
output-feedback (LQG/Kalman) designs are added to this report later, plant-input Ms/Mt still
would <b>not</b> automatically carry the same guarantee there (loop transfer recovery is a
separate, non-automatic step) &mdash; that's a mathematical fact about LTR, not a methodology
question, so it isn't something this confirmation resolves; it would need to be handled when/if
output-feedback designs are added.
</li>
</ol>

<p><b>P1 &mdash; important:</b></p>
<ol>
<li class="pass">
<b>[RESOLVED/OBSOLETE] Non-square plants have no closed-loop step response at all.</b>
Per the 2026-08-25 conversation with Professor, non-square plants are confirmed out of
scope for this catalog &mdash; there is no fix to build here, and the original ask (build an
alternative tracking scheme, e.g. a least-squares N&#772;, for non-square plants) doesn't apply.
Separately, this is now also moot for the current catalog specifically: the *2 revision of
<code>AIDistillationColumn.m</code> (the only plant that was previously non-square, 3 outputs /
2 inputs) reduces its output map to 2&times;11, making it square, so
{"<code>" + "</code>, <code>".join(non_square) + "</code>" if non_square else "no plant"}
in the current catalog {"is" if len(non_square) == 1 else "are"} excluded for this reason.
Left here (rather than deleted) as a record of what was asked and how it was resolved.
</li>
</ol>

<p><b>P2 &mdash; nice to have:</b></p>
<ol>
<li class="pass">
<b>[RESOLVED/OBSOLETE] {"<code>" + "</code>, <code>".join(singular_zero) + "</code>" if singular_zero else "(none in this catalog)"}
excluded for a different reason &mdash; a transmission zero at the origin</b>
(<code>[[A,B],[C,D]]</code> singular), not a shape mismatch. The ask here was just confirming
this exclusion is expected and not a sign something's broken &mdash; now confirmed: per the
professor's own note, AIDrone2 is expected to fail exactly this way (transmission zero at the
origin), and a dedicated worked-example memo
(<code>2026-08-25-aidrone2-worked-example-memo.pdf</code>) verifies the singular
<code>[[A,B],[C,D]]</code> block and the resulting failure directly. Left here as a record of
what was asked and how it was resolved.
</li>
<li>
<b>Is this report's format/metric set the right one to standardize on going forward?</b>
Meta-question, no urgency either way &mdash; flagging in case you'd rather see this
structured differently (e.g. fewer per-plant metrics with more plants, or vice versa)
before it becomes the template for future passes over this catalog.
</li>
</ol>

<h2>Results summary</h2>
<table>
<thead><tr><th>Plant</th><th>nx/nu/ny</th><th>Checks</th><th>Ms</th><th>Mt</th><th>Step response</th></tr></thead>
<tbody>
""")
    for p in plants:
        checks_str = "all pass" if p["all_checks_passed"] else (
            f"{sum(1 for c in p['checks'] if not c['passed'])} FAILED")
        step_str = "N/A (non-square)" if p["tracking_metrics"] is None else "see detail"
        parts.append(f"<tr><td><code>{_e(p['key'])}</code></td>"
                     f"<td>{p['nx']}/{p['nu']}/{p['ny']}</td>"
                     f"<td>{checks_str}</td>"
                     f"<td>{p['Ms']:.3g}</td><td>{p['Mt']:.3g}</td>"
                     f"<td>{step_str}</td></tr>\n")
    parts.append("</tbody></table>\n\n<h2>Per-plant detail</h2>\n")

    for p in plants:
        parts.append(f"<h3><code>{_e(p['key'])}</code> — {_e(p['name'])}</h3>\n")
        parts.append(f"<p class='section-note'>Source: <code>{_e(p['source_file'])}</code> "
                     f"({_e(p['citation'])})</p>\n")

        parts.append("<table class='no-break'><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>\n")
        for c in p["checks"]:
            cls = "pass" if c["passed"] else "fail"
            mark = "PASS" if c["passed"] else "FAIL"
            parts.append(f"<tr><td>{_e(c['name'])}</td>"
                         f"<td class='{cls}'>[{mark}] {_e(c['detail'])}</td></tr>\n")
        parts.append("</tbody></table>\n")

        parts.append(f"<p><b>Gain matrix K</b> ({p['nu']}&times;{p['nx']}):</p>\n")
        parts.append(f"<pre>{_e(_fmt_matrix(p['K']))}</pre>\n")

        rm = p["regulator_metrics"]
        parts.append("<table class='no-break'><thead><tr><th>Regulator step (x0=ones)</th><th>Value</th></tr></thead><tbody>\n")
        if rm.get("unstable"):
            parts.append("<tr><td colspan='2' class='fail'>UNSTABLE (diverging response)</td></tr>\n")
        else:
            parts.append(f"<tr><td>Settling (2%)</td><td>{rm['settling_2pct']:.3g} s</td></tr>\n")
            parts.append(f"<tr><td>Control energy (ISU = &int;u&#7502;u dt)</td><td>{rm['ISU']:.3g}</td></tr>\n")
            parts.append(f"<tr><td>|u|<sub>peak</sub></td><td>{rm['u_peak']:.3g}</td></tr>\n")
            parts.append(f"<tr><td>Final ||x||</td><td>{rm['final_state_norm']:.3g}</td></tr>\n")
        parts.append("</tbody></table>\n")

        parts.append("<table class='no-break'><thead><tr><th>Closed-loop step response (r=ones)</th><th>Overshoot</th><th>Rise (10-90%)</th><th>Settling (2%)</th></tr></thead><tbody>\n")
        if p["tracking_metrics"] is None:
            parts.append(f"<tr><td colspan='4' class='section-note'>not available: {_e(p['tracking_unavailable'])}</td></tr>\n")
        else:
            for j, m in enumerate(p["tracking_metrics"]):
                parts.append(f"<tr><td>y{j}</td><td>{m['Overshoot']:.3g}%</td>"
                             f"<td>{m['Rise']:.3g} s</td><td>{m['Settling']:.3g} s</td></tr>\n")
        parts.append("</tbody></table>\n")

        parts.append(f"<p>Sensitivity/complementary sensitivity at the plant input: "
                     f"<b>Ms = {p['Ms']:.3g}, Mt = {p['Mt']:.3g}</b></p>\n")

        plot_file, plot_label = _plot_step_response(p["key"], p["tracking_metrics"] is not None)
        parts.append(f"<p class='section-note'>Plot: {_e(plot_label)}</p>\n")
        parts.append(f"<p><img src=\"{plot_file}\" alt=\"{_e(p['key'])} step response\" "
                     f"style=\"max-width:100%;border:1px solid #b9c3cc;border-radius:5px;\"></p>\n")

    parts.append("""
<div class="appendix-title">Appendix: reproducing the numbers above</div>
<pre><code>cd src
python3 lqg_review.py       # writes examples/lqg/lqg_professor_review.json, docs/lqg_review.md
python3 gen_lqg_memo.py     # writes this report from that JSON
</code></pre>

</body>
</html>
""")
    return "".join(parts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant", default=None,
                        help="only report on this plant key (see lqg_examples.list_examples() "
                             "for valid keys); default is all plants in the JSON")
    args = parser.parse_args()

    with open(_JSON_IN) as f:
        payload = json.load(f)

    html_out = _HTML_OUT
    if args.plant is not None:
        available = [p["key"] for p in payload["plants"]]
        if args.plant not in available:
            raise SystemExit(f"no such plant {args.plant!r} in {_JSON_IN} — available: "
                             f"{available} (re-run lqg_review.py if it's missing)")
        payload = {**payload, "plants": [p for p in payload["plants"] if p["key"] == args.plant]}
        html_out = os.path.join(_OUT_DIR, f"{_DATE}-lqg-validation-report-{args.plant}.html")

    out = build_html(payload)
    with open(html_out, "w") as f:
        f.write(out)
    print(f"wrote {html_out}")
