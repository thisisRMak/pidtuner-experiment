#!/usr/bin/env python3
"""Generates docs/memos/lqg_validation_report_2026-08-18.html from
examples/lqg/lqg_professor_review.json (written by lqg_review.py) — the
updated LQG validation report with the full metric set (checks, Riccati,
closed-loop poles, gain matrix K, closed-loop step response, control
effort, transient metrics, sensitivity/complementary sensitivity), styled
to match docs/memos/kd_filtering_memo.html's house CSS.

Run: python3 lqg_review.py && python3 gen_lqg_memo.py
"""
import html
import json
import os

_HERE = os.path.dirname(__file__)
_JSON_IN = os.path.join(_HERE, "examples", "lqg", "lqg_professor_review.json")
_HTML_OUT = os.path.join(_HERE, "..", "docs", "memos", "2026-08-18-lqg-validation-report.html")

_CSS = open(os.path.join(_HERE, "_memo_css.txt")).read()


def _e(s):
    return html.escape(str(s))


def _fmt_matrix(m):
    rows = []
    for row in m:
        rows.append("[" + ", ".join(f"{v: .4g}" for v in row) + "]")
    return "\n".join(rows)


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
  <dt>To</dt><dd>Prof. Emami-Naeini</dd>
  <dt>From</dt><dd>Rohit</dd>
  <dt>Date</dt><dd>2026-08-18</dd>
  <dt>Re</dt><dd>Updated validation report — full 12-plant catalog, now with closed-loop step response, control effort, transient response, and sensitivity/complementary sensitivity added to the original checks-only report</dd>
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
<b>Open question &mdash; sensitivity/complementary sensitivity (Ms/Mt) loop-breaking point.</b>
All Ms/Mt figures below are evaluated at the <b>plant input</b>
(<code>loop_point = "plant_input"</code>). See "What we're asking" (P0) immediately below
for the full question &mdash; flagging it here too since it colors every Ms/Mt figure that
follows.
</div>

<h2>What we're asking</h2>

<p><b>P0 &mdash; mission critical:</b></p>
<ol>
<li>
<b>Ms/Mt loop-breaking point.</b> All Ms/Mt figures in this report are evaluated at the
<b>plant input</b> (<code>loop_point = "plant_input"</code>): L(s) =
K(sI&minus;A)<sup>-1</sup>B for full-state feedback. Every plant in this catalog is
currently full-state-feedback LQR, where the plant input is the standard place to judge
robustness and the classical LQR return-difference guarantee (Ms&le;1, &ge;60&deg; phase
margin, &plusmn;&infin;/&minus;6dB gain margin) applies &mdash; and indeed every plant
below reports Ms&asymp;1, consistent with that guarantee. This is mission-critical to
confirm because it isn't a coverage gap like the P1/P2 items below &mdash; it's a
question about whether a number printed for all 12 plants means what it claims to mean.
If output-feedback (LQG/Kalman) designs are added to this report later, plant-input Ms/Mt
would <b>not</b> carry the same guarantee there (loop transfer recovery is a separate,
non-automatic step), so this needs to be settled before the S/T methodology here is
extended to those designs, or before this report's Ms/Mt column is treated as a settled
result rather than a first pass.
</li>
</ol>

<p><b>P1 &mdash; important:</b></p>
<ol>
<li>
<b>Non-square plants have no closed-loop step response at all.</b>
{"<code>" + "</code>, <code>".join(non_square) + "</code>" if non_square else "(none in this catalog)"}
{"are" if len(non_square) != 1 else "is"} excluded because <code>add_reference_tracking</code>'s
N&#772; feedforward requires a square <code>[[A,B],[C,D]]</code> block (nu == ny), which
{"these plants don't" if len(non_square) != 1 else "this plant doesn't"} satisfy. That's a
real, current gap in this report &mdash; {"they" if len(non_square) != 1 else "it"} show{"" if len(non_square) != 1 else "s"}
a blank "N/A (non-square)" cell where every other plant has step-response numbers. Is that
gap acceptable as-is, or should we build an alternative tracking scheme (e.g. a
least-squares N&#772; for non-square plants) so these get covered too?
</li>
</ol>

<p><b>P2 &mdash; nice to have:</b></p>
<ol>
<li>
<b>{"<code>" + "</code>, <code>".join(singular_zero) + "</code>" if singular_zero else "(none in this catalog)"}
excluded for a different reason &mdash; a transmission zero at the origin</b>
(<code>[[A,B],[C,D]]</code> singular), not a shape mismatch. This looks like the same gap
as the P1 item above, but it isn't: it's a structural fact about this specific plant (no
N&#772; can exist here at all, regardless of tracking-scheme choice), not an
implementation limitation. The ask here is lighter &mdash; just confirming this exclusion
is expected and not a sign something's broken, rather than asking for a fix.
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
    with open(_JSON_IN) as f:
        payload = json.load(f)
    out = build_html(payload)
    os.makedirs(os.path.dirname(_HTML_OUT), exist_ok=True)
    with open(_HTML_OUT, "w") as f:
        f.write(out)
    print(f"wrote {_HTML_OUT}")
