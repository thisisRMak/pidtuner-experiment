#!/usr/bin/env python3
"""Generates two worked-example memo trios (.md/.html/.pdf-source-html):
  docs/memos/<YYYY-MM-DD>/<YYYY-MM-DD>-aidrone2-worked-example-memo.{md,html}
  docs/memos/<YYYY-MM-DD>/<YYYY-MM-DD>-genericrtp-worked-example-memo.{md,html}
<YYYY-MM-DD> is today's date (datetime.date.today(), _DATE below) -- each run
lands in its own dated snapshot folder rather than overwriting a fixed
2026-08-25 the way earlier revisions of this script did, so re-running the
script on a later date doesn't silently mislabel a new memo with a stale
date. Both memos cover the paper's minimum MIMO material per 2026-08-18-
execution-plan.md task 3: AIDrone2's expected transmission-zero-at-origin
failure, and genericRTP's worked LQR/LQG design (professor's stated
priority example). Runs cli_lqg.py as a subprocess (same command a reader
would type) so the numbers/plots in the memo can't drift from what the CLI
actually produces, and documents the matching Streamlit GUI steps.

Run: python3 gen_lqg_worked_examples_memos.py
(PDF rendering is a separate step — see the session's chrome --headless
--print-to-pdf invocation; not run automatically here.)
"""
import base64
import datetime
import html
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(__file__)
_DATE = datetime.date.today().isoformat()
_OUT_DIR = os.path.join(_HERE, "..", "docs", "memos", _DATE)
# Created here, not just inside write_pair() at the end of each build_*_memo(),
# because run_cli()'s --plot targets land in _OUT_DIR before write_pair ever
# runs -- with a fixed, always-already-existing 2026-08-25 folder this went
# unnoticed, but a fresh dated folder (see _DATE above) needs to exist before
# the first plot is saved into it.
os.makedirs(_OUT_DIR, exist_ok=True)
_CSS = open(os.path.join(_HERE, "_memo_css.txt")).read()


def _e(s):
    return html.escape(str(s))


def run_cli(*args, expect_error=False):
    cmd = [sys.executable, "cli_lqg.py", *args, "--json"]
    out = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True)
    stdout = out.stdout.strip()
    data = json.loads(stdout) if stdout else {}
    if expect_error:
        if "error" not in data:
            raise RuntimeError(f"expected an error from {cmd}, got: {stdout}")
    elif out.returncode != 0 or "error" in data:
        raise RuntimeError(f"cli_lqg.py failed: {cmd}\n{out.stderr}\n{stdout}")
    return data, " ".join(cmd[:-1])  # drop --json from the displayed command


def img_data_uri(path):
    with open(path, "rb") as f:
        b = f.read()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def fnum(x, nd=4):
    if x is None:
        return "&mdash;"
    return f"{x:,.{nd}f}"


def fmt_complex(pair, nd=4):
    re, im = pair
    sign = "+" if im >= 0 else "-"
    if abs(im) < 1e-9:
        return f"{re:.{nd}f}"
    return f"{re:.{nd}f} {sign} {abs(im):.{nd}f}j"


def poles_list_html(poles):
    return "".join(f"<li>{fmt_complex(p)}</li>" for p in sorted(poles, key=lambda z: z[0]))


def poles_list_md(poles):
    return "\n".join(f"- {fmt_complex(p)}" for p in sorted(poles, key=lambda z: z[0]))


def wrap_html(title, body):
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_e(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def write_pair(basename, title, body_html, body_md):
    os.makedirs(_OUT_DIR, exist_ok=True)
    html_path = os.path.join(_OUT_DIR, f"{basename}.html")
    with open(html_path, "w") as f:
        f.write(wrap_html(title, body_html))
    print(f"wrote {html_path}")
    md_path = os.path.join(_OUT_DIR, f"{basename}.md")
    with open(md_path, "w") as f:
        f.write(body_md)
    print(f"wrote {md_path}")
    return html_path


# ─────────────────────────────────────────────────────────────────────────
# AIDrone2
# ─────────────────────────────────────────────────────────────────────────

def build_drone_memo():
    plot_path = os.path.join(_OUT_DIR, "drone_regulator.png")
    reg_data, reg_cmd = run_cli("--plant-preset", "drone", "--method", "output_weighted",
                                "--sim", "state_feedback", "--plot", plot_path)
    err_data, err_cmd = run_cli("--plant-preset", "drone", "--method", "output_weighted",
                                "--reference-tracking", "--sim", "state_feedback",
                                expect_error=True)

    plot_uri = img_data_uri(plot_path)
    poles = reg_data["closed_loop_poles"]
    max_pole_real = max(p[0] for p in poles)

    body_html = f"""
<h1>Worked LQG Example &mdash; AIDrone2 (Expected Transmission-Zero Failure)</h1>

<dl class="memo-header">
  <dt>Date</dt><dd>{_e(_DATE)}</dd>
  <dt>Re</dt><dd>Full worked run of the drone plant (6 states, 2 inputs, 2 outputs) confirming
  the professor's flagged AIDrone2 failure &mdash; a transmission zero at the origin breaks the
  reference-tracking feedforward step, not the LQR design itself.</dd>
</dl>

<h2>1. Plant</h2>
<p>
Preset key <code>drone</code>, sourced from <code>AIDrone2.m</code> (A/B/C/D numerically
identical to the original <code>AIDrone.m</code> &mdash; see the 2026-08-25 revised-examples
memo &sect;1). 6 states, 2 inputs, 2 outputs. Suggested design: output-weighted LQR,
Q=C&#7511;C, R=I(2), per the source.</p>

<h2>2. LQR design and regulator response &mdash; succeeds</h2>
<p>The state-feedback design itself is well-posed and stable; nothing about the transmission
zero affects this step.</p>
<pre>{_e(reg_cmd)}</pre>
<table class="no-break">
  <tr><th>Check</th><th>Result</th></tr>
  <tr><td>Design</td><td class="pass">{'Stable' if reg_data['stable'] else 'Unstable'}</td></tr>
  <tr><td>Pre/post checks</td>
      <td class="{'pass' if reg_data['checks_all_passed'] else 'fail'}">{'All passed' if reg_data['checks_all_passed'] else 'Some failed'}</td></tr>
  <tr><td>Max closed-loop pole real part</td><td>{fnum(max_pole_real)}</td></tr>
  <tr><td>Settling (2%)</td><td>{fnum(reg_data['sim']['metrics']['settling_2pct'], 2)} s</td></tr>
  <tr><td>u_peak</td><td>{fnum(reg_data['sim']['metrics']['u_peak'], 3)}</td></tr>
  <tr><td>ISU</td><td>{fnum(reg_data['sim']['metrics']['ISU'], 3)}</td></tr>
</table>
<p>Closed-loop poles, real part ascending:</p>
<ul>{poles_list_html(poles)}</ul>
<p><img src="{plot_uri}" alt="Drone regulator state/control response" style="max-width:100%;border:1px solid #b9c3cc;border-radius:5px;"></p>

<h2>3. Reference tracking feedforward &mdash; fails as expected</h2>
<p>
Requesting the N&#772; reference-tracking feedforward gain requires inverting the Rosenbrock
system matrix <code>[[A,B],[C,D]]</code>. For this plant that matrix is singular (transmission
zero at the origin):
</p>
<pre>{_e(err_cmd)}</pre>
<div class="callout">
<strong>Result:</strong> <code>{_e(err_data.get('error', ''))}</code>
<br><br>
This is the expected, professor-flagged failure &mdash; not a bug. Treat it as an
asserted-failure test case in the validation catalog rather than something to chase to a fix.
</div>

<h2>4. Reproducing this in the Streamlit GUI</h2>
<ol>
<li>Launch the app: <code>streamlit run streamlit_app.py</code> (from <code>src/</code>).</li>
<li>Open the MIMO/LQG tab.</li>
<li>Plant source: <strong>Preset</strong> &rarr; select <code>drone</code> from the dropdown.
    The panel will confirm nx=6, nu=2, ny=2.</li>
<li>Method: <strong>Output-weighted LQR</strong>.</li>
<li>Leave "Reference tracking" <strong>unchecked</strong> and click design/simulate &mdash;
    reproduces &sect;2's stable regulator response and plot.</li>
<li>Re-run with "Reference tracking" <strong>checked</strong> (default reference is fine) &mdash;
    the panel surfaces the same singular-matrix error as &sect;3 via <code>st.error(...)</code>,
    since it calls the same <code>add_reference_tracking()</code> function as the CLI.</li>
</ol>
<p class="section-note">
Note: the GUI's LQG method only simulates the state-feedback regulator response (no
Kalman-filtered output-feedback simulation yet, per <code>streamlit_mimo_panel.py</code>'s
documented scope) &mdash; not relevant to this plant's LQR-only worked example, but worth knowing
if reproducing the genericRTP LQG memo's &sect;3.2 output-feedback run in the GUI instead of the
CLI.
</p>
"""

    body_md = f"""# Worked LQG Example — AIDrone2 (Expected Transmission-Zero Failure)

Date: {_DATE}
Re: Full worked run of the drone plant (6 states, 2 inputs, 2 outputs) confirming the
professor's flagged AIDrone2 failure — a transmission zero at the origin breaks the
reference-tracking feedforward step, not the LQR design itself.

---

## 1. Plant

Preset key `drone`, sourced from `AIDrone2.m` (A/B/C/D numerically identical to the original
`AIDrone.m` — see the 2026-08-25 revised-examples memo §1). 6 states, 2 inputs, 2 outputs.
Suggested design: output-weighted LQR, Q=CᵀC, R=I(2), per the source.

## 2. LQR design and regulator response — succeeds

The state-feedback design itself is well-posed and stable; nothing about the transmission zero
affects this step.

```
{reg_cmd}
```

| Check | Result |
|---|---|
| Design | {'Stable' if reg_data['stable'] else 'Unstable'} |
| Pre/post checks | {'All passed' if reg_data['checks_all_passed'] else 'Some failed'} |
| Max closed-loop pole real part | {fnum(max_pole_real)} |
| Settling (2%) | {fnum(reg_data['sim']['metrics']['settling_2pct'], 2)} s |
| u_peak | {fnum(reg_data['sim']['metrics']['u_peak'], 3)} |
| ISU | {fnum(reg_data['sim']['metrics']['ISU'], 3)} |

Closed-loop poles, real part ascending:

{poles_list_md(poles)}

![Drone regulator state/control response](drone_regulator.png)

## 3. Reference tracking feedforward — fails as expected

Requesting the N̄ reference-tracking feedforward gain requires inverting the Rosenbrock system
matrix `[[A,B],[C,D]]`. For this plant that matrix is singular (transmission zero at the origin):

```
{err_cmd}
```

**Result:** `{err_data.get('error', '')}`

This is the expected, professor-flagged failure — not a bug. Treat it as an asserted-failure test
case in the validation catalog rather than something to chase to a fix.

## 4. Reproducing this in the Streamlit GUI

1. Launch the app: `streamlit run streamlit_app.py` (from `src/`).
2. Open the MIMO/LQG tab.
3. Plant source: **Preset** → select `drone` from the dropdown. The panel will confirm
   nx=6, nu=2, ny=2.
4. Method: **Output-weighted LQR**.
5. Leave "Reference tracking" **unchecked** and click design/simulate — reproduces §2's stable
   regulator response and plot.
6. Re-run with "Reference tracking" **checked** (default reference is fine) — the panel surfaces
   the same singular-matrix error as §3 via `st.error(...)`, since it calls the same
   `add_reference_tracking()` function as the CLI.

Note: the GUI's LQG method only simulates the state-feedback regulator response (no
Kalman-filtered output-feedback simulation yet, per `streamlit_mimo_panel.py`'s documented scope)
— not relevant to this plant's LQR-only worked example, but worth knowing if reproducing the
genericRTP LQG memo's §3.2 output-feedback run in the GUI instead of the CLI.
"""

    return write_pair(f"{_DATE}-aidrone2-worked-example-memo",
                      "AIDrone2 Worked Example", body_html, body_md)


# ─────────────────────────────────────────────────────────────────────────
# genericRTP
# ─────────────────────────────────────────────────────────────────────────

def build_rtp_memo():
    lqr_plot = os.path.join(_OUT_DIR, "rtp_lqr.png")
    lqg_plot = os.path.join(_OUT_DIR, "rtp_lqg.png")
    per_channel_plot = os.path.join(_OUT_DIR, "rtp_per_channel_step.png")

    # Each design auto-crops its own plot independently (cli_lqg.py's
    # auto_plot_window(), no --plot-t-max override) -- tried sharing one
    # window (max of both) across §2/§3 on 2026-08-26, but that stretched
    # LQR's plot (whose own transient is much faster than LQG's) out to
    # LQG's slower window, making LQR look like an even more compressed
    # spike-then-flat than its own natural crop. Independent windows read
    # better for this pair even though the two plots then sit on different
    # timescales -- see each plot's own reported plot_t_max below.
    lqr_data, lqr_cmd = run_cli("--plant-preset", "generic_rtp", "--method", "output_weighted",
                                "--reference-tracking", "--sim", "state_feedback",
                                "--plot", lqr_plot)
    lqg_data, lqg_cmd = run_cli("--plant-preset", "generic_rtp", "--method", "lqg",
                                "--sim", "output_feedback", "--plot", lqg_plot)
    per_channel_data, per_channel_cmd = run_cli(
        "--plant-preset", "generic_rtp", "--method", "output_weighted",
        "--reference-tracking", "--sim", "per_channel_step", "--plot", per_channel_plot,
        "--plot-t-max", "5", "--plot-y-max", "1.1")

    # sensitivity (not yet CLI/GUI-exposed — computed directly via the library)
    import numpy as np
    from lqg_examples import load_example
    from lqg_design_methods import OutputWeightedLQR
    from lqg_frequency import compute_sensitivity
    ex = load_example("generic_rtp")
    res = OutputWeightedLQR(ex.plant).design()
    sens = compute_sensitivity(res)

    lqr_uri = img_data_uri(lqr_plot)
    lqg_uri = img_data_uri(lqg_plot)
    per_channel_uri = img_data_uri(per_channel_plot)
    lqr_poles = lqr_data["closed_loop_poles"]

    combined_overshoots = [m["Overshoot"] for m in lqr_data["sim"]["tracking_metrics"]]
    diag_overshoots = [entry["tracking_metrics"][j]["Overshoot"]
                       for j, entry in enumerate(per_channel_data["per_channel_step"])]
    overshoot_rows_html = "".join(
        f"<tr><td>y{j}</td><td>{fnum(c, 2)}%</td><td>{fnum(d, 2)}%</td></tr>"
        for j, (c, d) in enumerate(zip(combined_overshoots, diag_overshoots)))
    overshoot_rows_md = "\n".join(
        f"| y{j} | {fnum(c, 2)}% | {fnum(d, 2)}% |"
        for j, (c, d) in enumerate(zip(combined_overshoots, diag_overshoots)))

    body_html = f"""
<h1>Worked LQR/LQG Example &mdash; GenericRTP</h1>

<dl class="memo-header">
  <dt>Date</dt><dd>{_e(_DATE)}</dd>
  <dt>Re</dt><dd>Full worked LQR/LQG run on genericRTP (5&times;5 rapid thermal processing plant)
  &mdash; the professor's stated priority worked example for the paper's MIMO section.</dd>
</dl>

<h2>1. Plant</h2>
<p>
Preset key <code>generic_rtp</code>, sourced from <code>AIGeneric_RTP2.m</code> (matrices loaded
from <code>rtpsystem.dat</code>; A/B/C/D numerically identical to the original
<code>AIGeneric_RTP.m</code> &mdash; see the 2026-08-25 revised-examples memo &sect;1 and &sect;3).
15 states, 5 inputs, 5 outputs &mdash; the largest/highest-dimensional plant in the catalog.
Suggested design: output-weighted LQR, Q=C&#7511;C, R=I(5), per the source.
</p>

<h2>2. LQR design + reference tracking (state feedback)</h2>
<pre>{_e(lqr_cmd)}</pre>
<table class="no-break">
  <tr><th>Check</th><th>Result</th></tr>
  <tr><td>Design</td><td class="pass">{'Stable' if lqr_data['stable'] else 'Unstable'}</td></tr>
  <tr><td>Pre/post checks</td>
      <td class="{'pass' if lqr_data['checks_all_passed'] else 'fail'}">{'All passed' if lqr_data['checks_all_passed'] else 'Some failed'}</td></tr>
  <tr><td>Reference-tracking feedforward</td><td class="pass">Succeeds (no transmission zero at the origin for this plant)</td></tr>
  <tr><td>Settling (2%)</td><td>{fnum(lqr_data['sim']['metrics']['settling_2pct'], 1)} s</td></tr>
  <tr><td>u_peak</td><td>{fnum(lqr_data['sim']['metrics']['u_peak'], 3)}</td></tr>
  <tr><td>ISU</td><td>{fnum(lqr_data['sim']['metrics']['ISU'], 4)}</td></tr>
</table>
<p>15 closed-loop poles, real part ascending:</p>
<ul>{poles_list_html(lqr_poles)}</ul>
<p><img src="{lqr_uri}" alt="genericRTP LQR reference-tracking response" style="max-width:100%;border:1px solid #b9c3cc;border-radius:5px;"></p>
<p class="section-note">
Plot auto-cropped to <code>--plot-t-max {fnum(lqr_data['sim']['plot_t_max'], 1)}</code>s (this
design's own, independently-computed window &mdash; see &sect;3 below for LQG's) for readability
&mdash; the Settling (2%) figure above is still computed on the full ~1810s simulated duration
(auto_t_end() sizes that to genericRTP's slowest closed-loop pole), the crop only affects what
the plot displays.
</p>
<p class="section-note">
K is a 5&times;15 gain matrix &mdash; too large to usefully print inline; see the CLI's
<code>--json</code> output for the full matrix if needed.
</p>

<h3>2.1 Per-channel step response (MATLAB <code>step()</code> grid) &mdash; isolating cross-channel coupling</h3>
<p>
The plot above steps all 5 reference channels simultaneously (<code>r</code> = all-ones). MATLAB's
default <code>step(A-B*K, B*N&#772;, C, D)</code> on a MIMO state-space system instead produces a
grid: each reference channel is stepped <strong>individually</strong>, holding the other four at
zero, giving a 5&times;5 grid of single-channel responses across all 5 outputs. The new
<code>--sim per_channel_step</code> mode reproduces that grid:
</p>
<pre>{_e(per_channel_cmd)}</pre>
<p><img src="{per_channel_uri}" alt="genericRTP per-channel step response grid" style="max-width:100%;border:1px solid #b9c3cc;border-radius:5px;"></p>
<p class="section-note">
Every cell's axes are fixed to t &isin; [0, 5] s and y &isin; [0, 1.1] via
<code>--plot-t-max</code>/<code>--plot-y-max</code> (added alongside <code>per_channel_step</code>),
rather than each cell auto-scaling independently, so the 25 cells stay directly comparable at a
glance.
</p>
<p class="section-note">
Grid layout: column j is the response to stepping reference channel j alone; row i is output
y<sub>i</sub>(t). Diagonal cells (row i = column j) show each channel tracking its own commanded
step; off-diagonal cells show the cross-channel coupling terms that the combined plot above sums
together.
</p>
<p>
Comparing each output's overshoot in the combined step (&sect;2's plot) against that same output's
overshoot in its own isolated, diagonal per-channel step (this section's grid):
</p>
<table class="no-break">
  <tr><th>Output</th><th>Combined (5-channel) step overshoot</th><th>Isolated per-channel step overshoot</th></tr>
  {overshoot_rows_html}
</table>
<p class="section-note">
Every output's overshoot is measurably larger in the combined step than in its own isolated
per-channel response &mdash; roughly double, in every case. This is <strong>not</strong> a
different phenomenon appearing from nowhere: because the closed loop is linear, each output's
combined-step trajectory is exactly the sum of its five individual per-channel responses (one per
reference channel), which the per-channel grid decomposes. The extra overshoot in the combined
plot is the off-diagonal coupling terms above adding constructively on top of each channel's own
(smaller) diagonal overshoot &mdash; a transient that a single-channel MATLAB <code>step()</code>
call, run on one input at a time, would never surface on its own. For a report that's evaluating
"the" step response of this design, the per-channel grid is the more faithful reproduction of
MATLAB's own <code>step()</code> convention; the combined plot is a legitimate but different
question (what does the plant do if all five setpoints move at once).
</p>

<h2>3. LQG &mdash; adds a steady-state Kalman filter, output-feedback simulation</h2>
<p>
First-pass process/measurement noise weights (CLI defaults: Qw=0.01&middot;I(15),
Rv=0.1&middot;I(5) &mdash; not yet reconciled against the professor's own "robust servo design"
weighting in <code>AIGeneric_RTP2.m</code>'s integral-control section; see &sect;5).
</p>
<pre>{_e(lqg_cmd)}</pre>
<table class="no-break">
  <tr><th>Check</th><th>Result</th></tr>
  <tr><td>Design</td><td class="pass">{'Stable' if lqg_data['stable'] else 'Unstable'}</td></tr>
  <tr><td>Pre/post checks</td>
      <td class="{'pass' if lqg_data['checks_all_passed'] else 'fail'}">{'All passed' if lqg_data['checks_all_passed'] else 'Some failed'}</td></tr>
  <tr><td>Settling (2%), output feedback</td><td>{fnum(lqg_data['sim']['metrics']['settling_2pct'], 1)} s</td></tr>
  <tr><td>u_peak</td><td>{fnum(lqg_data['sim']['metrics']['u_peak'], 5)}</td></tr>
  <tr><td>ISU</td><td>{fnum(lqg_data['sim']['metrics']['ISU'], 6)}</td></tr>
</table>
<p><img src="{lqg_uri}" alt="genericRTP LQG output-feedback response" style="max-width:100%;border:1px solid #b9c3cc;border-radius:5px;"></p>
<p class="section-note">
Plot auto-cropped to <code>--plot-t-max {fnum(lqg_data['sim']['plot_t_max'], 1)}</code>s (this
design's own window, independent of &sect;2's LQR plot above &mdash; the two designs settle on
different timescales, so a shared window would stretch or compress one of them relative to its
own natural transient).
</p>

<h2>4. Sensitivity / robustness (plant-input loop transfer, LQR loop)</h2>
<p class="section-note">
Not yet exposed via CLI/GUI &mdash; computed directly against <code>lqg_frequency.
compute_sensitivity()</code> for this memo (same output-weighted LQR design as &sect;2).
</p>
<table class="no-break">
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>M<sub>s</sub> (peak &sigma;<sub>max</sub>(S))</td><td>{fnum(sens.Ms)}</td></tr>
  <tr><td>M<sub>t</sub> (peak &sigma;<sub>max</sub>(T))</td><td>{fnum(sens.Mt)}</td></tr>
</table>
<p>Both inside the classical LQR robustness guarantee (M<sub>s</sub>&le;2) for the full-state
plant-input loop transfer.</p>

<h3>4.1 Open question for the professor &mdash; does this match <code>AIGeneric_RTP2.m</code>'s own S/T/PM/GM?</h3>
<p>
<code>AIGeneric_RTP2.m</code> computes S, T, and PM/GM in a way that's more specific than what
our code currently does, in three respects:
</p>
<ol>
<li class="pass"><strong>[RESOLVED] Different loop, not just a different formula.</strong> The
source script's S/T (and PM/GM derived from them) are computed on the
<strong>integral-augmented closed loop</strong> &mdash; the block system <code>AAcl</code> that
includes the servo integrator state, the <code>K1</code>/<code>Ko</code> gain split, and the
Kalman filter gain <code>L</code>, all wired together (the "Robust Servo Design" section,
building <code>AAcl</code>/<code>BBcl</code>/<code>CCcl</code>/<code>DDcl</code>). Our current
<code>lqg_frequency.py</code> computes S/T on a plainer loop &mdash; LQR state feedback broken
at the plant input, no integral action, no Kalman filter folded in. Per the 2026-08-25
conversation with Professor, all phase/gain-margin calculations are with respect to the plant
input &mdash; so our plant-input convention is confirmed correct, and there's no need to
replicate <code>AAcl</code>'s integral-augmented construction just to match loop-breaking point.
(This doesn't by itself resolve items 2/3 below &mdash; the specific weighting and the PM/GM
formula itself are still open.)</li>
<li><strong>We don't compute PM/GM at all yet &mdash; only M<sub>s</sub>/M<sub>t</sub>.</strong>
The source script derives scalar phase/gain margins directly from the peak singular values:
<code>alpha = 1/max(&sigma;(S))</code>, then <code>PM1 = 2&middot;asin(alpha/2)</code>,
<code>GM1 = 1/(1-alpha)</code>, and the same from T for <code>PM2</code>/<code>GM2</code>,
combined into <code>GMplus</code>, <code>GMminus</code>, <code>PM = max(PM1, PM2)</code>. Our
code stops at reporting M<sub>s</sub>/M<sub>t</sub> above and never takes that next step to a
scalar PM/GM.</li>
<li><strong>The Q1/Am weighting for the integral-augmented LQR is bespoke to this plant.</strong>
<code>AIGeneric_RTP2.m</code> uses <code>Am = diag(-5,-5,-5,-5,-5)</code> and a specific
<code>Q1</code> with off-diagonal &minus;1 cross-terms &mdash; a different pattern from every
other <code>*2</code> file, which uses a plain <code>QQ = eye(n+m)</code> for its integral-control
section.</li>
</ol>
<div class="callout">
<strong>Open questions:</strong>
<ol>
<li class="pass">[RESOLVED, 2026-08-25] Should our reported S/T/PM/GM for genericRTP be computed
on the same integral-augmented, Kalman-filtered loop the source script uses (<code>AAcl</code>),
rather than the simpler LQR-at-plant-input loop we use now? &mdash; No: per the professor, all
phase/gain-margin calculations are with respect to the plant input, so we keep the current
plant-input convention and do not need to replicate <code>AAcl</code>'s construction.</li>
<li>Is the specific Q1/Am robust-servo weighting for genericRTP meant to be the "correct"/
recommended design for this plant specifically, or was it just an illustrative choice &mdash;
i.e., should our worked example replicate it exactly, or is a simpler generic weighting (what
we're using now) an acceptable stand-in for the paper?</li>
<li>Now that the loop-breaking point is settled (plant input): is <code>PM = max(PM1, PM2)</code>/
the <code>GMplus</code>/<code>GMminus</code> construction (derived from S and T's peak singular
values) what you want us reporting as "the" multivariable PM/GM for the paper, or is there a
different convention you'd prefer?</li>
</ol>
</div>

<h2>5. Reproducing this in the Streamlit GUI</h2>
<ol>
<li>Launch the app: <code>streamlit run streamlit_app.py</code> (from <code>src/</code>).</li>
<li>Open the MIMO/LQG tab.</li>
<li>Plant source: <strong>Preset</strong> &rarr; select <code>generic_rtp</code>. Confirms
    nx=15, nu=5, ny=5.</li>
<li>For &sect;2 (LQR + reference tracking): Method <strong>Output-weighted LQR</strong>, check
    "Reference tracking", leave the reference field blank (defaults to all-ones) &mdash;
    reproduces the design/response above.</li>
<li>For &sect;2.1 (per-channel step grid): after the &sect;2 design above, set Grid t_max to
    <code>5</code> and Grid y_max to <code>1.1</code>, then click
    "&#8862; Per-channel step response" &mdash; reproduces the 5&times;5 grid exactly, including
    its fixed axes (blank fields reproduce the same data auto-scaled per cell instead).</li>
<li>For &sect;3 (LQG): Method <strong>LQG (Kalman filter)</strong> &mdash; reproduces the design,
    but <strong>only the state-feedback regulator simulation is available in the GUI</strong>
    (no output-feedback sim yet, unlike the CLI's <code>--sim output_feedback</code> used above
    &mdash; see <code>streamlit_mimo_panel.py</code>'s documented scope gap). Use the CLI command
    in &sect;3 to reproduce the output-feedback plot exactly.</li>
<li>&sect;4's sensitivity/Ms/Mt numbers have no GUI or CLI surface yet &mdash; not reproducible
    outside a direct Python call to <code>lqg_frequency.compute_sensitivity()</code>.</li>
</ol>
"""

    body_md = f"""# Worked LQR/LQG Example — GenericRTP

Date: {_DATE}
Re: Full worked LQR/LQG run on genericRTP (5×5 rapid thermal processing plant) — the
professor's stated priority worked example for the paper's MIMO section.

---

## 1. Plant

Preset key `generic_rtp`, sourced from `AIGeneric_RTP2.m` (matrices loaded from
`rtpsystem.dat`; A/B/C/D numerically identical to the original `AIGeneric_RTP.m` — see the
2026-08-25 revised-examples memo §1 and §3). 15 states, 5 inputs, 5 outputs — the
largest/highest-dimensional plant in the catalog. Suggested design: output-weighted LQR,
Q=CᵀC, R=I(5), per the source.

## 2. LQR design + reference tracking (state feedback)

```
{lqr_cmd}
```

| Check | Result |
|---|---|
| Design | {'Stable' if lqr_data['stable'] else 'Unstable'} |
| Pre/post checks | {'All passed' if lqr_data['checks_all_passed'] else 'Some failed'} |
| Reference-tracking feedforward | Succeeds (no transmission zero at the origin for this plant) |
| Settling (2%) | {fnum(lqr_data['sim']['metrics']['settling_2pct'], 1)} s |
| u_peak | {fnum(lqr_data['sim']['metrics']['u_peak'], 3)} |
| ISU | {fnum(lqr_data['sim']['metrics']['ISU'], 4)} |

15 closed-loop poles, real part ascending:

{poles_list_md(lqr_poles)}

![genericRTP LQR reference-tracking response](rtp_lqr.png)

Plot auto-cropped to `--plot-t-max {fnum(lqr_data['sim']['plot_t_max'], 1)}`s (this design's own,
independently-computed window — see §3 below for LQG's) for readability — the Settling (2%)
figure above is still computed on the full ~1810s simulated duration (auto_t_end() sizes that to
genericRTP's slowest closed-loop pole), the crop only affects what the plot displays.

K is a 5×15 gain matrix — too large to usefully print inline; see the CLI's `--json` output for
the full matrix if needed.

### 2.1 Per-channel step response (MATLAB `step()` grid) — isolating cross-channel coupling

The plot above steps all 5 reference channels simultaneously (`r` = all-ones). MATLAB's default
`step(A-B*K, B*Nbar, C, D)` on a MIMO state-space system instead produces a grid: each reference
channel is stepped **individually**, holding the other four at zero, giving a 5×5 grid of
single-channel responses across all 5 outputs. The new `--sim per_channel_step` mode reproduces
that grid:

```
{per_channel_cmd}
```

![genericRTP per-channel step response grid](rtp_per_channel_step.png)

Every cell's axes are fixed to t ∈ [0, 5] s and y ∈ [0, 1.1] via `--plot-t-max`/`--plot-y-max`
(added alongside `per_channel_step`), rather than each cell auto-scaling independently, so the 25
cells stay directly comparable at a glance.

Grid layout: column j is the response to stepping reference channel j alone; row i is output
y_i(t). Diagonal cells (row i = column j) show each channel tracking its own commanded step;
off-diagonal cells show the cross-channel coupling terms that the combined plot above sums
together.

Comparing each output's overshoot in the combined step (§2's plot) against that same output's
overshoot in its own isolated, diagonal per-channel step (this section's grid):

| Output | Combined (5-channel) step overshoot | Isolated per-channel step overshoot |
|---|---|---|
{overshoot_rows_md}

Every output's overshoot is measurably larger in the combined step than in its own isolated
per-channel response — roughly double, in every case. This is **not** a different phenomenon
appearing from nowhere: because the closed loop is linear, each output's combined-step trajectory
is exactly the sum of its five individual per-channel responses (one per reference channel), which
the per-channel grid decomposes. The extra overshoot in the combined plot is the off-diagonal
coupling terms above adding constructively on top of each channel's own (smaller) diagonal
overshoot — a transient that a single-channel MATLAB `step()` call, run on one input at a time,
would never surface on its own. For a report that's evaluating "the" step response of this
design, the per-channel grid is the more faithful reproduction of MATLAB's own `step()`
convention; the combined plot is a legitimate but different question (what does the plant do if
all five setpoints move at once).

## 3. LQG — adds a steady-state Kalman filter, output-feedback simulation

First-pass process/measurement noise weights (CLI defaults: Qw=0.01·I(15), Rv=0.1·I(5) — not yet
reconciled against the professor's own "robust servo design" weighting in `AIGeneric_RTP2.m`'s
integral-control section; see §5).

```
{lqg_cmd}
```

| Check | Result |
|---|---|
| Design | {'Stable' if lqg_data['stable'] else 'Unstable'} |
| Pre/post checks | {'All passed' if lqg_data['checks_all_passed'] else 'Some failed'} |
| Settling (2%), output feedback | {fnum(lqg_data['sim']['metrics']['settling_2pct'], 1)} s |
| u_peak | {fnum(lqg_data['sim']['metrics']['u_peak'], 5)} |
| ISU | {fnum(lqg_data['sim']['metrics']['ISU'], 6)} |

![genericRTP LQG output-feedback response](rtp_lqg.png)

Plot auto-cropped to `--plot-t-max {fnum(lqg_data['sim']['plot_t_max'], 1)}`s (this design's own
window, independent of §2's LQR plot above — the two designs settle on different timescales, so a
shared window would stretch or compress one of them relative to its own natural transient).

## 4. Sensitivity / robustness (plant-input loop transfer, LQR loop)

Not yet exposed via CLI/GUI — computed directly against `lqg_frequency.compute_sensitivity()`
for this memo (same output-weighted LQR design as §2).

| Metric | Value |
|---|---|
| Ms (peak σmax(S)) | {fnum(sens.Ms)} |
| Mt (peak σmax(T)) | {fnum(sens.Mt)} |

Both inside the classical LQR robustness guarantee (Ms≤2) for the full-state plant-input loop
transfer.

### 4.1 Open question for the professor — does this match `AIGeneric_RTP2.m`'s own S/T/PM/GM?

`AIGeneric_RTP2.m` computes S, T, and PM/GM in a way that's more specific than what our code
currently does, in three respects:

1. **[RESOLVED] Different loop, not just a different formula.** The source script's S/T (and
   PM/GM derived from them) are computed on the **integral-augmented closed loop** — the block
   system `AAcl` that includes the servo integrator state, the `K1`/`Ko` gain split, and the
   Kalman filter gain `L`, all wired together (the "Robust Servo Design" section, building
   `AAcl`/`BBcl`/`CCcl`/`DDcl`). Our current `lqg_frequency.py` computes S/T on a plainer loop —
   LQR state feedback broken at the plant input, no integral action, no Kalman filter folded in.
   Per the 2026-08-25 conversation with Professor, all phase/gain-margin calculations are with
   respect to the plant input — so our plant-input convention is confirmed correct, and there's
   no need to replicate `AAcl`'s integral-augmented construction just to match loop-breaking
   point. (This doesn't by itself resolve items 2/3 below — the specific weighting and the PM/GM
   formula itself are still open.)
2. **We don't compute PM/GM at all yet — only Ms/Mt.** The source script derives scalar
   phase/gain margins directly from the peak singular values: `alpha = 1/max(σ(S))`, then
   `PM1 = 2·asin(alpha/2)`, `GM1 = 1/(1-alpha)`, and the same from T for `PM2`/`GM2`, combined
   into `GMplus`, `GMminus`, `PM = max(PM1, PM2)`. Our code stops at reporting Ms/Mt above and
   never takes that next step to a scalar PM/GM.
3. **The Q1/Am weighting for the integral-augmented LQR is bespoke to this plant.**
   `AIGeneric_RTP2.m` uses `Am = diag(-5,-5,-5,-5,-5)` and a specific `Q1` with off-diagonal −1
   cross-terms — a different pattern from every other `*2` file, which uses a plain
   `QQ = eye(n+m)` for its integral-control section.

**Open questions:**

1. **[RESOLVED, 2026-08-25]** Should our reported S/T/PM/GM for genericRTP be computed on the
   same integral-augmented, Kalman-filtered loop the source script uses (`AAcl`), rather than
   the simpler LQR-at-plant-input loop we use now? — No: per the professor, all phase/gain-margin
   calculations are with respect to the plant input, so we keep the current plant-input
   convention and do not need to replicate `AAcl`'s construction.
2. Is the specific Q1/Am robust-servo weighting for genericRTP meant to be the "correct"/
   recommended design for this plant specifically, or was it just an illustrative choice — i.e.,
   should our worked example replicate it exactly, or is a simpler generic weighting (what we're
   using now) an acceptable stand-in for the paper?
3. Now that the loop-breaking point is settled (plant input): is `PM = max(PM1, PM2)`/the
   `GMplus`/`GMminus` construction (derived from S and T's peak singular values) what you want us
   reporting as "the" multivariable PM/GM for the paper, or is there a different convention you'd
   prefer?

## 5. Reproducing this in the Streamlit GUI

1. Launch the app: `streamlit run streamlit_app.py` (from `src/`).
2. Open the MIMO/LQG tab.
3. Plant source: **Preset** → select `generic_rtp`. Confirms nx=15, nu=5, ny=5.
4. For §2 (LQR + reference tracking): Method **Output-weighted LQR**, check "Reference
   tracking", leave the reference field blank (defaults to all-ones) — reproduces the
   design/response above.
5. For §2.1 (per-channel step grid): after the §2 design above, set Grid t_max to `5` and Grid
   y_max to `1.1`, then click "⊞ Per-channel step response" — reproduces the 5×5 grid exactly,
   including its fixed axes (blank fields reproduce the same data auto-scaled per cell instead).
6. For §3 (LQG): Method **LQG (Kalman filter)** — reproduces the design, but **only the
   state-feedback regulator simulation is available in the GUI** (no output-feedback sim yet,
   unlike the CLI's `--sim output_feedback` used above — see `streamlit_mimo_panel.py`'s
   documented scope gap). Use the CLI command in §3 to reproduce the output-feedback plot
   exactly.
7. §4's sensitivity/Ms/Mt numbers have no GUI or CLI surface yet — not reproducible outside a
   direct Python call to `lqg_frequency.compute_sensitivity()`.
"""

    return write_pair(f"{_DATE}-genericrtp-worked-example-memo",
                      "GenericRTP Worked Example", body_html, body_md)


if __name__ == "__main__":
    build_drone_memo()
    build_rtp_memo()
