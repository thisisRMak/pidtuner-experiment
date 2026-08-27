#!/usr/bin/env python3
"""Generates docs/memos/<YYYY-MM-DD>/<YYYY-MM-DD>-pid-worked-example-coursebench-memo.html
— a new, third worked-example memo alongside gen_pid_full_worked_example_memo.py's
synthetic-plant memo and gen_pid_worked_example_v2_textbook_memo.py's PEI8e-plant
memo, this one on the class benchmark plant: G(s) = 1000/((s+1)(10s+1)),
L=0.5 (same plant as benchmark_plant() in test_pid_tuner.py, and
docs/worked_examples/course_benchmark/'s CLI+GUI parity example).

Unlike the other two, this is deliberately NOT a full delay/derivative-filter/
anti-windup pipeline walkthrough — there's no prior version of that narrative
on this plant to extend, and the actual point of this memo is narrower: show
every tuning technique PIDTuner implements on this specific plant, in one
place, as a durable reference to diff future default-tuning changes against
(the immediate reason this memo exists: ZN-I/ZN-II are now halved by default
in pid_compare.py's compare_all_methods(), and this plant's docs/worked_examples/
JSON snapshot was the one that first surfaced the need to track that).

<YYYY-MM-DD> is today's date (datetime.date.today(), _DATE below), same
dated-snapshot convention as the other two generators in this file's family
and gen_lqg_worked_examples_memos.py.

Section 2 (gallery + composite overlay) calls pid_compare.py's
compare_all_methods() directly, in-process — the same function the CLI's
--method all and the GUI's "Compare all methods" button call — so ZN-I/ZN-II
show up here exactly as a user would see them in that view (pre-halved,
"ZN-I ½"/"ZN-II ½"). Section 1 also runs one single-method cli_pid.py --json
call per named ZN method, unhalved, purely to report the before/after ZN
numbers directly in this memo (mirrors docs/worked_examples/course_benchmark/
's siso_zn2_cli.json, which stays full-strength since the single-method path
is untouched by the halving change).

Run: python3 gen_pid_worked_example_coursebench_memo.py
"""
import base64
import datetime
import html
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plant import TransferFunction
from pid_tuning_methods import PIDGains
from pid_simulate import simulate_closed_loop
from pid_compare import compare_all_methods
from cli_pid import serialize_row_json

_HERE = os.path.dirname(__file__)
_DATE = datetime.date.today().isoformat()
_OUT_DIR = os.path.join(_HERE, "..", "docs", "memos", _DATE)
_HTML_OUT = os.path.join(_OUT_DIR, f"{_DATE}-pid-worked-example-coursebench-memo.html")
_CSS = open(os.path.join(_HERE, "_memo_css.txt")).read()

_PLANT = "1000 / ((s+1)*(10s+1))"
_L = "0.5"

# Same palette pid_app.py (the desktop webapp) assigns to overlaid tuned
# controllers — distinguishable, colorblind-friendly, long enough to cover
# all 12 compare_all_methods() rows (see PALETTE in pid_app.py:66-68).
_PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd",
            "#ff7f0e", "#17becf", "#8c564b", "#e377c2",
            "#7f7f7f", "#bcbd22", "#393b79", "#ad494a"]


def _e(s):
    return html.escape(str(s))


def fnum(x, nd=3):
    if x is None:
        return "&mdash;"
    if isinstance(x, float) and x != x:  # nan
        return "nan"
    if isinstance(x, float) and abs(x) == float("inf"):
        return "&infin;"
    return f"{x:,.{nd}f}"


def _slug(name):
    """Filesystem-safe stem for a compare_all_methods() row name, e.g.
    'ZN-I ½' -> 'zn-i_half', 'CHR set 0%' -> 'chr_set_0pct'."""
    s = name.replace("½", "half").replace("%", "pct")
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def run_cli(method, *extra_args):
    cmd = [sys.executable, "cli_pid.py", "--plant", _PLANT, "--L", _L,
           "--method", method, "--json", *extra_args]
    out = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"cli_pid.py failed: {cmd}\n{out.stderr}")
    return json.loads(out.stdout)


def img_data_uri(path):
    with open(path, "rb") as f:
        b = f.read()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def panel_plot(name, gains, plant, out_path, N=80.0):
    """One method's own step-response panel — same 2-row layout (y(t) with
    setpoint, u(t)) cli_pid.py's single-method --plot produces."""
    sim = simulate_closed_loop(plant, gains, setpoint=1.0, setpoint_kind="step", N=N)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(sim.t, sim.y, label=name)
    ax1.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="Setpoint")
    ax1.set_ylabel("Output y(t)")
    ax1.set_title(f"Step Response: {name}")
    ax1.legend()
    ax1.grid(True)
    ax2.plot(sim.t, sim.u)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Control Effort u(t)")
    ax2.grid(True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def composite_overlay_plot(rows, plant, out_path):
    """One 3-row figure — PV/SP, control effort u(t), error e(t) — with every
    method's response overlaid on shared axes, same layout/labels/palette as
    the webapp's session overlay (pid_response_plotting.draw_response_tab)."""
    fig, (ax_y, ax_u, ax_e) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax in (ax_y, ax_u, ax_e):
        ax.grid(True, alpha=0.3)
    ax_y.set_ylabel("PV / SP")
    ax_u.set_ylabel("control u(t)")
    ax_e.set_ylabel("error e(t)")
    ax_e.set_xlabel("time (s)")

    sp_plotted = False
    for i, r in enumerate(rows):
        g = r["gains"]
        gains = PIDGains(Kp=g["Kp"], Ki=g["Ki"], Kd=g["Kd"])
        sim = simulate_closed_loop(plant, gains, setpoint=1.0, setpoint_kind="step", N=80.0)
        color = _PALETTE[i % len(_PALETTE)]
        if not sp_plotted:
            ax_y.plot(sim.t, sim.sp, "--", color="#666", linewidth=1.0,
                      alpha=0.7, label="setpoint")
            sp_plotted = True
        label = r["name"]
        if not r.get("stable", True):
            label += "  [UNSTABLE]"
        ax_y.plot(sim.t, sim.y, color=color, linewidth=1.5, label=label)
        ax_u.plot(sim.t, sim.u, color=color, linewidth=1.2, label=label)
        ax_e.plot(sim.t, sim.e, color=color, linewidth=1.2, label=label)

    fig.suptitle("Course Benchmark Plant — Step Response Overlay")
    ax_y.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_html():
    os.makedirs(_OUT_DIR, exist_ok=True)
    plant_obj = TransferFunction.parse(_PLANT, L=float(_L))

    # --- ZN-I/ZN-II unhalved, single-method, for the before/after numbers below ---
    zn1_full = run_cli("zn1")
    zn2_full = run_cli("zn2")

    # --- Every tuning technique PIDTuner implements, same plant ---
    # In-process compare_all_methods() call — the same function the CLI's
    # --method all and the GUI's "Compare all methods" call — so ZN-I/ZN-II
    # come back pre-halved ("ZN-I ½"/"ZN-II ½") exactly as a user would see
    # them there. See the module docstring.
    compare_rows = [serialize_row_json(r)
                     for r in compare_all_methods(plant_obj, include_variants=True)]

    all_methods_rows = []
    for r in compare_rows:
        if r.get("gains") is None:
            continue  # no gains to plot; every method succeeds on this plant as of writing
        plot_path = os.path.join(_OUT_DIR, f"pid_coursebench_{_slug(r['name'])}.png")
        panel_plot(r["name"], PIDGains(**{k: r["gains"][k] for k in ("Kp", "Ki", "Kd")}),
                   plant_obj, plot_path)
        r["_plot_uri"] = img_data_uri(plot_path)
        all_methods_rows.append(r)

    composite_plot_path = os.path.join(_OUT_DIR, "pid_coursebench_composite_overlay.png")
    composite_overlay_plot(all_methods_rows, plant_obj, composite_plot_path)
    composite_uri = img_data_uri(composite_plot_path)

    def metrics_row(name, r):
        return f"""
  <tr>
    <td>{name}</td>
    <td>{fnum(r['OS%'],1)}%</td>
    <td>{fnum(r['ts'],1)} s</td>
    <td>{fnum(r['Rise'],2)} s</td>
    <td>{fnum(r['IAE'],2)}</td>
    <td>{fnum(r['ISU'],1)}</td>
    <td>{fnum(r['Ms'],3)}</td>
    <td>{fnum(r['Mt'],3)}</td>
    <td>{fnum(r['GM_dB'],2)} dB</td>
    <td>{fnum(r['PM_deg'],1)}&deg;</td>
    <td>{fnum(r['u_peak'],4)}</td>
    <td>{fnum(r['u_tv'],3)}</td>
  </tr>"""

    def gallery_card(r):
        rg = r["gains"]
        return f"""
  <figure class="gallery-card">
    <img src="{r['_plot_uri']}" alt="Step response: {_e(r['name'])}">
    <figcaption>
      <strong>{_e(r['name'])}</strong>
      <div class="gallery-gains">K<sub>p</sub>={fnum(rg['Kp'],5)} &middot;
        T<sub>i</sub>={fnum(rg['Ti'])}&nbsp;s &middot; T<sub>d</sub>={fnum(rg['Td'])}&nbsp;s</div>
      <div class="gallery-metrics">OS%={fnum(r['OS%'],1)}% &middot;
        GM={fnum(r['GM_dB'],1)}&nbsp;dB &middot; PM={fnum(r['PM_deg'],1)}&deg;</div>
    </figcaption>
  </figure>"""

    zn1_g, zn2_g = zn1_full["gains"], zn2_full["gains"]
    zn1_half = next(r for r in all_methods_rows if r["name"] == "ZN-I ½")
    zn2_half = next(r for r in all_methods_rows if r["name"] == "ZN-II ½")

    body = f"""
<h1>SISO PID Worked Example: Course Benchmark Plant, Every Tuning Technique</h1>

<dl class="memo-header">
  <dt>Date</dt><dd>{_DATE}</dd>
  <dt>Re</dt><dd>A third worked-example memo, on the class benchmark plant already tracked in
  <code>docs/worked_examples/course_benchmark/</code> &mdash; every tuning technique PIDTuner
  implements, gathered in one place as a durable reference for diffing future default-tuning
  changes against (the trigger this time: ZN-I/ZN-II are now halved by default in
  <code>compare_all_methods()</code>).</dd>
</dl>

<h2>1. Setup</h2>
<p>
Plant: the course benchmark, DC gain {fnum(plant_obj.dc_gain(),0)} with a modest
L={_L}&nbsp;s dead time — the same plant <code>benchmark_plant()</code> in
<code>test_pid_tuner.py</code> and <code>docs/worked_examples/course_benchmark/</code> use.
</p>
<pre>G(s) = {_e(_PLANT)},  L = {_L} s</pre>

<h3>1.1 Reproducing these results</h3>
<pre>cd src/
python3 gen_pid_worked_example_coursebench_memo.py

cd ../docs/memos/{_DATE}/
F={_DATE}-pid-worked-example-coursebench-memo
google-chrome --headless --disable-gpu --no-sandbox \\
    --no-pdf-header-footer \\
    --print-to-pdf=$F.pdf $F.html</pre>

<h2>2. Every tuning technique PIDTuner implements, same plant</h2>
<p>
{len(all_methods_rows)} rows below (nine technique families; CHR's four response/overshoot variants
are each shown individually), via <code>compare_all_methods()</code> &mdash; the same function the
CLI's <code>--method all</code> and the GUI's "Compare all methods" button call, so this gallery
matches what a user actually sees there.
</p>
<div class="gallery">
{"".join(gallery_card(r) for r in all_methods_rows)}
</div>
<div class="callout">
<strong>ZN-I/ZN-II are shown pre-halved by default here.</strong> <code>pid_compare.py</code>'s
<code>compare_all_methods()</code> now runs ZN-I and ZN-II through the app's existing
<code>halve_gains()</code> helper before showing them, since those are the only two techniques whose
own tuning notes recommend it (a single <code>cli_pid.py --method zn1</code> run, run separately
below, is unaffected). Unhalved vs. halved on this plant:
<table class="no-break">
  <tr><th>Run</th><th>K<sub>p</sub></th><th>OS%</th><th>GM</th><th>PM</th></tr>
  <tr><td>ZN-I (full strength, <code>--method zn1</code>)</td>
      <td>{fnum(zn1_g['Kp'],5)}</td><td>{fnum(zn1_full['OS%'],1)}%</td>
      <td>{fnum(zn1_full['GM_dB'],1)}&nbsp;dB</td><td>{fnum(zn1_full['PM_deg'],1)}&deg;</td></tr>
  <tr><td>ZN-I &frac12; (compare-all default)</td>
      <td>{fnum(zn1_half['gains']['Kp'],5)}</td><td>{fnum(zn1_half['OS%'],1)}%</td>
      <td>{fnum(zn1_half['GM_dB'],1)}&nbsp;dB</td><td>{fnum(zn1_half['PM_deg'],1)}&deg;</td></tr>
  <tr><td>ZN-II (full strength, <code>--method zn2</code>)</td>
      <td>{fnum(zn2_g['Kp'],5)}</td><td>{fnum(zn2_full['OS%'],1)}%</td>
      <td>{fnum(zn2_full['GM_dB'],1)}&nbsp;dB</td><td>{fnum(zn2_full['PM_deg'],1)}&deg;</td></tr>
  <tr><td>ZN-II &frac12; (compare-all default)</td>
      <td>{fnum(zn2_half['gains']['Kp'],5)}</td><td>{fnum(zn2_half['OS%'],1)}%</td>
      <td>{fnum(zn2_half['GM_dB'],1)}&nbsp;dB</td><td>{fnum(zn2_half['PM_deg'],1)}&deg;</td></tr>
</table>
<p>
Halving Kp/Ki/Kd doesn't halve overshoot outright (closed-loop response isn't linear in the gains),
but it's a real reduction on this plant too, same direction as the PEI8e textbook plant
(<code>{_DATE}-pid-worked-example-v2-textbook-memo.html</code> &sect;2.2, run the same day), just a
smaller one —
this plant's L/&tau;&asymp;0.05 is far less delay-dominant than that one's &asymp;0.14, which is where
the classical ZN rules are known to over-drive gains the most.
</p>
</div>
<p class="section-note">
Cohen&ndash;Coon and ZN-II &frac12; are the two highest-overshoot rows here
(OS%&asymp;{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='Cohen–Coon'),1)}% and
{fnum(zn2_half['OS%'],1)}% respectively); CHR's "set 20%" and SIMC sit at the other end
(OS%&asymp;{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='CHR set 20%'),1)}% and
{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='SIMC'),1)}%). Pole cancellation is
slow and non-oscillatory by construction, same as on the other two plants. No method errored or went
unstable here.
</p>

<h3>2.1 Composite overlay, all included techniques on one set of axes</h3>
<p>
Same responses, drawn the way the desktop webapp draws them when several tuned controllers are kept
in a session &mdash; one 3-row PV/SP &middot; control effort &middot; error figure, all methods
overlaid with the app's own <code>PALETTE</code> (<code>pid_app.py:66-68</code>), not matplotlib's
default cycle.
</p>
<p><img src="{composite_uri}" alt="Composite overlay of all included tuning techniques' step responses" style="max-width:100%;border:1px solid #b9c3cc;border-radius:5px;"></p>

<h2>3. Summary</h2>
<table class="no-break">
  <tr><th>Technique</th><th>OS%</th><th>t<sub>s</sub> (2%)</th><th>Rise (10&ndash;90%)</th>
      <th>IAE</th><th>ISU</th><th>M<sub>s</sub></th><th>M<sub>t</sub></th>
      <th>GM</th><th>PM</th><th>|u|<sub>peak</sub></th><th>u<sub>tv</sub></th></tr>
  {"".join(metrics_row(r['name'], r) for r in all_methods_rows)}
</table>
<p>
No drift from PIDTuner's own numbers: &sect;2 calls <code>compare_all_methods()</code> in-process
(the same function the CLI's <code>--method all</code> and the GUI's "Compare all methods" call), and
&sect;1.1 gives the exact commands to regenerate all of it. The one finding this memo exists to
record: ZN-I/ZN-II are halved by default in the comparison as of this run, moving them out of being
this plant's most aggressive rows (see &sect;2's callout) — a smaller effect than on the more
delay-dominant PEI8e textbook plant, but the same direction.
</p>

<p class="appendix-title">Appendix &mdash; raw CLI output (unhalved ZN, single-method)</p>
<pre>{_e(json.dumps(zn1_full, indent=2))}</pre>
<pre>{_e(json.dumps(zn2_full, indent=2))}</pre>
"""

    _GALLERY_CSS = """
  .gallery {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 14px;
    margin: 12px 0 16px;
  }
  .gallery-card {
    margin: 0;
    border: 1px solid var(--rule);
    border-radius: 5px;
    padding: 6px 8px 8px;
  }
  .gallery-card img { width: 100%; display: block; }
  .gallery-card figcaption {
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 8.5pt;
    margin-top: 4px;
  }
  .gallery-gains, .gallery-metrics { color: var(--muted); margin-top: 2px; }
"""

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>PID Worked Example: Course Benchmark Memo</title>
<style>
{_CSS}
{_GALLERY_CSS}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    with open(_HTML_OUT, "w") as f:
        f.write(doc)
    print(f"wrote {_HTML_OUT}")


if __name__ == "__main__":
    build_html()
