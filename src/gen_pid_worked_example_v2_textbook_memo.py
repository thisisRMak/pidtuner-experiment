#!/usr/bin/env python3
"""Generates docs/memos/<YYYY-MM-DD>/<YYYY-MM-DD>-pid-worked-example-v2-textbook-memo.html
— the v2 companion to gen_pid_full_worked_example_memo.py's synthetic-plant
memo, same structure and same features exercised (delay-aware tuning, the
derivative filter Kd/N, back-calculation anti-windup Ka, every tuning
technique individually and as a composite overlay), but run against a real
textbook plant instead of a made-up one: the FOPDT surrogate PEI8e (Franklin,
Powell & Emami-Naeini, "Feedback Control of Dynamic Systems", 8th ed.)
Examples 4.9/4.10 use for the Example 2.18 heat exchanger — plant '1/(90s+1)',
L=13 — same surrogate 2026-08-18-zn-textbook-validation-memo.html already
validates zn1/zn2 against. This memo does NOT delete or modify the 2026-08-25
run of this same script, or the v1 synthetic-plant worked example.
<YYYY-MM-DD> is today's date (datetime.date.today(), _DATE below) -- each run
lands in its own dated snapshot folder rather than overwriting a fixed
2026-08-25 the way earlier revisions of this script did (same convention
gen_lqg_worked_examples_memos.py uses), so re-running it later never
silently mislabels a new memo with a stale date or clobbers an old one.

One thing came up on this plant that didn't on the synthetic one, and is
called out rather than smoothed over:
  - Stable pole cancellation can't run at all here (needs >=2 stable plant
    poles; this FOPDT surrogate only has one) — captured via
    compare_all_methods()'s own error field rather than skipped silently.

(2026-08-28 note: an earlier run of this memo also flagged Cohen-Coon's
response here as diverging to thousands of times the setpoint and never
settling — that was a simulate_closed_loop() discretization bug (the
timestep auto-picked from the plant's own dynamics under-resolved the
derivative filter's much faster pole), not a property of the Cohen-Coon
design itself. Fixed in pid_simulate.py; Cohen-Coon now settles normally
like every other row below and is back in the composite overlay.)

Section 2.2/2.3 (the gallery + composite overlay) call pid_compare.py's
compare_all_methods() directly, in-process, rather than shelling out to
cli_pid.py once per method (the 2026-08-25 run did the latter) — the two
approaches gave identical numbers back then, but ZN-I/ZN-II are now halved
by default in compare_all_methods() (see pid_compare.py; cli_pid.py
--method all's own --halve flag is a no-op there for the same reason) while
a lone `cli_pid.py --method zn1` is not, so this section has to go through
the same call the app's own "Compare all methods" view uses to actually
reflect that. Section 2.1 (AMIGO) and section 4 (saturation/anti-windup)
still run cli_pid.py's own --json/--plot output directly, since AMIGO's
single-method path is untouched by the ZN change. Section 3.2
(derivative-filter vs. measurement noise) drives pid_simulate.py's own
PIDState/pid_step/compute_metrics directly, same reasoning as v1 — no
--noise-sigma hook on the simulated response.

Run: python3 gen_pid_worked_example_v2_textbook_memo.py
"""
import base64
import datetime
import html
import json
import os
import re
import subprocess
import sys
import textwrap

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plant import TransferFunction
from pid_tuning_methods import PIDGains
from pid_simulate import PIDState, pid_step, compute_metrics, simulate_closed_loop
from lqg_simulate import auto_plot_window
from pid_compare import compare_all_methods
from cli_pid import serialize_row_json

_HERE = os.path.dirname(__file__)
_DATE = datetime.date.today().isoformat()
_OUT_DIR = os.path.join(_HERE, "..", "docs", "memos", _DATE)
_HTML_OUT = os.path.join(_OUT_DIR, f"{_DATE}-pid-worked-example-v2-textbook-memo.html")
_CSS = open(os.path.join(_HERE, "_memo_css.txt")).read()

_PLANT = "1/(90s+1)"
_L = "13"
_U_MIN, _U_MAX = -1.3, 1.3

# Same palette pid_app.py (the desktop webapp) assigns to overlaid tuned
# controllers — distinguishable, colorblind-friendly (see PALETTE in
# pid_app.py:66-68).
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


def panel_plot(name, gains, plant, out_path, N=80.0):
    """One method's own step-response panel — same 2-row layout (y(t) with
    setpoint, u(t)) cli_pid.py's single-method --plot produces, since this
    replaces what used to be a `cli_pid.py --method <name> --plot` subprocess
    call per gallery panel."""
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
    ax2.set_xlim(0.0, auto_plot_window(sim.t, sim.y, sim.u))
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_cli(method, *extra_args, plot=None, check=True):
    cmd = [sys.executable, "cli_pid.py", "--plant", _PLANT, "--L", _L,
           "--method", method, "--json", *extra_args]
    if plot:
        cmd += ["--plot", plot]
    out = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True)
    if check and out.returncode != 0:
        raise RuntimeError(f"cli_pid.py failed: {cmd}\n{out.stderr}")
    return json.loads(out.stdout), out.returncode


def img_data_uri(path):
    with open(path, "rb") as f:
        b = f.read()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def noisy_filter_run(gains, N, noise_sigma, seed, t_end=400.0):
    """Reruns the closed loop by hand (plant.discretize + pid_step, same
    building blocks simulate_closed_loop() uses), but with Gaussian sensor
    noise added only to the measurement handed to the controller — the
    plant's own state update still sees the true, noise-free u_eff.
    """
    plant = TransferFunction.parse(_PLANT, L=float(_L))
    dt = plant.auto_dt()
    t = np.arange(0.0, t_end + dt, dt)
    sp = np.ones_like(t)

    Ad, Bd, C, D = plant.discretize(dt)
    nx = Ad.shape[0]
    Bd_flat = Bd.flatten() if nx else np.zeros(0)
    d_scalar = float(np.atleast_2d(D).flatten()[0])
    delay = int(round(plant.L / dt))

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_sigma, len(t))

    state = PIDState()
    x = np.zeros(nx)
    y = np.zeros(len(t))
    u = np.zeros(len(t))
    for i in range(1, len(t)):
        pv_noisy = y[i - 1] + noise[i - 1]
        u[i] = pid_step(gains, state, sp[i], pv_noisy, dt, -1e6, 1e6, N=N)
        j = i - 1 - delay
        u_eff = u[j] if j >= 0 else 0.0
        if nx:
            x = Ad @ x + Bd_flat * u_eff
            y[i] = float((C @ x).item()) + d_scalar * u_eff
        else:
            y[i] = d_scalar * u_eff
    e = sp - y
    return compute_metrics(t, sp, y, e, u, sp_kind="step")


def composite_overlay_plot(rows, out_path):
    """One 3-row figure — PV/SP, control effort u(t), error e(t) — with every
    included method's response overlaid on shared axes, same layout/labels/
    palette as the webapp's session overlay (pid_response_plotting.
    draw_response_tab).
    """
    plant = TransferFunction.parse(_PLANT, L=float(_L))
    fig, (ax_y, ax_u, ax_e) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax in (ax_y, ax_u, ax_e):
        ax.grid(True, alpha=0.3)
    ax_y.set_ylabel("PV / SP")
    ax_u.set_ylabel("control u(t)")
    ax_e.set_ylabel("error e(t)")
    ax_e.set_xlabel("time (s)")

    sp_plotted = False
    xmax = 0.0
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
        xmax = max(xmax, auto_plot_window(sim.t, sim.y, sim.u, sim.e))

    fig.suptitle("PEI8e Ex. 4.9/4.10 Surrogate — Step Response Overlay")
    ax_y.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=7.5, ncol=2)
    ax_y.set_xlim(0.0, xmax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_html():
    os.makedirs(_OUT_DIR, exist_ok=True)

    # --- Step 1: delay-aware tuning (AMIGO), unsaturated ---
    amigo, _ = run_cli("amigo")
    g = amigo["gains"]
    gains = PIDGains(Kp=g["Kp"], Ki=g["Ki"], Kd=g["Kd"])
    Ti = g["Kp"] / g["Ki"] if g["Ki"] else float("inf")
    Td = g["Kd"] / g["Kp"] if g["Kp"] else 0.0

    # --- Step 1.2: every PID tuning technique PIDTuner implements, same plant ---
    # In-process compare_all_methods() call, not one subprocess per method —
    # see the module docstring for why this section specifically needs to go
    # through the same call the app's "Compare all methods" view uses.
    plant_obj = TransferFunction.parse(_PLANT, L=float(_L))
    compare_rows = [serialize_row_json(r)
                     for r in compare_all_methods(plant_obj, include_variants=True)]

    pole_cancel_row = next(r for r in compare_rows if r["name"] == "Pole cancellation")
    pole_cancel_error = pole_cancel_row.get("error")

    all_methods_rows = []
    for r in compare_rows:
        if r["name"] == "Pole cancellation":
            continue  # shown as a raw-error block instead, same as before
        plot_path = os.path.join(_OUT_DIR, f"pid_example_v2_{_slug(r['name'])}.png")
        panel_plot(r["name"], PIDGains(**{k: r["gains"][k] for k in ("Kp", "Ki", "Kd")}),
                   plant_obj, plot_path)
        r["_plot_uri"] = img_data_uri(plot_path)
        all_methods_rows.append(r)

    composite_rows = all_methods_rows
    composite_plot_path = os.path.join(_OUT_DIR, "pid_example_v2_composite_overlay.png")
    composite_overlay_plot(composite_rows, composite_plot_path)
    composite_uri = img_data_uri(composite_plot_path)

    # --- Step 2: derivative filter, unsaturated (clean step — no noise) ---
    amigo_n80, _ = run_cli("amigo", "--N", "80")
    amigo_n0, _ = run_cli("amigo", "--N", "0")

    # --- Step 2.2: derivative filter under measurement noise (custom sim) ---
    NOISE_SIGMA, SEED = 0.01, 42
    plant_dt = TransferFunction.parse(_PLANT, L=float(_L)).auto_dt()
    tau_d_80 = gains.Kd / (80.0 * gains.Kp)
    tau_d_1 = gains.Kd / (1.0 * gains.Kp)
    noisy_n80 = noisy_filter_run(gains, N=80.0, noise_sigma=NOISE_SIGMA, seed=SEED)
    noisy_n1 = noisy_filter_run(gains, N=1.0, noise_sigma=NOISE_SIGMA, seed=SEED)
    noisy_n0 = noisy_filter_run(gains, N=0.0, noise_sigma=NOISE_SIGMA, seed=SEED)

    # --- Step 3: actuator saturation + anti-windup ---
    cond_plot = os.path.join(_OUT_DIR, "pid_example_v2_conditional.png")
    backcalc_plot = os.path.join(_OUT_DIR, "pid_example_v2_backcalc_auto.png")

    sat_conditional, _ = run_cli("amigo", "--u-min", str(_U_MIN), "--u-max", str(_U_MAX),
                                  "--antiwindup", "conditional", plot=cond_plot)
    sat_backcalc_auto, _ = run_cli("amigo", "--u-min", str(_U_MIN), "--u-max", str(_U_MAX),
                                    "--antiwindup", "back_calc", plot=backcalc_plot)
    ka_values = (0.01, 0.3, 0.6, 1.0, 2.0)
    ka_sweep = {}
    for ka in ka_values:
        ka_sweep[ka], _ = run_cli("amigo", "--u-min", str(_U_MIN), "--u-max", str(_U_MAX),
                                   "--antiwindup", "back_calc", "--Ka", str(ka))

    cond_uri = img_data_uri(cond_plot)
    backcalc_uri = img_data_uri(backcalc_plot)

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
    <td>{fnum(r['u_peak'],2)}</td>
    <td>{fnum(r['u_tv'],1)}</td>
  </tr>"""

    def gallery_card(r):
        rg = r["gains"]
        r_ti = rg["Kp"] / rg["Ki"] if rg["Ki"] else float("inf")
        r_td = rg["Kd"] / rg["Kp"] if rg["Kp"] else 0.0
        return f"""
  <figure class="gallery-card">
    <img src="{r['_plot_uri']}" alt="Step response: {_e(r['name'])}">
    <figcaption>
      <strong>{_e(r['name'])}</strong>
      <div class="gallery-gains">K<sub>p</sub>={fnum(rg['Kp'])} &middot;
        T<sub>i</sub>={fnum(r_ti)}&nbsp;s &middot; T<sub>d</sub>={fnum(r_td)}&nbsp;s</div>
      <div class="gallery-metrics">OS%={fnum(r['OS%'],1)}% &middot;
        GM={fnum(r['GM_dB'],1)}&nbsp;dB &middot; PM={fnum(r['PM_deg'],1)}&deg;</div>
    </figcaption>
  </figure>"""

    def sat_row(name, ka_disp, tt_disp, m):
        return f"""
  <tr>
    <td>{name}</td>
    <td>{ka_disp}</td>
    <td>{tt_disp}</td>
    <td>{fnum(m['Overshoot'],2)}%</td>
    <td>{fnum(m['Settling'],1)} s</td>
    <td>{fnum(m['IAE'],2)}</td>
    <td>{fnum(m['ITAE'],1)}</td>
    <td>{fnum(m['u_peak'],3)}</td>
    <td>{fnum(m['ISU'],2)}</td>
  </tr>"""

    body = f"""
<h1>SISO PID Worked Example v2: Textbook Example (PEI8e Ex. 4.9/4.10 Heat Exchanger)</h1>

<dl class="memo-header">
  <dt>Date</dt><dd>{_DATE}</dd>
  <dt>Re</dt><dd>The same delay-aware-tuning &middot; derivative-filter (K<sub>d</sub>/N) &middot;
  back-calculation-anti-windup (K<sub>a</sub>) worked example as the v1 memo, rerun against a real
  textbook plant instead of a made-up one &mdash; PEI8e Examples 4.9/4.10's FOPDT surrogate for the
  Example 2.18 heat exchanger.</dd>
</dl>

<h2>1. Setup</h2>
<p>
<strong>Which textbook example, and why this plant.</strong> Franklin, Powell &amp; Emami-Naeini,
<em>Feedback Control of Dynamic Systems</em>, 8th edition (&ldquo;PEI8e&rdquo; in this repo's memos)
tunes P/PI/PID controllers for the heat exchanger of Example 2.18 in two places: Example 4.9
(Ziegler&ndash;Nichols reaction-curve method, quarter-decay ratio) and Example 4.10
(Ziegler&ndash;Nichols ultimate-gain/sensitivity method). Both approximate the true two-lag
heat-exchanger transfer function with a first-order-plus-dead-time (FOPDT) surrogate &mdash;
reaction rate R&asymp;1/90, apparent dead time L&asymp;13&nbsp;s &mdash; i.e.
<code>--plant '1/(90s+1)' --L 13</code>. That's the same surrogate
<code>2026-08-18-zn-textbook-validation-memo.html</code> already checks PIDTuner's <code>zn1</code>/
<code>zn2</code> gains against the book's closed-form numbers for; this memo is a different exercise
on the same plant &mdash; not another validation pass, but the v1 worked example's delay/filter/
anti-windup narrative rerun end-to-end on a real textbook plant instead of a synthetic one.
</p>
<pre>G(s) = {_e(_PLANT)},  L = {_L} s</pre>
<p>
DC gain is 1, so a unit setpoint step needs a steady-state control signal u<sub>ss</sub>=1/1=1. The
actuator is limited to <strong>u &isin; [{_U_MIN}, {_U_MAX}]</strong> &mdash; only 30% of headroom
above u<sub>ss</sub>, deliberately tight for the same reason as v1: so saturation and anti-windup
actually matter in &sect;4 rather than being a hypothetical.
</p>

<h3>1.1 Reproducing these results</h3>
<p>
Every number and plot in this memo comes from running PIDTuner's own code, not from re-deriving
anything by hand. The generator script below writes the <code>.html</code> plus every embedded PNG;
the <code>google-chrome</code> call after it turns that into this PDF. To regenerate both exactly:
</p>
<pre>cd src/
python3 gen_pid_worked_example_v2_textbook_memo.py

cd ../docs/memos/{_DATE}/
F={_DATE}-pid-worked-example-v2-textbook-memo
google-chrome --headless --disable-gpu --no-sandbox \\
    --no-pdf-header-footer \\
    --print-to-pdf=$F.pdf $F.html</pre>
<p class="section-note">
Tested with Python {sys.version.split()[0]}, numpy {np.__version__}, matplotlib {matplotlib.__version__}
(repo has no pinned <code>requirements.txt</code>, so exact versions aren't guaranteed to matter, just
what this run used) and Google Chrome for the headless PDF export. Every <code>cli_pid.py</code>
command shown inline through the rest of this memo (&sect;2, &sect;3.1, &sect;4) can also be run
standalone from <code>src/</code> to reproduce that one section's numbers without regenerating the
whole memo.
</p>

<h2>2. Step 1 &mdash; delay-aware tuning (AMIGO)</h2>
<p>
AMIGO (&Aring;str&ouml;m &amp; H&auml;gglund's M-constrained rules) is used here for the same reason
as v1: it tunes directly against the identified FOPDT dead time rather than treating L as an
afterthought, which matters at this plant's delay-to-lag ratio (L/&tau;&asymp;13/90&asymp;0.14 &mdash;
the same low-ratio region the 2026-08-18 ZN memo flags as where the classical ZN rules over-drive
gains).
</p>
<pre>python3 cli_pid.py --plant '{_e(_PLANT)}' --L {_L} --method amigo --json</pre>
<table class="no-break">
  <tr><th>K<sub>p</sub></th><th>T<sub>i</sub> (s)</th><th>T<sub>d</sub> (s)</th>
      <th>K<sub>i</sub></th><th>K<sub>d</sub></th></tr>
  <tr><td>{fnum(g['Kp'])}</td><td>{fnum(Ti)}</td>
      <td>{fnum(Td)}</td>
      <td>{fnum(g['Ki'])}</td><td>{fnum(g['Kd'])}</td></tr>
</table>
<h3>2.1 Unsaturated step response</h3>
<table class="no-break">
  <tr><th>Run</th><th>OS%</th><th>t<sub>s</sub> (2%)</th><th>Rise (10&ndash;90%)</th>
      <th>IAE</th><th>ISU</th><th>M<sub>s</sub></th><th>M<sub>t</sub></th>
      <th>GM</th><th>PM</th><th>|u|<sub>peak</sub></th><th>u<sub>tv</sub></th></tr>
  {metrics_row("AMIGO", amigo)}
</table>
<p class="section-note">
GM&asymp;{fnum(amigo['GM_dB'],1)}&nbsp;dB, PM&asymp;{fnum(amigo['PM_deg'],1)}&deg; &mdash; robust but
tighter than v1's synthetic plant, and u<sub>peak</sub>&asymp;{fnum(amigo['u_peak'],2)} is well above
the {_U_MAX} actuator limit set in &sect;1, which is exactly what makes &sect;4 worth doing on this
loop rather than a hypothetical one.
</p>

<h3>2.2 Every tuning technique PIDTuner implements, same plant</h3>
<p>
AMIGO is the technique carried through the rest of this memo (&sect;3&ndash;4), but it's one of nine
technique families the CLI can produce for this loop &mdash; {len(all_methods_rows)} rows below, since
CHR's four response/overshoot variants are each shown individually. This textbook plant surfaces one
thing the v1 synthetic-plant memo didn't: <strong>Stable pole cancellation can't run on this plant at
all</strong>: it needs at least two stable plant poles to cancel, and this FOPDT surrogate has only
one. Running it directly returns the CLI's own error rather than a plot, shown as-is rather than
silently dropped, since a method's failure mode on a specific plant is as informative as its success:
</p>
<pre>$ python3 cli_pid.py --plant '{_e(_PLANT)}' --L {_L} \\
    --method pole_cancellation --json
{{"error": "{_e(chr(10).join(textwrap.wrap(pole_cancel_error, 68)))}"}}</pre>
<p>
Among the rows that do run, <strong>Cohen&ndash;Coon is this plant's most aggressive tuning</strong>
&mdash; its gains (K<sub>p</sub>&asymp;{fnum(next(r['gains']['Kp'] for r in all_methods_rows if r['name']=='Cohen–Coon'))},
nearly 3&times; AMIGO's) give it the highest overshoot of any method here
(OS%&asymp;{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='Cohen–Coon'),1)}%,
&sect;2.3), but the response settles normally, at
t<sub>s</sub>&asymp;{fnum(next(r['ts'] for r in all_methods_rows if r['name']=='Cohen–Coon'),1)}&nbsp;s
&mdash; in the same range as every other row rather than an outlier by scale. Margins:
GM&asymp;{fnum(next(r['GM_dB'] for r in all_methods_rows if r['name']=='Cohen–Coon'),1)}&nbsp;dB,
PM&asymp;{fnum(next(r['PM_deg'] for r in all_methods_rows if r['name']=='Cohen–Coon'),1)}&deg;,
M<sub>s</sub>&asymp;{fnum(next(r['Ms'] for r in all_methods_rows if r['name']=='Cohen–Coon'),2)},
M<sub>t</sub>&asymp;{fnum(next(r['Mt'] for r in all_methods_rows if r['name']=='Cohen–Coon'),2)}.
</p>
<div class="gallery">
{"".join(gallery_card(r) for r in all_methods_rows)}
</div>
<div class="callout">
<strong>What's new since the 2026-08-25 run of this same memo: ZN-I/ZN-II are now halved by
default.</strong> <code>pid_compare.py</code>'s <code>compare_all_methods()</code> &mdash; what this
section and &sect;2.3 call, and what the CLI's <code>--method all</code>/the GUI's "Compare all
methods" button both use &mdash; now runs ZN-I and ZN-II through the app's existing
<code>halve_gains()</code> helper before showing them, since those are the only two techniques whose
own tuning notes recommend it (a single <code>cli_pid.py --method zn1</code> run is unaffected; that
path still needs the explicit <code>--halve</code> flag). The 2026-08-25 run of this memo showed
"Ziegler&ndash;Nichols I"/"II" at OS%&asymp;138.6%/116.9% (full-published-formula strength); this run
shows <strong>"ZN-I &frac12;"/"ZN-II &frac12;"</strong> at
OS%&asymp;{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='ZN-I ½'),1)}%/
{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='ZN-II ½'),1)}% instead &mdash; roughly
halving the gains doesn't halve the overshoot outright, but it does move both from "runs hot" into a
middle-of-the-pack range on this plant, as the panels below show directly.
</div>
<p class="section-note">
With ZN tamed, the widest spread here now belongs to <strong>CHR</strong>: its four
response/overshoot targets range from a well-damped
OS%&asymp;{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='CHR set 0%'),1)}% ("set 0%") up
to OS%&asymp;{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='CHR load 20%'),1)}%
("load 20%") &mdash; higher than either halved ZN variant &mdash; since CHR's load-disturbance targets
trade tracking overshoot for faster disturbance rejection, the same design tension ZN-I/II's own
"halve for tracking" note exists to manage. AMIGO/SIMC/Boyd/Tyreus&ndash;Luyben remain the more
uniformly delay-aware, margin-constrained group.
</p>

<h3>2.3 Composite overlay, all included techniques on one set of axes</h3>
<p>
Same responses, drawn the way the desktop webapp draws them when several tuned controllers are kept
in a session &mdash; one 3-row PV/SP &middot; control effort &middot; error figure, all methods
overlaid with the app's own <code>PALETTE</code> (<code>pid_app.py:66-68</code>), not matplotlib's
default cycle. Built with <code>simulate_closed_loop()</code> directly on each method's gains from
&sect;2.2, the same function <code>pid_app.py</code>'s session view and <code>cli_pid.py --plot</code>
both call. <strong>Stable pole cancellation is excluded</strong> &mdash; it never produced gains to
simulate (&sect;2.2) &mdash; every other technique is on this one shared set of axes.
</p>
<p><img src="{composite_uri}" alt="Composite overlay of the applicable tuning techniques' step responses" style="max-width:100%;border:1px solid #b9c3cc;border-radius:5px;"></p>
<p class="section-note">
Cohen&ndash;Coon is the visible extreme here (&sect;2.2), not ZN-I &frac12;/ZN-II &frac12; &mdash;
the flip from the 2026-08-25 run, where the two full-strength ZN curves were the standouts (see
&sect;2.2's callout).
</p>

<h2>3. Step 2 &mdash; derivative filtering (K<sub>d</sub>/N)</h2>
<p>
<code>pid_step()</code> applies the derivative term to <em>&minus;pv</em>, not to the error
(<code>pid_simulate.py:129-130</code>) &mdash; standard process-control practice specifically to avoid
a "derivative kick" when the setpoint itself steps. The filter's time constant is
&tau;<sub>d</sub>=K<sub>d</sub>/(N&middot;K<sub>p</sub>); <code>--N 0</code> disables it and falls back
to an unfiltered derivative of error instead.
</p>
<h3>3.1 A clean setpoint step doesn't exercise the filter at all</h3>
<pre>python3 cli_pid.py --plant '{_e(_PLANT)}' --L {_L} --method amigo --N 80 --json
python3 cli_pid.py --plant '{_e(_PLANT)}' --L {_L} --method amigo --N 0  --json</pre>
<table class="no-break">
  <tr><th>Run</th><th>OS%</th><th>t<sub>s</sub> (2%)</th><th>IAE</th><th>ISU</th>
      <th>|u|<sub>peak</sub></th><th>u<sub>tv</sub></th></tr>
  <tr><td>N=80 (filtered)</td><td>{fnum(amigo_n80['OS%'],2)}%</td><td>{fnum(amigo_n80['ts'],1)} s</td>
      <td>{fnum(amigo_n80['IAE'],4)}</td><td>{fnum(amigo_n80['ISU'],4)}</td>
      <td>{fnum(amigo_n80['u_peak'],5)}</td><td>{fnum(amigo_n80['u_tv'],4)}</td></tr>
  <tr><td>N=0 (ideal derivative)</td><td>{fnum(amigo_n0['OS%'],2)}%</td><td>{fnum(amigo_n0['ts'],1)} s</td>
      <td>{fnum(amigo_n0['IAE'],4)}</td><td>{fnum(amigo_n0['ISU'],4)}</td>
      <td>{fnum(amigo_n0['u_peak'],5)}</td><td>{fnum(amigo_n0['u_tv'],4)}</td></tr>
</table>
<p class="section-note">
Identical again, same reason as v1: the derivative acts on the plant output, which is continuous for a
clean setpoint step, so N has nothing to bite on.
</p>

<h3>3.2 What N is actually for, and why the default barely shows it on this plant</h3>
<p>
Same noise-injection approach as v1 &mdash; reruns the closed loop by hand on
<code>plant.discretize()</code>/<code>PIDState</code>/<code>pid_step()</code>, the same building
blocks <code>simulate_closed_loop()</code> uses, adding Gaussian sensor noise (&sigma;={NOISE_SIGMA},
seed={SEED}) onto only the measurement handed to the controller. But this plant's
<code>auto_dt()</code> samples much coarser (dt&asymp;{fnum(plant_dt,2)}&nbsp;s) than v1's, and
AMIGO's K<sub>d</sub> here is large relative to K<sub>p</sub>, so the default N=80 filter's time
constant &tau;<sub>d</sub>=K<sub>d</sub>/(80&middot;K<sub>p</sub>)&asymp;{fnum(tau_d_80,4)}&nbsp;s
is <em>far below one sample interval</em> &mdash; the discrete filter barely does anything at N=80
on this plant. Dropping to N=1 (&tau;<sub>d</sub>&asymp;{fnum(tau_d_1,2)}&nbsp;s, now above dt) shows
the effect v1 saw at its default N:
</p>
<table class="no-break">
  <tr><th>Run</th><th>&tau;<sub>d</sub> (s)</th><th>IAE</th><th>ISU</th><th>|u|<sub>peak</sub></th><th>u<sub>tv</sub></th></tr>
  <tr><td>N=80 (filtered, default)</td><td>{fnum(tau_d_80,4)}</td><td>{fnum(noisy_n80['IAE'],3)}</td><td>{fnum(noisy_n80['ISU'],3)}</td>
      <td>{fnum(noisy_n80['u_peak'],3)}</td><td>{fnum(noisy_n80['u_tv'],2)}</td></tr>
  <tr><td>N=1 (heavier filter)</td><td>{fnum(tau_d_1,2)}</td><td>{fnum(noisy_n1['IAE'],3)}</td><td>{fnum(noisy_n1['ISU'],3)}</td>
      <td>{fnum(noisy_n1['u_peak'],3)}</td><td>{fnum(noisy_n1['u_tv'],2)}</td></tr>
  <tr><td>N=0 (ideal derivative)</td><td>0</td><td>{fnum(noisy_n0['IAE'],3)}</td><td>{fnum(noisy_n0['ISU'],3)}</td>
      <td>{fnum(noisy_n0['u_peak'],3)}</td><td>{fnum(noisy_n0['u_tv'],2)}</td></tr>
</table>
<div class="callout">
<strong>Verdict &mdash; the default N doesn't automatically transfer across plants.</strong> At the
default N=80, control-signal chatter (u<sub>tv</sub>) only drops from {fnum(noisy_n0['u_tv'],1)}
(N=0) to {fnum(noisy_n80['u_tv'],1)}, roughly
{fnum(100*(1-noisy_n80['u_tv']/noisy_n0['u_tv']),0)}% &mdash; a much smaller win than v1's &asymp;23%
on the same &sigma;=0.01 noise, because &tau;<sub>d</sub> at N=80 is small compared to this plant's
coarser sample time. Dropping to N=1 recovers a comparable win
({fnum(100*(1-noisy_n1['u_tv']/noisy_n0['u_tv']),0)}% reduction, u<sub>tv</sub>
{fnum(noisy_n0['u_tv'],1)}&rarr;{fnum(noisy_n1['u_tv'],1)}). The lesson isn't that N=80 is wrong
&mdash; IAE is unaffected either way &mdash; it's that "the filter helps under noise" doesn't come
with a plant-independent N; how much it helps depends on &tau;<sub>d</sub> relative to the loop's own
sample time, which the default doesn't know about.
</div>

<h2>4. Step 3 &mdash; actuator saturation &amp; back-calculation anti-windup (K<sub>a</sub>)</h2>
<p>
&sect;2.1 already showed AMIGO's unsaturated u<sub>peak</sub>&asymp;{fnum(amigo['u_peak'],2)} exceeds
the &plusmn;{_U_MAX} actuator limit from &sect;1, so this loop saturates for real. Same check as v1:
<code>compute_back_calc_Ka()</code> auto-derives K<sub>a</sub>=1/T<sub>t</sub>,
T<sub>t</sub>=&radic;(T<sub>i</sub>T<sub>d</sub>) &mdash; &Aring;str&ouml;m &amp; H&auml;gglund's rule
of thumb &mdash; checked here against plain conditional integration and a K<sub>a</sub> sweep.
</p>
<pre>python3 cli_pid.py --plant '{_e(_PLANT)}' --L {_L} --method amigo \\
    --u-min {_U_MIN} --u-max {_U_MAX} --antiwindup conditional --json
python3 cli_pid.py --plant '{_e(_PLANT)}' --L {_L} --method amigo \\
    --u-min {_U_MIN} --u-max {_U_MAX} --antiwindup back_calc --json</pre>

<h3>4.1 Conditional integration vs. back-calculation, auto and swept K<sub>a</sub></h3>
<table class="no-break">
  <tr><th>Strategy</th><th>K<sub>a</sub></th><th>T<sub>t</sub> (s)</th><th>OS%</th>
      <th>Settling (2%)</th><th>IAE</th><th>ITAE</th><th>|u|<sub>peak</sub></th><th>ISU</th></tr>
  {sat_row("Conditional (freeze integral)", "&mdash;", "&mdash;", sat_conditional['saturated_sim']['metrics'])}
  {sat_row("Back-calc, auto K<sub>a</sub>", fnum(sat_backcalc_auto['saturated_sim']['Ka'],4),
           fnum(sat_backcalc_auto['saturated_sim']['Tt'],3), sat_backcalc_auto['saturated_sim']['metrics'])}
  {"".join(sat_row(f"Back-calc, K<sub>a</sub>={ka}", fnum(ka_sweep[ka]['saturated_sim']['Ka'],4),
                    fnum(ka_sweep[ka]['saturated_sim']['Tt'],3), ka_sweep[ka]['saturated_sim']['metrics'])
           for ka in ka_values)}
</table>
<div class="callout">
<strong>Verdict &mdash; same finding as v1, on a real textbook plant.</strong>
The auto-derived K<sub>a</sub>&asymp;{fnum(sat_backcalc_auto['saturated_sim']['Ka'],4)}
(T<sub>t</sub>&asymp;{fnum(sat_backcalc_auto['saturated_sim']['Tt'],1)}&nbsp;s) does <em>worse</em>
than plain conditional integration here too &mdash;
OS%&asymp;{fnum(sat_backcalc_auto['saturated_sim']['metrics']['Overshoot'],1)}% vs.
{fnum(sat_conditional['saturated_sim']['metrics']['Overshoot'],1)}%, ITAE
{fnum(sat_backcalc_auto['saturated_sim']['metrics']['ITAE'],0)} vs.
{fnum(sat_conditional['saturated_sim']['metrics']['ITAE'],0)}. Pushing K<sub>a</sub> up by hand
({'&rarr;'.join(str(k) for k in ka_values)}) again steadily closes the gap toward conditional
integration's numbers, confirming this isn't a one-plant fluke: the auto rule's
T<sub>t</sub>&asymp;{fnum(sat_backcalc_auto['saturated_sim']['Tt'],1)}&nbsp;s assumes a slower unwind
than this loop's actuator actually needs, on both the synthetic v1 plant and this real textbook one.
For this loop too, <strong>conditional integration is the better default</strong>, and back-calculation
needs a hand-tuned K<sub>a</sub> well above the auto value to compete.
</div>

<h3>4.2 Step response: conditional (winner) vs. back-calc auto K<sub>a</sub></h3>
<p><img src="{cond_uri}" alt="AMIGO, saturated actuator, conditional anti-windup" style="max-width:100%;border:1px solid #b9c3cc;border-radius:5px;"></p>
<p class="section-note">Conditional integration: minimal overshoot, integral frozen while u sits at the rail.</p>
<p><img src="{backcalc_uri}" alt="AMIGO, saturated actuator, back-calculation anti-windup, auto Ka" style="max-width:100%;border:1px solid #b9c3cc;border-radius:5px;"></p>
<p class="section-note">Back-calculation, auto K<sub>a</sub>: visibly slower to unwind, larger overshoot before it settles.</p>

<h2>5. Summary</h2>
<table class="no-break">
  <tr><th>Feature</th><th>What was checked</th><th>Result</th></tr>
  <tr><td>Delay-aware tuning (<code>--L</code>, AMIGO)</td>
      <td>Gains vs. robust margin, unsaturated response</td>
      <td class="pass">GM&asymp;{fnum(amigo['GM_dB'],1)}&nbsp;dB, PM&asymp;{fnum(amigo['PM_deg'],1)}&deg; &mdash; robust</td></tr>
  <tr><td>All 9 tuning technique families (&sect;2.2&ndash;2.3)</td>
      <td>Gallery + composite overlay on the textbook plant, via <code>compare_all_methods()</code></td>
      <td class="fail">Pole cancellation can't run (plant has only 1 stable pole); Cohen&ndash;Coon is this plant's highest-overshoot method but settles normally (&sect;2.2); ZN-I/II now shown pre-halved by default (see &sect;2.2)</td></tr>
  <tr><td>Derivative filter (<code>--N</code>)</td>
      <td>Clean step (&sect;3.1) vs. noisy measurement at N=80 and N=1 (&sect;3.2)</td>
      <td class="fail">Default N=80 gives a much smaller noise-rejection win here than on v1's plant &mdash; &tau;<sub>d</sub> is plant-dependent</td></tr>
  <tr><td>Anti-windup (<code>--antiwindup</code>, <code>--Ka</code>)</td>
      <td>Conditional vs. back-calc (auto and swept K<sub>a</sub>) under saturation</td>
      <td class="fail">Auto K<sub>a</sub> underperforms conditional again &mdash; same finding as v1, now confirmed on a second, real plant</td></tr>
</table>
<p>
Same rigor as v1: &sect;2.1 and &sect;4 are direct <code>cli_pid.py --json</code> output, &sect;2.2/2.3
call <code>compare_all_methods()</code> in-process (the same function the CLI's <code>--method all</code>
and the GUI's "Compare all methods" call), &sect;3.2's noise-injected comparison drives the exact
<code>pid_step()</code>/<code>compute_metrics()</code> functions <code>simulate_closed_loop()</code>
itself calls, and &sect;1.1 gives the exact commands to regenerate all of it. Two findings carry
forward from v1 with a second data point behind them now: the back-calculation K<sub>a</sub>
auto-derivation needs checking against conditional integration rather than trusted blindly, and the
derivative filter's default N doesn't transfer across plants without checking &tau;<sub>d</sub> against
the loop's own sample time. One finding is specific to this plant: pole cancellation's
stable-pole-count requirement isn't always met. One finding is new since the 2026-08-25 run of this memo: with
ZN-I/ZN-II now halved by default in the comparison (&sect;2.2), they're no longer this plant's
overshoot outliers &mdash; CHR's "load" variants are.
</p>

<p class="appendix-title">Appendix &mdash; raw CLI output</p>
<pre>{_e(json.dumps(amigo, indent=2))}</pre>
<pre>{_e(json.dumps(sat_conditional, indent=2))}</pre>
<pre>{_e(json.dumps(sat_backcalc_auto, indent=2))}</pre>
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
<title>PID Worked Example v2 (Textbook) Memo</title>
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
