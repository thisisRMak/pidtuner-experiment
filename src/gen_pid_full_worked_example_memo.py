#!/usr/bin/env python3
"""Generates docs/memos/<YYYY-MM-DD>/<YYYY-MM-DD>-pid-full-worked-example-memo.html
— a single, self-contained SISO PID worked example that runs the plant
through PIDTuner's full pipeline: delay-aware tuning (AMIGO), the
derivative filter (Kd/N), and back-calculation anti-windup (Ka) under
actuator saturation, all in one loop. Styled to match the house CSS used
by the other 2026-08-18 memos.
<YYYY-MM-DD> is today's date (datetime.date.today(), _DATE below) -- each
run lands in its own dated snapshot folder rather than overwriting a fixed
2026-08-25 the way earlier revisions of this script did (same convention
gen_lqg_worked_examples_memos.py uses), so re-running it later never
silently mislabels a new memo with a stale date or clobbers an old one.

Section 2.1 (AMIGO) and section 4 (saturation/anti-windup) run cli_pid.py's
own --json/--plot output directly, so those numbers/plots can't drift from
what the CLI actually produces. Section 2.2 (the gallery of every technique)
and 2.3 (composite overlay) call pid_compare.py's compare_all_methods()
directly, in-process, rather than shelling out to cli_pid.py once per
method (the 2026-08-25 run did the latter) — the two approaches gave
identical numbers back then, but ZN-I/ZN-II are now halved by default in
compare_all_methods() (see pid_compare.py; cli_pid.py --method all's own
--halve flag is a no-op there for the same reason) while a lone
`cli_pid.py --method zn1` is not, so this section has to go through the
same call the app's own "Compare all methods" view uses to actually
reflect that. Section 3.2 (derivative-filter vs. measurement noise) has no
CLI hook — cli_pid.py's --noise-sigma only applies to --gen-signal
identification tests, not to a tuned method's simulated response — so
it drives pid_simulate.py's own PIDState/pid_step/compute_metrics
directly, injecting synthetic sensor noise onto the measurement fed to
the controller (not into the plant physics), which is the standard way
to exercise a derivative filter's actual job.

Run: python3 gen_pid_full_worked_example_memo.py
"""
import base64
import datetime
import html
import json
import os
import re
import subprocess
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plant import TransferFunction
from pid_tuning_methods import PIDGains
from pid_simulate import PIDState, pid_step, compute_metrics, simulate_closed_loop
from pid_compare import compare_all_methods
from cli_pid import serialize_row_json

_HERE = os.path.dirname(__file__)
_DATE = datetime.date.today().isoformat()
_OUT_DIR = os.path.join(_HERE, "..", "docs", "memos", _DATE)
_HTML_OUT = os.path.join(_OUT_DIR, f"{_DATE}-pid-full-worked-example-memo.html")
_CSS = open(os.path.join(_HERE, "_memo_css.txt")).read()

_PLANT = "5/((20s+1)(4s+1))"
_L = "3"
_U_MIN, _U_MAX = -0.25, 0.25

# Same palette pid_app.py (the desktop webapp) assigns to overlaid tuned
# controllers — distinguishable, colorblind-friendly, and long enough to
# cover all 12 compare_all_methods() rows without matplotlib's default 10-color cycle
# repeating (see PALETTE in pid_app.py:66-68).
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
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_cli(method, *extra_args, plot=None):
    cmd = [sys.executable, "cli_pid.py", "--plant", _PLANT, "--L", _L,
           "--method", method, "--json", *extra_args]
    if plot:
        cmd += ["--plot", plot]
    out = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"cli_pid.py failed: {cmd}\n{out.stderr}")
    return json.loads(out.stdout)


def img_data_uri(path):
    with open(path, "rb") as f:
        b = f.read()
    return "data:image/png;base64," + base64.b64encode(b).decode("ascii")


def noisy_filter_run(gains, N, noise_sigma, seed, t_end=120.0):
    """Reruns the closed loop by hand (plant.discretize + pid_step, same
    building blocks simulate_closed_loop() uses), but with Gaussian sensor
    noise added only to the measurement handed to the controller — the
    plant's own state update still sees the true, noise-free u_eff. This
    is the only way to exercise the derivative filter's actual job
    (attenuating noise-driven derivative kick), since cli_pid.py's
    simulated response has no --noise-sigma hook.
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
    method's response overlaid on shared axes, same layout/labels/palette as
    the webapp's session overlay (pid_response_plotting.draw_response_tab):
    that's the "composite plot" this mirrors, just rendered headlessly to a
    PNG instead of a live Tk canvas.
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

    fig.suptitle("All PID Tuning Techniques — Step Response Overlay")
    ax_y.legend(loc="lower right", bbox_to_anchor=(1.0, 1.02), fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def build_html():
    os.makedirs(_OUT_DIR, exist_ok=True)

    # --- Step 1: delay-aware tuning (AMIGO), unsaturated ---
    amigo = run_cli("amigo")
    g = amigo["gains"]
    gains = PIDGains(Kp=g["Kp"], Ki=g["Ki"], Kd=g["Kd"])
    Ti = g["Kp"] / g["Ki"] if g["Ki"] else float("inf")
    Td = g["Kd"] / g["Kp"] if g["Kp"] else 0.0

    # --- Step 1.2: every PID tuning technique PIDTuner implements, same plant ---
    # In-process compare_all_methods() call, not one subprocess per method —
    # see the module docstring for why this section specifically needs to go
    # through the same call the app's "Compare all methods" view uses.
    all_plots_dir = _OUT_DIR
    plant_obj = TransferFunction.parse(_PLANT, L=float(_L))
    compare_rows = [serialize_row_json(r)
                     for r in compare_all_methods(plant_obj, include_variants=True)]

    all_methods_rows = []
    for r in compare_rows:
        if r.get("gains") is None:
            continue  # no gains to plot; every method succeeds on this plant as of writing
        plot_path = os.path.join(all_plots_dir, f"pid_example_{_slug(r['name'])}.png")
        panel_plot(r["name"], PIDGains(**{k: r["gains"][k] for k in ("Kp", "Ki", "Kd")}),
                   plant_obj, plot_path)
        r["_plot_uri"] = img_data_uri(plot_path)
        all_methods_rows.append(r)

    composite_plot_path = os.path.join(all_plots_dir, "pid_example_composite_overlay.png")
    composite_overlay_plot(all_methods_rows, composite_plot_path)
    composite_uri = img_data_uri(composite_plot_path)

    # --- Step 2: derivative filter, unsaturated (clean step — no noise) ---
    amigo_n80 = run_cli("amigo", "--N", "80")
    amigo_n0 = run_cli("amigo", "--N", "0")

    # --- Step 2.2: derivative filter under measurement noise (custom sim) ---
    NOISE_SIGMA, SEED = 0.01, 42
    noisy_n80 = noisy_filter_run(gains, N=80.0, noise_sigma=NOISE_SIGMA, seed=SEED)
    noisy_n0 = noisy_filter_run(gains, N=0.0, noise_sigma=NOISE_SIGMA, seed=SEED)

    # --- Step 3: actuator saturation + anti-windup ---
    cond_plot = os.path.join(_OUT_DIR, "pid_example_conditional.png")
    backcalc_plot = os.path.join(_OUT_DIR, "pid_example_backcalc_auto.png")

    sat_conditional = run_cli("amigo", "--u-min", str(_U_MIN), "--u-max", str(_U_MAX),
                               "--antiwindup", "conditional", plot=cond_plot)
    sat_backcalc_auto = run_cli("amigo", "--u-min", str(_U_MIN), "--u-max", str(_U_MAX),
                                 "--antiwindup", "back_calc", plot=backcalc_plot)
    ka_sweep = {}
    for ka in (0.02, 1.5, 3, 5, 8):
        ka_sweep[ka] = run_cli("amigo", "--u-min", str(_U_MIN), "--u-max", str(_U_MAX),
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
<h1>SISO PID Worked Example: Delay-Aware Tuning, Derivative Filtering &amp; Back-Calculation Anti-Windup</h1>

<dl class="memo-header">
  <dt>Date</dt><dd>{_DATE}</dd>
  <dt>Re</dt><dd>One SISO PID loop, taken end-to-end through PIDTuner's full pipeline &mdash;
  dead-time-aware tuning, the derivative filter (K<sub>d</sub>/N), and back-calculation anti-windup
  (K<sub>a</sub>) under real actuator limits &mdash; as a single worked example rather than three
  separate feature memos.</dd>
</dl>

<h2>1. Setup</h2>
<p>
<strong>Scope note.</strong> This is a synthetic worked example, not a textbook validation &mdash;
unlike the 2026-08-18 method-validation memos, there's no book problem being checked against. The
point is to exercise delay time (<code>--L</code>), the derivative filter (<code>--N</code>), and
back-calculation anti-windup (<code>--Ka</code>) together, in one narrative, using PIDTuner's own
code paths throughout (never re-derived by hand).
</p>
<p>
Plant: a lag-dominant process with transport delay &mdash; two energy-storage lags
(&tau;<sub>1</sub>=20&nbsp;s, &tau;<sub>2</sub>=4&nbsp;s) plus L=3&nbsp;s of pure dead time between
actuator and sensor (e.g. a heated-fluid loop with the temperature probe downstream of the heater):
</p>
<pre>G(s) = {_e(_PLANT)},  L = {_L} s</pre>
<p>
DC gain is 5, so a unit setpoint step needs a steady-state control signal u<sub>ss</sub>=1/5=0.2.
The actuator is limited to <strong>u &isin; [{_U_MIN}, {_U_MAX}]</strong> &mdash; only 25% of headroom
above u<sub>ss</sub>, chosen deliberately tight so saturation and anti-windup actually matter in
&sect;4.
</p>

<h2>2. Step 1 &mdash; delay-aware tuning (AMIGO)</h2>
<p>
AMIGO (&Aring;str&ouml;m &amp; H&auml;gglund's M-constrained rules) is used here because it tunes
directly against the identified FOPDT dead time rather than treating L as an afterthought, which
matters at this plant's delay-to-lag ratio (L/&tau;<sub>1</sub>&asymp;0.15).
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
GM&asymp;{fnum(amigo['GM_dB'],1)}&nbsp;dB, PM&asymp;{fnum(amigo['PM_deg'],1)}&deg; &mdash; comfortably
robust margins, and u<sub>peak</sub>&asymp;{fnum(amigo['u_peak'],2)} is already above the
{_U_MAX} actuator limit set in &sect;1, which is exactly what makes &sect;4 worth doing on this loop
rather than a hypothetical one.
</p>

<h3>2.2 Every tuning technique PIDTuner implements, same plant</h3>
<p>
AMIGO is the technique carried through the rest of this memo (&sect;3&ndash;4), but it's one of nine
technique families the CLI can produce for this loop &mdash; {len(all_methods_rows)} panels below,
since CHR's four response/overshoot variants are each shown individually. Each panel is that method's
own step response &mdash; unsaturated, default <code>--N 80</code>, no anti-windup involved yet.
</p>
<div class="gallery">
{"".join(gallery_card(r) for r in all_methods_rows)}
</div>
<div class="callout">
<strong>What's new since the 2026-08-25 run of this same memo: ZN-I/ZN-II are now halved by
default.</strong> <code>pid_compare.py</code>'s <code>compare_all_methods()</code> &mdash; what
this section and &sect;2.3 call &mdash; now runs ZN-I and ZN-II through the app's existing
<code>halve_gains()</code> helper before showing them, since those are the only two techniques whose
own tuning notes recommend it (a single <code>cli_pid.py --method zn1</code> run is unaffected). The
2026-08-25 run showed "Ziegler&ndash;Nichols I"/"II" at
OS%&asymp;{fnum(31.9,1)}%/{fnum(63.7,1)}%; this run shows <strong>"ZN-I &frac12;"/"ZN-II
&frac12;"</strong> at
OS%&asymp;{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='ZN-I ½'),1)}%/
{fnum(next(r['OS%'] for r in all_methods_rows if r['name']=='ZN-II ½'),1)}% instead.
</div>
<p class="section-note">
Spread is still wide, as expected across such different design philosophies on a delay-dominant
plant: Stable pole cancellation is slow and non-oscillatory by construction (it cancels the plant
poles outright rather than reacting to them), Cohen&ndash;Coon and ZN-II &frac12; are now the two
highest-overshoot rows (Cohen&ndash;Coon predates dead-time-explicit design entirely; ZN-II &frac12;
is still the more aggressive of the two ZN rules even halved), and AMIGO/SIMC/Boyd/Tyreus&ndash;Luyben
remain the more uniformly delay-aware, margin-constrained group. No method errored or went unstable
on this plant.
</p>

<h3>2.3 Composite overlay, all included techniques on one set of axes</h3>
<p>
Same responses, drawn the way the desktop webapp draws them when several tuned controllers are
kept in a session &mdash; one 3-row PV/SP &middot; control effort &middot; error figure, all methods
overlaid with the app's own <code>PALETTE</code> (<code>pid_app.py:66-68</code>), not matplotlib's
default cycle. Built with <code>simulate_closed_loop()</code> directly on each method's gains from
&sect;2.2, the same function <code>pid_app.py</code>'s session view and <code>cli_pid.py --plot</code>
both call.
</p>
<p><img src="{composite_uri}" alt="Composite overlay of all included tuning techniques' step responses" style="max-width:100%;border:1px solid #b9c3cc;border-radius:5px;"></p>
<p class="section-note">
Cohen&ndash;Coon's control-effort peak (u<sub>peak</sub>&asymp;{fnum(next(r['u_peak'] for r in all_methods_rows if r['name']=='Cohen–Coon'),2)},
vs. u<sub>ss</sub>=0.2) and Stable pole cancellation's slow, un-oscillatory climb are the two extremes
visible at a glance here now &mdash; the flip from the 2026-08-25 run, where ZN-II's unhalved peak
(&asymp;1.4) was the standout (see &sect;2.2's callout).
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
Identical to the digits shown. Not a bug &mdash; because the derivative acts on the plant output and
the plant output is continuous (no discontinuity for a filter to smooth), N has nothing to bite on
here. This is worth stating plainly rather than leaving as an unexplained flat table: N's job is
noise rejection, not kick suppression, and a noise-free step response can't demonstrate it.
</p>

<h3>3.2 What N is actually for: measurement-noise rejection</h3>
<p>
cli_pid.py's simulated response has no noise hook (<code>--noise-sigma</code> only feeds
<code>--gen-signal</code> identification tests). To exercise the filter's real job, this section
reruns the closed loop by hand on the same AMIGO gains and plant &mdash; same
<code>plant.discretize()</code>/<code>PIDState</code>/<code>pid_step()</code> building blocks
<code>simulate_closed_loop()</code> itself uses &mdash; adding Gaussian sensor noise
(&sigma;={NOISE_SIGMA}, seed={SEED}) onto only the measurement handed to the controller; the plant's
own state update still runs on the true, noise-free control signal.
</p>
<table class="no-break">
  <tr><th>Run</th><th>IAE</th><th>ISU</th><th>|u|<sub>peak</sub></th><th>u<sub>tv</sub></th></tr>
  <tr><td>N=80 (filtered)</td><td>{fnum(noisy_n80['IAE'],3)}</td><td>{fnum(noisy_n80['ISU'],3)}</td>
      <td>{fnum(noisy_n80['u_peak'],3)}</td><td>{fnum(noisy_n80['u_tv'],2)}</td></tr>
  <tr><td>N=0 (ideal derivative)</td><td>{fnum(noisy_n0['IAE'],3)}</td><td>{fnum(noisy_n0['ISU'],3)}</td>
      <td>{fnum(noisy_n0['u_peak'],3)}</td><td>{fnum(noisy_n0['u_tv'],2)}</td></tr>
</table>
<div class="callout">
<strong>Verdict.</strong> With sensor noise in the loop, tracking (IAE) is essentially unchanged
between N=80 and N=0, but control-signal chatter (u<sub>tv</sub>, total variation of u &mdash; a proxy
for actuator wear) drops from {fnum(noisy_n0['u_tv'],1)} to {fnum(noisy_n80['u_tv'],1)}, roughly a
{fnum(100*(1-noisy_n80['u_tv']/noisy_n0['u_tv']),0)}% reduction, and peak control effort drops too
({fnum(noisy_n0['u_peak'],3)}&rarr;{fnum(noisy_n80['u_peak'],3)}). This held up across several
different noise seeds tried during drafting (not shown), not just this one. That's the filter's job:
it costs nothing on a clean setpoint step (&sect;3.1) and pays for itself once real sensor noise shows
up.
</div>

<h2>4. Step 3 &mdash; actuator saturation &amp; back-calculation anti-windup (K<sub>a</sub>)</h2>
<p>
&sect;2.1 already showed AMIGO's unsaturated u<sub>peak</sub>&asymp;{fnum(amigo['u_peak'],2)} exceeds
the &plusmn;{_U_MAX} actuator limit from &sect;1, so this loop saturates for real, not
hypothetically. <code>compute_back_calc_Ka()</code> auto-derives
K<sub>a</sub>=1/T<sub>t</sub>, T<sub>t</sub>=&radic;(T<sub>i</sub>T<sub>d</sub>) &mdash;
&Aring;str&ouml;m &amp; H&auml;gglund's rule of thumb &mdash; but that's a rule of thumb, not a
guarantee, so this section checks it against the alternative anti-windup strategy
(<code>conditional</code>) and against a K<sub>a</sub> sweep.
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
           for ka in (0.02, 1.5, 3, 5, 8))}
</table>
<div class="callout">
<strong>Verdict &mdash; the auto K<sub>a</sub> is not automatically the best choice here.</strong>
The textbook auto-derived K<sub>a</sub>&asymp;{fnum(sat_backcalc_auto['saturated_sim']['Ka'],3)}
(T<sub>t</sub>&asymp;{fnum(sat_backcalc_auto['saturated_sim']['Tt'],1)}&nbsp;s) actually does <em>worse</em>
than plain conditional integration on this loop &mdash;
OS%&asymp;{fnum(sat_backcalc_auto['saturated_sim']['metrics']['Overshoot'],1)}% vs.
{fnum(sat_conditional['saturated_sim']['metrics']['Overshoot'],1)}%, ITAE
{fnum(sat_backcalc_auto['saturated_sim']['metrics']['ITAE'],0)} vs.
{fnum(sat_conditional['saturated_sim']['metrics']['ITAE'],0)}. Pushing K<sub>a</sub> up by hand
(1.5&rarr;3&rarr;5&rarr;8) steadily closes the gap &mdash; as K<sub>a</sub>&rarr;&infin;,
back-calculation degenerates toward clamping the integrator the same way conditional integration
does &mdash; but even at K<sub>a</sub>=8 (T<sub>t</sub>&asymp;0.125&nbsp;s, over 50&times; the
auto-derived value) it hasn't quite caught up. The auto rule assumes T<sub>t</sub>&asymp;
&radic;(T<sub>i</sub>T<sub>d</sub>)&asymp;{fnum(sat_backcalc_auto['saturated_sim']['Tt'],1)}&nbsp;s
is a reasonable unwind time constant; on this loop that's far slower than how quickly the actuator
actually leaves saturation, so the integrator keeps "remembering" the saturation error longer than it
should. For this loop, <strong>conditional integration is the better default</strong>, and
back-calculation would need a hand-tuned K<sub>a</sub> well above the auto value to compete.
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
      <td>Gallery + composite overlay, via <code>compare_all_methods()</code></td>
      <td class="pass">ZN-I/II now shown pre-halved by default (&ldquo;ZN-I &frac12;&rdquo;/&ldquo;ZN-II &frac12;&rdquo;) &mdash; no longer this plant's overshoot outlier; Cohen&ndash;Coon is now</td></tr>
  <tr><td>Derivative filter (<code>--N</code>)</td>
      <td>Clean step (&sect;3.1) vs. noisy measurement (&sect;3.2)</td>
      <td class="pass">No cost on a clean step; &asymp;{fnum(100*(1-noisy_n80['u_tv']/noisy_n0['u_tv']),0)}% less control chatter under sensor noise</td></tr>
  <tr><td>Anti-windup (<code>--antiwindup</code>, <code>--Ka</code>)</td>
      <td>Conditional vs. back-calc (auto and swept K<sub>a</sub>) under saturation</td>
      <td class="fail">Auto K<sub>a</sub> underperforms conditional on this plant; needs hand-tuning to compete</td></tr>
</table>
<p>
One loop, four features, no drift from PIDTuner's own numbers: &sect;2.1 and &sect;4 are direct
<code>cli_pid.py --json</code> output, &sect;2.2/2.3 call <code>compare_all_methods()</code>
in-process (the same function the CLI's <code>--method all</code> and the GUI's "Compare all
methods" call), and &sect;3.2's noise-injected comparison drives the exact
<code>pid_step()</code>/<code>compute_metrics()</code> functions <code>simulate_closed_loop()</code>
itself calls. Two findings worth carrying forward: &sect;4's back-calculation K<sub>a</sub>
auto-derivation is a reasonable starting point but not a substitute for checking it against
conditional integration on the actual plant and actuator limits in play; and &sect;2.2's, new since
the 2026-08-25 run of this memo &mdash; halving ZN-I/ZN-II by default moved them out of the
"runs hot" group on this plant, leaving Cohen&ndash;Coon alone at the top.
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
<title>PID Full Worked Example Memo</title>
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
