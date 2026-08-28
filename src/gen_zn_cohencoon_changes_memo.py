#!/usr/bin/env python3
"""Generates docs/memos/<YYYY-MM-DD>/<YYYY-MM-DD>-zn-cohencoon-changes-memo.html
— a short internal memo on what changed in code for ZN-I/ZN-II and
Cohen-Coon over 2026-08-27/08-28, styled to match the house CSS the other
memo generators in this file's family use (_memo_css.txt). Unlike the
textbook-validation memos, this one is a change-history note, not a
verification against a book example — no To/From, same as
gen_pid_full_worked_example_memo.py's convention for internal memos.

The CLI timing figure is measured live, right here (subprocess + wall
clock around cli_pid.py --method all), so it can't drift from what the
CLI actually takes on the machine that runs this script. The GUI timing
figure is NOT reproducible by this script — it requires driving an actual
browser against a running Streamlit process (Playwright), which isn't a
repo dependency — so it's recorded as a one-time measurement with its own
methodology spelled out in the text instead of silently presented as if
it were live-computed the same way.

Run: python3 gen_zn_cohencoon_changes_memo.py
"""
import datetime
import html
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(__file__)
_DATE = datetime.date.today().isoformat()
_OUT_DIR = os.path.join(_HERE, "..", "docs", "memos", _DATE)
_HTML_OUT = os.path.join(_OUT_DIR, f"{_DATE}-zn-cohencoon-changes-memo.html")
_CSS = open(os.path.join(_HERE, "_memo_css.txt")).read()

_PLANT = "1/(90s+1)"
_L = "13"

# One-time browser-automated measurement (Playwright driving the actual
# streamlit_app.py "Compare all methods" button on _PLANT/_L, post-fix) —
# not reproducible by this script alone, see module docstring.
_GUI_TOAST_S = 11.09
_GUI_SETTLED_S = 12.43


def _e(s):
    return html.escape(str(s))


def time_cli_compare_all():
    """Wall-clock time for `cli_pid.py --method all --json` on _PLANT/_L —
    the CLI path's own compare-all, timed the same way `time` would."""
    cmd = [sys.executable, "cli_pid.py", "--plant", _PLANT, "--L", _L,
           "--method", "all", "--json"]
    t0 = time.perf_counter()
    out = subprocess.run(cmd, cwd=_HERE, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    if out.returncode != 0:
        raise RuntimeError(f"cli_pid.py failed: {cmd}\n{out.stderr}")
    return elapsed


def build_html():
    cli_s = time_cli_compare_all()

    body = f"""
<h1>ZN-I/II and Cohen&ndash;Coon: Code Changes, 2026-08-27 to 2026-08-28</h1>

<dl class="memo-header">
  <dt>Date</dt><dd>{_DATE}</dd>
  <dt>Re</dt><dd>What changed in code for Ziegler&ndash;Nichols I/II and Cohen&ndash;Coon over
  these two dates, with emphasis on Cohen&ndash;Coon's discretization saga, plus a CLI-vs-GUI
  timing comparison for the textbook worked example (ZN-I, ZN-II, Cohen&ndash;Coon on
  <code>{_e(_PLANT)}</code>, L={_L} &mdash; PEI8e Ex. 4.9/4.10).</dd>
</dl>

<h2>1. 2026-08-27 &mdash; ZN-I/ZN-II halved by default in "compare all"</h2>
<p>
<code>compare_all_methods()</code> (<code>src/pid_compare.py</code>, commit <code>eb1e8d7</code>)
now runs ZN-I/ZN-II through <code>halve_gains()</code> automatically, labeling those rows
<strong>"ZN-I &frac12;"/"ZN-II &frac12;"</strong> &mdash; the only two methods whose own tuning
notes recommend halving for setpoint tracking. <code>cli_pid.py --method all --halve</code>
became a documented no-op on that path (single-method <code>--halve</code> and the GUI checkboxes
are unaffected).
</p>
<div class="callout">
<strong>Cohen&ndash;Coon was explicitly considered and deferred</strong> from this same treatment:
its notes say "can be aggressive on setpoint tracking" (same reaction-curve family as ZN-I) but
don't point at a halve toggle the way ZN-I/ZN-II's notes do. That question was left open for a
separate session &mdash; the work below grew out of that follow-up, but turned out to be about
something else entirely (discretization, not halving); whether Cohen&ndash;Coon should get the
same halve-by-default treatment is still unresolved.
</div>

<h2>2. 2026-08-28 &mdash; Cohen&ndash;Coon: two opposite discretization bugs, one fix each</h2>
<p>
<strong>Trigger.</strong> Professor flagged Cohen&ndash;Coon on <code>{_e(_PLANT)}</code>, L={_L}
as not converging. The tool's gains (K<sub>p</sub>=9.53, T<sub>i</sub>=30.05, T<sub>d</sub>=4.59)
were within ~0.5% of a hand-built MATLAB simulation of the same design &mdash; that gap is
expected, the CLI re-identifies &tau;/L from a simulated step response rather than trusting the
exact analytic values &mdash; but where MATLAB's continuous-time simulation settled cleanly at
t<sub>s</sub>&asymp;119&nbsp;s, the tool's own simulated step response diverged to
~450,000% overshoot.
</p>

<h3>2.1 Issue 1 &mdash; under-discretized</h3>
<p>
<code>simulate_closed_loop()</code>'s timestep came from <code>plant.auto_dt()</code>
(<code>src/plant.py</code>), which sizes <code>dt</code> from the <strong>plant's</strong> poles
and dead time only &mdash; it has no visibility into the <strong>controller's</strong> derivative
filter pole (&tau;<sub>d</sub>=K<sub>d</sub>/(N&middot;K<sub>p</sub>)). For this case
<code>auto_dt()</code> picked dt=2.6&nbsp;s; the filter's own time constant was
&tau;<sub>d</sub>&asymp;0.057&nbsp;s, about 45&times; faster. At that resolution the discrete
PID/plant loop is a genuinely different (numerically unstable) system from the continuous one it's
meant to approximate &mdash; confirmed by rerunning the identical discrete loop at progressively
finer dt and watching it converge onto MATLAB's answer.
</p>
<p class="section-note">
Fix: bound dt by &tau;<sub>d</sub>/5 in <code>simulate_closed_loop()</code>
(<code>src/pid_simulate.py</code>), reusing the same /5 margin <code>auto_dt()</code> already
applies to dead time. Re-simulating the reported case now settles at t<sub>s</sub>&asymp;124&nbsp;s.
</p>

<h3>2.2 Issue 2 &mdash; over-discretized</h3>
<p>
That bound has no floor. Running "Compare all methods" in the Streamlit UI on the <em>same</em>
plant then stopped completing at all. Cause: the <strong>Boyd</strong> tuner (run alongside
Cohen&ndash;Coon in every "compare all") returned K<sub>d</sub>&asymp;0.0001 for this plant &mdash;
not real derivative action but a boundary artifact of how its LP search box is sized (seeded from
SIMC's K<sub>d</sub>=0 here, giving a [0, 1e-4] search box for K<sub>d</sub>, and the optimizer
landed on the edge). Feeding that into the new &tau;<sub>d</sub>/5 bound collapsed dt to
~1.3&times;10<sup>&minus;7</sup>&nbsp;s; over this plant's ~1415&nbsp;s auto t<sub>end</sub> that's
~10 billion steps &mdash; not slow, effectively infinite.
</p>
<p class="section-note">
Fix: a hard 300,000-step ceiling in <code>simulate_closed_loop()</code> that coarsens dt to hit the
cap instead of hanging, whatever drove dt down in the first place.
</p>

<h3>2.3 Optimization &mdash; getting some of that time back</h3>
<p>
Separately, the Streamlit panel's "compare all" was paying for a <strong>third</strong>
<code>simulate_closed_loop()</code> call per method: two already happen inside
<code>compare_all_methods()</code> for scoring (deliberately against a wide-open, unconstrained
actuator so every method is judged on equal footing), and the panel re-simulated a third time just
to get a plottable trace under its own actuator/sim-setting widgets.
</p>
<p class="section-note">
Fix: <code>metric_row()</code>/<code>compare_all_methods()</code> gained an opt-in
<code>return_sim=True</code> exposing the scoring simulation they already compute. The panel reuses
it for the plot only when doing so is <em>provably</em> identical to a fresh widget-bounded
simulation (same step setpoint/amplitude/N, no custom t<sub>end</sub>, and the trace never actually
needed bounds wider than the widgets' own u<sub>min</sub>/u<sub>max</sub>); otherwise it falls back
to a fresh simulation, unchanged from before. Cuts "compare all"'s <code>simulate_closed_loop</code>
calls from 3 to 2 per method.
</p>
<div class="callout">
Same-day, unrelated to discretization: the SISO response plot's shared x-axis is now auto-cropped
to the settled window &mdash; reusing the LQG track's <code>auto_plot_window()</code> &mdash; since
the auto t<sub>end</sub> is sized for a correct settling-time measurement, not for viewing, and was
squashing every trace into the first slice of a mostly-flat plot.
</div>

<h2>3. Time analysis &mdash; CLI vs. GUI, textbook worked example</h2>
<p>
Total wall-clock time to run the full comparison (all applicable tuning methods, including
ZN-I &frac12;, ZN-II &frac12;, and Cohen&ndash;Coon) on <code>{_e(_PLANT)}</code>, L={_L}, measured
after all fixes above. Not a re-analysis of the results themselves &mdash; see the ZN
textbook-validation memo (regenerated {_DATE}) for those.
</p>
<table class="no-break">
  <tr><th>Path</th><th>Total time</th><th>Method</th></tr>
  <tr><td>CLI (<code>cli_pid.py --method all --json</code>)</td>
      <td><strong>{cli_s:.1f}s</strong></td>
      <td>Live-measured by this script (subprocess wall-clock)</td></tr>
  <tr><td>GUI (Streamlit "Compare all methods" click &rarr; fully rendered plot)</td>
      <td><strong>{_GUI_SETTLED_S:.1f}s</strong></td>
      <td>One-time measurement, browser-automated (Playwright); toast at
      {_GUI_TOAST_S:.1f}s, visually settled {_GUI_SETTLED_S - _GUI_TOAST_S:.1f}s later &mdash;
      not reproducible by this script</td></tr>
</table>
<p class="section-note">
The two are close &mdash; the GUI's extra time is the matplotlib figure rendering/transmitting
after the same underlying computation the CLI does directly to stdout.
</p>
"""

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ZN/Cohen&ndash;Coon Changes Memo</title>
<style>
{_CSS}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    os.makedirs(os.path.dirname(_HTML_OUT), exist_ok=True)
    with open(_HTML_OUT, "w") as f:
        f.write(doc)
    print(f"wrote {_HTML_OUT}")


if __name__ == "__main__":
    build_html()
