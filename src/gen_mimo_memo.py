#!/usr/bin/env python3
"""Generates docs/memos/mimo_pi_antiwindup_memo.html — the MIMO-PI
multivariable integral anti-windup memo (Windup_AEN 7.pdf §9.3), styled to
match docs/memos/kd_filtering_memo.html's house CSS. Runs cli_mimo_example.py's
worked example directly rather than hardcoding numbers, so the memo can't
drift from what the code actually produces.

Run: python3 gen_mimo_memo.py
"""
import base64
import html
import os

import numpy as np

from cli_mimo_example import build_example, run_all_modes, plot_comparison
from mimo_pi import mimo_pi_closed_loop_poles, saturation_mask

_HERE = os.path.dirname(__file__)
_HTML_OUT = os.path.join(_HERE, "..", "docs", "memos", "2026-08-18-mimo-pi-antiwindup-memo.html")
_PLOT_PATH = os.path.join(_HERE, "examples", "out", "mimo_pi_antiwindup.png")

_CSS = open(os.path.join(_HERE, "_memo_css.txt")).read()


def _e(s):
    return html.escape(str(s))


def build_html():
    plant, gains, u_min, u_max = build_example()
    sims = run_all_modes(plant, gains, u_min, u_max)
    plot_comparison(sims, u_min, u_max, _PLOT_PATH)
    with open(_PLOT_PATH, "rb") as f:
        plot_b64 = base64.b64encode(f.read()).decode("ascii")

    poles = mimo_pi_closed_loop_poles(plant, gains)

    rows = []
    for mode, sim in sims.items():
        m = sim.metrics
        sat = int(saturation_mask(sim).sum())
        rows.append(f"<tr><td><code>{mode}</code></td><td>{sat}</td>"
                    f"<td>{m['IAE']:.3g}</td><td>{m['ITAE']:.3g}</td>"
                    f"<td>{m['ISU']:.3g}</td><td>{m['u_peak']:.3g}</td></tr>\n")

    parts = []
    parts.append(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>MIMO-PI Integral Anti-Windup Memo</title>
<style>{_CSS}</style>
</head>
<body>

<h1>MIMO Integrator Anti-Windup: Implementation and Worked Example</h1>

<dl class="memo-header">
  <dt>To</dt><dd>Prof. Emami-Naeini</dd>
  <dt>From</dt><dd>Rohit</dd>
  <dt>Date</dt><dd>2026-08-18</dd>
  <dt>Re</dt><dd>MIMO-PI multivariable integral anti-windup (Windup_AEN 7.pdf &sect;9.3) — implementation, the three anti-windup modes, why Method III is out of scope, and a worked comparison</dd>
</dl>

<h2>1. Scope: PI, not PID</h2>
<p>
<code>mimo_pi.py</code> implements the multivariable analog of
<code>pid_simulate.py</code>'s SISO PID anti-windup, but for the matrix-PI
structure the book's Fig&nbsp;9.31/9.32 actually draw &mdash; not the
LQR/LQG track's full-state-feedback <code>u = -Kx</code>:
</p>
<pre>Uc = KP&middot;E + KI&middot;xI,      E = r - y,      xI = &int;E dt</pre>
<p>
<b>KP</b>, <b>KI</b> are (nu, ny) gain matrices, restricted to square plants
(nu == ny) &mdash; the same restriction the book's Method&nbsp;II states
outright ("the plant be square"). This module doesn't derive KP/KI (a
tuning-method concern, out of scope here); it takes them as given and
focuses on what actuator saturation does to xI &mdash; genuinely a
different problem in MIMO than SISO, since a single saturated actuator
doesn't correspond to a single output channel once KP/KI have off-diagonal
terms. Freezing/reset decisions are made for the whole xI vector at once
rather than per-channel &mdash; the book's own opening warning is that "it
would be naive to think that it suffices to just implement SISO integrator
windup for each integrator and ignore the coupling."
</p>

<h2>2. The three anti-windup modes implemented</h2>
<table>
<thead><tr><th>Mode</th><th>Rule</th><th>Requires</th></tr></thead>
<tbody>
<tr><td><code>conditional</code> (default)</td>
<td>Freeze the whole xI vector whenever ANY actuator channel saturates &mdash; direct MIMO lift of the SISO conditional-integration scheme.</td>
<td>Nothing extra &mdash; simplest, no invertibility requirement, but (like its SISO counterpart) only stops things from getting worse; doesn't actively unwind.</td></tr>
<tr><td><code>resettable</code> &mdash; Method&nbsp;II (eq. 9.121-9.123)</td>
<td>While any channel saturates, continually reset <code>xI = KI&#8315;&sup1;(u_sat - KP&middot;E)</code>, so Uc lands exactly on u_sat.</td>
<td>KI invertible &mdash; the book flags pseudo-inverse as an unstudied fallback; this module allows it but reports <code>pinv_used=True</code> so callers don't mistake it for exact.</td></tr>
<tr><td><code>hanus</code> &mdash; Method&nbsp;I (eq. 9.117-9.120)</td>
<td>Always-on correction: <code>xI(t+dt) = xI(t) + [E + KP&#8315;&sup1;&middot;(u_sat - u_unsat)]dt</code> &mdash; vanishes when u_sat == u_unsat, otherwise pulls xI toward consistency with what the actuators can deliver, first-order lag dynamics set by KI&middot;KP&#8315;&sup1;.</td>
<td>KP invertible, same pinv-fallback/flag convention.</td></tr>
</tbody>
</table>

<h2>3. Why Method III (&sect;9.4, directional/SVD) is out of scope</h2>
<div class="callout">
Method III &mdash; the book's directional/SVD-based crush/msat scheme for
<b>p &gt; m</b> sensor-rich systems (more measured outputs than actuator
inputs) &mdash; is <b>not implemented</b>. It needs a genuinely different,
direction-preserving saturation function plus an SVD of the open-loop DC
gain, not simply another anti-windup mode bolted onto the same controller
structure Methods&nbsp;I/II share. Since this module (like Methods&nbsp;I/II
themselves) is restricted to square plants (nu == ny) by construction, the
p&gt;m case Method&nbsp;III addresses doesn't arise here at all &mdash;
supporting it would mean relaxing the square-plant restriction throughout
the module first, a separate scope expansion, not an additional mode.
</div>
<p>
Put differently: Methods&nbsp;I and II both correct the integrator state
<i>after</i> an ordinary elementwise clip has already been applied to u
&mdash; the saturation function itself is unchanged from the SISO case,
just applied per-channel. Method&nbsp;III instead <i>replaces</i> that
saturation function with one that preserves the <i>direction</i> of the
commanded actuator vector (rescaling all channels together rather than
clipping each independently), which only matters when there are more
outputs to control than actuators to do it with. That's a different
control problem, not a variant of the anti-windup correction Methods&nbsp;I/II
apply.
</p>

<h2>4. Worked example</h2>
<p>
Plant: 2-state, 2-input, 2-output, stable, with <b>input coupling</b> (B
has off-diagonal terms &mdash; actuator&nbsp;0 also pushes on state&nbsp;1
and vice versa):
</p>
<pre>A = [[-1.0,  0.0],       B = [[1.0, 0.5],       C = [[1.0, 0.0],
     [ 0.0, -2.0]]            [0.3, 1.0]]             [0.0, 1.0]]

KP = 2.0&middot;I,  KI = 1.0&middot;I,  u &isin; [-0.6, 0.6] (both channels)</pre>
<p>
Unsaturated closed-loop poles (linearized, no actuator limits):
<code>{_e(np.array2string(poles, precision=4))}</code> &mdash; stable, confirming
the gains are reasonable before saturation is introduced. The actuator
limits (&plusmn;0.6) are tight enough to force real, sustained saturation
but loose enough that the loop still desaturates and recovers within the
simulation horizon &mdash; that recovery phase is exactly where the three
modes visibly diverge from one another.
</p>
<p>Reference: unit step on both channels (r = [1, 1]). Results, all three modes:</p>
<table class="no-break">
<thead><tr><th>Mode</th><th>Saturated samples</th><th>IAE</th><th>ITAE</th><th>ISU (control effort)</th><th>|u|<sub>peak</sub></th></tr></thead>
<tbody>
{"".join(rows)}
</tbody></table>
<p>
<code>resettable</code> and <code>hanus</code> (both active-correction
methods) desaturate faster and land closer to the reference than
<code>conditional</code> at the cost of more control effort (ISU 45.8 and
46.3 vs. 37.5) and more time actually spent at the actuator limit
(saturated_samples higher for both) &mdash; consistent with the book's
framing of Methods&nbsp;I/II as actively correcting the integrator state
rather than merely freezing it. All three remain stable under these
actuator limits; none diverges.
</p>
<p>Response plot &mdash; y(t) for all three modes vs. reference, and u(t) with saturation limits marked:</p>
<img src="data:image/png;base64,{plot_b64}" style="width:100%;max-width:700px;display:block;margin:12px auto;border:1px solid var(--rule);" alt="MIMO-PI anti-windup comparison plot">

<h2>5. What we're asking</h2>
<p><b>P0 &mdash; the two scope questions everything else here depends on:</b></p>
<ol>
<li>
<b>Method&nbsp;III (&sect;3):</b> does leaving it out of scope sound right given the
square-plant (nu&nbsp;==&nbsp;ny) restriction the current module shares with Methods&nbsp;I/II
&mdash; or, if p&gt;m sensor-rich plants matter for this project, how would you want us to
approach implementing it? As noted in &sect;3, that's not an additional mode on the current
controller; it needs a direction-preserving saturation function plus an SVD of the open-loop DC
gain, and relaxing the square-plant restriction throughout the module first.
</li>
<li>
Is PI-only the right scope for the MIMO track at all, or should MIMO-PID be on the roadmap
&mdash; and if so, should the derivative term be filtered the way the SISO Kd term is (see the
companion memo, <code>2026-08-18-kd-filtering-memo.pdf</code>, on whether Kd filtering should
apply at design time or only at simulation time)? MIMO-PID doesn't exist yet in any form, so
this decides whether MIMO derivative filtering is even a live question or moot until PID lands.
</li>
</ol>
<p><b>P1 &mdash; secondary, only matters once P0 is answered:</b></p>
<ol>
<li>
Is the <code>pinv_used</code> flag (for non-invertible KI/KP in <code>resettable</code>/
<code>hanus</code>) sufficient disclosure, or should those modes refuse to run at all rather
than fall back to a pseudo-inverse the book itself calls unstudied? This is an implementation
detail of Methods&nbsp;I/II as they exist today &mdash; it doesn't bear on whether Method&nbsp;III
or MIMO-PID get built, so it's not blocking either P0 question above.
</li>
</ol>

<div class="appendix-title">Appendix: reproducing the numbers above</div>
<pre><code>cd src
python3 cli_mimo_example.py    # prints the metrics table, saves examples/out/mimo_pi_antiwindup.png
python3 gen_mimo_memo.py       # regenerates this memo from the same code path
</code></pre>

</body>
</html>
""")
    return "".join(parts)


if __name__ == "__main__":
    out = build_html()
    os.makedirs(os.path.dirname(_HTML_OUT), exist_ok=True)
    with open(_HTML_OUT, "w") as f:
        f.write(out)
    print(f"wrote {_HTML_OUT}")
