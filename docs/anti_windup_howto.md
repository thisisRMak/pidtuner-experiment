# Anti-Windup How-To: Conditional-Integration vs. Back-Calculation

A practical guide to trying PIDTuner's two anti-windup modes in both the
CLI and the GUI. For the full explanation and formulas, see the
"Anti-windup" section of `pidtuner/README.md` and the module docstring in
`pidtuner/simulate.py`.

## The short version

Actuator saturation (the controller asking for more than `u_min`/`u_max`
allow) makes a plain integrator "wind up" — it keeps accumulating error it
can't act on, then overshoots badly once it finally can. PIDTuner offers
two fixes, selectable per simulation:

- **`conditional`** (default, unchanged from before): freeze the integral
  while saturated.
- **`back_calc`**: keep integrating, corrected by the saturation error
  itself (`Ka*(u_sat - u_unsat)`). `Ka` is auto-derived from the tuned
  gains (`Ka = 1/Tt`) unless you override it: `Tt = sqrt(Ti*Td)` for full
  PID, which is Åström & Hägglund's own stated rule of thumb (*Advanced
  PID Control*, Ch. 3); `Tt = Ti` for PI-only (Td = 0) gains, which is
  this codebase's own fallback for the degenerate `Tt = sqrt(Ti*0) = 0`
  case, not something the textbook recommends.

Neither mode does anything unless the actuator actually saturates —
`u_min`/`u_max` need to be tight enough for the specific method's gains.

## CLI examples

All commands run from the `pidtuner/` directory. They use the course's
benchmark plant, `1000/((s+1)(10s+1))`, tuned with Ziegler-Nichols I —
gains small enough that `u_min=-0.003, u_max=0.003` reliably saturates.

**1. Compare both modes side by side**, same plant/method/saturation:

```bash
python3 cli.py --plant "1000/((s+1)(10s+1))" --method zn1 \
  --u-min -0.003 --u-max 0.003 --antiwindup conditional

python3 cli.py --plant "1000/((s+1)(10s+1))" --method zn1 \
  --u-min -0.003 --u-max 0.003 --antiwindup back_calc
```

Each prints an extra "Saturated-actuator simulation" block on top of the
normal (always-unsaturated) comparison metrics. On this plant,
conditional settles with ~7% overshoot; back-calculation's auto-derived
`Ka` actually overshoots to ~63% — a concrete reminder that `Ka` is a
real design tradeoff, not a free upgrade. Compare both on your own plant
rather than assuming one always wins.

**2. Override `Ka` by hand:**

```bash
python3 cli.py --plant "1000/((s+1)(10s+1))" --method zn1 \
  --u-min -0.003 --u-max 0.003 --antiwindup back_calc --Ka 3.0
```

**3. Save plots** of the actual saturated response for a visual diff:

```bash
python3 cli.py --plant "1000/((s+1)(10s+1))" --method zn1 \
  --u-min -0.003 --u-max 0.003 --antiwindup back_calc \
  --plot examples/out/antiwindup_backcalc.png

python3 cli.py --plant "1000/((s+1)(10s+1))" --method zn1 \
  --u-min -0.003 --u-max 0.003 --antiwindup conditional \
  --plot examples/out/antiwindup_conditional.png
```

**4. All 9 methods at once**, saturation applied uniformly, JSON output
(each stable row gets an extra `saturated_sim` key):

```bash
python3 cli.py --plant "1000/((s+1)(10s+1))" --method all \
  --u-min -0.003 --u-max 0.003 --antiwindup back_calc --json
```

The same bound is applied to all 9 methods, but they don't all produce
gains of the same magnitude, so some rows may not actually saturate at
that bound — `saturated_sim` is added to *every* stable row regardless,
since it's conditioned only on whether you asked for saturation bounds at
all, not on whether that specific row's gains actually hit them. Each
`saturated_sim` block carries a `"saturated": true/false` field for
exactly this reason: when `false`, `Ka`/`Tt` come back `null` rather than
showing a derived number that never actually engaged (back_calc's
correction term is zero unless the actuator is pinned at `u_min`/`u_max` —
see `test_modes_identical_when_never_saturated` in `test_pid_tuner.py`),
and its metrics will just numerically match the ordinary (always-
unbounded) row above it. The single-method CLI path also prints a stderr
note in this case ("the actuator never reached --u-min/--u-max... Ka not
reported"); the GUI's session-list label likewise reads
`[back_calc: never saturated]` instead of showing a `Ka` value.

**5. The no-op warnings** — both print a note to stderr instead of
silently doing nothing:

```bash
# back_calc with no saturation bounds set:
python3 cli.py --plant "1000/((s+1)(10s+1))" --method simc --antiwindup back_calc

# --Ka with the default (conditional) mode:
python3 cli.py --plant "1000/((s+1)(10s+1))" --method simc --Ka 0.5
```

## GUI walkthrough

```bash
python3 app.py
```

1. Plant tab: enter `1000/((s+1)(10s+1))` (or use the default).
2. Method: **2. Ziegler-Nichols I**.
3. In **Closed-loop simulation**, set `u min = -0.003`, `u max = 0.003`.
4. Set the **Anti-windup** dropdown to `conditional`, leave `Ka override`
   blank, click **Tune & simulate**.
5. Switch **Anti-windup** to `back_calc` (still blank `Ka` = auto-derive),
   click **Tune & simulate** again — same plant, method, setpoint, and
   `u_min`/`u_max` as step 4.
6. Both runs now sit in the session overlay list, and the legend
   disambiguates them automatically — e.g. `Ziegler-Nichols I` vs.
   `Ziegler-Nichols I [back_calc, Ka=0.938]` — so you don't have to guess
   from color alone.
7. On the Response tab's **control effort `u(t)`** panel, saturated
   samples (where `u` is pinned at `u_min`/`u_max`) are marked with small
   dots in each trace's color, with a "saturated (u at u_min/u_max)"
   legend entry — both traces will show the same clipped plateau (that
   part is identical by design), so look at the **top PV/SP panel**
   instead for where the two modes actually diverge: overshoot magnitude
   and settling time.
8. The result panel (bottom-left, below the Tune button) also prints the
   derived `Ka`/`Tt` whenever `back_calc` is active.

### Why the two modes can look identical if you're not careful

- **`u_min`/`u_max` too loose for this method's gains**: if the
  unsaturated command never actually reaches the bounds, both modes
  reduce to plain integration and the two runs are bit-for-bit identical.
  Check the metrics readout's `|u|_peak` — it should equal `u_max` (or
  `u_min` in magnitude) if saturation actually occurred.
- **Looking at the wrong panel**: the `u(t)` plateau is identical between
  modes by construction (both clip to the same bound) — the difference
  shows up in the PV/SP panel's overshoot/settling, not in how hard
  `u(t)` is clipped.
- **Comparing across a "Compare all methods" click**: that button applies
  the *same* anti-windup setting to every method in one shot — to compare
  modes you need two separate single-method tunes with the dropdown
  flipped in between.
