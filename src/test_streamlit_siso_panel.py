"""Regression tests for the Streamlit SISO PID panel — no browser, just
Streamlit's AppTest harness driving streamlit_app.py's actual widget tree.

Run with:
    python test_streamlit_siso_panel.py
or:
    python -m unittest test_streamlit_siso_panel -v

These exercise the panel end-to-end (select a method, set its args,
click Tune, inspect the resulting session-state entry) rather than
calling the backend functions directly — that's what caught the one
real bug found so far this way: method 1's manual pole-cancellation
widgets had an inverted sign convention (defaulted to negative pole
values when StablePoleCancellation requires positive ones), which no
amount of backend-only testing would have surfaced since the bug was
in the widget wiring, not the tuning math.

What this does NOT cover (see docs/gui_plan.md "Testing debt"):
  - A real browser (AppTest only inspects the server-side element tree,
    which is exactly what missed the dark-theme-invisible-heatmap and
    nested-st.tabs rendering bugs found earlier by hand).
  - A side-by-side numeric diff against pid_app.py (blocked in headless
    CI environments without a display server for Tkinter).
"""

import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = __file__.replace("test_streamlit_siso_panel.py", "streamlit_app.py")

DEFAULT_PLANT = "1000 / ((s+1)*(10s+1))"  # poles at s = -1, -0.1

# (method label, non-default widget args) — mirrors _render_method_args'
# key names in streamlit_siso_panel.py. Pole values for method 1 are
# *positive* (p1=1.0 cancels the plant's pole at s=-1, p2=0.1 cancels
# s=-0.1) — StablePoleCancellation's own convention, see pid_tuning_methods.py.
METHOD_CASES = [
    ("1. Stable pole cancellation",
     {"pc_mode": "manual", "pc_p1": 1.0, "pc_p2": 0.1, "pc_kd": "2.0"}),
    ("2. Ziegler–Nichols I (step / FOPDT)",
     {"zn1_step": 2.0, "zn1_noise": 0.05}),
    ("3. Ziegler–Nichols II (ultimate gain)",
     {"zn2_source": "relay", "zn2_relay_h": 2.0, "zn2_relay_T": 80.0}),
    ("4. AMIGO (FOPDT)",
     {"amigo_integrating": True}),
    ("5. SIMC (FOPDT)",
     {"simc_tau_c": "0.5", "simc_tau2": "1.0"}),
    ("6. Boyd (convex-concave)",
     {"boyd_Ms": 1.6, "boyd_Mt": 1.6}),
    ("7. Cohen–Coon (FOPDT)",
     {"cc_step": 2.0, "cc_noise": 0.02}),
    ("8. Chien–Hrones–Reswick (FOPDT)",
     {"chr_response": "load", "chr_overshoot": 20}),
    ("9. Tyreus–Luyben (ultimate gain)",
     {"tl_source": "relay", "tl_relay_h": 2.0, "tl_relay_T": 80.0, "tl_pi": True}),
]


def _fresh_app():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    return at


def _siso_tab(at):
    # No more tabs — SISO/PID + Manual is streamlit_unified_panel.py's
    # default state, so nothing needs selecting first; this just keeps
    # the existing call sites (tab.button(...), tab.text_input(...), ...)
    # working unchanged.
    return at


def _n_entries(at):
    import streamlit_gui_state as gs
    return len(at.session_state[gs.CONTROLLERS_KEY])


def _set_widget(tab, key, value):
    """Find whichever widget kind owns this key and set its value —
    the panel mixes number_input/text_input/radio/checkbox for the
    method-arg widgets, keyed identically to streamlit_siso_panel.py."""
    for kind in ("number_input", "text_input", "radio", "checkbox"):
        try:
            getattr(tab, kind)(key=key).set_value(value)
            return
        except Exception:
            continue
    raise AssertionError(f"no widget found for key={key!r}")


class TestAllTuningMethodsThroughUI(unittest.TestCase):
    """Every method, clicked through the actual UI with non-default
    arguments, each expected to add exactly one session entry."""

    def test_all_nine_methods(self):
        at = _fresh_app()
        for i, (method, args) in enumerate(METHOD_CASES, start=1):
            tab = _siso_tab(at)
            tab.selectbox(key="siso_method").set_value(method)
            at.run(timeout=30)
            self.assertFalse(at.exception, f"{method}: exception after selecting method")

            tab = _siso_tab(at)
            for k, v in args.items():
                _set_widget(tab, k, v)
            at.run(timeout=30)
            self.assertFalse(at.exception, f"{method}: exception after setting args")

            before = _n_entries(at)
            tab = _siso_tab(at)
            tab.button(key="siso_tune").click()
            at.run(timeout=60)
            self.assertFalse(at.exception, f"{method}: exception after Tune & simulate")
            self.assertEqual(_n_entries(at), before + 1,
                             f"{method}: expected one new session entry")


class TestMatlabPlantForm(unittest.TestCase):
    def test_matlab_coefficients_tune(self):
        at = _fresh_app()
        tab = _siso_tab(at)
        tab.radio(key="siso_plant_form").set_value("MATLAB coefficients")
        at.run(timeout=30)
        tab = _siso_tab(at)
        tab.text_input(key="siso_gain").set_value("500")
        tab.text_input(key="siso_num").set_value("[1]")
        tab.text_input(key="siso_den").set_value("[5, 6, 1]")
        at.run(timeout=30)
        self.assertFalse(at.exception)

        before = _n_entries(at)
        tab = _siso_tab(at)
        tab.button(key="siso_tune").click()
        at.run(timeout=60)
        self.assertFalse(at.exception)
        self.assertEqual(_n_entries(at), before + 1)


class TestBackCalcAntiWindup(unittest.TestCase):
    def test_back_calc_with_actual_saturation(self):
        at = _fresh_app()
        tab = _siso_tab(at)
        tab.radio(key="antiwindup").set_value("back_calc")
        tab.number_input(key="u_min").set_value(-0.001)
        tab.number_input(key="u_max").set_value(0.001)
        at.run(timeout=30)
        self.assertFalse(at.exception)

        tab = _siso_tab(at)
        tab.button(key="siso_tune").click()
        at.run(timeout=60)
        self.assertFalse(at.exception)

        import streamlit_gui_state as gs
        entries = at.session_state[gs.CONTROLLERS_KEY]
        self.assertTrue(entries)
        self.assertEqual(entries[-1].sim.antiwindup, "back_calc")


class TestErrorPaths(unittest.TestCase):
    def test_rhp_pole_cancellation_rejected_cleanly(self):
        at = _fresh_app()
        tab = _siso_tab(at)
        tab.text_input(key="siso_tf_expr").set_value("1/(s-1)")  # RHP pole at s=1
        at.run(timeout=30)
        tab = _siso_tab(at)
        tab.selectbox(key="siso_method").set_value("1. Stable pole cancellation")
        at.run(timeout=30)

        before = _n_entries(at)
        tab = _siso_tab(at)
        tab.button(key="siso_tune").click()
        at.run(timeout=30)
        self.assertFalse(at.exception, "RHP-pole rejection should not raise")
        self.assertEqual(_n_entries(at), before, "no entry should be added")
        errs = [e.value for e in _siso_tab(at).error]
        self.assertTrue(any("RHP" in e or "unsafe" in e for e in errs),
                        f"expected an RHP/unsafe error message, got {errs}")

    def test_malformed_symbolic_plant_disables_tune(self):
        at = _fresh_app()
        tab = _siso_tab(at)
        tab.text_input(key="siso_tf_expr").set_value("this is not a transfer function ((")
        at.run(timeout=30)
        self.assertFalse(at.exception)
        tab = _siso_tab(at)
        self.assertTrue(tab.button(key="siso_tune").disabled)
        self.assertTrue(tab.button(key="siso_compare_all").disabled)

    def test_malformed_matlab_coefficients_disables_tune(self):
        at = _fresh_app()
        tab = _siso_tab(at)
        tab.radio(key="siso_plant_form").set_value("MATLAB coefficients")
        at.run(timeout=30)
        tab = _siso_tab(at)
        tab.text_input(key="siso_num").set_value("not-a-list")
        at.run(timeout=30)
        self.assertFalse(at.exception)
        tab = _siso_tab(at)
        self.assertTrue(tab.button(key="siso_tune").disabled)

    def test_garbage_numeric_field_fails_cleanly(self):
        at = _fresh_app()
        tab = _siso_tab(at)
        tab.selectbox(key="siso_method").set_value("1. Stable pole cancellation")
        at.run(timeout=30)
        tab = _siso_tab(at)
        tab.radio(key="pc_mode").set_value("manual")
        at.run(timeout=30)
        tab = _siso_tab(at)
        tab.text_input(key="pc_kd").set_value("garbage")
        at.run(timeout=30)
        self.assertFalse(at.exception)

        before = _n_entries(at)
        tab = _siso_tab(at)
        tab.button(key="siso_tune").click()
        at.run(timeout=30)
        self.assertFalse(at.exception)
        self.assertEqual(_n_entries(at), before)


class TestCircleSwatchIsRealUnicodeNotAShortcode(unittest.TestCase):
    """Regression: the color swatch used to be a markdown :color_circle:
    shortcode, on the assumption Streamlit's frontend converts it the
    way GitHub's does. Confirmed live in a real browser (AppTest can't
    see this -- it only inspects the raw markdown source, never what a
    browser does with it) that only :large_blue_circle:/:red_circle:/
    :black_circle: actually convert -- every other color rendered as
    literal, unconverted shortcode text. Now a literal Unicode
    character, so there's no conversion step left to fail."""

    def test_circle_emoji_covers_every_palette_color_with_a_real_character(self):
        import streamlit_gui_state as gs
        from streamlit_siso_panel import _circle_emoji

        for hex_color in gs.PALETTE:
            swatch = _circle_emoji(hex_color)
            self.assertNotIn(":", swatch, f"{hex_color} still returns shortcode-shaped text")

    def test_no_leftover_shortcode_text_in_a_real_session_list(self):
        at = _fresh_app()
        tab = _siso_tab(at)
        tab.button(key="siso_compare_all").click()
        at.run(timeout=60)
        self.assertGreater(_n_entries(at), 5)  # several distinct palette colors in play
        for m in at.markdown:
            self.assertNotRegex(m.value, r":\w+_circle:",
                               f"shortcode-shaped text leaked into a rendered row: {m.value!r}")


class TestLlmEntryTagIsABadge(unittest.TestCase):
    """Regression: the 🤖 LLM tag on a source="llm" session-list row used
    to be plain markdown text appended after the label, which rendered
    tiny/monochrome and easy to miss next to the color swatch — not a
    regression, but reported live as hard to spot. Now uses Streamlit's
    :color-badge[...] markdown directive so it renders as a distinct
    colored pill."""

    def test_llm_sourced_entry_renders_a_violet_badge(self):
        import streamlit_gui_state as gs

        at = _fresh_app()
        tab = _siso_tab(at)
        tab.button(key="siso_compare_all").click()
        at.run(timeout=60)
        entries = at.session_state[gs.CONTROLLERS_KEY]
        self.assertGreater(len(entries), 0)
        entries[0].source = "llm"
        at.run(timeout=30)

        rows = [m.value for m in at.markdown if ":violet-badge[🤖 LLM]" in m.value]
        self.assertEqual(len(rows), 1)


class TestSessionListBulkActions(unittest.TestCase):
    """Regression test for the widget-key/state desync bug: bulk actions
    (select/deselect all) must actually change what the checkboxes show,
    not just the underlying ControllerEntry.enabled — see
    streamlit_siso_panel.py's _render_session_list for the fix (the
    checkbox's own session_state key, not a value= argument, is the
    single source of truth after the first render)."""

    def test_deselect_all_then_select_all_sync_widgets(self):
        import streamlit_gui_state as gs

        at = _fresh_app()
        tab = _siso_tab(at)
        tab.button(key="siso_compare_all").click()
        at.run(timeout=60)
        self.assertFalse(at.exception)
        n = _n_entries(at)
        self.assertGreater(n, 1)

        tab = _siso_tab(at)
        tab.button(key="siso_deselect_all").click()
        at.run(timeout=30)
        entries = at.session_state[gs.CONTROLLERS_KEY]
        self.assertTrue(all(not e.enabled for e in entries))
        tab = _siso_tab(at)
        cbs = [c for c in tab.get("checkbox") if c.key and c.key.startswith("siso_en_")]
        self.assertTrue(all(c.value is False for c in cbs))

        tab = _siso_tab(at)
        tab.button(key="siso_select_all").click()
        at.run(timeout=30)
        entries = at.session_state[gs.CONTROLLERS_KEY]
        self.assertTrue(all(e.enabled for e in entries))
        tab = _siso_tab(at)
        cbs = [c for c in tab.get("checkbox") if c.key and c.key.startswith("siso_en_")]
        self.assertTrue(all(c.value is True for c in cbs))

    def test_remove_unchecked_and_clear_all(self):
        import streamlit_gui_state as gs

        at = _fresh_app()
        tab = _siso_tab(at)
        tab.button(key="siso_compare_all").click()
        at.run(timeout=60)
        n = _n_entries(at)

        tab = _siso_tab(at)
        cb = [c for c in tab.get("checkbox") if c.key and c.key.startswith("siso_en_")][0]
        cb.set_value(False)
        at.run(timeout=30)
        tab = _siso_tab(at)
        tab.button(key="siso_remove_unchecked").click()
        at.run(timeout=30)
        self.assertEqual(_n_entries(at), n - 1)

        tab = _siso_tab(at)
        tab.button(key="siso_clear_all").click()
        at.run(timeout=30)
        self.assertEqual(_n_entries(at), 0)


class TestPlantSurvivesATrackSwitch(unittest.TestCase):
    """Regression: Streamlit deletes a widget's session_state entry
    whenever that widget isn't instantiated on a script run -- since
    render_controls() only runs for the active Track, a typed-in plant/
    delay/method used to reset to the hardcoded default the moment the
    user switched to the MIMO track and back. See streamlit_gui_state.
    preserve_widget_state/snapshot_widget_state."""

    def test_plant_L_and_method_survive_a_round_trip_to_mimo_and_back(self):
        at = _fresh_app()
        at.text_input(key="siso_tf_expr").set_value("1/(90s+1)").run(timeout=30)
        at.number_input(key="siso_L").set_value(13.0).run(timeout=30)
        at.selectbox(key="siso_method").set_value("5. SIMC (FOPDT)").run(timeout=30)

        at.segmented_control(key="unified_track").set_value("MIMO / LQG").run(timeout=30)
        at.segmented_control(key="unified_track").set_value("SISO / PID").run(timeout=30)

        self.assertEqual(at.exception[:], [])
        self.assertEqual(at.text_input(key="siso_tf_expr").value, "1/(90s+1)")
        self.assertEqual(at.number_input(key="siso_L").value, 13.0)
        self.assertEqual(at.selectbox(key="siso_method").value, "5. SIMC (FOPDT)")


class TestMethodArgsSurviveAMethodSwitch(unittest.TestCase):
    """Regression: same class of bug as TestPlantSurvivesATrackSwitch above,
    but for a method switch rather than a Track switch -- only one
    method's args render at a time even within an active SISO session, so
    a typed-in Boyd Ms/Mt used to reset to the hardcoded default the
    moment the user selected a different method and back. See
    streamlit_gui_state.preserve_widget_state/snapshot_widget_state and
    streamlit_siso_panel.py's _METHOD_ARG_KEYS."""

    def test_boyd_args_survive_a_round_trip_to_another_method_and_back(self):
        at = _fresh_app()
        at.selectbox(key="siso_method").set_value("6. Boyd (convex-concave)").run(timeout=30)
        at.number_input(key="boyd_Ms").set_value(1.6).run(timeout=30)
        at.number_input(key="boyd_Mt").set_value(1.7).run(timeout=30)

        at.selectbox(key="siso_method").set_value("1. Stable pole cancellation").run(timeout=30)
        at.selectbox(key="siso_method").set_value("6. Boyd (convex-concave)").run(timeout=30)

        self.assertEqual(at.exception[:], [])
        self.assertEqual(at.number_input(key="boyd_Ms").value, 1.6)
        self.assertEqual(at.number_input(key="boyd_Mt").value, 1.7)


class TestAbsorbLlmRowsReusesOrResimulates(unittest.TestCase):
    """absorb_llm_rows must mirror _do_compare_all's reuse-vs-resimulate
    check (see both functions' docstrings): row["sim"] is always a
    wide-open-actuator trace, reusable as-is only when this panel's own
    sim settings can't tell the difference -- otherwise it has to
    re-simulate against the panel's actual u_min/u_max, same as a manual
    "Compare all methods" click would. Drives the real chat -> plot_calls
    -> absorb_llm_rows path (session.plot_calls appended by hand, same as
    test_streamlit_llm_panel.py's TestLlmEntriesJoinTheSharedSessionList)
    rather than calling absorb_llm_rows directly, since it now also
    depends on gs.preserve_widget_state() restoring this panel's sim
    settings from their shadow copies -- see its own docstring."""

    def test_constrained_actuator_limits_trigger_a_resimulation(self):
        import numpy as np
        import streamlit_gui_state as gs
        from supervisor_session_pid import Session
        from supervisor_tools_whitebox_pid import run_whitebox_benchmark

        real_result = run_whitebox_benchmark(DEFAULT_PLANT, return_sim=True)
        unconstrained = [r for r in real_result["_sim_rows"] if r.get("sim") is not None]
        # Not a vacuous check: at least one method's wide-open trace must
        # actually need bounds tighter than +/-0.005 for this test to
        # prove anything (this plant's large DC gain keeps every method's
        # unconstrained u small, so the bound has to be tight to bite).
        self.assertTrue(any(np.max(np.abs(r["sim"].u)) > 0.005 for r in unconstrained))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True), \
             patch("streamlit_llm_panel.load_dotenv"):
            at = AppTest.from_file(APP_PATH).run(timeout=30)
            at.number_input(key="u_min").set_value(-0.005).run(timeout=30)
            at.number_input(key="u_max").set_value(0.005).run(timeout=30)

            at.segmented_control(key="unified_mode").set_value("LLM Supervisor").run(timeout=30)
            session = at.session_state["llm_session_obj"]
            session.plot_calls.append({
                "kind": "siso", "plant": DEFAULT_PLANT, "delay": 0.0,
                "rows": real_result["_sim_rows"],
            })
            with patch.object(Session, "handle_user_message", return_value="ok"):
                at.chat_input[0].set_value("go").run(timeout=30)
            self.assertEqual(at.exception[:], [])

            entries = [e for e in at.session_state[gs.CONTROLLERS_KEY] if e.source == "llm"]
            self.assertGreater(len(entries), 0)
            for e in entries:
                self.assertLessEqual(float(np.max(e.sim.u)), 0.005 + 1e-9)
                self.assertGreaterEqual(float(np.min(e.sim.u)), -0.005 - 1e-9)

    def test_entries_carry_the_raw_reloadable_plant_identity(self):
        import streamlit_gui_state as gs
        from supervisor_session_pid import Session
        from supervisor_tools_whitebox_pid import run_whitebox_benchmark

        real_result = run_whitebox_benchmark("1/(90s+1)", delay=13.0, return_sim=True)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True), \
             patch("streamlit_llm_panel.load_dotenv"):
            at = AppTest.from_file(APP_PATH).run(timeout=30)
            at.segmented_control(key="unified_mode").set_value("LLM Supervisor").run(timeout=30)
            session = at.session_state["llm_session_obj"]
            session.plot_calls.append({
                "kind": "siso", "plant": "1/(90s+1)", "delay": 13.0,
                "rows": real_result["_sim_rows"],
            })
            with patch.object(Session, "handle_user_message", return_value="ok"):
                at.chat_input[0].set_value("go").run(timeout=30)
            self.assertEqual(at.exception[:], [])

            entries = [e for e in at.session_state[gs.CONTROLLERS_KEY] if e.source == "llm"]
            self.assertGreater(len(entries), 0)
            self.assertTrue(all(e.plant_tf == "1/(90s+1)" for e in entries))
            self.assertTrue(all(e.plant_L == 13.0 for e in entries))


class TestLlmPlantCarryoverAffordance(unittest.TestCase):
    """Regression: the Manual controls offer a one-click way to pick up
    the plant the LLM Supervisor last analyzed -- see
    streamlit_siso_panel._render_llm_plant_carryover(). Mirrors
    test_streamlit_llm_panel.TestApiKeySurvivesAModeSwitch's Mode
    round-trip style."""

    def test_load_this_plant_seeds_tf_expr_and_L_then_hides_itself(self):
        import streamlit_gui_state as gs
        from supervisor_session_pid import Session
        from supervisor_tools_whitebox_pid import run_whitebox_benchmark

        real_result = run_whitebox_benchmark("1/(90s+1)", delay=13.0, return_sim=True)

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True), \
             patch("streamlit_llm_panel.load_dotenv"):
            at = AppTest.from_file(APP_PATH).run(timeout=30)
            at.segmented_control(key="unified_mode").set_value("LLM Supervisor").run(timeout=30)
            session = at.session_state["llm_session_obj"]
            session.plot_calls.append({
                "kind": "siso", "plant": "1/(90s+1)", "delay": 13.0,
                "rows": real_result["_sim_rows"],
            })
            with patch.object(Session, "handle_user_message", return_value="Ran it."):
                at.chat_input[0].set_value("tune it").run(timeout=30)

            at.segmented_control(key="unified_mode").set_value("Manual").run(timeout=30)
            self.assertEqual(at.exception[:], [])

            self.assertNotEqual(at.text_input(key="siso_tf_expr").value, "1/(90s+1)")
            captions = [c.value for c in at.caption]
            self.assertTrue(any("LLM Supervisor last analyzed" in c for c in captions))

            at.button(key="siso_load_llm_plant").click().run(timeout=30)
            self.assertEqual(at.exception[:], [])
            self.assertEqual(at.text_input(key="siso_tf_expr").value, "1/(90s+1)")
            self.assertEqual(at.number_input(key="siso_L").value, 13.0)

            captions = [c.value for c in at.caption]
            self.assertFalse(any("LLM Supervisor last analyzed" in c for c in captions),
                             "affordance must not still offer to load the plant it just loaded")


if __name__ == "__main__":
    unittest.main()
