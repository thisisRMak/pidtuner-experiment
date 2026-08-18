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

import unittest

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
    return at.tabs[0]


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


if __name__ == "__main__":
    unittest.main()
