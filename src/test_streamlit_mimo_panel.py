"""Regression tests for the Streamlit MIMO LQR/LQG panel — see
test_streamlit_siso_panel.py's module docstring for what AppTest does
and does not cover; the same caveats apply here (no real browser, no
side-by-side diff against a CLI/Tkinter equivalent since there isn't
one to diff against for this panel — cli_lqg.py is text/JSON/PNG only).

Run with:
    python test_streamlit_mimo_panel.py
or:
    python -m unittest test_streamlit_mimo_panel -v
"""

import unittest

from streamlit.testing.v1 import AppTest

APP_PATH = __file__.replace("test_streamlit_mimo_panel.py", "streamlit_app.py")

# 'airc' is the default (alphabetically first) preset — a small aircraft
# model, cheap to design/simulate repeatedly across every test here.
DEFAULT_PRESET = "airc"

METHOD_CASES = [
    ("LQR (suggested Q/R)", {}),
    ("LQR (custom Q/R diagonal)", {"mimo_Q_diag": "1.0", "mimo_R_diag": "1.0"}),
    ("Output-weighted LQR", {"mimo_Qy_scale": 2.0, "mimo_ow_R_scale": 0.5}),
    ("Bryson's rule", {"mimo_x_max": "1.0", "mimo_u_max": "1.0"}),
    ("LQG (Kalman filter)", {"mimo_Qw_scale": 0.05, "mimo_Rv_scale": 0.2}),
]


def _fresh_app():
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=30)
    return at


def _mimo_tab(at):
    return at.tabs[1]


def _n_entries(at):
    import streamlit_gui_state as gs
    return len([e for e in at.session_state[gs.CONTROLLERS_KEY] if e.kind == "mimo"])


def _set_widget(tab, key, value):
    for kind in ("number_input", "text_input", "radio", "checkbox", "selectbox"):
        try:
            getattr(tab, kind)(key=key).set_value(value)
            return
        except Exception:
            continue
    raise AssertionError(f"no widget found for key={key!r}")


class TestAllMethodsThroughUI(unittest.TestCase):
    def test_all_five_methods(self):
        at = _fresh_app()
        for method, args in METHOD_CASES:
            tab = _mimo_tab(at)
            tab.selectbox(key="mimo_method").set_value(method)
            at.run(timeout=30)
            self.assertFalse(at.exception, f"{method}: exception after selecting method")

            tab = _mimo_tab(at)
            for k, v in args.items():
                _set_widget(tab, k, v)
            at.run(timeout=30)
            self.assertFalse(at.exception, f"{method}: exception after setting args")

            before = _n_entries(at)
            tab = _mimo_tab(at)
            tab.button(key="mimo_design").click()
            at.run(timeout=60)
            self.assertFalse(at.exception, f"{method}: exception after Design & simulate")
            self.assertEqual(_n_entries(at), before + 1, f"{method}: expected one new entry")


class TestCompareAll(unittest.TestCase):
    def test_compare_all_regulator_family(self):
        at = _fresh_app()
        tab = _mimo_tab(at)
        tab.button(key="mimo_compare_all").click()
        at.run(timeout=60)
        self.assertFalse(at.exception)
        self.assertEqual(_n_entries(at), 4)  # LQR, output-weighted, Bryson, LQG


class TestReferenceTracking(unittest.TestCase):
    def test_reference_tracking_on_square_plant(self):
        # airc is not guaranteed square; find a preset with nu == ny.
        from lqg_examples import list_examples, load_example
        square_key = None
        for key in list_examples():
            ex = load_example(key)
            if ex.plant.nu == ex.plant.ny:
                square_key = key
                break
        if square_key is None:
            self.skipTest("no square-plant preset available (nu == ny)")

        at = _fresh_app()
        tab = _mimo_tab(at)
        tab.selectbox(key="mimo_preset").set_value(square_key)
        at.run(timeout=30)
        tab = _mimo_tab(at)
        tab.checkbox(key="mimo_ref_tracking").set_value(True)
        at.run(timeout=30)
        self.assertFalse(at.exception)

        before = _n_entries(at)
        tab = _mimo_tab(at)
        tab.button(key="mimo_design").click()
        at.run(timeout=60)
        self.assertFalse(at.exception)
        self.assertEqual(_n_entries(at), before + 1)

        import streamlit_gui_state as gs
        entries = [e for e in at.session_state[gs.CONTROLLERS_KEY] if e.kind == "mimo"]
        self.assertIsNotNone(entries[-1].sim.tracking_metrics)

    def test_reference_tracking_on_nonsquare_plant_fails_cleanly(self):
        from lqg_examples import list_examples, load_example
        nonsquare_key = None
        for key in list_examples():
            ex = load_example(key)
            if ex.plant.nu != ex.plant.ny:
                nonsquare_key = key
                break
        if nonsquare_key is None:
            self.skipTest("no non-square-plant preset available")

        at = _fresh_app()
        tab = _mimo_tab(at)
        tab.selectbox(key="mimo_preset").set_value(nonsquare_key)
        at.run(timeout=30)
        tab = _mimo_tab(at)
        tab.checkbox(key="mimo_ref_tracking").set_value(True)
        at.run(timeout=30)

        before = _n_entries(at)
        tab = _mimo_tab(at)
        tab.button(key="mimo_design").click()
        at.run(timeout=30)
        self.assertFalse(at.exception, "non-square reference tracking should not raise")
        self.assertEqual(_n_entries(at), before, "no entry should be added")
        errs = [e.value for e in _mimo_tab(at).error]
        self.assertTrue(any("square" in e for e in errs), f"expected a square-plant error, got {errs}")


class TestErrorPaths(unittest.TestCase):
    def test_malformed_broadcast_field_fails_cleanly(self):
        at = _fresh_app()
        tab = _mimo_tab(at)
        tab.selectbox(key="mimo_method").set_value("LQR (custom Q/R diagonal)")
        at.run(timeout=30)
        tab = _mimo_tab(at)
        tab.text_input(key="mimo_Q_diag").set_value("not a number")
        at.run(timeout=30)
        self.assertFalse(at.exception)

        before = _n_entries(at)
        tab = _mimo_tab(at)
        tab.button(key="mimo_design").click()
        at.run(timeout=30)
        self.assertFalse(at.exception)
        self.assertEqual(_n_entries(at), before)

    def test_wrong_length_broadcast_field_fails_cleanly(self):
        at = _fresh_app()
        tab = _mimo_tab(at)
        tab.selectbox(key="mimo_method").set_value("LQR (custom Q/R diagonal)")
        at.run(timeout=30)
        tab = _mimo_tab(at)
        # airc's nx is > 2 (aircraft models), so exactly 2 values is neither
        # "1 to broadcast" nor "nx values" — should fail the length check.
        tab.text_input(key="mimo_Q_diag").set_value("1.0 2.0")
        at.run(timeout=30)

        before = _n_entries(at)
        tab = _mimo_tab(at)
        tab.button(key="mimo_design").click()
        at.run(timeout=30)
        self.assertFalse(at.exception)
        self.assertEqual(_n_entries(at), before)


class TestSessionListBulkActions(unittest.TestCase):
    def test_deselect_all_then_select_all_sync_widgets(self):
        import streamlit_gui_state as gs

        at = _fresh_app()
        tab = _mimo_tab(at)
        tab.button(key="mimo_compare_all").click()
        at.run(timeout=60)
        self.assertFalse(at.exception)

        tab = _mimo_tab(at)
        tab.button(key="mimo_deselect_all").click()
        at.run(timeout=30)
        entries = [e for e in at.session_state[gs.CONTROLLERS_KEY] if e.kind == "mimo"]
        self.assertTrue(all(not e.enabled for e in entries))
        tab = _mimo_tab(at)
        cbs = [c for c in tab.get("checkbox") if c.key and c.key.startswith("mimo_en_")]
        self.assertTrue(all(c.value is False for c in cbs))

        tab = _mimo_tab(at)
        tab.button(key="mimo_select_all").click()
        at.run(timeout=30)
        entries = [e for e in at.session_state[gs.CONTROLLERS_KEY] if e.kind == "mimo"]
        self.assertTrue(all(e.enabled for e in entries))

    def test_clear_all(self):
        at = _fresh_app()
        tab = _mimo_tab(at)
        tab.button(key="mimo_compare_all").click()
        at.run(timeout=60)
        self.assertGreater(_n_entries(at), 0)

        tab = _mimo_tab(at)
        tab.button(key="mimo_clear_all").click()
        at.run(timeout=30)
        self.assertEqual(_n_entries(at), 0)


class TestSisoMimoStateIsolation(unittest.TestCase):
    """The two panels share one controllers list, keyed by `kind` — make
    sure MIMO actions never touch SISO entries and vice versa."""

    def test_mimo_clear_all_does_not_touch_siso_entries(self):
        import streamlit_gui_state as gs

        at = _fresh_app()
        siso_tab = at.tabs[0]
        siso_tab.button(key="siso_compare_all").click()
        at.run(timeout=60)
        n_siso = len([e for e in at.session_state[gs.CONTROLLERS_KEY] if e.kind == "siso"])
        self.assertGreater(n_siso, 0)

        mimo_tab = _mimo_tab(at)
        mimo_tab.button(key="mimo_compare_all").click()
        at.run(timeout=60)

        mimo_tab = _mimo_tab(at)
        mimo_tab.button(key="mimo_clear_all").click()
        at.run(timeout=30)

        entries = at.session_state[gs.CONTROLLERS_KEY]
        self.assertEqual(len([e for e in entries if e.kind == "mimo"]), 0)
        self.assertEqual(len([e for e in entries if e.kind == "siso"]), n_siso)


if __name__ == "__main__":
    unittest.main()
