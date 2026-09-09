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

import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from lqg_examples import list_examples

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
    # MIMO/LQG isn't streamlit_unified_panel.py's default Track (SISO/PID
    # is) -- select it once here so every test below finds MIMO's
    # controls in the tree, same as before when both tabs' widgets
    # existed unconditionally.
    at.segmented_control(key="unified_track").set_value("MIMO / LQG").run(timeout=30)
    return at


def _mimo_tab(at):
    return at


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


class TestPerChannelStep(unittest.TestCase):
    def test_per_channel_step_after_reference_tracking_design(self):
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
        tab = _mimo_tab(at)
        tab.button(key="mimo_design").click()
        at.run(timeout=60)
        self.assertFalse(at.exception)

        tab = _mimo_tab(at)
        tab.button(key="mimo_per_channel_step_btn").click()
        at.run(timeout=60)
        self.assertFalse(at.exception, "per-channel step response should not raise")
        self.assertIn("mimo_per_channel_step", at.session_state)

    def test_per_channel_step_with_t_max_y_max(self):
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
        tab = _mimo_tab(at)
        tab.button(key="mimo_design").click()
        at.run(timeout=60)
        self.assertFalse(at.exception)

        tab = _mimo_tab(at)
        _set_widget(tab, "mimo_pcs_t_max", "5")
        _set_widget(tab, "mimo_pcs_y_max", "1.1")
        at.run(timeout=30)
        tab = _mimo_tab(at)
        tab.button(key="mimo_per_channel_step_btn").click()
        at.run(timeout=60)
        self.assertFalse(at.exception, "t_max/y_max should not raise")
        _, _, t_max, y_max = at.session_state["mimo_per_channel_step"]
        self.assertEqual(t_max, 5.0)
        self.assertEqual(y_max, 1.1)

    def test_per_channel_step_without_reference_tracking_fails_cleanly(self):
        at = _fresh_app()
        tab = _mimo_tab(at)
        tab.button(key="mimo_design").click()
        at.run(timeout=60)
        self.assertFalse(at.exception)

        tab = _mimo_tab(at)
        tab.button(key="mimo_per_channel_step_btn").click()
        at.run(timeout=30)
        self.assertFalse(at.exception, "missing reference tracking should not raise")
        errs = [e.value for e in _mimo_tab(at).error]
        self.assertTrue(any("Reference tracking" in e for e in errs), f"expected a reference-tracking error, got {errs}")


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


class TestLlmEntryTagIsABadge(unittest.TestCase):
    """Regression: same fix as streamlit_siso_panel.py's -- the 🤖 LLM
    tag now renders as a :violet-badge[...] pill instead of plain
    markdown text, so it's not tiny/monochrome next to the swatch."""

    def test_llm_sourced_entry_renders_a_violet_badge(self):
        import streamlit_gui_state as gs

        at = _fresh_app()
        tab = _mimo_tab(at)
        tab.button(key="mimo_compare_all").click()
        at.run(timeout=60)
        entries = [e for e in at.session_state[gs.CONTROLLERS_KEY] if e.kind == "mimo"]
        self.assertGreater(len(entries), 0)
        entries[0].source = "llm"
        at.run(timeout=30)

        rows = [m.value for m in at.markdown if ":violet-badge[🤖 LLM]" in m.value]
        self.assertEqual(len(rows), 1)


class TestLlmPlantCarryoverAffordance(unittest.TestCase):
    """Regression: the Manual controls offer a one-click way to pick up
    the custom plant the LLM Supervisor last analyzed -- see
    streamlit_mimo_panel._render_llm_plant_carryover(). Mirrors
    test_streamlit_siso_panel.TestLlmPlantCarryoverAffordance's own test,
    but for MIMO's custom-plant A/B/C/D widgets rather than SISO's
    tf_expr/L (a preset plant reloads via plant_preset alone -- see
    gs.ControllerEntry's own note -- so the custom-plant path is the one
    that actually needs the matrix-literal round trip exercised here)."""

    def test_load_this_plant_seeds_custom_matrices_then_hides_itself(self):
        import streamlit_gui_state as gs
        from supervisor_session_lqg import LQGSession
        from supervisor_tools_lqg import run_lqg_benchmark

        custom_plant = {"name": "My widget", "A": [[0, 1], [-2, -3]], "B": [[0], [1]],
                        "C": [[1, 0]], "D": [[0]]}
        real_result = run_lqg_benchmark(custom_plant=custom_plant, return_sim=True)
        self.assertTrue(real_result["ok"], real_result.get("error"))

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True), \
             patch("streamlit_llm_panel.load_dotenv"):
            at = AppTest.from_file(APP_PATH).run(timeout=30)
            at.segmented_control(key="unified_track").set_value("MIMO / LQG").run(timeout=30)
            at.segmented_control(key="unified_mode").set_value("LLM Supervisor").run(timeout=30)
            session = at.session_state["llm_session_obj"]
            session.plot_calls.append({
                "kind": "mimo", "plant": real_result["plant_name"],
                "plant_preset": real_result["plant_preset"],
                "custom_plant_literals": real_result["_custom_plant_literals"],
                "rows": real_result["_sim_rows"],
            })
            with patch.object(LQGSession, "handle_user_message", return_value="Ran it."):
                at.chat_input[0].set_value("tune it").run(timeout=30)

            at.segmented_control(key="unified_mode").set_value("Manual").run(timeout=30)
            self.assertEqual(at.exception[:], [])

            self.assertEqual(at.radio(key="mimo_plant_source").value, "Preset")
            captions = [c.value for c in at.caption]
            self.assertTrue(any("LLM Supervisor last analyzed" in c for c in captions))

            at.button(key="mimo_load_llm_plant").click().run(timeout=30)
            self.assertEqual(at.exception[:], [])
            self.assertEqual(at.radio(key="mimo_plant_source").value,
                             "Custom (MATLAB matrix entry)")
            self.assertEqual(at.text_area(key="mimo_custom_A").value, "[0 1; -2 -3]")
            self.assertEqual(at.text_area(key="mimo_custom_B").value, "[0; 1]")
            self.assertEqual(at.text_area(key="mimo_custom_C").value, "[1 0]")
            self.assertEqual(at.text_area(key="mimo_custom_D").value, "[0]")

            captions = [c.value for c in at.caption]
            self.assertFalse(any("LLM Supervisor last analyzed" in c for c in captions),
                             "affordance must not still offer to load the plant it just loaded")


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

        # Not _fresh_app() -- that starts on MIMO/LQG, but this needs to
        # start on SISO/PID (the unified panel's actual default) to
        # populate SISO entries before ever switching Track.
        at = AppTest.from_file(APP_PATH)
        at.run(timeout=30)
        at.button(key="siso_compare_all").click()
        at.run(timeout=60)
        n_siso = len([e for e in at.session_state[gs.CONTROLLERS_KEY] if e.kind == "siso"])
        self.assertGreater(n_siso, 0)

        at.segmented_control(key="unified_track").set_value("MIMO / LQG").run(timeout=30)
        mimo_tab = _mimo_tab(at)
        mimo_tab.button(key="mimo_compare_all").click()
        at.run(timeout=60)

        mimo_tab = _mimo_tab(at)
        mimo_tab.button(key="mimo_clear_all").click()
        at.run(timeout=30)

        entries = at.session_state[gs.CONTROLLERS_KEY]
        self.assertEqual(len([e for e in entries if e.kind == "mimo"]), 0)
        self.assertEqual(len([e for e in entries if e.kind == "siso"]), n_siso)


class TestPresetSurvivesATrackSwitch(unittest.TestCase):
    """Regression: same class of bug as test_streamlit_siso_panel.py's
    TestPlantSurvivesATrackSwitch, MIMO side -- see streamlit_gui_state.
    preserve_widget_state/snapshot_widget_state."""

    def test_preset_and_method_survive_a_round_trip_to_siso_and_back(self):
        at = _fresh_app()  # starts on MIMO/LQG
        presets = list_examples()
        non_default = next(p for p in presets if p != DEFAULT_PRESET)
        _mimo_tab(at).selectbox(key="mimo_preset").set_value(non_default).run(timeout=30)
        _mimo_tab(at).selectbox(key="mimo_method").set_value("Bryson's rule").run(timeout=30)

        at.segmented_control(key="unified_track").set_value("SISO / PID").run(timeout=30)
        at.segmented_control(key="unified_track").set_value("MIMO / LQG").run(timeout=30)

        self.assertEqual(at.exception[:], [])
        self.assertEqual(at.selectbox(key="mimo_preset").value, non_default)
        self.assertEqual(at.selectbox(key="mimo_method").value, "Bryson's rule")


class TestMethodArgsSurviveAMethodSwitch(unittest.TestCase):
    """Regression: same class of bug as TestPresetSurvivesATrackSwitch
    above, but for a method switch rather than a Track switch -- only one
    method's args render at a time even within an active MIMO session, so
    a typed-in custom Q diagonal used to reset to the hardcoded default
    the moment the user selected a different method and back. See
    streamlit_gui_state.preserve_widget_state/snapshot_widget_state and
    streamlit_mimo_panel.py's _METHOD_ARG_KEYS."""

    def test_custom_qr_args_survive_a_round_trip_to_another_method_and_back(self):
        at = _fresh_app()  # starts on MIMO/LQG
        tab = _mimo_tab(at)
        tab.selectbox(key="mimo_method").set_value("LQR (custom Q/R diagonal)").run(timeout=30)
        tab = _mimo_tab(at)
        tab.text_input(key="mimo_Q_diag").set_value("2.0").run(timeout=30)
        tab = _mimo_tab(at)
        tab.text_input(key="mimo_R_diag").set_value("3.0").run(timeout=30)

        tab = _mimo_tab(at)
        tab.selectbox(key="mimo_method").set_value("Bryson's rule").run(timeout=30)
        tab = _mimo_tab(at)
        tab.selectbox(key="mimo_method").set_value("LQR (custom Q/R diagonal)").run(timeout=30)

        self.assertEqual(at.exception[:], [])
        self.assertEqual(at.text_input(key="mimo_Q_diag").value, "2.0")
        self.assertEqual(at.text_input(key="mimo_R_diag").value, "3.0")


if __name__ == "__main__":
    unittest.main()
