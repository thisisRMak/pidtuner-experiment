"""Unit tests for the LQR/LQG supervisor layer -- no live Ollama required.
LQG-flavored counterpart to test_supervisor_pid.py; same structure (tool
wrapper tests, priorities worksheet tests, session loop tests via a
scripted fake LLM client), no live-model end-to-end coverage.

Run with:
    python test_supervisor_lqg.py
or:
    python -m unittest test_supervisor_lqg -v
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from lqg_examples import list_examples

from supervisor_common_lqg import PRIORITY_CATEGORIES_LQG, LQGPrioritiesWorksheet
from supervisor_session_lqg import (
    FALLBACK_MESSAGE, PARTIAL_FALLBACK_PREFIX, PARTIAL_FALLBACK_SUFFIX, LQGSession,
)
from supervisor_tools_lqg import RUN_LQG_BENCHMARK_SCHEMA, run_lqg_benchmark


# ─────────────────────────────────────────────────────────────────────────────
# Tool wrapper function
# ─────────────────────────────────────────────────────────────────────────────

class TestRunLqgBenchmark(unittest.TestCase):
    def test_returns_four_rows(self):
        result = run_lqg_benchmark("aircraft_hall")
        self.assertTrue(result["ok"])
        self.assertEqual([r["name"] for r in result["rows"]],
                         ["LQR (suggested Q/R)", "Output-weighted LQR",
                          "Bryson's rule", "LQG (Kalman filter)"])

    def test_rows_are_json_safe_and_rounded(self):
        result = run_lqg_benchmark("aircraft_hall")
        for r in result["rows"]:
            self.assertTrue(r["stable"])
            self.assertTrue(r["all_checks_passed"])
            self.assertIsInstance(r["ISU"], float)
            self.assertNotIn("numpy", type(r["ISU"]).__module__)
            self.assertNotIn("numpy", type(r["K"][0][0]).__module__)

    def test_lqg_row_has_kalman_flag(self):
        result = run_lqg_benchmark("aircraft_hall")
        lqg_row = result["rows"][-1]
        self.assertIn("kalman_estimator_stable", lqg_row)
        self.assertTrue(lqg_row["kalman_estimator_stable"])
        # non-LQG rows shouldn't claim a Kalman filter they don't have
        for r in result["rows"][:-1]:
            self.assertNotIn("kalman_estimator_stable", r)

    def test_custom_x_max_u_max_used_for_bryson(self):
        default = run_lqg_benchmark("chemical_reactor")
        custom = run_lqg_benchmark("chemical_reactor", x_max=[1, 2, 3, 4], u_max=[5, 5])
        default_K = default["rows"][2]["K"]
        custom_K = custom["rows"][2]["K"]
        self.assertNotEqual(default_K, custom_K)

    def test_unknown_preset_reports_error_not_exception(self):
        result = run_lqg_benchmark("not_a_real_plant")
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_custom_plant_is_benchmarked(self):
        # 1/((s+1)(10s+1)) in controllable canonical form.
        result = run_lqg_benchmark(custom_plant={
            "A": [[0, 1], [-0.1, -1.1]], "B": [[0], [1]],
            "C": [[0.1, 0]], "D": [[0]],
        })
        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual([r["name"] for r in result["rows"]],
                         ["LQR (suggested Q/R)", "Output-weighted LQR",
                          "Bryson's rule", "LQG (Kalman filter)"])
        self.assertEqual((result["nx"], result["nu"], result["ny"]), (2, 1, 1))

    def test_custom_plant_return_sim_carries_matrix_literals(self):
        """streamlit_mimo_panel.py's "Load this plant" affordance needs
        the actual matrices back for a custom plant (unlike a preset,
        reloadable from plant_preset alone) -- see run_lqg_benchmark's
        own note next to _custom_plant_literals."""
        result = run_lqg_benchmark(custom_plant={
            "A": [[0, 1], [-0.1, -1.1]], "B": [[0], [1]],
            "C": [[0.1, 0]], "D": [[0]],
        }, return_sim=True)
        self.assertTrue(result["ok"], result.get("error"))
        literals = result["_custom_plant_literals"]
        self.assertEqual(literals["A"], "[0 1; -0.1 -1.1]")
        self.assertEqual(literals["B"], "[0; 1]")
        self.assertEqual(literals["C"], "[0.1 0]")
        self.assertEqual(literals["D"], "[0]")

    def test_preset_plant_return_sim_has_no_custom_plant_literals(self):
        """A preset is reloadable from plant_preset (the catalog key)
        alone -- no matrix text needed, unlike a custom plant."""
        result = run_lqg_benchmark("aircraft_hall", return_sim=True)
        self.assertTrue(result["ok"], result.get("error"))
        self.assertNotIn("_custom_plant_literals", result)

    def test_custom_plant_no_return_sim_has_no_custom_plant_literals(self):
        result = run_lqg_benchmark(custom_plant={
            "A": [[0, 1], [-0.1, -1.1]], "B": [[0], [1]], "C": [[0.1, 0]],
        })
        self.assertTrue(result["ok"], result.get("error"))
        self.assertNotIn("_custom_plant_literals", result)

    def test_custom_plant_D_defaults_to_zero(self):
        result = run_lqg_benchmark(custom_plant={
            "A": [[0, 1], [-0.1, -1.1]], "B": [[0], [1]], "C": [[0.1, 0]],
        })
        self.assertTrue(result["ok"], result.get("error"))

    def test_custom_plant_bad_shape_reports_error_not_exception(self):
        result = run_lqg_benchmark(custom_plant={
            "A": [[0, 1], [-0.1, -1.1]], "B": [[0]], "C": [[0.1, 0]],
        })
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_neither_plant_preset_nor_custom_plant_reports_error(self):
        result = run_lqg_benchmark()
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_both_plant_preset_and_custom_plant_reports_error(self):
        result = run_lqg_benchmark(plant_preset="aircraft_hall", custom_plant={
            "A": [[0]], "B": [[1]], "C": [[1]],
        })
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_schema_has_no_required_top_level_since_two_ways_to_pick_a_plant(self):
        params = RUN_LQG_BENCHMARK_SCHEMA["function"]["parameters"]
        self.assertNotIn("plant_preset", params.get("required", []))
        self.assertIn("custom_plant", params["properties"])

    def test_am_diag_omitted_skips_model_following(self):
        result = run_lqg_benchmark("aircraft_hall")
        names = [r["name"] for r in result["rows"]]
        self.assertNotIn("Implicit model-following", names)
        self.assertNotIn("Explicit model-following", names)

    def test_am_diag_given_adds_both_model_following_rows(self):
        result = run_lqg_benchmark("aircraft_hall", am_diag=[0.1, 0.07])
        self.assertTrue(result["ok"])
        names = [r["name"] for r in result["rows"]]
        self.assertEqual(names, ["LQR (suggested Q/R)", "Output-weighted LQR",
                                 "Bryson's rule", "LQG (Kalman filter)",
                                 "Implicit model-following", "Explicit model-following"])
        implicit_row, explicit_row = result["rows"][4], result["rows"][5]
        self.assertTrue(implicit_row["stable"])
        self.assertTrue(explicit_row["stable"])
        self.assertIn("K", implicit_row)
        self.assertIn("K1", explicit_row)
        self.assertIn("K2", explicit_row)
        self.assertNotIn("K", explicit_row)

    def test_am_diag_wrong_length_reports_error(self):
        result = run_lqg_benchmark("aircraft_hall", am_diag=[0.1, 0.07, 0.03])
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_am_diag_nonpositive_reports_error(self):
        result = run_lqg_benchmark("aircraft_hall", am_diag=[0.1, -0.07])
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_schema_includes_am_diag_and_q1_scale(self):
        props = RUN_LQG_BENCHMARK_SCHEMA["function"]["parameters"]["properties"]
        self.assertIn("am_diag", props)
        self.assertIn("q1_scale", props)

    def test_schema_includes_custom_weights_and_reference(self):
        props = RUN_LQG_BENCHMARK_SCHEMA["function"]["parameters"]["properties"]
        self.assertIn("Q_diag", props)
        self.assertIn("R_diag", props)
        self.assertIn("reference", props)

    def test_Q_diag_R_diag_adds_custom_row(self):
        result = run_lqg_benchmark("aircraft_hall", Q_diag=[1, 1, 1, 1, 1], R_diag=[0.1, 0.1])
        self.assertTrue(result["ok"])
        names = [r["name"] for r in result["rows"]]
        self.assertIn("Custom LQR (Q_diag/R_diag)", names)

    def test_Q_diag_without_R_diag_reports_error(self):
        result = run_lqg_benchmark("aircraft_hall", Q_diag=[1, 1, 1, 1, 1])
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_Q_diag_list_R_diag_list_adds_one_row_per_pair(self):
        result = run_lqg_benchmark("aircraft_hall",
                                    Q_diag_list=[[1, 1, 1, 1, 1], [5, 5, 1, 1, 1]],
                                    R_diag_list=[[1, 1], [1, 1]])
        self.assertTrue(result["ok"], result.get("error"))
        names = [r["name"] for r in result["rows"]]
        self.assertEqual(len(names), 6)
        self.assertTrue(any("Custom LQR 1" in n for n in names))
        self.assertTrue(any("Custom LQR 2" in n for n in names))

    def test_Q_diag_list_without_R_diag_list_reports_error(self):
        result = run_lqg_benchmark("aircraft_hall", Q_diag_list=[[1, 1, 1, 1, 1]])
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_schema_includes_sweep_params(self):
        props = RUN_LQG_BENCHMARK_SCHEMA["function"]["parameters"]["properties"]
        self.assertIn("Q_diag_list", props)
        self.assertIn("R_diag_list", props)

    def test_reference_adds_tracking_metrics(self):
        result = run_lqg_benchmark("aircraft_hall", reference=[1.0, -0.5])
        self.assertTrue(result["ok"])
        for row in result["rows"]:
            self.assertIn("tracking_metrics", row)
            self.assertEqual(len(row["tracking_metrics"]), 2)

    def test_no_reference_means_no_tracking_metrics_key(self):
        result = run_lqg_benchmark("aircraft_hall")
        for row in result["rows"]:
            self.assertNotIn("tracking_metrics", row)

    def test_reference_on_non_square_plant_reports_error(self):
        result = run_lqg_benchmark("chemical_reactor", reference=[1.0, 1.0, 1.0, 1.0])
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_iteration_workflow_different_weights_change_metrics(self):
        # The scenario the supervisor is actually meant to support: call
        # once, look at the result, propose different weights, call again.
        first = run_lqg_benchmark("aircraft_hall", Q_diag=[1, 1, 1, 1, 1],
                                  R_diag=[10, 10], reference=[1.0, -0.5])
        second = run_lqg_benchmark("aircraft_hall", Q_diag=[1, 1, 1, 1, 1],
                                   R_diag=[0.01, 0.01], reference=[1.0, -0.5])
        first_overshoot = first["rows"][-1]["tracking_metrics"][0]["Overshoot"]
        second_overshoot = second["rows"][-1]["tracking_metrics"][0]["Overshoot"]
        self.assertNotEqual(first_overshoot, second_overshoot)

    def test_schema_enum_matches_catalog(self):
        props = RUN_LQG_BENCHMARK_SCHEMA["function"]["parameters"]["properties"]
        self.assertEqual(set(props["plant_preset"]["enum"]), set(list_examples()))


# ─────────────────────────────────────────────────────────────────────────────
# Priorities worksheet
# ─────────────────────────────────────────────────────────────────────────────

class TestLQGPrioritiesWorksheet(unittest.TestCase):
    def test_partial_updates_leave_other_fields_unchanged(self):
        w = LQGPrioritiesWorksheet()
        w.update(top_priority="speed")
        w.update(hard_constraints=["no Kalman filter"])
        self.assertEqual(w.top_priority, "speed")
        self.assertEqual(w.hard_constraints, ["no Kalman filter"])

    def test_invalid_top_priority_rejected(self):
        w = LQGPrioritiesWorksheet()
        result = w.update(top_priority="vibes")
        self.assertFalse(result["ok"])
        self.assertIsNone(w.top_priority)

    def test_valid_top_priority_covers_all_categories(self):
        for cat in PRIORITY_CATEGORIES_LQG:
            w = LQGPrioritiesWorksheet()
            result = w.update(top_priority=cat)
            self.assertTrue(result["ok"])


# ─────────────────────────────────────────────────────────────────────────────
# Session loop plumbing (scripted fake LLM client, no live Ollama)
# ─────────────────────────────────────────────────────────────────────────────

def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def _response(content="", tool_calls=None):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))


class ScriptedClient:
    def __init__(self, script):
        self._script = list(script)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append((len(messages), sorted(t["function"]["name"] for t in (tools or []))))
        return self._script.pop(0)


def _fake_lqg_tool():
    def fn(plant_preset, x_max=None, u_max=None, return_sim=False):
        rows = [
            {"name": "LQR (suggested Q/R)", "stable": True, "K": [[1.0, 2.0]]},
            {"name": "LQG (Kalman filter)", "stable": False, "K": [[1.0, 2.0]]},
        ]
        result = {"ok": True, "plant_preset": plant_preset, "plant_name": f"Plant {plant_preset}",
                 "rows": rows}
        if return_sim:
            # Mirrors the real tool's return_sim=True shape: raw
            # ComparisonRow-like objects (object attributes, not dict
            # keys -- unlike the PID whitebox tool's rows), each with a
            # .sim attached (a stand-in object here -- the wrapper only
            # cares that it's not None).
            result["_sim_rows"] = [SimpleNamespace(name=r["name"], sim=SimpleNamespace())
                                   for r in rows]
        return result
    return (RUN_LQG_BENCHMARK_SCHEMA, fn)


class TestLQGSessionToolLoop(unittest.TestCase):
    def _session(self, script):
        return LQGSession(ScriptedClient(script), lqg_tool=_fake_lqg_tool())

    def test_no_tool_calls_returns_content_directly(self):
        session = self._session([_response(content="hello")])
        reply = session.handle_user_message("hi")
        self.assertEqual(reply, "hello")

    def test_benchmark_tool_available_from_the_first_turn(self):
        """Unlike the PID Session (gated on tf_known), the LQG tool needs no
        prior tool call to unlock -- it should be offered immediately."""
        session = self._session([_response(content="hi")])
        session.handle_user_message("go")
        first_call_tools = session.client.calls[0][1]
        self.assertIn("run_lqg_benchmark", first_call_tools)

    def test_benchmark_call_populates_known_stable_methods_only_for_stable_rows(self):
        script = [
            _response(tool_calls=[_tool_call("run_lqg_benchmark", {"plant_preset": "aircraft_hall"})]),
            _response(content="benchmarked"),
        ]
        session = self._session(script)
        session.handle_user_message("go")
        self.assertIn("LQR (suggested Q/R)", session.known_stable_methods)
        self.assertNotIn("LQG (Kalman filter)", session.known_stable_methods)

    def test_finalize_recommendation_rejects_unknown_method(self):
        script = [
            _response(tool_calls=[_tool_call("finalize_recommendation",
                                              {"method_name": "Made Up Method", "rationale": "because"})]),
            _response(content="ok"),
        ]
        session = self._session(script)
        session.handle_user_message("go")
        tool_msgs = [m for m in session.messages if isinstance(m, dict) and m.get("role") == "tool"]
        self.assertIn('"ok":false', tool_msgs[-1]["content"].replace(" ", ""))

    def test_max_tool_hops_cap_returns_fallback(self):
        script = [_response(tool_calls=[_tool_call("set_priorities", {"top_priority": "speed"})])] * 10
        session = self._session(script)
        reply = session.handle_user_message("keep calling tools forever")
        self.assertEqual(reply, FALLBACK_MESSAGE)

    def test_hop_exhaustion_after_successful_benchmarks_surfaces_last_result(self):
        """Even when every tool call succeeds, a model that keeps iterating
        -- exactly what the system prompt's own '3-5 iterations is normal,
        don't stop at the first proposal' guidance invites -- can still
        exhaust the hop budget before ever emitting plain content. See
        docs/memos/2026-09-07's hop-exhaustion memo. The reply must carry
        the last real benchmark result, not the content-free bare
        FALLBACK_MESSAGE from the case above (no benchmark ever
        succeeded)."""
        script = [_response(tool_calls=[
            _tool_call("run_lqg_benchmark", {"plant_preset": "aircraft_hall"})
        ])] * 10
        session = self._session(script)
        reply = session.handle_user_message("keep iterating forever")
        self.assertNotEqual(reply, FALLBACK_MESSAGE)
        self.assertTrue(reply.startswith(PARTIAL_FALLBACK_PREFIX))
        self.assertIn("LQR (suggested Q/R): stable", reply)
        self.assertIn("LQG (Kalman filter): UNSTABLE", reply)

    def test_hop_exhaustion_with_benchmark_invites_continuation(self):
        """The partial-fallback reply should read as an offer to keep
        going, not a dead end -- see docs/memos/2026-09-07's robustness
        memo (user feedback on a live transcript)."""
        script = [_response(tool_calls=[
            _tool_call("run_lqg_benchmark", {"plant_preset": "aircraft_hall"})
        ])] * 10
        session = self._session(script)
        reply = session.handle_user_message("keep iterating forever")
        self.assertTrue(reply.endswith(PARTIAL_FALLBACK_SUFFIX))
        self.assertIn("continue", PARTIAL_FALLBACK_SUFFIX.lower())

    def test_continuing_after_hop_exhaustion_does_not_crash_the_next_turn(self):
        """The internal message history after exhaustion ends mid tool-
        exchange (no final plain-content turn) -- confirm a follow-up user
        message still gets a normal reply rather than erroring, since
        nothing in the session appends the partial-fallback text itself
        back into the message history."""
        script = [_response(tool_calls=[
            _tool_call("run_lqg_benchmark", {"plant_preset": "aircraft_hall"})
        ])] * 10 + [_response(content="Continuing: here is my recommendation.")]
        session = self._session(script)
        session.handle_user_message("keep iterating forever")  # exhausts, returns partial fallback
        reply = session.handle_user_message("continue")
        self.assertEqual(reply, "Continuing: here is my recommendation.")

    def test_hop_exhaustion_with_only_failed_benchmarks_still_uses_bare_fallback(self):
        # _fake_lqg_tool's fn requires plant_preset; omitting it makes
        # every call raise (caught by _dispatch_tool as {"ok": False}),
        # so no benchmark ever "succeeds" -- the bare FALLBACK_MESSAGE
        # (no result to summarize) must still be what comes back.
        script = [_response(tool_calls=[_tool_call("run_lqg_benchmark", {})])] * 10
        session = self._session(script)
        reply = session.handle_user_message("keep failing forever")
        self.assertEqual(reply, FALLBACK_MESSAGE)


class TestLQGSessionPlotCalls(unittest.TestCase):
    """plot_calls -- the side-channel streamlit_llm_panel.py drains into
    plottable session entries after each turn (see LQGSession._wrap_
    benchmark's docstring). Must never leak into what the model sees.
    capture_plots=True here since these tests are exercising that
    capture; the default-off case (a plain CLI-style LQGSession) is
    covered separately below."""

    def _session(self, script):
        return LQGSession(ScriptedClient(script), lqg_tool=_fake_lqg_tool(), capture_plots=True)

    def test_benchmark_call_populates_plot_calls_with_sim_and_plant(self):
        script = [
            _response(tool_calls=[_tool_call("run_lqg_benchmark", {"plant_preset": "aircraft_hall"})]),
            _response(content="done"),
        ]
        session = self._session(script)
        session.handle_user_message("go")
        self.assertEqual(len(session.plot_calls), 1)
        call = session.plot_calls[0]
        self.assertEqual(call["kind"], "mimo")
        # plant_name, not the plant_preset key -- see test_custom_plant_
        # gets_its_own_name_not_the_generic_preset_key below for why.
        self.assertEqual(call["plant"], "Plant aircraft_hall")
        self.assertEqual(call["plant_preset"], "aircraft_hall")
        self.assertEqual(call["rows"][0].name, "LQR (suggested Q/R)")
        self.assertIsNotNone(call["rows"][0].sim)
        # _fake_lqg_tool never sets _custom_plant_literals -- see
        # test_custom_plant_literals_land_in_plot_calls_not_the_model
        # below for the field actually populated.
        self.assertIsNone(call["custom_plant_literals"])

    def test_custom_plant_gets_its_own_name_not_the_generic_preset_key(self):
        """Regression: _wrap_benchmark used to tag every custom-plant call
        with the literal string "custom" (the constant _build_custom_
        example always uses for plant_preset/ex.key), making two different
        custom plants benchmarked in one conversation indistinguishable in
        the session list. Now tags with plant_name instead, which is
        always the actual, distinguishing name."""
        def fn(custom_plant, return_sim=False):
            result = {"ok": True, "plant_preset": "custom",
                      "plant_name": custom_plant.get("name") or "Custom plant", "rows": []}
            if return_sim:
                result["_sim_rows"] = []
            return result

        session = LQGSession(ScriptedClient([
            _response(tool_calls=[_tool_call("run_lqg_benchmark",
                                             {"custom_plant": {"name": "My widget", "A": [[0]],
                                                               "B": [[1]], "C": [[1]]}})]),
            _response(content="done"),
        ]), lqg_tool=(RUN_LQG_BENCHMARK_SCHEMA, fn), capture_plots=True)
        session.handle_user_message("go")
        # No rows -- sim_rows is [] (falsy-but-not-None), so plot_calls
        # still gets populated per the "is not None" guard.
        self.assertEqual(session.plot_calls[0]["plant"], "My widget")

    def test_custom_plant_literals_land_in_plot_calls_not_the_model(self):
        """_custom_plant_literals must reach plot_calls (for streamlit_
        mimo_panel.py's "Load this plant" affordance) but never the JSON
        sent to the model -- same treatment as _sim_rows."""
        def fn(custom_plant, return_sim=False):
            result = {"ok": True, "plant_preset": "custom",
                      "plant_name": custom_plant.get("name") or "Custom plant", "rows": []}
            if return_sim:
                result["_sim_rows"] = []
                result["_custom_plant_literals"] = {
                    "A": "[0 1; -2 -3]", "B": "[0; 1]", "C": "[1 0]", "D": "[0]"}
            return result

        session = LQGSession(ScriptedClient([
            _response(tool_calls=[_tool_call("run_lqg_benchmark",
                                             {"custom_plant": {"name": "My widget", "A": [[0]],
                                                               "B": [[1]], "C": [[1]]}})]),
            _response(content="done"),
        ]), lqg_tool=(RUN_LQG_BENCHMARK_SCHEMA, fn), capture_plots=True)
        session.handle_user_message("go")

        call = session.plot_calls[0]
        self.assertEqual(call["plant_preset"], "custom")
        self.assertEqual(call["custom_plant_literals"],
                         {"A": "[0 1; -2 -3]", "B": "[0; 1]", "C": "[1 0]", "D": "[0]"})

        tool_msgs = [m for m in session.messages if isinstance(m, dict) and m.get("role") == "tool"]
        benchmark_msg = next(m for m in tool_msgs if m["tool_name"] == "run_lqg_benchmark")
        self.assertNotIn("_custom_plant_literals", benchmark_msg["content"])

    def test_tool_result_sent_to_the_model_is_json_safe(self):
        script = [
            _response(tool_calls=[_tool_call("run_lqg_benchmark", {"plant_preset": "aircraft_hall"})]),
            _response(content="done"),
        ]
        session = self._session(script)
        session.handle_user_message("go")
        tool_msgs = [m for m in session.messages if isinstance(m, dict) and m.get("role") == "tool"]
        benchmark_msg = next(m for m in tool_msgs if m["tool_name"] == "run_lqg_benchmark")
        self.assertNotIn("_sim_rows", benchmark_msg["content"])
        json.loads(benchmark_msg["content"])  # must not have raised building it, either


class TestLQGSessionPlotCallsDefaultOff(unittest.TestCase):
    """capture_plots defaults to False -- a plain CLI-style LQGSession
    (cli_supervisor_lqg.py, which never drains plot_calls) must never
    accumulate simulated trajectories for the life of the process."""

    def test_benchmark_call_never_populates_plot_calls_by_default(self):
        script = [
            _response(tool_calls=[_tool_call("run_lqg_benchmark", {"plant_preset": "aircraft_hall"})]),
            _response(content="done"),
        ]
        session = LQGSession(ScriptedClient(script), lqg_tool=_fake_lqg_tool())
        session.handle_user_message("go")
        self.assertEqual(session.plot_calls, [])


if __name__ == "__main__":
    unittest.main()
