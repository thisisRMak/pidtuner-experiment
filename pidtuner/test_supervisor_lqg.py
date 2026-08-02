"""Unit tests for the LQR/LQG supervisor layer -- no live Ollama required.
LQG-flavored counterpart to test_supervisor.py; same structure (tool
wrapper tests, priorities worksheet tests, session loop tests via a
scripted fake LLM client), no live-model end-to-end coverage.

Run with:
    python test_supervisor_lqg.py
or:
    python -m unittest test_supervisor_lqg -v
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from lqg_examples import list_examples

from supervisor_common_lqg import PRIORITY_CATEGORIES_LQG, LQGPrioritiesWorksheet
from supervisor_session_lqg import FALLBACK_MESSAGE, LQGSession
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
    def fn(plant_preset, x_max=None, u_max=None):
        return {"ok": True, "rows": [
            {"name": "LQR (suggested Q/R)", "stable": True, "K": [[1.0, 2.0]]},
            {"name": "LQG (Kalman filter)", "stable": False, "K": [[1.0, 2.0]]},
        ]}
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


if __name__ == "__main__":
    unittest.main()
