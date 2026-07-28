"""Unit tests for the LLM supervisor layer -- no live Ollama required.

Run with:
    python test_supervisor.py
or:
    python -m unittest test_supervisor -v

Covers everything deterministic: the tool wrapper functions, entity
isolation (supervisor_tools_blackbox.py must never import plant.py, same
contract as blackbox.py/signal_format.py), the priorities worksheet, and the
Session tool-call loop plumbing via a scripted fake LLM client. End-to-end
behavior against the real local qwen3-coder:30b model is a separate, manual
verification step (see docs/refactor_prompt.md discussion / plan) -- it
can't be made deterministic and isn't covered here.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

from compare import compare_all_methods
from plant import TransferFunction
from signal_format import save_signal
from signal_source import SignalGenerator

from supervisor_common import PRIORITY_CATEGORIES, PrioritiesWorksheet
from supervisor_session import FALLBACK_MESSAGE, Session
from supervisor_tools_blackbox import RUN_BLACKBOX_BENCHMARK_SCHEMA, run_blackbox_benchmark
from supervisor_tools_whitebox import RUN_WHITEBOX_BENCHMARK_SCHEMA, run_whitebox_benchmark
from test_pid_tuner import _module_imports_plant, benchmark_plant


# ─────────────────────────────────────────────────────────────────────────────
# Entity isolation
# ─────────────────────────────────────────────────────────────────────────────

class TestSupervisorToolIsolation(unittest.TestCase):
    def test_no_plant_import(self):
        import supervisor_tools_blackbox
        self.assertFalse(
            _module_imports_plant(supervisor_tools_blackbox.__file__),
            "supervisor_tools_blackbox.py must never import plant.py",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tool wrapper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestRunWhiteboxBenchmark(unittest.TestCase):
    def test_matches_compare_all_methods_row_count(self):
        direct_rows = compare_all_methods(benchmark_plant(), include_variants=True)
        result = run_whitebox_benchmark("1000 / ((s+1)*(10s+1))", delay=0.5)
        self.assertTrue(result["ok"])
        self.assertEqual(len(result["rows"]), len(direct_rows))
        self.assertEqual([r["name"] for r in result["rows"]], [r["name"] for r in direct_rows])

    def test_rows_are_json_safe_and_rounded(self):
        result = run_whitebox_benchmark("1000 / ((s+1)*(10s+1))", delay=0.5)
        stable_rows = [r for r in result["rows"] if r["stable"]]
        self.assertTrue(stable_rows)
        for r in stable_rows:
            self.assertIsInstance(r["gains"]["Kp"], float)
            self.assertIsInstance(r["OS%"], (float, int, type(None)))
            # No numpy scalar types should have leaked through.
            self.assertNotIn("numpy", type(r["gains"]["Kp"]).__module__)

    def test_bad_plant_string_reports_error_not_exception(self):
        result = run_whitebox_benchmark("not a valid transfer function (((")
        self.assertFalse(result["ok"])
        self.assertIn("error", result)


class TestRunBlackboxBenchmark(unittest.TestCase):
    def _signal_paths(self, tmpdir):
        plant = benchmark_plant()
        gen = SignalGenerator(plant)
        step_path = os.path.join(tmpdir, "step.npz")
        relay_path = os.path.join(tmpdir, "relay.npz")
        save_signal(gen.step_test(step_amp=1.0), step_path)
        save_signal(gen.relay_test(h=1.0), relay_path)
        return step_path, relay_path

    def test_returns_rows_and_model(self):
        with tempfile.TemporaryDirectory() as d:
            step_path, relay_path = self._signal_paths(d)
            result = run_blackbox_benchmark(step_signal_path=step_path, relay_signal_path=relay_path)
        self.assertTrue(result["ok"])
        self.assertIn("model", result)
        self.assertTrue(result["rows"])
        available = [r for r in result["rows"] if r["available"]]
        self.assertTrue(available)
        for r in available:
            self.assertTrue(r["black_box"])

    def test_no_signal_paths_reports_error(self):
        result = run_blackbox_benchmark()
        self.assertFalse(result["ok"])
        self.assertIn("error", result)

    def test_bad_path_reports_error_not_exception(self):
        result = run_blackbox_benchmark(step_signal_path="/nonexistent/path.npz")
        self.assertFalse(result["ok"])
        self.assertIn("error", result)


# ─────────────────────────────────────────────────────────────────────────────
# Priorities worksheet
# ─────────────────────────────────────────────────────────────────────────────

class TestPrioritiesWorksheet(unittest.TestCase):
    def test_partial_updates_leave_other_fields_unchanged(self):
        w = PrioritiesWorksheet()
        w.update(tf_known=True)
        w.update(top_priority="speed")
        self.assertTrue(w.tf_known)
        self.assertEqual(w.top_priority, "speed")

    def test_invalid_top_priority_rejected(self):
        w = PrioritiesWorksheet()
        result = w.update(top_priority="vibes")
        self.assertFalse(result["ok"])
        self.assertIsNone(w.top_priority)

    def test_valid_top_priority_covers_all_categories(self):
        for cat in PRIORITY_CATEGORIES:
            w = PrioritiesWorksheet()
            result = w.update(top_priority=cat)
            self.assertTrue(result["ok"])

    def test_tf_known_locked_after_first_set(self):
        w = PrioritiesWorksheet()
        w.update(tf_known=True)
        result = w.update(tf_known=False)
        self.assertFalse(result["ok"])
        self.assertTrue(w.tf_known)  # unchanged

    def test_tf_known_repeated_same_value_is_fine(self):
        w = PrioritiesWorksheet()
        w.update(tf_known=True)
        result = w.update(tf_known=True)
        self.assertTrue(result["ok"])


# ─────────────────────────────────────────────────────────────────────────────
# Session loop plumbing (scripted fake LLM client, no live Ollama)
# ─────────────────────────────────────────────────────────────────────────────

def _tool_call(name, arguments):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def _response(content="", tool_calls=None):
    return SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=tool_calls))


class ScriptedClient:
    """Fake OllamaClient: .chat() returns the next pre-programmed response,
    ignoring the actual messages/tools passed in (recorded for assertions)."""

    def __init__(self, script):
        self._script = list(script)
        self.calls = []  # list of (messages_snapshot_len, tool_names) per call

    def chat(self, messages, tools=None):
        self.calls.append((len(messages), sorted(t["function"]["name"] for t in (tools or []))))
        return self._script.pop(0)


def _fake_whitebox_tool():
    def fn(plant_tf, delay=0.0):
        return {"ok": True, "rows": [{"name": "SIMC", "stable": True, "gains": {"Kp": 1, "Ki": 1, "Kd": 0}}]}
    return (RUN_WHITEBOX_BENCHMARK_SCHEMA, fn)


def _fake_blackbox_tool():
    def fn(step_signal_path=None, relay_signal_path=None):
        return {"ok": True, "rows": [{"name": "ZN-I", "available": True, "black_box": True,
                                       "gains": {"Kp": 1, "Ki": 1, "Kd": 0}}]}
    return (RUN_BLACKBOX_BENCHMARK_SCHEMA, fn)


class TestSessionToolLoop(unittest.TestCase):
    def _session(self, script):
        return Session(ScriptedClient(script), whitebox_tool=_fake_whitebox_tool(),
                        blackbox_tool=_fake_blackbox_tool())

    def test_no_tool_calls_returns_content_directly(self):
        session = self._session([_response(content="hello")])
        reply = session.handle_user_message("hi")
        self.assertEqual(reply, "hello")

    def test_tool_call_then_final_answer(self):
        script = [
            _response(tool_calls=[_tool_call("set_priorities", {"tf_known": True})]),
            _response(content="got it"),
        ]
        session = self._session(script)
        reply = session.handle_user_message("I know my TF")
        self.assertEqual(reply, "got it")
        self.assertTrue(session.worksheet.tf_known)
        # A 'tool' role message with the result should be in the transcript.
        tool_msgs = [m for m in session.messages if isinstance(m, dict) and m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["tool_name"], "set_priorities")

    def test_active_tools_gated_before_and_after_tf_known(self):
        script = [
            _response(tool_calls=[_tool_call("set_priorities", {"tf_known": True})]),
            _response(content="done"),
        ]
        session = self._session(script)
        session.handle_user_message("known TF")
        first_call_tools, second_call_tools = [c[1] for c in session.client.calls]
        self.assertNotIn("run_whitebox_benchmark", first_call_tools)
        self.assertIn("run_whitebox_benchmark", second_call_tools)
        self.assertNotIn("run_blackbox_benchmark", second_call_tools)

    def test_benchmark_call_populates_known_stable_methods(self):
        script = [
            _response(tool_calls=[_tool_call("set_priorities", {"tf_known": True})]),
            _response(tool_calls=[_tool_call("run_whitebox_benchmark", {"plant_tf": "1/(s+1)"})]),
            _response(content="benchmarked"),
        ]
        session = self._session(script)
        session.handle_user_message("go")
        self.assertIn("SIMC", session.known_stable_methods)

    def test_finalize_recommendation_rejects_unknown_method(self):
        script = [
            _response(tool_calls=[_tool_call("set_priorities", {"tf_known": True})]),
            _response(tool_calls=[_tool_call("finalize_recommendation",
                                              {"method_name": "Made Up Method", "rationale": "because"})]),
            _response(content="ok"),
        ]
        session = self._session(script)
        session.handle_user_message("go")
        tool_msgs = [m for m in session.messages if isinstance(m, dict) and m.get("role") == "tool"]
        self.assertIn('"ok":false', tool_msgs[-1]["content"].replace(" ", ""))

    def test_dispatch_rejects_tool_outside_active_set(self):
        """Defensive check: even if a (buggy or adversarial) model output
        requests a benchmark tool before tf_known is set, dispatch must
        refuse rather than silently running it."""
        session = self._session([])
        result = session._dispatch_tool("run_whitebox_benchmark", {"plant_tf": "1/(s+1)"})
        self.assertFalse(result["ok"])

    def test_max_tool_hops_cap_returns_fallback(self):
        script = [_response(tool_calls=[_tool_call("set_priorities", {"top_priority": "speed"})])] * 10
        session = self._session(script)
        reply = session.handle_user_message("keep calling tools forever")
        self.assertEqual(reply, FALLBACK_MESSAGE)


if __name__ == "__main__":
    unittest.main()
