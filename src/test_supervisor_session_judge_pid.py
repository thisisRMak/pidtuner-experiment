"""Unit tests for LLM-as-judge orchestration -- no live API required.
Mirrors test_supervisor_llm_anthropic.py's style/conventions.

Run with:
    python test_supervisor_session_judge_pid.py
or:
    python -m unittest test_supervisor_session_judge_pid -v

Real supervisor_session_pid.Session instances are used throughout (not a
hand-rolled stand-in), driven by a scripted fake llm_client so no network
call happens -- this exercises serialize_candidate_trace() against
Session's actual .messages shape, not a guessed approximation of it. The
judge side is exercised through the real JudgeSession.handle_user_message()
loop, including the real ThreadPoolExecutor fan-out (small candidate
counts, no mocking of the threading itself) -- a genuine, if light,
concurrency exercise, not just a single-threaded stand-in.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from supervisor_session_pid import Session
from supervisor_session_judge_pid import JudgeSession, serialize_candidate_trace, summarize_candidate_round

WHITEBOX_SCHEMA = {
    "type": "function",
    "function": {"name": "run_whitebox_benchmark", "description": "", "parameters": {"type": "object", "properties": {}}},
}
BLACKBOX_SCHEMA = {
    "type": "function",
    "function": {"name": "run_blackbox_benchmark", "description": "", "parameters": {"type": "object", "properties": {}}},
}


def _fake_whitebox(**kwargs):
    return {
        "ok": True,
        "rows": [
            {"name": "Tyreus-Luyben", "stable": True, "OS%": 2.662},
            {"name": "CHR set 0%", "stable": True, "OS%": 3.986},
        ],
    }


def _fake_blackbox(**kwargs):
    return {"ok": True, "rows": []}


class _Msg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _TC:
    def __init__(self, name, arguments):
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _Resp:
    def __init__(self, message):
        self.message = message


class _ScriptedClient:
    """Plays back one _Msg per .chat() call, in order -- a fake llm_client
    for Session, standing in for AnthropicClient/OpenAIClient/GeminiClient
    without hitting a real API. `.script` is public and mutable so a test
    can extend it between rounds (see TestJudgeSessionRevision)."""

    def __init__(self, script):
        self.script = list(script)

    def chat(self, messages, tools=None):
        return _Resp(self.script.pop(0))


class _RecordingJudgeClient:
    """Records every outgoing messages list (for assertions on what the
    judge was actually shown) and plays back one scripted reply per call."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        return _Resp(_Msg(self.replies.pop(0)))


class _FakeProviderError(Exception):
    """Stands in for anthropic.APIStatusError/genai_errors.ServerError/etc
    -- every provider exception this project handles at the top level
    carries a `.message` with the provider's own text (see
    supervisor_session_judge_pid._short_error)."""

    def __init__(self, message):
        super().__init__(message)
        self.message = message


class _AlwaysFailingClient:
    """A candidate client that raises on every .chat() call -- e.g. a real
    provider capacity error, which in practice hits exactly one of N
    independent candidates, not all of them."""

    def __init__(self, message="This model is currently experiencing high demand."):
        self._message = message

    def chat(self, messages, tools=None):
        raise _FakeProviderError(self._message)


class _NeverCalledJudgeClient:
    """Fails the test loudly if the judge is ever actually invoked --
    used to assert the all-candidates-failed path skips the judge call
    entirely rather than asking it to arbitrate over zero real answers."""

    def chat(self, messages, tools=None):
        raise AssertionError("the judge must not be called when every candidate failed")


def _make_candidate(method_name, rationale, final_text):
    """A Session that: sets priorities, runs the whitebox benchmark, then
    finalizes with the given method/rationale, matching the real tool-
    gating order supervisor_session_pid.Session._active_tools() enforces
    (the benchmark tool isn't offered until tf_known is set)."""
    script = [
        _Msg("", tool_calls=[_TC("set_priorities", {"tf_known": True, "top_priority": "overshoot"})]),
        _Msg("", tool_calls=[_TC("run_whitebox_benchmark", {"plant_tf": "1/(90s+1)", "delay": 13})]),
        _Msg("", tool_calls=[_TC("finalize_recommendation", {"method_name": method_name, "rationale": rationale})]),
        _Msg(final_text, tool_calls=None),
    ]
    client = _ScriptedClient(script)
    return Session(client, whitebox_tool=(WHITEBOX_SCHEMA, _fake_whitebox), blackbox_tool=(BLACKBOX_SCHEMA, _fake_blackbox))


class TestSerializeCandidateTrace(unittest.TestCase):
    def test_includes_tool_calls_results_and_final_text_excludes_system_and_user(self):
        session = SimpleNamespace(messages=[
            {"role": "system", "content": "SYSTEM PROMPT TEXT"},
            {"role": "user", "content": "hello there"},
            _Msg("", tool_calls=[_TC("run_whitebox_benchmark", {"plant_tf": "1/(90s+1)"})]),
            {"role": "tool", "tool_name": "run_whitebox_benchmark", "content": '{"ok": true, "rows": []}'},
            _Msg("final answer text", tool_calls=None),
        ])
        out = serialize_candidate_trace("Claude (Anthropic): Claude Haiku 4.5", session)
        self.assertIn("=== Candidate: Claude (Anthropic): Claude Haiku 4.5 ===", out)
        self.assertIn("run_whitebox_benchmark", out)
        self.assertIn('"plant_tf": "1/(90s+1)"', out)
        self.assertIn('{"ok": true, "rows": []}', out)
        self.assertIn("final answer text", out)
        self.assertNotIn("hello there", out, "user turns are already given to the judge once, not per candidate")
        self.assertNotIn("SYSTEM PROMPT TEXT", out, "the judge prompt already describes what a candidate is")


class TestJudgeSessionFanOut(unittest.TestCase):
    def _build(self):
        candidates = [
            ("Claude (Anthropic): Claude Haiku 4.5",
             _make_candidate("Tyreus-Luyben", "Lowest overshoot, 2.662%, stable.", "I recommend Tyreus-Luyben.")),
            ("ChatGPT (OpenAI): GPT-5.6 Luna",
             _make_candidate("CHR set 0%", "3.986% overshoot, healthy Ms.", "I recommend CHR set 0%.")),
        ]
        judge_client = _RecordingJudgeClient(["Tyreus-Luyben is the better pick (2.662% vs 3.986% overshoot)."])
        return candidates, judge_client, JudgeSession(candidates, judge_client)

    def test_all_candidates_receive_the_identical_user_text(self):
        candidates, _, judge = self._build()
        judge.handle_user_message("1/(90s+1), delay 13, minimize overshoot.")
        for _, session in candidates:
            self.assertEqual(session.messages[1]["content"], "1/(90s+1), delay 13, minimize overshoot.")

    def test_judge_sees_both_candidates_tool_traces_and_no_tools_offered(self):
        _, judge_client, judge = self._build()
        judge.handle_user_message("1/(90s+1), delay 13, minimize overshoot.")
        self.assertEqual(len(judge_client.calls), 1)
        call = judge_client.calls[0]
        self.assertIsNone(call["tools"], "the judge is never given tools of its own -- see module docstring")
        trace_text = call["messages"][-1]["content"]
        self.assertIn("Tyreus-Luyben", trace_text)
        self.assertIn("CHR set 0%", trace_text)
        self.assertIn('"plant_tf": "1/(90s+1)"', trace_text)
        self.assertIn("=== Candidate: Claude (Anthropic): Claude Haiku 4.5 ===", trace_text)
        self.assertIn("=== Candidate: ChatGPT (OpenAI): GPT-5.6 Luna ===", trace_text)

    def test_returns_the_judges_reply(self):
        _, _, judge = self._build()
        reply = judge.handle_user_message("1/(90s+1), delay 13, minimize overshoot.")
        self.assertEqual(reply, "Tyreus-Luyben is the better pick (2.662% vs 3.986% overshoot).")

    def test_dialogue_persists_only_plain_text_not_the_trace_dump(self):
        _, _, judge = self._build()
        judge.handle_user_message("1/(90s+1), delay 13, minimize overshoot.")
        self.assertEqual(len(judge.dialogue), 2)
        self.assertEqual(judge.dialogue[0], {"role": "user", "content": "1/(90s+1), delay 13, minimize overshoot."})
        self.assertEqual(judge.dialogue[1]["role"], "assistant")
        self.assertNotIn("Candidate model traces", judge.dialogue[0]["content"])


class TestJudgeSessionRevision(unittest.TestCase):
    def test_a_second_round_carries_a_candidates_revised_pick_to_the_judge(self):
        candidates = [
            ("Claude (Anthropic): Claude Haiku 4.5",
             _make_candidate("Tyreus-Luyben", "Lowest overshoot.", "I recommend Tyreus-Luyben.")),
            ("ChatGPT (OpenAI): GPT-5.6 Luna",
             _make_candidate("CHR set 0%", "3.986% overshoot.", "I recommend CHR set 0%.")),
        ]
        judge_client = _RecordingJudgeClient([
            "Tyreus-Luyben is the better pick.",
            "GPT-5.6 has revised to Tyreus-Luyben too -- both candidates now agree.",
        ])
        judge = JudgeSession(candidates, judge_client)
        judge.handle_user_message("1/(90s+1), delay 13, minimize overshoot.")

        # Round 2: the second candidate (originally wrong) revises its pick.
        # "Keep advancing everyone" means no source change is needed to let
        # this happen -- Session.handle_user_message is just called again.
        _, chatgpt_session = candidates[1]
        chatgpt_session.client.script.extend([
            _Msg("", tool_calls=[_TC("finalize_recommendation", {"method_name": "Tyreus-Luyben", "rationale": "On reflection, Tyreus-Luyben is strictly better."})]),
            _Msg("I now recommend Tyreus-Luyben.", tool_calls=None),
        ])
        _, claude_session = candidates[0]
        claude_session.client.script.extend([
            _Msg("", tool_calls=[_TC("finalize_recommendation", {"method_name": "Tyreus-Luyben", "rationale": "Still the best."})]),
            _Msg("Still Tyreus-Luyben.", tool_calls=None),
        ])

        reply = judge.handle_user_message("Please confirm your final answer.")

        self.assertEqual(len(judge_client.calls), 2)
        round2_trace = judge_client.calls[1]["messages"][-1]["content"]
        self.assertIn("On reflection, Tyreus-Luyben is strictly better.", round2_trace,
                      "the revised finalize call must reach the judge")
        # The original (now-superseded) pick legitimately still appears too
        # -- full history is preserved, not truncated; it's the judge
        # prompt's job (not this code) to reason about the latest state.
        self.assertIn("CHR set 0%", round2_trace)

        self.assertEqual(len(judge.dialogue), 4)
        self.assertEqual(judge.dialogue[2], {"role": "user", "content": "Please confirm your final answer."})
        self.assertEqual(reply, "GPT-5.6 has revised to Tyreus-Luyben too -- both candidates now agree.")


class TestSummarizeCandidateRound(unittest.TestCase):
    def test_extracts_calls_finalized_method_and_latest_reply(self):
        session = SimpleNamespace(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
            _Msg("", tool_calls=[_TC("run_whitebox_benchmark", {"plant_tf": "1/(90s+1)"})]),
            {"role": "tool", "tool_name": "run_whitebox_benchmark", "content": "{}"},
            _Msg("", tool_calls=[_TC("finalize_recommendation", {"method_name": "Tyreus-Luyben", "rationale": "best"})]),
            _Msg("I recommend Tyreus-Luyben.", tool_calls=None),
        ])
        out = summarize_candidate_round(session, since=2)
        self.assertEqual(out["finalized"], "Tyreus-Luyben")
        self.assertEqual(out["reply"], "I recommend Tyreus-Luyben.")
        self.assertEqual([name for name, _ in out["calls"]], ["run_whitebox_benchmark", "finalize_recommendation"])

    def test_since_excludes_earlier_rounds(self):
        session = SimpleNamespace(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "round 1"},
            _Msg("round 1 reply", tool_calls=None),
            {"role": "user", "content": "round 2"},
            _Msg("", tool_calls=[_TC("finalize_recommendation", {"method_name": "CHR set 0%"})]),
            _Msg("round 2 reply", tool_calls=None),
        ])
        out = summarize_candidate_round(session, since=3)
        self.assertEqual(out["finalized"], "CHR set 0%")
        self.assertEqual(out["reply"], "round 2 reply")


class TestRoundsHistory(unittest.TestCase):
    def test_accumulates_one_entry_per_round(self):
        candidates = [
            ("Claude (Anthropic): Claude Haiku 4.5",
             _make_candidate("Tyreus-Luyben", "Lowest overshoot.", "I recommend Tyreus-Luyben.")),
            ("ChatGPT (OpenAI): GPT-5.6 Luna",
             _make_candidate("CHR set 0%", "3.986% overshoot.", "I recommend CHR set 0%.")),
        ]
        judge_client = _RecordingJudgeClient(["Round 1 verdict.", "Round 2 verdict."])
        judge = JudgeSession(candidates, judge_client)

        judge.handle_user_message("round 1 text")
        for _, session in candidates:
            session.client.script.extend([
                _Msg("", tool_calls=[_TC("finalize_recommendation", {"method_name": "Tyreus-Luyben", "rationale": "revised"})]),
                _Msg("revised reply", tool_calls=None),
            ])
        judge.handle_user_message("round 2 text")

        self.assertEqual(len(judge.rounds_history), 2)
        self.assertEqual(judge.rounds_history[0]["user_text"], "round 1 text")
        self.assertEqual(judge.rounds_history[0]["judge_reply"], "Round 1 verdict.")
        self.assertEqual(judge.rounds_history[1]["user_text"], "round 2 text")
        self.assertEqual(judge.rounds_history[1]["judge_reply"], "Round 2 verdict.")
        # Each round's own candidate snapshot, not a shared/aliased list
        # that later mutates underneath an earlier entry.
        self.assertEqual(judge.rounds_history[0]["candidates"][0]["finalized"], "Tyreus-Luyben")
        self.assertEqual(judge.rounds_history[1]["candidates"][1]["finalized"], "Tyreus-Luyben")

    def test_starts_empty_on_a_fresh_object(self):
        judge = JudgeSession([], _NeverCalledJudgeClient())
        self.assertEqual(judge.rounds_history, [])

    def test_all_candidates_failing_still_records_a_round(self):
        candidates = [
            ("Claude (Anthropic): Claude Haiku 4.5",
             Session(_AlwaysFailingClient("down"), whitebox_tool=(WHITEBOX_SCHEMA, _fake_whitebox), blackbox_tool=(BLACKBOX_SCHEMA, _fake_blackbox))),
        ]
        judge = JudgeSession(candidates, _NeverCalledJudgeClient())
        judge.handle_user_message("hello")
        self.assertEqual(len(judge.rounds_history), 1)
        self.assertIn("All 1 candidates failed", judge.rounds_history[0]["judge_reply"])


class TestPerCandidateResilience(unittest.TestCase):
    def test_one_candidates_provider_error_does_not_sink_the_round(self):
        good = _make_candidate("Tyreus-Luyben", "Lowest overshoot.", "I recommend Tyreus-Luyben.")
        also_good = _make_candidate("Tyreus-Luyben", "Agreed.", "I also recommend Tyreus-Luyben.")
        failing = Session(_AlwaysFailingClient(),
                           whitebox_tool=(WHITEBOX_SCHEMA, _fake_whitebox),
                           blackbox_tool=(BLACKBOX_SCHEMA, _fake_blackbox))
        candidates = [
            ("Claude (Anthropic): Claude Haiku 4.5", good),
            ("ChatGPT (OpenAI): GPT-5.6 Luna", also_good),
            ("Gemini (Google): Gemini 3.5 Flash-Lite", failing),
        ]
        judge_client = _RecordingJudgeClient(["Both responding candidates agree on Tyreus-Luyben."])
        judge = JudgeSession(candidates, judge_client)

        reply = judge.handle_user_message("1/(90s+1), delay 13, minimize overshoot.")

        self.assertEqual(reply, "Both responding candidates agree on Tyreus-Luyben.")
        self.assertEqual(len(judge_client.calls), 1, "the judge must still be called when some candidates succeeded")
        trace_text = judge_client.calls[0]["messages"][-1]["content"]
        self.assertIn(
            "[FAILED TO RESPOND THIS ROUND -- provider error: "
            "This model is currently experiencing high demand.]",
            trace_text,
        )

        by_label = {item["label"]: item for item in judge.last_round}
        self.assertEqual(by_label["Gemini (Google): Gemini 3.5 Flash-Lite"]["error"],
                          "This model is currently experiencing high demand.")
        self.assertEqual(by_label["Claude (Anthropic): Claude Haiku 4.5"]["finalized"], "Tyreus-Luyben")

    def test_all_candidates_failing_skips_the_judge_call_entirely(self):
        candidates = [
            ("Claude (Anthropic): Claude Haiku 4.5",
             Session(_AlwaysFailingClient("rate limited"), whitebox_tool=(WHITEBOX_SCHEMA, _fake_whitebox), blackbox_tool=(BLACKBOX_SCHEMA, _fake_blackbox))),
            ("ChatGPT (OpenAI): GPT-5.6 Luna",
             Session(_AlwaysFailingClient("503 unavailable"), whitebox_tool=(WHITEBOX_SCHEMA, _fake_whitebox), blackbox_tool=(BLACKBOX_SCHEMA, _fake_blackbox))),
        ]
        judge = JudgeSession(candidates, _NeverCalledJudgeClient())

        reply = judge.handle_user_message("1/(90s+1), delay 13, minimize overshoot.")

        self.assertIn("All 2 candidates failed to respond this round", reply)
        self.assertIn("rate limited", reply)
        self.assertIn("503 unavailable", reply)
        self.assertEqual(len(judge.dialogue), 2, "dialogue continuity is kept even without a real judge call")
        self.assertEqual(judge.dialogue[1]["content"], reply)
        self.assertTrue(all("error" in item for item in judge.last_round))


if __name__ == "__main__":
    unittest.main()
