"""Unit tests for the OpenAI-backed LLM client -- no live OpenAI API
required. Mirrors test_supervisor_llm_anthropic.py's style/conventions,
adapted where OpenAI's Chat Completions API genuinely differs (see
supervisor_llm_openai.py's module docstring): no tool-schema translation
to test (tools pass straight through), no system_text extraction, no
same-turn tool-result batching, and `function.arguments` round-trips as a
JSON string rather than a dict.

Run with:
    python test_supervisor_llm_openai.py
or:
    python -m unittest test_supervisor_llm_openai -v

End-to-end behavior against the real OpenAI API is a separate, manual
verification step, same as for the Anthropic client -- not covered here.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from supervisor_llm_openai import (
    OpenAIClient,
    DEFAULT_MODEL,
    _AssistantMessage,
    _translate_messages,
)


def _tool_call(id, name, arguments_json):
    """A SimpleNamespace matching the shape of one
    response.choices[0].message.tool_calls[] entry -- .id, .type,
    .function.name, .function.arguments (a JSON *string*, per OpenAI's
    convention)."""
    return SimpleNamespace(
        id=id, type="function",
        function=SimpleNamespace(name=name, arguments=arguments_json),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Message translation / tool_call_id round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateMessages(unittest.TestCase):
    def test_system_message_passed_through_as_a_normal_message(self):
        """Unlike Anthropic (a separate system= parameter), Chat Completions
        takes role: "system" as an ordinary message -- no extraction."""
        msgs = _translate_messages([{"role": "system", "content": "SYS"}])
        self.assertEqual(msgs, [{"role": "system", "content": "SYS"}])

    def test_plain_user_message_passed_through(self):
        msgs = _translate_messages([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ])
        self.assertEqual(msgs, [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ])

    def test_single_tool_call_round_trip_resolves_real_id(self):
        raw = [{"id": "call_123", "type": "function",
                "function": {"name": "run_whitebox_benchmark",
                             "arguments": '{"plant_tf": "1/(90s+1)", "delay": 13}'}}]
        assistant = _AssistantMessage(content="", raw_tool_calls=raw)

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "tune 1/(90s+1), L=13"},
            assistant,
            {"role": "tool", "tool_name": "run_whitebox_benchmark", "content": '{"ok": true}'},
        ]
        msgs = _translate_messages(messages)

        self.assertEqual(msgs[1], {"role": "user", "content": "tune 1/(90s+1), L=13"})
        self.assertEqual(msgs[2], {"role": "assistant", "content": None, "tool_calls": raw})
        self.assertEqual(msgs[3], {
            "role": "tool", "tool_call_id": "call_123", "content": '{"ok": true}',
        })

    def test_multiple_tool_calls_in_one_hop_are_separate_messages(self):
        """No same-turn batching requirement here, unlike Anthropic's single
        user message carrying multiple tool_result blocks -- each Chat
        Completions tool result is its own independent message."""
        raw = [
            {"id": "id_a", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
            {"id": "id_b", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}},
        ]
        assistant = _AssistantMessage(content="", raw_tool_calls=raw)

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "do both"},
            assistant,
            {"role": "tool", "tool_name": "tool_a", "content": "A result"},
            {"role": "tool", "tool_name": "tool_b", "content": "B result"},
        ]
        msgs = _translate_messages(messages)

        self.assertEqual(msgs[-2], {"role": "tool", "tool_call_id": "id_a", "content": "A result"})
        self.assertEqual(msgs[-1], {"role": "tool", "tool_call_id": "id_b", "content": "B result"})

    def test_second_hop_assistant_turn_appended_after_tool_result(self):
        raw = [{"id": "call_1", "type": "function",
                "function": {"name": "set_priorities", "arguments": '{"tf_known": true}'}}]
        assistant_1 = _AssistantMessage(content="", raw_tool_calls=raw)
        assistant_2 = _AssistantMessage(content="Got it, tuning now.", raw_tool_calls=[])

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "I know the TF"},
            assistant_1,
            {"role": "tool", "tool_name": "set_priorities", "content": '{"ok": true}'},
            assistant_2,
        ]
        msgs = _translate_messages(messages)
        self.assertEqual(msgs[-1], {"role": "assistant", "content": "Got it, tuning now."})

    def test_duplicate_tool_name_in_one_turn_resolves_each_result_to_its_own_id(self):
        """Regression test ported from test_supervisor_llm_anthropic.py's
        equivalent, for the same underlying risk (docs/memos/2026-09-07/
        2026-09-07-supervisor-robustness-memo.md, section 3, and that
        memo's TODO #1: OpenAI's tool-calling is id-based the same way
        Anthropic's is, so a hand-rolled client needs the identical
        per-name-FIFO fix, not just Anthropic). A name-keyed dict collapses
        both onto the last id; the fix is a per-name FIFO queue, consumed
        in call order."""
        raw = [
            {"id": "call_AAA", "type": "function",
             "function": {"name": "run_lqg_benchmark", "arguments": '{"plant_preset": "aircraft_hall"}'}},
            {"id": "call_BBB", "type": "function",
             "function": {"name": "run_lqg_benchmark", "arguments": '{"plant_preset": "aircraft_hall", "Q_diag": [5, 1]}'}},
        ]
        assistant = _AssistantMessage(content="", raw_tool_calls=raw)

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "compare fastest vs lowest overshoot"},
            assistant,
            {"role": "tool", "tool_name": "run_lqg_benchmark", "content": '{"ok":true,"A":1}'},
            {"role": "tool", "tool_name": "run_lqg_benchmark", "content": '{"ok":true,"B":2}'},
        ]
        msgs = _translate_messages(messages)

        ids = [m["tool_call_id"] for m in msgs[-2:]]
        self.assertEqual(ids, ["call_AAA", "call_BBB"])

    def test_three_duplicate_tool_names_resolve_in_call_order(self):
        raw = [{"id": f"call_{i}", "type": "function",
                "function": {"name": "run_lqg_benchmark", "arguments": "{}"}} for i in range(3)]
        assistant = _AssistantMessage(content="", raw_tool_calls=raw)
        messages = [
            {"role": "system", "content": "SYS"},
            assistant,
            *[{"role": "tool", "tool_name": "run_lqg_benchmark", "content": f"r{i}"} for i in range(3)],
        ]
        msgs = _translate_messages(messages)
        ids = [m["tool_call_id"] for m in msgs[-3:]]
        self.assertEqual(ids, ["call_0", "call_1", "call_2"])

    def test_two_names_each_called_twice_dont_cross_contaminate(self):
        tc = lambda id_, name: {"id": id_, "type": "function", "function": {"name": name, "arguments": "{}"}}
        raw = [tc("a1", "tool_a"), tc("b1", "tool_b"), tc("a2", "tool_a"), tc("b2", "tool_b")]
        assistant = _AssistantMessage(content="", raw_tool_calls=raw)
        messages = [
            {"role": "system", "content": "SYS"},
            assistant,
            {"role": "tool", "tool_name": "tool_a", "content": "A-first"},
            {"role": "tool", "tool_name": "tool_b", "content": "B-first"},
            {"role": "tool", "tool_name": "tool_a", "content": "A-second"},
            {"role": "tool", "tool_name": "tool_b", "content": "B-second"},
        ]
        msgs = _translate_messages(messages)
        ids = [m["tool_call_id"] for m in msgs[-4:]]
        self.assertEqual(ids, ["a1", "b1", "a2", "b2"])

    def test_tool_result_with_no_matching_tool_call_gets_none_id(self):
        """Defensive case, shouldn't happen in practice given Session's own
        loop shape, but the lookup must not raise -- a missing id surfaces
        as a None tool_call_id rather than a KeyError crashing the
        session."""
        msgs = _translate_messages([
            {"role": "system", "content": "SYS"},
            {"role": "tool", "tool_name": "nonexistent", "content": "x"},
        ])
        self.assertIsNone(msgs[1]["tool_call_id"])


# ─────────────────────────────────────────────────────────────────────────────
# _AssistantMessage -- arguments-as-JSON-string parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestAssistantMessageArgumentParsing(unittest.TestCase):
    def test_tool_calls_exposes_parsed_dict_not_raw_json_string(self):
        """Session._dispatch_tool calls fn(**(arguments or {})) -- it needs
        a dict, matching Ollama's/Anthropic's already-a-dict shape, even
        though OpenAI hands back arguments as a JSON string (confirmed
        against chat_completion_message_function_tool_call.py's own field
        docstring)."""
        raw = [{"id": "call_1", "type": "function",
                "function": {"name": "run_whitebox_benchmark",
                             "arguments": '{"plant_tf": "1/(90s+1)", "delay": 13}'}}]
        msg = _AssistantMessage(content="", raw_tool_calls=raw)
        self.assertEqual(msg.tool_calls[0].function.name, "run_whitebox_benchmark")
        self.assertEqual(msg.tool_calls[0].function.arguments, {"plant_tf": "1/(90s+1)", "delay": 13})

    def test_empty_arguments_string_parses_to_empty_dict(self):
        raw = [{"id": "call_1", "type": "function",
                "function": {"name": "set_priorities", "arguments": ""}}]
        msg = _AssistantMessage(content="", raw_tool_calls=raw)
        self.assertEqual(msg.tool_calls[0].function.arguments, {})

    def test_no_tool_calls_leaves_tool_calls_none(self):
        msg = _AssistantMessage(content="hello", raw_tool_calls=[])
        self.assertIsNone(msg.tool_calls)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAIClient construction (no network -- OpenAI() doesn't call out until a
# request is actually made)
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenAIClientConstruction(unittest.TestCase):
    def test_default_model_is_cheapest_tier(self):
        """Regression guard mirroring test_supervisor_llm_anthropic.py's
        Opus-default-mistake guard -- see supervisor_llm_openai.py's module
        docstring for why gpt-5.6-luna, not gpt-5.6-sol/gpt-6-astra."""
        self.assertEqual(DEFAULT_MODEL, "gpt-5.6-luna")

    def test_stores_model_and_max_completion_tokens(self):
        client = OpenAIClient(api_key="sk-fake", model="gpt-5.6-terra", max_completion_tokens=1234)
        self.assertEqual(client.model, "gpt-5.6-terra")
        self.assertEqual(client.max_completion_tokens, 1234)


if __name__ == "__main__":
    unittest.main()
