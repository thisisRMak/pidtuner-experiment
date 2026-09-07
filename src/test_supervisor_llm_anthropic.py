"""Unit tests for the Claude-backed LLM client -- no live Anthropic API
required. Mirrors test_supervisor_pid.py's style/conventions.

Run with:
    python test_supervisor_llm_anthropic.py
or:
    python -m unittest test_supervisor_llm_anthropic -v

Covers everything deterministic: the Ollama/OpenAI-shaped tool schema ->
Anthropic input_schema translation, and the trickiest part of this module
by its own docstring's account -- _translate_messages' round-trip of
assistant turns and tool results, including real tool_use_id correlation
and the multi-tool-call-in-one-hop batching rule (return all tool_result
blocks in a single user message). End-to-end behavior against the real
Anthropic API is a separate, manual/scripted verification step (see
docs/aituner_plan.md) -- it can't be made deterministic and isn't covered
here.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from supervisor_llm_anthropic import (
    AnthropicClient,
    DEFAULT_MODEL,
    _AssistantMessage,
    _to_anthropic_tools,
    _translate_messages,
)


def _block(type, **kw):
    return SimpleNamespace(type=type, **kw)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema translation
# ─────────────────────────────────────────────────────────────────────────────

class TestToAnthropicTools(unittest.TestCase):
    def test_translates_name_description_parameters(self):
        ollama_tools = [{
            "type": "function",
            "function": {
                "name": "run_whitebox_benchmark",
                "description": "run it",
                "parameters": {
                    "type": "object",
                    "properties": {"plant_tf": {"type": "string"}},
                    "required": ["plant_tf"],
                },
            },
        }]
        converted = _to_anthropic_tools(ollama_tools)
        self.assertEqual(converted, [{
            "name": "run_whitebox_benchmark",
            "description": "run it",
            "input_schema": {
                "type": "object",
                "properties": {"plant_tf": {"type": "string"}},
                "required": ["plant_tf"],
            },
        }])

    def test_missing_description_defaults_to_empty_string(self):
        ollama_tools = [{"type": "function", "function": {
            "name": "set_priorities", "parameters": {"type": "object", "properties": {}},
        }}]
        converted = _to_anthropic_tools(ollama_tools)
        self.assertEqual(converted[0]["description"], "")

    def test_translates_multiple_tools_in_order(self):
        make = lambda name: {"type": "function", "function": {
            "name": name, "description": "", "parameters": {"type": "object", "properties": {}},
        }}
        converted = _to_anthropic_tools([make("a"), make("b")])
        self.assertEqual([t["name"] for t in converted], ["a", "b"])


# ─────────────────────────────────────────────────────────────────────────────
# Message translation / tool_use_id round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestTranslateMessages(unittest.TestCase):
    def test_system_message_extracted(self):
        system_text, msgs = _translate_messages([{"role": "system", "content": "SYS"}])
        self.assertEqual(system_text, "SYS")
        self.assertEqual(msgs, [])

    def test_plain_user_message_passed_through(self):
        _, msgs = _translate_messages([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ])
        self.assertEqual(msgs, [{"role": "user", "content": "hello"}])

    def test_single_tool_call_round_trip_resolves_real_id(self):
        tool_use = _block("tool_use", id="toolu_123", name="run_whitebox_benchmark",
                           input={"plant_tf": "1/(90s+1)", "delay": 13})
        assistant = _AssistantMessage(content="", tool_use_blocks=[tool_use], raw_blocks=[tool_use])

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "tune 1/(90s+1), L=13"},
            assistant,
            {"role": "tool", "tool_name": "run_whitebox_benchmark", "content": '{"ok": true}'},
        ]
        _, msgs = _translate_messages(messages)

        self.assertEqual(msgs[0], {"role": "user", "content": "tune 1/(90s+1), L=13"})
        self.assertEqual(msgs[1], {"role": "assistant", "content": [tool_use]})
        self.assertEqual(msgs[2], {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "toolu_123", "content": '{"ok": true}'}],
        })

    def test_multiple_tool_calls_in_one_hop_batch_into_one_user_message(self):
        """The Anthropic API requires all tool_result blocks for one turn to
        land in a single user message (see claude-api skill's Common
        Pitfalls -- splitting them trains the model to stop parallel tool
        use). Session doesn't do this today (one tool active at a time), but
        the mechanism itself must get this right if that ever changes."""
        tu_a = _block("tool_use", id="id_a", name="tool_a", input={})
        tu_b = _block("tool_use", id="id_b", name="tool_b", input={})
        assistant = _AssistantMessage(content="", tool_use_blocks=[tu_a, tu_b], raw_blocks=[tu_a, tu_b])

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "do both"},
            assistant,
            {"role": "tool", "tool_name": "tool_a", "content": "A result"},
            {"role": "tool", "tool_name": "tool_b", "content": "B result"},
        ]
        _, msgs = _translate_messages(messages)

        self.assertEqual(msgs[-1]["role"], "user")
        self.assertEqual(msgs[-1]["content"], [
            {"type": "tool_result", "tool_use_id": "id_a", "content": "A result"},
            {"type": "tool_result", "tool_use_id": "id_b", "content": "B result"},
        ])

    def test_second_hop_assistant_turn_appended_after_tool_result(self):
        tool_use = _block("tool_use", id="toolu_1", name="set_priorities", input={"tf_known": True})
        assistant_1 = _AssistantMessage(content="", tool_use_blocks=[tool_use], raw_blocks=[tool_use])
        text_block = _block("text", text="Got it, tuning now.")
        assistant_2 = _AssistantMessage(content="Got it, tuning now.", tool_use_blocks=[], raw_blocks=[text_block])

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "I know the TF"},
            assistant_1,
            {"role": "tool", "tool_name": "set_priorities", "content": '{"ok": true}'},
            assistant_2,
        ]
        _, msgs = _translate_messages(messages)
        self.assertEqual(msgs[-1], {"role": "assistant", "content": [text_block]})

    def test_tool_result_with_no_matching_tool_use_gets_none_id(self):
        """Defensive case, shouldn't happen in practice given Session's own
        loop shape, but the lookup must not raise -- a missing id surfaces
        as a None tool_use_id (which the real API would then reject with a
        clear error) rather than a KeyError crashing the session."""
        _, msgs = _translate_messages([
            {"role": "system", "content": "SYS"},
            {"role": "tool", "tool_name": "nonexistent", "content": "x"},
        ])
        self.assertIsNone(msgs[0]["content"][0]["tool_use_id"])


# ─────────────────────────────────────────────────────────────────────────────
# AnthropicClient construction (no network -- Anthropic() doesn't call out
# until a request is actually made)
# ─────────────────────────────────────────────────────────────────────────────

class TestAnthropicClientConstruction(unittest.TestCase):
    def test_default_model_is_cheapest_tier(self):
        """Regression guard for the Opus-default mistake this session
        already made once -- see streamlit_llm_panel.py's model picker."""
        self.assertEqual(DEFAULT_MODEL, "claude-haiku-4-5")

    def test_stores_model_and_max_tokens(self):
        client = AnthropicClient(api_key="sk-ant-fake", model="claude-sonnet-5", max_tokens=1234)
        self.assertEqual(client.model, "claude-sonnet-5")
        self.assertEqual(client.max_tokens, 1234)


if __name__ == "__main__":
    unittest.main()
