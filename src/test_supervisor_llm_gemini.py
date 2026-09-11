"""Unit tests for the Gemini-backed LLM client -- no live Gemini API
required. Mirrors test_supervisor_llm_anthropic.py/test_supervisor_llm_openai.py's
style/conventions, adapted where google-genai's actual (installed-package-
verified, not just documented) behavior genuinely differs -- see
supervisor_llm_gemini.py's module docstring: no id-based matching (Gemini's
own shipped reference implementation doesn't use it despite docs claiming
otherwise), role="user" for function-response turns (not "tool", despite
the SDK's own README example showing that), one batched Content per turn
like Anthropic (not one message per result like OpenAI), and
FunctionCall.args is already a dict (like Anthropic, unlike OpenAI's
JSON-string convention).

Run with:
    python test_supervisor_llm_gemini.py
or:
    python -m unittest test_supervisor_llm_gemini -v

End-to-end behavior against the real Gemini API is a separate, manual
verification step, same as for the Anthropic/OpenAI clients -- not covered
here.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from google.genai import types

from supervisor_llm_gemini import (
    GeminiClient,
    DEFAULT_MODEL,
    _AssistantMessage,
    _to_gemini_tool,
    _translate_messages,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool schema translation
# ─────────────────────────────────────────────────────────────────────────────

class TestToGeminiTool(unittest.TestCase):
    def test_translates_name_description_parameters_json_schema(self):
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
        tool = _to_gemini_tool(ollama_tools)
        self.assertEqual(len(tool.function_declarations), 1)
        decl = tool.function_declarations[0]
        self.assertEqual(decl.name, "run_whitebox_benchmark")
        self.assertEqual(decl.description, "run it")
        self.assertEqual(decl.parameters_json_schema, {
            "type": "object",
            "properties": {"plant_tf": {"type": "string"}},
            "required": ["plant_tf"],
        })

    def test_missing_description_defaults_to_empty_string(self):
        ollama_tools = [{"type": "function", "function": {
            "name": "set_priorities", "parameters": {"type": "object", "properties": {}},
        }}]
        decl = _to_gemini_tool(ollama_tools).function_declarations[0]
        self.assertEqual(decl.description, "")

    def test_multiple_tools_become_one_tool_with_multiple_declarations(self):
        make = lambda name: {"type": "function", "function": {
            "name": name, "description": "", "parameters": {"type": "object", "properties": {}},
        }}
        tool = _to_gemini_tool([make("a"), make("b")])
        self.assertEqual([d.name for d in tool.function_declarations], ["a", "b"])


# ─────────────────────────────────────────────────────────────────────────────
# Message translation / same-turn batching (no id matching -- see module
# docstring for why this is deliberate, not an oversight)
# ─────────────────────────────────────────────────────────────────────────────

def _function_call_part(name, args):
    return types.Part(function_call=types.FunctionCall(name=name, args=args))


class TestTranslateMessages(unittest.TestCase):
    def test_system_message_extracted_not_sent_as_a_content(self):
        system_text, contents = _translate_messages([{"role": "system", "content": "SYS"}])
        self.assertEqual(system_text, "SYS")
        self.assertEqual(contents, [])

    def test_plain_user_message_becomes_user_role_content(self):
        _, contents = _translate_messages([
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hello"},
        ])
        self.assertEqual(len(contents), 1)
        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "hello")

    def test_single_tool_call_round_trip(self):
        call_part = _function_call_part("run_whitebox_benchmark", {"plant_tf": "1/(90s+1)", "delay": 13})
        raw_content = types.Content(role="model", parts=[call_part])
        assistant = _AssistantMessage(content="", tool_calls=[], raw_content=raw_content)

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "tune 1/(90s+1), L=13"},
            assistant,
            {"role": "tool", "tool_name": "run_whitebox_benchmark", "content": '{"ok": true}'},
        ]
        _, contents = _translate_messages(messages)

        self.assertEqual(contents[0].role, "user")
        self.assertEqual(contents[0].parts[0].text, "tune 1/(90s+1), L=13")
        self.assertIs(contents[1], raw_content)
        self.assertEqual(contents[2].role, "user")
        self.assertEqual(len(contents[2].parts), 1)
        fr = contents[2].parts[0].function_response
        self.assertEqual(fr.name, "run_whitebox_benchmark")
        self.assertEqual(fr.response, {"ok": True})
        self.assertIsNone(fr.id, "no id-based matching -- see module docstring")

    def test_multiple_tool_calls_in_one_hop_batch_into_one_content_in_call_order(self):
        """Mirrors Anthropic's same-turn-batching requirement, not OpenAI's
        one-message-per-result shape -- confirmed against google-genai's
        own shipped _extra_utils.get_function_response_parts, which builds
        exactly one Content per turn."""
        call_a = _function_call_part("tool_a", {})
        call_b = _function_call_part("tool_b", {})
        raw_content = types.Content(role="model", parts=[call_a, call_b])
        assistant = _AssistantMessage(content="", tool_calls=[], raw_content=raw_content)

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "do both"},
            assistant,
            {"role": "tool", "tool_name": "tool_a", "content": '{"A": 1}'},
            {"role": "tool", "tool_name": "tool_b", "content": '{"B": 2}'},
        ]
        _, contents = _translate_messages(messages)

        self.assertEqual(contents[-1].role, "user")
        names = [p.function_response.name for p in contents[-1].parts]
        responses = [p.function_response.response for p in contents[-1].parts]
        self.assertEqual(names, ["tool_a", "tool_b"])
        self.assertEqual(responses, [{"A": 1}, {"B": 2}])

    def test_duplicate_tool_name_in_one_turn_preserves_call_order(self):
        """The Gemini equivalent of the Anthropic/OpenAI duplicate-tool-
        name regression tests -- here correctness comes from order
        preservation within one batched Content, not id matching (Gemini's
        shipped reference implementation doesn't use ids either, see
        module docstring)."""
        call_a = _function_call_part("run_lqg_benchmark", {"plant_preset": "aircraft_hall"})
        call_b = _function_call_part("run_lqg_benchmark", {"plant_preset": "aircraft_hall", "Q_diag": [5, 1]})
        raw_content = types.Content(role="model", parts=[call_a, call_b])
        assistant = _AssistantMessage(content="", tool_calls=[], raw_content=raw_content)

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "compare fastest vs lowest overshoot"},
            assistant,
            {"role": "tool", "tool_name": "run_lqg_benchmark", "content": '{"ok":true,"A":1}'},
            {"role": "tool", "tool_name": "run_lqg_benchmark", "content": '{"ok":true,"B":2}'},
        ]
        _, contents = _translate_messages(messages)

        responses = [p.function_response.response for p in contents[-1].parts]
        self.assertEqual(responses, [{"ok": True, "A": 1}, {"ok": True, "B": 2}])

    def test_second_hop_assistant_turn_appended_after_tool_result(self):
        call_part = _function_call_part("set_priorities", {"tf_known": True})
        raw_content_1 = types.Content(role="model", parts=[call_part])
        assistant_1 = _AssistantMessage(content="", tool_calls=[], raw_content=raw_content_1)
        raw_content_2 = types.Content(role="model", parts=[types.Part(text="Got it, tuning now.")])
        assistant_2 = _AssistantMessage(content="Got it, tuning now.", tool_calls=[], raw_content=raw_content_2)

        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "I know the TF"},
            assistant_1,
            {"role": "tool", "tool_name": "set_priorities", "content": '{"ok": true}'},
            assistant_2,
        ]
        _, contents = _translate_messages(messages)
        self.assertIs(contents[-1], raw_content_2)


# ─────────────────────────────────────────────────────────────────────────────
# _AssistantMessage -- args round-trip (already a dict, unlike OpenAI)
# ─────────────────────────────────────────────────────────────────────────────

class TestAssistantMessageToolCalls(unittest.TestCase):
    def test_tool_calls_exposes_args_dict_directly(self):
        call_part = _function_call_part("run_whitebox_benchmark", {"plant_tf": "1/(90s+1)", "delay": 13})
        raw_content = types.Content(role="model", parts=[call_part])
        msg = _AssistantMessage(
            content="",
            tool_calls=[type("_TC", (), {"function": type("_F", (), {
                "name": "run_whitebox_benchmark",
                "arguments": {"plant_tf": "1/(90s+1)", "delay": 13},
            })()})()],
            raw_content=raw_content,
        )
        self.assertEqual(msg.tool_calls[0].function.arguments, {"plant_tf": "1/(90s+1)", "delay": 13})

    def test_empty_tool_calls_list_becomes_none(self):
        msg = _AssistantMessage(content="hello", tool_calls=[], raw_content=types.Content(role="model"))
        self.assertIsNone(msg.tool_calls)


# ─────────────────────────────────────────────────────────────────────────────
# .chat() request construction -- self._client.models.generate_content is
# mocked out, no real network call.
# ─────────────────────────────────────────────────────────────────────────────

def _fake_candidate(text="", function_calls=(), finish_reason=types.FinishReason.STOP):
    parts = [types.Part(text=text)] if text else []
    parts += [types.Part(function_call=types.FunctionCall(name=n, args=a)) for n, a in function_calls]
    content = types.Content(role="model", parts=parts) if parts else None
    return types.Candidate(content=content, finish_reason=finish_reason)


def _fake_response(candidate):
    return types.GenerateContentResponse(candidates=[candidate])


class TestChatToolCallExtraction(unittest.TestCase):
    def _client_with_mocked_generate(self, response):
        client = GeminiClient(api_key="fake-key")
        client._client = MagicMock()
        client._client.models.generate_content.return_value = response
        return client

    def test_tool_calls_extracted_with_dict_args(self):
        candidate = _fake_candidate(function_calls=[("run_whitebox_benchmark", {"plant_tf": "1/(90s+1)"})])
        client = self._client_with_mocked_generate(_fake_response(candidate))
        resp = client.chat([{"role": "user", "content": "hi"}],
                            tools=[{"type": "function", "function": {"name": "run_whitebox_benchmark", "parameters": {}}}])
        self.assertEqual(resp.message.tool_calls[0].function.name, "run_whitebox_benchmark")
        self.assertEqual(resp.message.tool_calls[0].function.arguments, {"plant_tf": "1/(90s+1)"})

    def test_automatic_function_calling_disabled_when_tools_given(self):
        candidate = _fake_candidate(text="hi")
        client = self._client_with_mocked_generate(_fake_response(candidate))
        client.chat([{"role": "user", "content": "hi"}],
                    tools=[{"type": "function", "function": {"name": "f", "parameters": {}}}])
        config = client._client.models.generate_content.call_args.kwargs["config"]
        self.assertTrue(config.automatic_function_calling.disable)

    def test_automatic_function_calling_untouched_when_no_tools(self):
        candidate = _fake_candidate(text="hi")
        client = self._client_with_mocked_generate(_fake_response(candidate))
        client.chat([{"role": "user", "content": "hi"}], tools=None)
        config = client._client.models.generate_content.call_args.kwargs["config"]
        self.assertIsNone(config.automatic_function_calling)
        self.assertIsNone(config.tools)

    def test_refusal_like_finish_reason_returns_friendly_message(self):
        candidate = _fake_candidate(finish_reason=types.FinishReason.PROHIBITED_CONTENT)
        client = self._client_with_mocked_generate(_fake_response(candidate))
        resp = client.chat([{"role": "user", "content": "hi"}])
        self.assertIn("not able to help", resp.message.content)
        self.assertIsNone(resp.message.tool_calls)


# ─────────────────────────────────────────────────────────────────────────────
# GeminiClient construction (no network -- Client() doesn't call out until a
# request is actually made)
# ─────────────────────────────────────────────────────────────────────────────

class TestGeminiClientConstruction(unittest.TestCase):
    def test_default_model_is_cheapest_tier(self):
        """Regression guard mirroring the other two clients' Opus/Sol-
        default-mistake guard -- see supervisor_llm_gemini.py's module
        docstring for why gemini-3.5-flash-lite, not gemini-3.1-pro-preview."""
        self.assertEqual(DEFAULT_MODEL, "gemini-3.5-flash-lite")

    def test_stores_model_and_max_output_tokens(self):
        client = GeminiClient(api_key="fake-key", model="gemini-3.8-flash", max_output_tokens=1234)
        self.assertEqual(client.model, "gemini-3.8-flash")
        self.assertEqual(client.max_output_tokens, 1234)


if __name__ == "__main__":
    unittest.main()
