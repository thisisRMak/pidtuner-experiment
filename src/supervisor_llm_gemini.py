"""Thin wrapper over the Gemini API (google-genai SDK), shaped to satisfy
the same interface supervisor_llm.OllamaClient does -- `.chat(messages,
tools=None)` returning an object with `.message.content` (str) and
`.message.tool_calls` (a list of objects with `.function.name`/
`.function.arguments`) -- so supervisor_session_pid.Session and
supervisor_session_lqg.LQGSession work against this client completely
unmodified. Structured the same way supervisor_llm_anthropic.AnthropicClient
and supervisor_llm_openai.OpenAIClient are; this one differs where Gemini's
own API genuinely differs, verified against the *installed* google-genai
2.23.0 package's own source (not just docs/README, which turned out to
disagree with the shipped code on the one point that matters most here --
see the role note below) as of this writing (2026-09):

- **No tool-schema translation needed for the schema shape itself.**
  `FunctionDeclaration` has a `parameters_json_schema` field that accepts
  the same plain `{"type": "object", "properties": {...}}` dict shape
  already used throughout supervisor_tools_*.py (confirmed via
  `types.FunctionDeclaration.model_fields` on the installed package) --
  no Schema-object-with-uppercase-enum-types translation required, unlike
  what older Gemini SDK generations needed.
- **Function-call matching is positional, not id-based, despite `id`
  fields existing on both `FunctionCall` and `FunctionResponse`.** Web
  docs (and even this SDK's own README example) claim/show an id-keyed
  matching scheme; the installed package's *actual, shipped* automatic-
  function-calling implementation (`google/genai/_extra_utils.py`'s
  `get_function_response_parts`, called from `models.py`) never sets `id`
  on the `FunctionResponse` it builds, and correctness instead comes from
  building the response `Part`s in the same order the `FunctionCall`
  `Part`s appeared, all inside one `Content`. This client follows the
  same shipped behavior, not the docs/README -- see `_translate_messages`.
- **`role="user"` for function-response turns, not `"tool"`.** Confirmed
  the same way: `google/genai/models.py`'s own reference implementation
  builds `types.Content(role='user', parts=func_response_parts)` --
  despite the README's own manual-function-calling example showing
  `role='tool'` and `Content.role`'s field docstring itself saying
  "Must be either 'user' or 'model'". Three sources, two different
  answers; trusted the code that actually ships and runs over the doc
  page and the docstring.
- **One `Content` per turn carries every function response, like
  Anthropic's same-turn batching requirement** (not OpenAI's one-message-
  per-result shape) -- matches the shipped `_extra_utils.py` behavior
  above: all of one turn's `function_response` `Part`s go in a single
  `Content(role="user", ...)`.
- **`FunctionCall.args` is already a dict**, like Anthropic's
  `tool_use.input` -- the opposite of OpenAI's JSON-string convention.
  `FunctionResponse.response` also needs a dict, so this client
  `json.loads()`s Session's `json.dumps()`-encoded tool-result string once
  going the other direction.
- **`automatic_function_calling` explicitly disabled whenever tools are
  passed**, not left at its default. Session/LQGSession own tool dispatch
  entirely (the same reason the Anthropic/OpenAI clients are manual, not
  using either SDK's own agentic-loop convenience); this SDK's automatic
  mode only triggers when raw Python callables are passed as tools rather
  than schema-only `FunctionDeclaration`s, so it likely wouldn't fire here
  regardless -- but a real, filed SDK issue (googleapis/python-genai#1818)
  describes this config persisting unexpectedly across calls on the same
  `Client`, so it's set defensively rather than assumed inert.
- **System instruction is a separate `GenerateContentConfig` field**
  (`system_instruction`), like Anthropic's `system=` -- not a message
  role, unlike OpenAI's Chat Completions.

Model default: gemini-3.5-flash-lite (cheapest current tier per Google's
own pricing docs, mirroring the Haiku/gpt-5.6-luna default choice in the
other two clients -- gemini-3.1-pro-preview deliberately not offered as a
default/option here, same "most expensive tier, no reason to offer it"
reasoning as excluding Opus/gpt-5.6-sol/gpt-6-astra).
"""

from __future__ import annotations

from collections import defaultdict
import json

from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-3.5-flash-lite"
DEFAULT_MAX_OUTPUT_TOKENS = 16000

# Finish reasons that mean "the model declined to produce real content" --
# Gemini has no single unified "refusal" stop reason the way Anthropic's
# Messages API does, so this is the closest analogue: any of these means
# candidate.content isn't a normal reply to surface as-is.
_REFUSAL_FINISH_REASONS = {
    types.FinishReason.SAFETY,
    types.FinishReason.PROHIBITED_CONTENT,
    types.FinishReason.BLOCKLIST,
    types.FinishReason.SPII,
    types.FinishReason.RECITATION,
}


class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _Function(name, arguments)


class _AssistantMessage:
    """What this client appends as `resp.message`. `content`/`tool_calls`
    are the Ollama-shaped surface Session reads; `raw_content` is the
    exact `types.Content` the API returned (role="model") -- extra data
    Session never touches but this client's own `_translate_messages`
    replays verbatim on the next hop, the same role _AssistantMessage
    plays in the Anthropic/OpenAI clients."""

    def __init__(self, content, tool_calls, raw_content):
        self.content = content
        self.tool_calls = tool_calls or None
        self.raw_content = raw_content


class _Response:
    def __init__(self, message):
        self.message = message


def _to_gemini_tool(tools):
    """Ollama/OpenAI-shaped {"type": "function", "function": {...}} tool
    schemas -> one types.Tool bundling a FunctionDeclaration per tool,
    using parameters_json_schema (see module docstring) so the existing
    plain-dict schemas in supervisor_tools_*.py need no reshaping."""
    declarations = [
        types.FunctionDeclaration(
            name=t["function"]["name"],
            description=t["function"].get("description", ""),
            parameters_json_schema=t["function"]["parameters"],
        )
        for t in tools
    ]
    return types.Tool(function_declarations=declarations)


def _translate_messages(messages):
    """Session's `self.messages` list -> (system_text, gemini_contents).
    See module docstring for why this batches a turn's tool results into
    one role="user" Content in call order, rather than id-matching or
    per-message results -- both would depart from google-genai's own
    shipped (not just documented) behavior."""
    system_text = None
    gemini_contents = []
    pending_response_parts = []

    def flush_tool_results():
        if pending_response_parts:
            gemini_contents.append(types.Content(role="user", parts=list(pending_response_parts)))
            pending_response_parts.clear()

    for entry in messages:
        if isinstance(entry, _AssistantMessage):
            flush_tool_results()
            gemini_contents.append(entry.raw_content)
            continue

        role = entry.get("role")
        if role == "system":
            system_text = entry["content"]
        elif role == "tool":
            pending_response_parts.append(types.Part.from_function_response(
                name=entry["tool_name"], response=json.loads(entry["content"]),
            ))
        else:
            flush_tool_results()
            gemini_contents.append(types.Content(role=role, parts=[types.Part(text=entry["content"])]))

    flush_tool_results()
    return system_text, gemini_contents


class GeminiClient:
    def __init__(self, api_key, model=DEFAULT_MODEL, max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self._client = genai.Client(api_key=api_key)

    def chat(self, messages, tools=None):
        """Return a _Response whose `.message` matches OllamaClient's
        `.chat()` return shape -- see module docstring."""
        system_text, contents = _translate_messages(messages)
        config_kwargs = {"max_output_tokens": self.max_output_tokens}
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if tools:
            config_kwargs["tools"] = [_to_gemini_tool(tools)]
            config_kwargs["automatic_function_calling"] = types.AutomaticFunctionCallingConfig(disable=True)

        response = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config_kwargs),
        )

        candidate = response.candidates[0]
        raw_content = candidate.content
        if candidate.finish_reason in _REFUSAL_FINISH_REASONS or raw_content is None:
            return _Response(_AssistantMessage(
                "I'm not able to help with that request.", [], raw_content,
            ))

        parts = raw_content.parts or []
        text = "".join(p.text for p in parts if p.text)
        tool_calls = [
            _ToolCall(p.function_call.name, p.function_call.args or {})
            for p in parts if p.function_call
        ]
        return _Response(_AssistantMessage(text, tool_calls, raw_content))
