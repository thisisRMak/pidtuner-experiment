"""Thin wrapper over the Claude Messages API, shaped to satisfy the same
interface supervisor_llm.OllamaClient does -- `.chat(messages, tools=None)`
returning an object with `.message.content` (str) and `.message.tool_calls`
(a list of objects with `.function.name`/`.function.arguments`) -- so
supervisor_session_pid.Session and supervisor_session_lqg.LQGSession work
against this client completely unmodified.

Why this isn't just "swap the client": Claude's tool-calling API is
ID-based (each `tool_use` block carries an `id`, matched by
`tool_result.tool_use_id` on the way back), but Session threads tool
results back by `tool_name` only (see supervisor_llm.py's docstring for
why -- that's an Ollama-package limitation, not a Session design choice).
This client resolves the mismatch itself, invisibly to Session:

- Every assistant turn this client returns is a `_AssistantMessage`
  instance carrying the *real* Anthropic content blocks (including each
  tool_use block's real `id`) alongside the Ollama-shaped `.tool_calls`
  Session expects. Session appends this object into its own `messages`
  list verbatim (it never introspects it) -- exactly like it does with
  OllamaClient's raw `ollama.Message`.
- On the *next* `.chat()` call, this client walks that same mixed
  `messages` list itself (not Ollama, not Session) and recognizes its own
  `_AssistantMessage` objects by type, pulling the real tool_use blocks
  straight off them -- so no name-based guessing is needed for the
  assistant-turn round-trip.
- The one place name-based matching still happens: correlating Session's
  `{"role": "tool", "tool_name": ...}` result dicts back to a
  `tool_use_id`. This is resolved against the tool_use blocks of the
  immediately preceding assistant turn only (never across turns), which
  is safe as long as that turn didn't call the same tool name twice --
  true today, since Session's `_active_tools()` only ever offers one
  benchmark tool at a time (see supervisor_session_pid.py) and LQGSession
  offers exactly one tool total. Revisit this note if that ever changes.

Model default: claude-haiku-4-5 (cheapest tier; streamlit_llm_panel.py's
model picker offers Haiku/Sonnet and always passes model= explicitly, so
this default only matters for a bare AnthropicClient() elsewhere, e.g. a
future CLI script or a test). Tool-use response shape verified against the
claude-api skill's Python docs as of this writing -- re-verify if the
Anthropic SDK's tool-use response shape changes.
"""

from __future__ import annotations

import anthropic

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 16000


class _Function:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, arguments):
        self.function = _Function(name, arguments)


class _AssistantMessage:
    """What this client appends as `resp.message`. `content`/`tool_calls`
    are the Ollama-shaped surface Session reads; `raw_blocks`/
    `tool_use_blocks` are extra data Session never touches but this
    client's own `_translate_messages` reads back out on the next hop."""

    def __init__(self, content, tool_use_blocks, raw_blocks):
        self.content = content
        self.tool_calls = [
            _ToolCall(b.name, b.input) for b in tool_use_blocks
        ] or None
        self.tool_use_blocks = tool_use_blocks
        self.raw_blocks = raw_blocks


class _Response:
    def __init__(self, message):
        self.message = message


def _to_anthropic_tools(tools):
    """Ollama/OpenAI-shaped {"type": "function", "function": {...}} tool
    schemas, as used throughout supervisor_tools_*.py, translated to
    Anthropic's flat {"name", "description", "input_schema"} shape."""
    converted = []
    for t in tools:
        fn = t["function"]
        converted.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn["parameters"],
        })
    return converted


def _translate_messages(messages):
    """Session's `self.messages` list -> (system_text, anthropic_messages).
    See module docstring for how assistant turns and tool results round-trip."""
    system_text = None
    anthropic_messages = []
    pending_tool_results = []
    last_tool_use_by_name = {}

    def flush_tool_results():
        if pending_tool_results:
            anthropic_messages.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results.clear()

    for entry in messages:
        if isinstance(entry, _AssistantMessage):
            flush_tool_results()
            anthropic_messages.append({"role": "assistant", "content": entry.raw_blocks})
            last_tool_use_by_name = {b.name: b.id for b in entry.tool_use_blocks}
            continue

        role = entry.get("role")
        if role == "system":
            system_text = entry["content"]
        elif role == "tool":
            tool_use_id = last_tool_use_by_name.get(entry["tool_name"])
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": entry["content"],
            })
        else:
            flush_tool_results()
            anthropic_messages.append({"role": role, "content": entry["content"]})

    flush_tool_results()
    return system_text, anthropic_messages


class AnthropicClient:
    def __init__(self, api_key, model=DEFAULT_MODEL, max_tokens=DEFAULT_MAX_TOKENS):
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def chat(self, messages, tools=None):
        """Return a _Response whose `.message` matches OllamaClient's
        `.chat()` return shape -- see module docstring."""
        system_text, anthropic_messages = _translate_messages(messages)
        kwargs = {}
        if tools:
            kwargs["tools"] = _to_anthropic_tools(tools)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_text,
            messages=anthropic_messages,
            **kwargs,
        )

        if response.stop_reason == "refusal":
            return _Response(_AssistantMessage(
                "I'm not able to help with that request.", [], response.content,
            ))

        text = "".join(b.text for b in response.content if b.type == "text")
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        return _Response(_AssistantMessage(text, tool_use_blocks, response.content))
