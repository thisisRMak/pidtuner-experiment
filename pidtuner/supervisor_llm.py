"""Thin wrapper over the local Ollama chat API. No PID-domain knowledge and
no tool implementations live here -- just message/tool plumbing.

Empirically verified against the locally installed `ollama` package
(v0.6.2) and a running `qwen3-coder:30b`: tool_calls come back as
`resp.message.tool_calls`, each a `ToolCall(function=Function(name=...,
arguments=<dict>))` -- this package version does NOT expose a per-call id,
unlike Ollama's raw HTTP /api/chat response. Tool results are threaded back
by `tool_name`, not by call id (see supervisor_session.py).
"""

from __future__ import annotations

import ollama

DEFAULT_MODEL = "qwen3-coder:30b"
DEFAULT_NUM_CTX = 8192
DEFAULT_KEEP_ALIVE = "30m"


class OllamaClient:
    def __init__(self, model=DEFAULT_MODEL, host=None,
                 num_ctx=DEFAULT_NUM_CTX, keep_alive=DEFAULT_KEEP_ALIVE):
        self.model = model
        self.num_ctx = num_ctx
        self.keep_alive = keep_alive
        self._client = ollama.Client(host=host) if host else ollama

    def chat(self, messages, tools=None):
        """Return the raw ollama.ChatResponse for one round of the loop."""
        return self._client.chat(
            model=self.model,
            messages=messages,
            tools=tools,
            options={"num_ctx": self.num_ctx},
            keep_alive=self.keep_alive,
        )
