"""The supervisor conversation loop: message-list management, active-tools
gating (which benchmark tool, if any, the model is currently allowed to
call), and the grounding cache that ties finalize_recommendation to method
names a benchmark tool actually returned.

No imports from plant.py, signal_format.py, or pid_blackbox.py -- the white-box
and black-box tool implementations are injected by the caller as
(schema, callable) pairs, so this module never has to choose sides on the
entity-isolation boundary itself.
"""

from __future__ import annotations

import json

from supervisor_common_pid import (
    FINALIZE_RECOMMENDATION_SCHEMA,
    SET_PRIORITIES_SCHEMA,
    PrioritiesWorksheet,
    make_finalize_recommendation_tool,
    make_set_priorities_tool,
)
from supervisor_prompts_pid import SYSTEM_PROMPT

MAX_TOOL_HOPS = 6

FALLBACK_MESSAGE = (
    "I'm having trouble finishing that with the tools available -- could you "
    "rephrase, or ask me to just run the benchmark directly?"
)


class Session:
    """One conversation.

    `whitebox_tool` / `blackbox_tool` are `(schema_dict, callable)` pairs.
    The callable is called as `fn(**arguments)` and must return a plain
    JSON-safe dict (see supervisor_tools_whitebox.run_whitebox_benchmark /
    supervisor_tools_blackbox.run_blackbox_benchmark).
    """

    def __init__(self, llm_client, whitebox_tool, blackbox_tool,
                 max_tool_hops=MAX_TOOL_HOPS):
        self.client = llm_client
        self.max_tool_hops = max_tool_hops
        self.worksheet = PrioritiesWorksheet()
        self.known_stable_methods = set()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        self._whitebox_schema, whitebox_fn = whitebox_tool
        self._blackbox_schema, blackbox_fn = blackbox_tool

        self._tool_fns = {
            self._whitebox_schema["function"]["name"]: self._wrap_benchmark(whitebox_fn),
            self._blackbox_schema["function"]["name"]: self._wrap_benchmark(blackbox_fn),
            "set_priorities": make_set_priorities_tool(self.worksheet),
            "finalize_recommendation": make_finalize_recommendation_tool(self.known_stable_methods),
        }

    def _wrap_benchmark(self, fn):
        """Feed the grounding cache from every successful benchmark call,
        whichever entity it came from -- whitebox rows use 'stable',
        blackbox rows use 'available'."""

        def _wrapped(**kwargs):
            result = fn(**kwargs)
            if result.get("ok") and "rows" in result:
                for row in result["rows"]:
                    if row.get("stable", row.get("available")):
                        self.known_stable_methods.add(row["name"])
            return result

        return _wrapped

    def _active_tools(self):
        """Recomputed every loop iteration. Exactly one benchmark tool is
        ever offered at a time, gated on worksheet.tf_known -- this is what
        makes it structurally impossible for a black-box conversation to
        call the white-box tool (and vice versa), not just discouraged by
        the system prompt."""
        tools = [SET_PRIORITIES_SCHEMA, FINALIZE_RECOMMENDATION_SCHEMA]
        if self.worksheet.tf_known is True:
            tools.insert(0, self._whitebox_schema)
        elif self.worksheet.tf_known is False:
            tools.insert(0, self._blackbox_schema)
        return tools

    def _dispatch_tool(self, name, arguments):
        active_names = {t["function"]["name"] for t in self._active_tools()}
        if name not in active_names:
            return {
                "ok": False,
                "error": f"{name!r} is not available right now (call set_priorities first).",
            }
        fn = self._tool_fns.get(name)
        if fn is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        try:
            return fn(**(arguments or {}))
        except Exception as exc:  # noqa: BLE001 - a bad tool call must not crash the session
            return {"ok": False, "error": str(exc)}

    def handle_user_message(self, text: str) -> str:
        self.messages.append({"role": "user", "content": text})
        for _ in range(self.max_tool_hops):
            resp = self.client.chat(self.messages, tools=self._active_tools())
            self.messages.append(resp.message)
            tool_calls = resp.message.tool_calls or []
            if not tool_calls:
                return resp.message.content or ""
            for tc in tool_calls:
                result = self._dispatch_tool(tc.function.name, tc.function.arguments)
                self.messages.append({
                    "role": "tool",
                    "tool_name": tc.function.name,
                    "content": json.dumps(result, separators=(",", ":")),
                })
        return FALLBACK_MESSAGE
