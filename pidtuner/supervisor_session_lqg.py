"""The LQR/LQG supervisor conversation loop -- LQG-flavored counterpart to
supervisor_session.py.

Deliberately a separate, smaller class rather than a reuse of
supervisor_session.Session: Session's constructor hardcodes two named tool
slots (whitebox_tool/blackbox_tool) gated by a single boolean
(worksheet.tf_known), which is a PID-specific concept (does the user know
their transfer function or not?) with no LQG analog -- there's one
benchmark tool here, always available, nothing to gate. Forcing this
through Session's two-slot shape would mean either faking a boolean that
means nothing or reworking Session's gating logic and risking the
(tested, working) PID path. This mirrors the project's existing choice to
keep cli.py/cli_blackbox.py/cli_lqg.py as separate flat scripts rather than
a unified dispatcher (docs/lqg_plan.md "Decisions") -- same tradeoff, same
call: duplication now over a shared abstraction bent to fit a shape it
wasn't designed for. Revisit if a third domain needs this pattern too.
"""

from __future__ import annotations

import json

from supervisor_common import FINALIZE_RECOMMENDATION_SCHEMA, make_finalize_recommendation_tool
from supervisor_common_lqg import (
    SET_PRIORITIES_LQG_SCHEMA,
    LQGPrioritiesWorksheet,
    make_set_priorities_lqg_tool,
)
from supervisor_prompts_lqg import SYSTEM_PROMPT_LQG

MAX_TOOL_HOPS = 6

FALLBACK_MESSAGE = (
    "I'm having trouble finishing that with the tools available -- could you "
    "rephrase, or ask me to just run the benchmark directly?"
)


class LQGSession:
    """One conversation.

    `lqg_tool` is a `(schema_dict, callable)` pair, same contract as
    supervisor_session.Session's whitebox_tool/blackbox_tool: the callable
    is called as `fn(**arguments)` and must return a plain JSON-safe dict
    (see supervisor_tools_lqg.run_lqg_benchmark).
    """

    def __init__(self, llm_client, lqg_tool, max_tool_hops=MAX_TOOL_HOPS):
        self.client = llm_client
        self.max_tool_hops = max_tool_hops
        self.worksheet = LQGPrioritiesWorksheet()
        self.known_stable_methods = set()
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT_LQG}]

        self._lqg_schema, lqg_fn = lqg_tool

        self._tool_fns = {
            self._lqg_schema["function"]["name"]: self._wrap_benchmark(lqg_fn),
            "set_priorities": make_set_priorities_lqg_tool(self.worksheet),
            "finalize_recommendation": make_finalize_recommendation_tool(self.known_stable_methods),
        }

    def _wrap_benchmark(self, fn):
        def _wrapped(**kwargs):
            result = fn(**kwargs)
            if result.get("ok") and "rows" in result:
                for row in result["rows"]:
                    if row.get("stable"):
                        self.known_stable_methods.add(row["name"])
            return result

        return _wrapped

    def _active_tools(self):
        """Unlike Session._active_tools, nothing is gated here -- the one
        benchmark tool is always available, there's no mode to lock in
        first."""
        return [self._lqg_schema, SET_PRIORITIES_LQG_SCHEMA, FINALIZE_RECOMMENDATION_SCHEMA]

    def _dispatch_tool(self, name, arguments):
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
