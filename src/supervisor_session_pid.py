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
        # Every successful sim-capable benchmark call, most recent last --
        # see _wrap_benchmark. A caller like streamlit_llm_panel.py drains
        # this after handle_user_message() to turn it into plottable
        # session entries; nothing here ever reaches the model.
        self.plot_calls = []
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        self._whitebox_schema, whitebox_fn = whitebox_tool
        self._blackbox_schema, blackbox_fn = blackbox_tool

        self._tool_fns = {
            self._whitebox_schema["function"]["name"]: self._wrap_benchmark(
                whitebox_fn, kind="siso", plant_of=lambda kwargs: kwargs.get("plant_tf")),
            self._blackbox_schema["function"]["name"]: self._wrap_benchmark(blackbox_fn),
            "set_priorities": make_set_priorities_tool(self.worksheet),
            "finalize_recommendation": make_finalize_recommendation_tool(self.known_stable_methods),
        }

    def _wrap_benchmark(self, fn, kind=None, plant_of=None):
        """Feed the grounding cache from every successful benchmark call,
        whichever entity it came from -- whitebox rows use 'stable',
        blackbox rows use 'available'.

        `kind`/`plant_of` are given only for a benchmark tool that can
        produce plottable simulation traces -- currently just the
        white-box tool. The black-box tool has no ground-truth plant to
        simulate against (see supervisor_tools_blackbox_pid.py's isolation
        contract), so it's called plain, with no return_sim and nothing
        appended to plot_calls. When given, every successful call asks the
        tool for return_sim=True and pops "_sim_rows" (row["sim"] intact,
        not JSON-safe) off the result before it goes anywhere near
        json.dumps/the model -- the model only ever sees exactly what it
        saw before this existed. plot_calls also carries the call's raw
        "delay" kwarg (0.0 if the model never passed one) alongside
        "plant" -- both are the exact strings/numbers TransferFunction.
        parse() takes, not a display-formatted plant, so a caller can
        re-derive the plant this call actually ran against."""

        def _wrapped(**kwargs):
            call_kwargs = dict(kwargs, return_sim=True) if kind else kwargs
            result = fn(**call_kwargs)
            sim_rows = result.pop("_sim_rows", None) if kind else None
            if result.get("ok") and "rows" in result:
                for row in result["rows"]:
                    if row.get("stable", row.get("available")):
                        self.known_stable_methods.add(row["name"])
            if result.get("ok") and sim_rows is not None:
                self.plot_calls.append({
                    "kind": kind, "plant": plant_of(kwargs),
                    "delay": kwargs.get("delay", 0.0), "rows": sim_rows,
                })
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
