"""The LQR/LQG supervisor conversation loop -- LQG-flavored counterpart to
supervisor_session_pid.py.

Deliberately a separate, smaller class rather than a reuse of
supervisor_session.Session: Session's constructor hardcodes two named tool
slots (whitebox_tool/blackbox_tool) gated by a single boolean
(worksheet.tf_known), which is a PID-specific concept (does the user know
their transfer function or not?) with no LQG analog -- there's one
benchmark tool here, always available, nothing to gate. Forcing this
through Session's two-slot shape would mean either faking a boolean that
means nothing or reworking Session's gating logic and risking the
(tested, working) PID path. This mirrors the project's existing choice to
keep cli_pid.py/cli_pid_blackbox.py/cli_lqg.py as separate flat scripts rather than
a unified dispatcher (docs/lqg_plan.md "Decisions") -- same tradeoff, same
call: duplication now over a shared abstraction bent to fit a shape it
wasn't designed for. Revisit if a third domain needs this pattern too.
"""

from __future__ import annotations

import json

from supervisor_common_pid import FINALIZE_RECOMMENDATION_SCHEMA, make_finalize_recommendation_tool
from supervisor_common_lqg import (
    SET_PRIORITIES_LQG_SCHEMA,
    LQGPrioritiesWorksheet,
    make_set_priorities_lqg_tool,
)
from supervisor_prompts_lqg import SYSTEM_PROMPT_LQG

MAX_TOOL_HOPS = 8
# Derived, not guessed -- the original 6 was introduced for the PID track
# (commit 83887e6, no iterative-tuning concept, all 9 methods in one call)
# and copied here unmodified (94389de) without being re-derived for a
# workflow that can legitimately need more. Worst-realistic-case count for
# this track's own documented job ("Your job, in order" above), now that
# run_lqg_benchmark's Q_diag_list/R_diag_list lets several weightings be
# compared in one call instead of one call each (see
# docs/memos/2026-09-07's robustness memo, "sweep" section):
#   1 establish the plant
# + 1-2 record priorities (set_priorities, possibly called more than once
#        as fields trickle in)
# + 1-2 sweep several weightings, then at most one narrower follow-up
#        sweep/refinement if the first didn't satisfy the goal
# + 0-1 re-run with `reference` set, if the user wants overshoot/rise/
#        settling and didn't give a reference value up front (observed
#        live -- see the memo's Live test 4)
# + 1 finalize_recommendation
# = 4 typical, 7 generous worst case. 8 leaves one hop of margin above
# that derived worst case, not a round number picked by feel.

FALLBACK_MESSAGE = (
    "I'm having trouble finishing that with the tools available -- could you "
    "rephrase, or ask me to just run the benchmark directly?"
)

PARTIAL_FALLBACK_PREFIX = (
    "I ran out of turns before I could weigh these against your priorities "
    "and finish -- here's the last benchmark I ran, so this isn't wasted:"
)

PARTIAL_FALLBACK_SUFFIX = (
    "\n\nJust say \"continue\" and I'll pick up from here with this same "
    "data -- nothing above needs to be repeated."
)


def _summarize_benchmark_result(result: dict) -> str:
    """Plain-text summary of a run_lqg_benchmark result, for the case where
    the conversation loop exhausts its hop budget mid-iteration (see
    docs/memos/2026-09-07's hop-exhaustion memo) -- gives the user the last
    real data point instead of nothing, even though no method was actually
    recommended. Deliberately terse; this is a fallback, not the normal
    conversational summary the LLM would otherwise produce."""
    lines = [
        f"Plant: {result.get('plant_preset')} "
        f"(nx={result.get('nx')}, nu={result.get('nu')}, ny={result.get('ny')})"
    ]
    for row in result.get("rows", []):
        status = "stable" if row.get("stable") else "UNSTABLE"
        checks = ("checks passed" if row.get("all_checks_passed")
                  else f"{row.get('n_checks_failed')} check(s) failed")
        metrics = [f"{k}={row[k]}" for k in ("settling_2pct", "ISU", "u_peak")
                   if row.get(k) is not None]
        metrics_str = f" ({', '.join(metrics)})" if metrics else ""
        lines.append(f"- {row.get('name')}: {status}, {checks}{metrics_str}")
    return "\n".join(lines)


class LQGSession:
    """One conversation.

    `lqg_tool` is a `(schema_dict, callable)` pair, same contract as
    supervisor_session.Session's whitebox_tool/blackbox_tool: the callable
    is called as `fn(**arguments)` and must return a plain JSON-safe dict
    (see supervisor_tools_lqg.run_lqg_benchmark).
    """

    def __init__(self, llm_client, lqg_tool, max_tool_hops=MAX_TOOL_HOPS, capture_plots=False):
        self.client = llm_client
        self.max_tool_hops = max_tool_hops
        self.worksheet = LQGPrioritiesWorksheet()
        self.known_stable_methods = set()
        # Every successful benchmark call, most recent last -- see
        # _wrap_benchmark. A caller like streamlit_llm_panel.py drains
        # this after handle_user_message() to turn it into plottable
        # session entries; nothing here ever reaches the model. Only
        # populated when capture_plots=True -- a plain CLI conversation
        # (cli_supervisor_lqg.py) has no drain step, so left at the
        # default it would otherwise retain every simulated trajectory
        # for the life of the process.
        self.capture_plots = capture_plots
        self.plot_calls = []
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT_LQG}]

        self._lqg_schema, lqg_fn = lqg_tool
        self._benchmark_tool_name = self._lqg_schema["function"]["name"]

        self._tool_fns = {
            self._benchmark_tool_name: self._wrap_benchmark(lqg_fn),
            "set_priorities": make_set_priorities_lqg_tool(self.worksheet),
            "finalize_recommendation": make_finalize_recommendation_tool(self.known_stable_methods),
        }

    def _wrap_benchmark(self, fn):
        """Unlike supervisor_session_pid.Session, there's only one
        benchmark tool here and it always has a real plant (preset or
        user-supplied custom A/B/C/D) -- so whenever self.capture_plots
        is True (see __init__), this asks for return_sim=True and pops
        "_sim_rows" (each row's .sim intact, not JSON-safe) and
        "_custom_plant_literals" (present only for a custom plant --
        see run_lqg_benchmark) off the result before it goes anywhere
        near json.dumps/the model. capture_plots=False (the default)
        asks for return_sim=False instead, skipping that simulation
        work entirely for a caller (e.g. cli_supervisor_lqg.py) with no
        drain step to use it.

        dict(kwargs, return_sim=self.capture_plots) rather than
        fn(**kwargs, return_sim=self.capture_plots): the latter raises
        TypeError if a tool-calling model ever echoes its own return_sim
        argument back (nothing in RUN_LQG_BENCHMARK_SCHEMA forbids that)
        -- dict(...) lets an injected kwarg win instead of crashing.
        Tagging with result.get("plant_name") rather than "plant_preset":
        the preset *key* collapses every custom plant to the constant
        "custom" (see _build_custom_example), so two different custom
        plants benchmarked in one conversation would otherwise get
        identical, indistinguishable plot_calls tags -- plant_name is
        always the actual, distinguishing name."""

        def _wrapped(**kwargs):
            result = fn(**dict(kwargs, return_sim=self.capture_plots))
            sim_rows = result.pop("_sim_rows", None)
            custom_plant_literals = result.pop("_custom_plant_literals", None)
            if result.get("ok") and "rows" in result:
                for row in result["rows"]:
                    if row.get("stable"):
                        self.known_stable_methods.add(row["name"])
            if result.get("ok") and sim_rows is not None:
                self.plot_calls.append({
                    "kind": "mimo", "plant": result.get("plant_name"),
                    "plant_preset": result.get("plant_preset"),
                    "custom_plant_literals": custom_plant_literals,
                    "rows": sim_rows,
                })
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
        last_benchmark_result = None
        for _ in range(self.max_tool_hops):
            resp = self.client.chat(self.messages, tools=self._active_tools())
            self.messages.append(resp.message)
            tool_calls = resp.message.tool_calls or []
            if not tool_calls:
                return resp.message.content or ""
            for tc in tool_calls:
                result = self._dispatch_tool(tc.function.name, tc.function.arguments)
                if tc.function.name == self._benchmark_tool_name and result.get("ok"):
                    last_benchmark_result = result
                self.messages.append({
                    "role": "tool",
                    "tool_name": tc.function.name,
                    "content": json.dumps(result, separators=(",", ":")),
                })
        # Hop budget exhausted. If at least one benchmark call succeeded
        # along the way, surface it instead of a content-free dead end --
        # see docs/memos/2026-09-07's hop-exhaustion memo for why this is
        # reachable even when every tool call the model made succeeded.
        if last_benchmark_result is not None:
            return (f"{PARTIAL_FALLBACK_PREFIX}\n\n"
                    f"{_summarize_benchmark_result(last_benchmark_result)}"
                    f"{PARTIAL_FALLBACK_SUFFIX}")
        return FALLBACK_MESSAGE
