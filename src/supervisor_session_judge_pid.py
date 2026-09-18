"""LLM-as-judge orchestration for PID/SISO Judge Mode: fans one user
message out to N already-constructed candidate `Session`s in lockstep,
then asks a separate judge model to mediate/arbitrate over their current
state. See docs/memos/2026-09-14/2026-09-14-recommendation-rationale-
discrepancy-memo.md for the motivating failure mode (a model finalizing a
recommendation that contradicts its own tool data) and supervisor_llm_
{anthropic,openai,gemini}.py for the client interface this reuses as-is.

Deliberately does NOT give the judge any tools of its own. The whole
point is that the judge fact-checks candidates against numbers *they*
already fetched -- every benchmark result a judge could need is already
sitting in a candidate's trace, since all candidates were given the same
prompt against the same deterministic benchmark. Giving the judge its own
whitebox/blackbox tool would let it run a redundant benchmark call
instead of reading the ones already made, adding cost and a second grader
of the same facts for no benefit -- so `judge_client.chat()` is always
called with `tools=None`, and this module never imports the tool schemas.

Session model: fresh, synchronized start, then lockstep. `JudgeSession`
doesn't build candidate Sessions or the judge client itself -- the caller
(streamlit_judge_panel.py) constructs all of them up front, exactly the
way streamlit_llm_panel.py already builds a single-provider Session, and
hands this class already-built objects to orchestrate. This keeps
provider dispatch in exactly one place (streamlit_llm_panel._new_session /
streamlit_judge_panel._build_judge_client), not duplicated here.

One candidate's failure must never sink the whole round: a provider's own
transient error (rate limit, a real capacity 503, ...) is exactly the
kind of thing that happens to exactly one of N independent API calls, not
all of them at once. `_fan_out` catches per-candidate, so a live 503 from
one provider still lets the other candidates -- and the judge, reasoning
over whoever did respond -- produce a normal answer. Every round's outcome
(what each candidate did, or how it failed) is kept on `self.last_round`
for the caller to display -- see streamlit_judge_panel.py's transparency
expander.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

from supervisor_prompts_judge_pid import JUDGE_SYSTEM_PROMPT


def serialize_candidate_trace(label: str, session, error: str | None = None) -> str:
    """One candidate's tool calls, tool results, and latest reply text, as
    a plain-text block the judge reads -- not the user turns (every
    candidate got the identical user messages the judge already has via
    its own dialogue history, so repeating them once per candidate would
    just burn tokens) and not the system prompt (the judge prompt already
    describes what a candidate is).

    `error`, when given, means this candidate failed to respond THIS round
    (see _fan_out) -- its trace still reflects whatever it did in earlier
    rounds (so the judge isn't blind to its prior finalized pick, if any),
    with the failure appended as an explicit note rather than silently
    reusing a stale reply as if it were current.

    Walks `session.messages` duck-typed on `.content`/`.tool_calls` for
    assistant turns, not any provider-specific `_AssistantMessage` type --
    every client in supervisor_llm_{anthropic,openai,gemini}.py (and
    supervisor_llm.py's OllamaClient) already exposes exactly that surface
    on the object it appends as an assistant turn, by design (see each
    module's own docstring) -- so this works across all of them
    unmodified, the same way supervisor_session_pid.Session itself does.
    """
    lines = [f"=== Candidate: {label} ==="]
    for entry in session.messages:
        if isinstance(entry, dict):
            role = entry.get("role")
            if role == "tool":
                lines.append(f"  tool result ({entry['tool_name']}): {entry['content']}")
            elif role != "system" and role != "user":
                lines.append(f"  {role}: {entry['content']}")
            continue
        tool_calls = getattr(entry, "tool_calls", None) or []
        for tc in tool_calls:
            lines.append(f"  assistant tool call: {tc.function.name}({json.dumps(tc.function.arguments)})")
        text = getattr(entry, "content", "") or ""
        if text:
            lines.append(f"  assistant: {text}")
    if error:
        lines.append(f"  [FAILED TO RESPOND THIS ROUND -- provider error: {error}]")
    return "\n".join(lines)


def summarize_candidate_round(session, since: int) -> dict:
    """Short, human-facing summary of what one candidate did since message
    index `since` (its `session.messages` length right before this round's
    fan-out) -- for the UI's transparency panel, not the judge's prompt
    (see serialize_candidate_trace for that, which is full-history and
    machine-oriented). `finalized` is the method_name argument of a
    finalize_recommendation call made this round, if any -- pulled from
    the real tool-call arguments, not guessed from prose, so the UI can
    show "-> recommends X" without parsing free text."""
    calls = []
    finalized = None
    reply_text = ""
    for entry in session.messages[since:]:
        if isinstance(entry, dict):
            continue
        for tc in getattr(entry, "tool_calls", None) or []:
            args = tc.function.arguments or {}
            calls.append((tc.function.name, args))
            if tc.function.name == "finalize_recommendation":
                finalized = args.get("method_name")
        text = getattr(entry, "content", "") or ""
        if text:
            reply_text = text
    return {"calls": calls, "finalized": finalized, "reply": reply_text}


def _short_error(exc: Exception) -> str:
    """A short, human-readable message for a candidate's failed call this
    round. Every provider exception this project handles at the top level
    (streamlit_judge_panel.py's own except cascade) already carries a
    `.message` with the provider's real text (e.g. "This model is
    currently experiencing high demand...") -- reused here via getattr
    rather than importing anthropic/openai/genai_errors' types, since this
    module doesn't otherwise need to know provider exception hierarchies
    at all."""
    message = getattr(exc, "message", None)
    if message:
        return str(message)
    return str(exc) or type(exc).__name__


class JudgeSession:
    """One judged conversation: N candidate `(label, Session)` pairs plus
    a bare judge client. `.dialogue` is the plain human<->judge exchange
    only (never the trace dump) -- rebuilt fresh into the judge's *outgoing*
    message each round from the candidates' current state instead of
    accumulated, so context grows with "current state" (bounded) rather
    than "sum over every round so far" (unbounded). `.last_round`, set at
    the end of every handle_user_message() call, is a list of per-candidate
    dicts ({"label", "error"} or {"label", "calls", "finalized", "reply"})
    -- the caller's own record of "what actually happened," for display."""

    def __init__(self, candidates, judge_client):
        self.candidates = candidates
        self.judge_client = judge_client
        self.dialogue = []
        self.last_round = []

    def _fan_out(self, text: str) -> dict:
        """Returns {label: error_message} for whichever candidates failed
        this round -- empty if everyone succeeded. Each candidate's own
        Session.handle_user_message() call is independent (see the module
        docstring on why concurrent calls across candidates are safe), so
        one raising doesn't stop or corrupt the others."""
        errors = {}

        def _one(item):
            label, session = item
            try:
                session.handle_user_message(text)
            except Exception as exc:  # noqa: BLE001 - one candidate's failure must not sink the round
                errors[label] = _short_error(exc)

        with ThreadPoolExecutor(max_workers=len(self.candidates)) as pool:
            list(pool.map(_one, self.candidates))
        return errors

    def handle_user_message(self, text: str) -> str:
        before = {label: len(session.messages) for label, session in self.candidates}
        errors = self._fan_out(text)

        self.last_round = [
            {"label": label, "error": errors[label]} if label in errors
            else {"label": label, **summarize_candidate_round(session, before[label])}
            for label, session in self.candidates
        ]

        if len(errors) == len(self.candidates):
            # Nothing to judge -- every candidate failed this round (e.g.
            # a shared provider outage, or all N hitting rate limits at
            # once). Skip the judge call entirely rather than asking it to
            # arbitrate over zero real answers.
            failure_lines = "\n".join(f"- {label}: {msg}" for label, msg in errors.items())
            reply = (
                f"All {len(self.candidates)} candidates failed to respond this round -- "
                f"nothing to judge yet. Try again in a moment.\n\n{failure_lines}"
            )
            self.dialogue.append({"role": "user", "content": text})
            self.dialogue.append({"role": "assistant", "content": reply})
            return reply

        trace_block = "\n\n".join(
            serialize_candidate_trace(label, session, error=errors.get(label))
            for label, session in self.candidates
        )
        judge_messages = [{"role": "system", "content": JUDGE_SYSTEM_PROMPT}]
        judge_messages += self.dialogue
        judge_messages.append({
            "role": "user",
            "content": (
                f"{text}\n\n---\nCandidate model traces as of this round "
                f"(tool calls, results, and each candidate's current reply):\n{trace_block}"
            ),
        })

        resp = self.judge_client.chat(judge_messages, tools=None)
        judge_reply = resp.message.content or ""

        self.dialogue.append({"role": "user", "content": text})
        self.dialogue.append({"role": "assistant", "content": judge_reply})
        return judge_reply
