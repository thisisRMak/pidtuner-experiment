#!/usr/bin/env python3
"""Interactive CLI for LLM-as-judge MIMO/LQG Judge Mode: fans one user
message out to N candidate supervisor_session_lqg.LQGSession instances and
asks a separate judge model to mediate every round. Thin over
supervisor_session_judge_pid.JudgeSession, the same way cli_supervisor_lqg.py
is thin over supervisor_session_lqg.LQGSession -- the orchestration itself
(fan-out, per-candidate error isolation, judge arbitration) is not
reimplemented here; see that module's docstring for how it works.
MIMO/LQG track counterpart to cli_supervisor_judge_pid.py (SISO/PID
track), the same way cli_supervisor_lqg.py already mirrors
cli_supervisor_pid.py for the single-provider case.

Provider/key resolution follows cli_supervisor_pid.py's own
_resolve_anthropic_key/_resolve_openai_key/_resolve_gemini_key pattern
exactly (an explicit --api-key-<provider> wins outright, otherwise
.env/the process environment), just with three separate flags instead of
one shared --api-key -- a judge conversation routinely spans more than
one provider at once (candidates and judge can each be a different
vendor), where a single --api-key flag would be ambiguous about which
provider it's for.

Candidates and judge are given as provider[:model] specs (e.g.
"anthropic:claude-haiku-4-5" or bare "anthropic" for that provider's
default model) via repeatable --candidate and a single required --judge.
Same two invariants cli_supervisor_judge_pid.py's own --validate-selection
enforce -- at least 2 candidates, and the judge's exact (provider, model)
pair must differ from every candidate's -- are checked explicitly at
startup here (argparse's own per-flag `type` validation covers "is this
a known provider"; the cross-flag invariants can't be expressed that way
and are checked after parsing), not left to fail later inside
JudgeSession or silently produce a judge that's really just another
candidate.

Only the judge's reply is printed by default, matching the SISO/PID CLI's
"only the judge's reply is the primary output" rule -- per-candidate
detail (what each candidate actually did this round: tool calls, its
finalized pick if any, its reply) is opt-in, either automatically every
turn via --verbose or on demand via /candidates, both rendering the same
JudgeSession.last_round data as plain text.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

import supervisor_cli_output as clio
import supervisor_cli_plots as clip
from supervisor_llm_anthropic import AnthropicClient, DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from supervisor_llm_openai import OpenAIClient, DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from supervisor_llm_gemini import GeminiClient, DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from supervisor_session_lqg import LQGSession
from supervisor_session_judge_pid import JudgeSession
from supervisor_prompts_judge_lqg import JUDGE_SYSTEM_PROMPT_LQG
from supervisor_tools_lqg import RUN_LQG_BENCHMARK_SCHEMA, run_lqg_benchmark

LQG_TOOL = (RUN_LQG_BENCHMARK_SCHEMA, run_lqg_benchmark)

VALID_PROVIDERS = ("anthropic", "openai", "gemini")
DISPLAY_NAME = {"anthropic": "Anthropic", "openai": "OpenAI", "gemini": "Gemini"}
DEFAULT_MODEL = {
    "anthropic": ANTHROPIC_DEFAULT_MODEL,
    "openai": OPENAI_DEFAULT_MODEL,
    "gemini": GEMINI_DEFAULT_MODEL,
}


def _parse_provider_model(spec: str) -> tuple[str, str | None]:
    """argparse `type=` for --candidate/--judge: "provider" or
    "provider:model". Raises ArgumentTypeError (argparse's own mechanism
    for a bad flag value) for an unknown provider, so a typo is caught at
    parse time with a normal argparse usage error rather than surfacing
    later as an opaque failure while building a client."""
    provider, _, model = spec.partition(":")
    if provider not in VALID_PROVIDERS:
        raise argparse.ArgumentTypeError(
            f"invalid provider {provider!r} in {spec!r} -- must be one of {', '.join(VALID_PROVIDERS)}"
        )
    return provider, (model or None)


def _resolve_anthropic_key(explicit_key):
    """Resolve an Anthropic API key: an explicit --api-key-anthropic wins
    outright and skips the lookup below entirely; otherwise fall back to
    .env/the process environment's ANTHROPIC_API_KEY. Mirrors
    cli_supervisor_pid.py's own _resolve_anthropic_key exactly."""
    if explicit_key:
        return explicit_key
    load_dotenv()
    return os.environ.get("ANTHROPIC_API_KEY")


def _resolve_openai_key(explicit_key):
    """Same resolution order as _resolve_anthropic_key above, for
    OPENAI_API_KEY instead."""
    if explicit_key:
        return explicit_key
    load_dotenv()
    return os.environ.get("OPENAI_API_KEY")


def _resolve_gemini_key(explicit_key):
    """Same resolution order as _resolve_anthropic_key above, for Gemini --
    GOOGLE_API_KEY first, then GEMINI_API_KEY, matching the google-genai
    SDK's own precedence (see cli_supervisor_pid.py's identical function
    for where that precedence was confirmed)."""
    if explicit_key:
        return explicit_key
    load_dotenv()
    return os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")


_RESOLVE_KEY = {
    "anthropic": _resolve_anthropic_key,
    "openai": _resolve_openai_key,
    "gemini": _resolve_gemini_key,
}
_ENV_VAR_HELP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GOOGLE_API_KEY (or GEMINI_API_KEY)",
}


def _resolve_key(provider: str, args) -> str:
    explicit = {"anthropic": args.api_key_anthropic, "openai": args.api_key_openai,
                "gemini": args.api_key_gemini}[provider]
    key = _RESOLVE_KEY[provider](explicit)
    if not key:
        print(f"error: no {DISPLAY_NAME[provider]} API key found -- pass --api-key-{provider}, "
              f"set {_ENV_VAR_HELP[provider]}, or add it to a .env file", file=sys.stderr)
        sys.exit(1)
    return key


def _client_for(provider: str, model: str, api_key: str):
    if provider == "openai":
        return OpenAIClient(api_key=api_key, model=model)
    if provider == "gemini":
        return GeminiClient(api_key=api_key, model=model)
    return AnthropicClient(api_key=api_key, model=model)


def _label(provider: str, model: str) -> str:
    return f"{provider}:{model}"


def _validate_selection(candidates, judge):
    """Resolves every provider[:model] spec's default model, then checks
    the two invariants cli_supervisor_judge_pid.py's own version also
    enforces: at least 2 candidates, and the judge's exact
    (provider, model) pair differs from every candidate's -- resolved
    pairs, not raw specs, so a candidate given as bare "anthropic" and a
    judge given as "anthropic:<that provider's actual default>" are still
    caught as the same pair. Exits with a hard error (not a silent
    fallback) on any violation, same as a missing API key does in
    _resolve_key -- this is meant to be caught at startup, not three
    turns into a conversation. Returns the resolved (provider, model)
    lists on success."""
    if not candidates or len(candidates) < 2:
        got = len(candidates) if candidates else 0
        print(f"error: need at least 2 --candidate entries (got {got})", file=sys.stderr)
        sys.exit(1)

    resolved_candidates = [(p, m or DEFAULT_MODEL[p]) for p, m in candidates]
    seen = set()
    for pair in resolved_candidates:
        if pair in seen:
            print(f"error: duplicate candidate {_label(*pair)!r} -- each --candidate must be distinct",
                  file=sys.stderr)
            sys.exit(1)
        seen.add(pair)

    judge_provider, judge_model = judge
    resolved_judge = (judge_provider, judge_model or DEFAULT_MODEL[judge_provider])
    if resolved_judge in resolved_candidates:
        print(f"error: --judge {_label(*resolved_judge)!r} must differ from every candidate's "
              f"exact provider+model pair -- it matches a candidate", file=sys.stderr)
        sys.exit(1)

    return resolved_candidates, resolved_judge


def _new_judge_session(candidate_clients, judge_client) -> JudgeSession:
    candidate_sessions = [
        (label, LQGSession(client, lqg_tool=LQG_TOOL, capture_plots=True))
        for label, client in candidate_clients
    ]
    return JudgeSession(candidate_sessions, judge_client, judge_system_prompt=JUDGE_SYSTEM_PROMPT_LQG)


def _print_candidate_rounds(rounds, log_fh=None) -> None:
    """Plain-text rendering of JudgeSession.last_round -- the CLI
    equivalent of cli_supervisor_judge_pid.py's own /candidates output.
    Routed through log_print so /candidates and --verbose output land in
    --log-file too, not just stdout."""
    if not rounds:
        clio.log_print(log_fh, "(no candidate data yet -- send a message first)")
        return
    for item in rounds:
        label = item["label"]
        if item.get("error"):
            clio.log_print(log_fh, f"  {label}: FAILED to respond -- {item['error']}")
            continue
        header = f"  {label}"
        if item.get("finalized"):
            header += f" -> recommends {item['finalized']}"
        clio.log_print(log_fh, header)
        for name, call_args in item.get("calls", []):
            arg_text = ", ".join(f"{k}={v!r}" for k, v in call_args.items())
            clio.log_print(log_fh, f"    called {name}({arg_text})")
        if item.get("reply"):
            clio.log_print(log_fh, f"    {item['reply']}")


def main():
    parser = argparse.ArgumentParser(
        description="PIDTuner LLM-as-judge CLI: N candidate LQR/LQG supervisors fan out a message "
                     "in lockstep, and a separate judge model arbitrates. MIMO/LQG track counterpart "
                     "to cli_supervisor_judge_pid.py (SISO/PID track)."
    )
    parser.add_argument("--candidate", action="append", dest="candidates", type=_parse_provider_model,
                         metavar="PROVIDER[:MODEL]",
                         help="A candidate model, as provider[:model] -- provider is one of "
                              f"{', '.join(VALID_PROVIDERS)}; model defaults to that provider's "
                              "cheapest tier when omitted. Repeatable; give at least 2, e.g. "
                              "--candidate anthropic:claude-haiku-4-5 --candidate openai:gpt-5.6-luna")
    parser.add_argument("--judge", type=_parse_provider_model, required=True, metavar="PROVIDER[:MODEL]",
                         help="The judge model, as provider[:model]. Required -- no default, must "
                              "be chosen deliberately, and must differ from every candidate's "
                              "exact provider+model pair.")
    parser.add_argument("--api-key-anthropic", default=None,
                         help="Anthropic API key; skips the .env/ANTHROPIC_API_KEY lookup when given")
    parser.add_argument("--api-key-openai", default=None,
                         help="OpenAI API key; skips the .env/OPENAI_API_KEY lookup when given")
    parser.add_argument("--api-key-gemini", default=None,
                         help="Gemini API key; skips the .env/GOOGLE_API_KEY lookup when given")
    parser.add_argument("--verbose", action="store_true",
                         help="Print each candidate's breakdown (tool calls, finalized pick, reply) "
                              "after every judge reply, not just on /candidates")
    parser.add_argument("--log-file", default=None,
                         help="Append the full session transcript (both what you type and every "
                              "printed reply) to this file -- unlike `| tee`, this also captures "
                              "what you type at an interactive prompt, which a pipe on stdout never sees")
    args = parser.parse_args()

    resolved_candidates, resolved_judge = _validate_selection(args.candidates, args.judge)

    keys = {}
    for provider, _ in resolved_candidates + [resolved_judge]:
        if provider not in keys:
            keys[provider] = _resolve_key(provider, args)

    candidate_clients = [(_label(p, m), _client_for(p, m, keys[p])) for p, m in resolved_candidates]
    judge_provider, judge_model = resolved_judge
    judge_client = _client_for(judge_provider, judge_model, keys[judge_provider])
    judge_session = _new_judge_session(candidate_clients, judge_client)
    run_paths = clio.RunPaths.new("judge-lqg")
    log_fh = clio.open_session_log(args.log_file)

    clio.log_print(log_fh, f"PIDTuner LLM judge (LQR/LQG track). Candidates: "
                            f"{', '.join(label for label, _ in candidate_clients)}. Judge: {_label(*resolved_judge)}.")
    clio.log_print(log_fh, "Tell me which preset plant you're working with and what matters most to you. Type "
                            "/candidates to see what each candidate said last round, /reset to start over, /quit to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        clio.log_user_input(log_fh, "you> ", text)
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            judge_session = _new_judge_session(candidate_clients, judge_client)
            clio.log_print(log_fh, "(session reset)")
            continue
        if text == "/candidates":
            _print_candidate_rounds(judge_session.last_round, log_fh=log_fh)
            continue
        try:
            reply = judge_session.handle_user_message(text)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive on unexpected errors
            clio.log_print(log_fh, f"(error talking to the model: {exc})", file=sys.stderr)
            continue
        clio.log_print(log_fh, f"\njudge> {reply}")
        if args.verbose:
            _print_candidate_rounds(judge_session.last_round, log_fh=log_fh)

        run_paths.next_turn()
        try:
            for label, session in judge_session.candidates:
                calls, session.plot_calls = session.plot_calls, []
                clip.save_lqg_turn_plots(calls, run_paths, candidate_label=label, log_fh=log_fh)
            clio.write_judge_report(run_paths, "MIMO / LQG", judge_session, log_fh=log_fh)
        except Exception as exc:  # noqa: BLE001 - a save bug must not crash the REPL
            clio.log_print(log_fh, f"(warning: failed to save plot/report: {exc})", file=sys.stderr)


if __name__ == "__main__":
    main()
