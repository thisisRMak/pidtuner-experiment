#!/usr/bin/env python3
"""Interactive supervisor: talks to a user about their priorities, runs the
existing PIDTuner benchmark (white-box or black-box, whichever applies),
and recommends a single technique grounded in the real computed metrics.

Thin over supervisor_session.Session, the same way cli_pid.py/cli_pid_blackbox.py
are thin over pid_compare.py/pid_blackbox.py.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

from supervisor_llm import DEFAULT_KEEP_ALIVE, DEFAULT_MODEL, DEFAULT_NUM_CTX, OllamaClient
from supervisor_llm_anthropic import AnthropicClient, DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from supervisor_session_pid import Session
from supervisor_tools_blackbox_pid import RUN_BLACKBOX_BENCHMARK_SCHEMA, run_blackbox_benchmark
from supervisor_tools_whitebox_pid import RUN_WHITEBOX_BENCHMARK_SCHEMA, run_whitebox_benchmark

WHITEBOX_TOOL = (RUN_WHITEBOX_BENCHMARK_SCHEMA, run_whitebox_benchmark)
BLACKBOX_TOOL = (RUN_BLACKBOX_BENCHMARK_SCHEMA, run_blackbox_benchmark)


def _new_session(client) -> Session:
    return Session(client, whitebox_tool=WHITEBOX_TOOL, blackbox_tool=BLACKBOX_TOOL)


def _resolve_anthropic_key(explicit_key):
    """Resolve an Anthropic API key: an explicit --api-key wins outright and
    skips the lookup below entirely (scriptability -- no surprise .env pickup
    when a caller already passed one); otherwise fall back to .env/the
    process environment's ANTHROPIC_API_KEY. Same two sources and order as
    streamlit_llm_panel.py's `_configured_key`, minus its Streamlit-only
    st.secrets fallback (no equivalent on the CLI)."""
    if explicit_key:
        return explicit_key
    load_dotenv()
    return os.environ.get("ANTHROPIC_API_KEY")


def _build_client(args):
    if args.provider == "anthropic":
        api_key = _resolve_anthropic_key(args.api_key)
        if not api_key:
            print("error: no Anthropic API key found -- pass --api-key, set ANTHROPIC_API_KEY, "
                  "or add it to a .env file", file=sys.stderr)
            sys.exit(1)
        model = args.model or ANTHROPIC_DEFAULT_MODEL
        return AnthropicClient(api_key=api_key, model=model)
    model = args.model or DEFAULT_MODEL
    return OllamaClient(model=model, host=args.host, num_ctx=args.num_ctx, keep_alive=args.keep_alive)


def main():
    parser = argparse.ArgumentParser(
        description="PIDTuner LLM supervisor: a conversational recommendation "
                     "layer over the existing 9-technique benchmark."
    )
    parser.add_argument("--provider", choices=["ollama", "anthropic"], default="ollama",
                         help="LLM backend to use (default: ollama)")
    parser.add_argument("--model", default=None,
                         help=f"Model tag/id -- default is {DEFAULT_MODEL} for --provider ollama or "
                              f"{ANTHROPIC_DEFAULT_MODEL} for --provider anthropic")
    parser.add_argument("--api-key", default=None,
                         help="Anthropic API key (--provider anthropic only); skips the .env/"
                              "ANTHROPIC_API_KEY lookup when given")
    parser.add_argument("--host", default=None, help="Ollama host URL (--provider ollama only; default: local daemon)")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX,
                         help=f"Ollama context window (--provider ollama only; default: {DEFAULT_NUM_CTX})")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE,
                         help=f"Ollama keep_alive (--provider ollama only; default: {DEFAULT_KEEP_ALIVE})")
    args = parser.parse_args()

    client = _build_client(args)
    session = _new_session(client)

    print("PIDTuner supervisor. Tell me about your plant and what matters most "
          "to you. Type /reset to start over, /quit to exit.")
    while True:
        try:
            text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit"):
            break
        if text == "/reset":
            session = _new_session(client)
            print("(session reset)")
            continue
        try:
            reply = session.handle_user_message(text)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive on unexpected errors
            print(f"(error talking to the model: {exc})", file=sys.stderr)
            continue
        print(f"\nsupervisor> {reply}")


if __name__ == "__main__":
    main()
