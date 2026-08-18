#!/usr/bin/env python3
"""Interactive supervisor: talks to a user about their priorities, runs the
existing PIDTuner benchmark (white-box or black-box, whichever applies),
and recommends a single technique grounded in the real computed metrics.

Thin over supervisor_session.Session, the same way cli_pid.py/cli_pid_blackbox.py
are thin over pid_compare.py/pid_blackbox.py.
"""

from __future__ import annotations

import argparse
import sys

from supervisor_llm import DEFAULT_KEEP_ALIVE, DEFAULT_MODEL, DEFAULT_NUM_CTX, OllamaClient
from supervisor_session_pid import Session
from supervisor_tools_blackbox_pid import RUN_BLACKBOX_BENCHMARK_SCHEMA, run_blackbox_benchmark
from supervisor_tools_whitebox_pid import RUN_WHITEBOX_BENCHMARK_SCHEMA, run_whitebox_benchmark

WHITEBOX_TOOL = (RUN_WHITEBOX_BENCHMARK_SCHEMA, run_whitebox_benchmark)
BLACKBOX_TOOL = (RUN_BLACKBOX_BENCHMARK_SCHEMA, run_blackbox_benchmark)


def _new_session(client: OllamaClient) -> Session:
    return Session(client, whitebox_tool=WHITEBOX_TOOL, blackbox_tool=BLACKBOX_TOOL)


def main():
    parser = argparse.ArgumentParser(
        description="PIDTuner LLM supervisor: a conversational recommendation "
                     "layer over the existing 9-technique benchmark."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=None, help="Ollama host URL (default: local daemon)")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help=f"Context window (default: {DEFAULT_NUM_CTX})")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE, help=f"Ollama keep_alive (default: {DEFAULT_KEEP_ALIVE})")
    args = parser.parse_args()

    client = OllamaClient(model=args.model, host=args.host,
                           num_ctx=args.num_ctx, keep_alive=args.keep_alive)
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
