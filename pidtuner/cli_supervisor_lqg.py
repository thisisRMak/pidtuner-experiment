#!/usr/bin/env python3
"""Interactive LQR/LQG supervisor: talks to a user about their priorities,
runs the LQR/output-weighted-LQR/Bryson/LQG benchmark against a named preset
plant, and recommends a single technique grounded in the real computed
metrics.

Thin over supervisor_session_lqg.LQGSession, the same way cli_supervisor_pid.py
is thin over supervisor_session.Session.
"""

from __future__ import annotations

import argparse
import sys

from supervisor_llm import DEFAULT_KEEP_ALIVE, DEFAULT_MODEL, DEFAULT_NUM_CTX, OllamaClient
from supervisor_session_lqg import LQGSession
from supervisor_tools_lqg import RUN_LQG_BENCHMARK_SCHEMA, run_lqg_benchmark

LQG_TOOL = (RUN_LQG_BENCHMARK_SCHEMA, run_lqg_benchmark)


def _new_session(client: OllamaClient) -> LQGSession:
    return LQGSession(client, lqg_tool=LQG_TOOL)


def main():
    parser = argparse.ArgumentParser(
        description="LQR/LQG supervisor: a conversational recommendation "
                     "layer over the LQR/output-weighted-LQR/Bryson/LQG benchmark."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model tag (default: {DEFAULT_MODEL})")
    parser.add_argument("--host", default=None, help="Ollama host URL (default: local daemon)")
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help=f"Context window (default: {DEFAULT_NUM_CTX})")
    parser.add_argument("--keep-alive", default=DEFAULT_KEEP_ALIVE, help=f"Ollama keep_alive (default: {DEFAULT_KEEP_ALIVE})")
    args = parser.parse_args()

    client = OllamaClient(model=args.model, host=args.host,
                           num_ctx=args.num_ctx, keep_alive=args.keep_alive)
    session = _new_session(client)

    print("LQR/LQG supervisor. Tell me which preset plant you're working with "
          "and what matters most to you. Type /reset to start over, /quit to exit.")
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
