#!/usr/bin/env bash
# Demonstrates cli_supervisor_pid.py: a conversational REPL that asks about your
# plant/priorities, runs the same white-box/black-box benchmark under the
# hood, and recommends one tuning technique grounded in the real metrics.
#
# Requires a local Ollama daemon with a tool-calling model pulled, e.g.:
#   ollama pull qwen3-coder:30b
# The REPL is normally interactive; here we pipe a scripted conversation
# into stdin so the whole thing runs non-interactively as a demo.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v ollama >/dev/null 2>&1; then
    echo "Skipping: 'ollama' is not installed. Install it from https://ollama.com" \
         "and run 'ollama pull qwen3-coder:30b' to try this demo." >&2
    exit 0
fi

if ! python3 -c "import ollama" >/dev/null 2>&1; then
    echo "Skipping: the Python 'ollama' package is not installed in this" \
         "environment. Run 'pip install ollama' to try this demo." >&2
    exit 0
fi

if ! ollama list 2>/dev/null | grep -q "qwen3-coder"; then
    echo "Skipping: no qwen3-coder model found. Run 'ollama pull qwen3-coder:30b' first." >&2
    exit 0
fi

echo "### Scripted supervisor session (white-box plant, prioritizing low overshoot) ###"
python3 cli_supervisor_pid.py <<'EOF'
My plant is 1000/((s+1)(10s+1)). I care most about minimizing overshoot, settling time is secondary.
/quit
EOF
