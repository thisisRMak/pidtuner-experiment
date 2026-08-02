#!/usr/bin/env bash
# Demonstrates cli_supervisor_lqg.py: a conversational REPL that asks which
# preset plant/priorities you have, runs the LQR/output-weighted-LQR/
# Bryson/LQG benchmark under the hood, and recommends one technique
# grounded in the real computed metrics + correctness checks.
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

echo "### Scripted LQG supervisor session (aircraft_hall, prioritizing control effort) ###"
python3 cli_supervisor_lqg.py <<'EOF'
I'm working with the aircraft_hall preset plant. I care most about minimizing control effort, and I'd rather avoid a Kalman filter if I can measure the full state directly.
/quit
EOF
