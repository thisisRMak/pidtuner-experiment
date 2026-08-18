# CLI/dev container: dependency parity with the conda environment
# (environment.yml) — numpy/scipy/matplotlib, Tk, and ollama — for running
# the CLI scripts and the ollama-backed LLM supervisor without installing
# conda locally. This does NOT run the Tkinter GUI: there's no display
# server inside the container, so pid_app.py's dependency chain imports
# cleanly but tk.Tk() itself has nowhere to render (X11 forwarding/VNC are
# deliberately out of scope — see docs/lqg_plan.md-adjacent discussion).
# conda/environment.yml/build.yml are unaffected by this file; it's an
# additive, parallel option, not a replacement for either.
#
# Build (from src/):  docker build -t controldesign .
# Run (interactive):  docker run -it controldesign
# Run one script:     docker run controldesign python3 cli_pid.py --plant "1000/((s+1)(10s+1))" --method simc
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends python3-tk \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir numpy scipy matplotlib ollama

COPY . /app

CMD ["bash"]
