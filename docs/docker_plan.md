# Docker

`docker/cli-ollama-tk-parity` builds a `controldesign` image
(`python:3.11-slim` + `apt-get install python3-tk` + `pip install numpy
scipy matplotlib ollama streamlit`) giving dependency parity with the
conda environment, without installing conda. Covers every CLI script,
the ollama-backed LLM supervisor, and the Streamlit web GUI. The Tkinter
GUI (`pid_app.py`) imports cleanly but can't render — no display server
in a container, out of scope by design, not a gap to close.

Verified in-container: full test suite (318 tests) passes, `cli_pid.py`
produces real tuning output, `--help` works on the blackbox/supervisor
CLIs, and Streamlit responds to `curl` with real HTML after `docker run
-p 8501:8501 ... streamlit run streamlit_app.py --server.address=0.0.0.0
--server.port=8501` (see `src/Dockerfile`'s header comment for the exact
commands — `--server.address=0.0.0.0` is required or the host can't
reach it).

Current verdict: not yet a day-to-day conda replacement — no volume
mount (every code change needs a rebuild before it's testable
in-container), and it doesn't reduce the PyInstaller CI matrix (that
ships a double-click native binary, which a container can't be). Keep
it as a periodic parity/portability check; keep developing in conda.

## Candidate next steps (unordered, pick up if relevant)

- **Dev-loop ergonomics** — bind-mount `src/` (`-v $(pwd):/app` or a
  `docker-compose.yml`) so code changes don't need a rebuild. Only worth
  it once someone wants to iterate *inside* Docker regularly.
- **CI parity job** — a separate, additional GitHub Actions workflow
  building the image and running the test suite in it on every push/PR,
  as an automated "did anything start depending on conda-only behavior"
  check. Not a replacement for `build.yml`.
- **Multi-stage build / smaller image, health checks, port config** —
  worth it once Docker becomes an actual deployment target rather than
  a local convenience.
- **Ollama sidecar** (bundled via `docker-compose`) — deliberately
  excluded; Ollama stays externally reached via `--host`. Revisit only
  if handing this off to someone without their own Ollama setup.
- **Registry/publishing** — currently local-build-only. Decide if/when
  worth publishing (Docker Hub, GHCR) once there's a consumer who isn't
  building it locally from source.
- **Image naming/tagging** — currently just `controldesign:latest`, no
  version scheme.
