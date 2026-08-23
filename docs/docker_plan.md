# Docker — next steps

**Status: the `docker/cli-ollama-tk-parity` branch now also serves the
Streamlit GUI over HTTP (see "Streamlit GUI update" below) — that closes
the "can't run the GUI at all" gap called out below. The Tkinter-in-Docker
non-goal and the remaining candidates below are otherwise unaffected.**

## Context

`docker/cli-ollama-tk-parity` (commit `4c9c230`) built and verified a
`controldesign` image: `python:3.11-slim` + `apt-get install python3-tk` +
`pip install numpy scipy matplotlib ollama` — dependency parity with the
conda environment for every CLI script and the ollama-backed LLM
supervisor, without running the Tkinter GUI itself (no display server in
a container; out of scope by design, not a gap to close). Verified: full
295-test suite passes in-container, `cli_pid.py` produces real tuning
output, `--help` on the blackbox and supervisor CLIs works, `pid_app.py`
imports cleanly.

Reflecting on it once built (see prior turn): it's not yet useful as a
day-to-day development replacement for conda — no volume mount (every
code change needs a rebuild before it's testable in-container), it can't
run the GUI at all, and it doesn't reduce the PyInstaller CI matrix (that
exists to ship a double-click native binary, which a container can't be).
Current verdict: keep it as a periodic parity/portability check, keep
developing in conda day-to-day, and treat this document as *where Docker
goes next*, not an argument for switching now.

## Streamlit GUI update

A follow-up turn added Streamlit GUI support to the same
`docker/cli-ollama-tk-parity` branch (merged in `main`'s Streamlit-GUI
commits first, then extended the Dockerfile): `pip install` now includes
`streamlit`, the image `EXPOSE`s port 8501, and the Dockerfile header
documents the run command. The CLI/ollama/Tk parity behavior is
unchanged — this is additive.

Run it:

```
docker build -t controldesign .
docker run -p 8501:8501 controldesign streamlit run streamlit_app.py --server.address=0.0.0.0 --server.port=8501
```

then open `http://localhost:8501`. `--server.address=0.0.0.0` is required
— without it Streamlit only binds loopback inside the container and
nothing on the host can reach it. Existing CLI-only usage
(`docker run controldesign python3 cli_pid.py ...`) still works unchanged.

Verified in-container: full test suite (318 tests, `python3 -m unittest
discover -p 'test_*.py'`) passes, `cli_pid.py` still produces real tuning
output, and `streamlit run ... -p 8501:8501` responds to `curl` with
real Streamlit HTML (HTTP 200) a few seconds after start. The Tkinter
GUI (`pid_app.py`) staying unusable in Docker (no display server) is
unchanged and still expected — not something this update touches.

## Candidate next steps (unordered — pick up whichever becomes relevant)

- **Dev-loop ergonomics** — bind-mount `src/` into the container
  (`docker run -v $(pwd):/app ...` or a `docker-compose.yml` with a
  `volumes:` entry) so code changes are testable without a rebuild. Only
  worth doing once someone actually wants to iterate *inside* Docker
  regularly — not needed for the current "occasional parity check" use.
- **CI parity job** — a lightweight, separate GitHub Actions workflow that
  builds the image and runs the test suite in it on every push/PR, purely
  as an automated "did anything silently start depending on conda-only
  behavior" check. Explicitly *not* a replacement for `build.yml` — a new,
  additional, cheap job.
- **Tie-in to the GUI decision (`docs/gui_plan.md`)** — if a web-based GUI
  option is chosen, Docker becomes the actual deployment target rather
  than a side artifact. That's the point real investment becomes
  worthwhile: multi-stage builds for a smaller image, health checks, port
  configuration, maybe a `docker-compose.yml` if a second service (e.g. a
  database, or a bundled Ollama sidecar — see below) ever becomes
  necessary. Nothing here should be built ahead of that decision.
- **Ollama sidecar, reconsidered** — deliberately excluded from the
  current image (Ollama stays externally reached via `--host`). Worth
  revisiting only for a specific future need: e.g., handing this off to
  someone without their own Ollama setup, where "one `docker compose up`
  gets you everything, including the LLM" might matter more than image
  size/simplicity. Not needed today.
- **Registry/publishing** — currently a local-only image (`docker build`,
  no push anywhere). Decide if/when it's worth publishing (Docker Hub,
  GHCR) — only relevant once there's an actual consumer who isn't you
  building it locally from source.
- **Image naming/tagging** — currently just `controldesign:latest`, no
  version scheme. Revisit once this is more than a local experiment (tags
  tracking git commits/releases, etc.).

## Non-goals of this document

- No implementation — no Dockerfile changes, no new files, no touching
  the `docker/cli-ollama-tk-parity` branch.
- No decision on *which* of the candidates above to do first — that's a
  follow-up-session call, likely driven by whichever of `docs/gui_plan.md`
  or `docs/aituner_plan.md` moves first.
- Does not revisit the conda-vs-Docker-for-dev question — already
  answered (keep conda for now) in the prior turn's reflection; this
  document assumes that answer holds until something concrete changes it.
