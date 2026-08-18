# GUI Strategy — Streamlit plan

**Status: decision made (Streamlit). Build plan Steps 1-3 are done and
committed (skeleton, session-state schema, SISO PID panel + heatmap/
radar comparison views). Steps 4-7 (MIMO LQR/LQG, LLM chat, packaging,
container check) are next. See "Testing debt" below for what's still
unverified in what's shipped so far.**

## Decision

Streamlit, single Python process, covering SISO PID + MIMO LQR/LQG +
the LLM conversational supervisor in one app. No new Tkinter work for
LQG/MIMO. Rationale and the full options survey that led here are kept
below under "Options considered" for reference; skip straight to
"Target audience" and "Build plan" if you just need the spec.

## Target audience (revised)

This is **not** a classroom/student tool. The audience is the control
theory / CS / robotics research community — people who will `git clone`
the repo and expect to reproduce specific results, not double-click an
executable. This changes what "accessible" means here:

- The bar is "clones cleanly and reproduces the same numbers," not
  "zero technical friction for a non-technical user." A documented
  `pip install` / `streamlit run` or `docker compose up` path is the
  right bar — not a per-OS native binary.
- Reproducibility is a **backend/packaging** concern (pinned deps,
  fixed seeds, versioned control-library dependencies, a documented
  entry point) — not something the GUI framework choice fixes or
  breaks. The GUI is a convenient way to explore the tool; the
  reproducibility guarantee lives in the scripted/CLI path and the
  environment spec, which must stay correct independent of the GUI.
- This weakens the case for the current PyInstaller 3-OS build
  pipeline (`.github/workflows/build.yml`) — that packaging effort
  exists for a zero-install *student* audience. A research audience is
  better served by a container image or a plain `pip install` than by
  per-OS binaries. Not deciding here to drop it, but it's no longer a
  reason to keep `pid_app.py`/Tkinter as the primary interface.
- `pid_app.py` (Tkinter) can keep working unchanged as a legacy/offline
  option during migration — no big-bang cutover required — but it is
  not the thing new LQG/MIMO UI work should be added to.

## What the Streamlit app needs to satisfy

- SISO PID: the existing fixed menu of 9 named tuning methods,
  low-dimensional widgets (sliders, a couple of pole entries) — direct
  port of `pid_app.py`'s panel logic onto Streamlit widgets, calling
  the same UI-agnostic `pid_tuning_methods.py` / `pid_simulate.py` /
  `pid_compare.py` / `pid_blackbox.py` functions.
- MIMO LQR/LQG: scalar/broadcast knobs onto `Q`/`R`/`N`, deliberately
  *not* raw matrix editors — see `docs/lqg_plan.md` "CLI vs. GUI" for
  why matrix-shaped widgets don't reduce cognitive load the way the
  PID panels do. Reuses the LQG-track backend functions the same way.
- LLM conversational supervisor: a chat panel driving the same
  tuning/simulation backend via the multi-provider supervisor work
  (`docs/aituner_plan.md`) — independent of this GUI choice as a
  backend concern, but it's the third first-class panel in this app
  alongside SISO and MIMO, not a bolt-on.
- The CLI/LLM-supervisor workflows stay viable independent of the GUI
  — already true, must remain true.
- The one identified hard technical problem, common to any reactive-UI
  framework: the "tuned controllers" session list (per-row
  enable/disable checkboxes, add/remove, cached simulation results
  surviving reruns), **plus** now a second stateful surface — chat
  history for the LLM panel — that has the same rerun-survival
  requirement. Both need explicit `st.session_state` design (see Build
  plan below); neither is a blocker, but don't improvise it inline.

## Build plan

1. **DONE.** App skeleton and multi-panel structure. `streamlit_app.py`,
   `st.tabs` for the three top-level sections (SISO PID / MIMO LQR-LQG /
   LLM chat) — resolved in favor of tabs over a sidebar radio (see
   Resolved decisions below). *Check: app launches, all three sections
   render, no backend calls yet. Verified.*
2. **DONE.** Session-state design for the tuned-controllers list.
   `streamlit_gui_state.py` — `ControllerEntry` dataclass (kind/label/
   params/result/sim/enabled/id/mrow) plus a full mutation contract
   (`add_controller`, `remove_controller`, `set_enabled`, and
   kind-scoped variants `get_by_kind`/`clear_by_kind`/
   `remove_unchecked_by_kind`/`set_all_enabled_by_kind` so SISO and the
   upcoming MIMO panel don't each hand-filter the shared list) plus
   ephemeral chat-history helpers. *Check: add/remove/toggle round-trip
   verified directly; later exercised live through the SISO panel with
   no desync between panel state and rendered checkboxes (see Testing
   debt — this took two passes to get right, see history for the
   widget-key gotcha that caused the first pass to silently revert
   bulk-select actions).*
3. **DONE.** SISO PID panel (`streamlit_siso_panel.py`), ported from
   `pid_app.py` widget-by-widget, plus heatmap/radar comparison views
   (`streamlit_siso_comparison_views.py`, reusing `pid_compare.py`'s
   plain data functions with Streamlit-native rendering — an HTML
   table for the heatmap since Streamlit has no colored-cell grid
   widget, `st.pyplot` for the radar). *Check: all 9 tuning methods
   individually exercised end-to-end through the UI (`AppTest`,
   selectbox → args → click → verify one new session entry, no
   exceptions) with non-default arguments per method; MATLAB-
   coefficient plant form; `back_calc` anti-windup with genuine
   actuator saturation; 4 error paths (RHP-pole rejection, malformed
   plant expression, malformed MATLAB coefficients, a garbage numeric
   field) all fail cleanly via `st.error` with no phantom session
   entries. NOT done: a literal side-by-side run against `pid_app.py`
   on the same inputs — see Testing debt. These checks are a real
   regression suite in `test_streamlit_siso_panel.py` (run via `python
   test_streamlit_siso_panel.py`), not one-off scratch scripts — extend
   it rather than re-deriving this coverage by hand when Step 4's MIMO
   panel needs the analogous tests.*
4. **NEXT.** MIMO LQR/LQG panel, scalar/broadcast `Q`/`R`/`N` knobs per
   the existing LQG-track backend. *Check: a known-good LQG example from
   `test_lqg_frequency.py` or equivalent reproduces matching
   gains/plots through the UI.*
5. **LLM chat panel**, wired to the multi-provider supervisor
   (`docs/aituner_plan.md`), sharing the Step 2 session-state pattern
   for chat history. *Check: a conversational tuning request actually
   invokes the backend and populates the tuned-controllers list (not
   just chat text).*
6. **Reproducibility packaging**: pin dependency versions actually
   exercised by this app (Streamlit + existing backend deps), document
   the `pip install` / `streamlit run` entry point in the README, and
   confirm a fixed-seed run reproduces the same numeric result twice.
   *Check: a clean clone + documented install steps + one documented
   command reproduces a specific published result.*
7. **Container story** (ties into the existing Docker work): confirm
   the Streamlit app runs unmodified in the container that already
   proved the backend has no conda dependency. *Check: `docker compose
   up` (or equivalent) serves the app, reachable at localhost, all
   three panels functional.*

Each step should land as its own reviewable change — consistent with
how the Docker work was kept small and atomic.

## Testing debt (open TODOs, deliberately deferred)

Verification for Steps 1-3 leaned entirely on Streamlit's `AppTest`
harness (server-side element-tree inspection) plus direct backend
calls — real, but not the whole picture. Two rendering-layer bugs
(dark-theme text invisible on the heatmap, a nested-`st.tabs` bug that
`AppTest` couldn't see at all) only surfaced from actually looking at
the running app, which is exactly the class of check still missing
below:

- **No side-by-side numeric diff against `pid_app.py`.** Two tuning
  formulas were checked directly against the backend, and the same
  backend calls are shared by both UIs, but the two actual UIs have
  never been run on the same plant/method/settings and diffed. Blocked
  in this environment specifically — `pid_app.py` needs a real X11/
  Wayland display (or `Xvfb`) that isn't available here.
- **No real-browser check.** Everything so far is `AppTest` or
  curl-for-200; no Playwright/Chromium run against the live page.
  Given both bugs found so far were rendering-only, this is the
  highest-value gap left. Needs `playwright` installed (not present in
  this environment).
- **No multi-session isolation check.** `st.session_state` is per-session
  by Streamlit's design, but two concurrent browser sessions against
  the same running process have never actually been tested against
  each other.
- **Docker container run untested** — Step 7 above, blocked on Steps
  4-6 landing first, but noted here too since it's a real gap in
  today's confidence, not just a future step.

## Resolved decisions

- **Tabs, not sidebar** — `st.tabs` for the three top-level panels.
  Within a panel with its own sub-views (SISO's Response/Heatmap/Radar),
  use a radio switch, not a nested `st.tabs` — Streamlit allows nesting
  tabs at the Python level with no error, but the inner tab bar can
  render invisible/non-interactive in the actual frontend. Found live,
  not caught by `AppTest`.
- **Chat history: ephemeral**, `st.session_state` only, lost on
  refresh — no persistence layer for v1.
- **File naming: flat `src/`, `streamlit_` prefix** — not a `src/webapp/`
  subfolder. Matches this repo's existing flat+prefix convention
  (`pid_*`, `lqg_*`, `cli_*`, `supervisor_*`, several of which already
  have 10+ files flat). A folder move later is a cheap, fully
  reversible `git mv` if the Streamlit family genuinely outgrows flat
  — not foreclosed, just not earning its cost yet at 4 files.

## Open questions still unresolved

- Whether `pid_app.py` gets a deprecation notice pointing at the new
  app once full parity (not just SISO) is reached, or stays silently
  maintained.

## Options considered (reference, decision already made above)

| Option | Effort to first working version | Layout control | Multi-user/remote | GUI-in-Docker story | Notes |
|---|---|---|---|---|---|
| **Stay Tkinter (status quo)** | Zero — already built | Full (native widgets) | No — single-user desktop only | N/A, doesn't apply | Zero migration cost, already proven, offline-capable. But: needs a 3-OS PyInstaller build per release, no remote/multi-user access, dated look, tightly couples distribution to conda/Tk-bundling. Fine as a legacy/offline option during migration, not the primary interface going forward. |
| **Streamlit (chosen)** | Low — single Python process, direct reuse of existing compute, matplotlib inline | Moderate — Streamlit's own widget/layout system, not pixel-level | Yes, but each user's session reruns the whole script on every interaction | Good — exactly the "run in a container, view as a webpage" story that motivated Docker in the first place | Fastest realistic path to something running, and matches a research/reproducibility audience well. The session-list/chat-history rerun problem is real design work — scoped explicitly in the Build plan above, not a blocker. |
| **Gradio** | Low, similar to Streamlit for simple cases | Weak for this app's shape — built for single-function ML demos, not multi-tab/multi-panel dashboards | Yes | Good | Ruled out earlier specifically because of the multi-tab/multi-panel layout mismatch. |
| **NiceGUI / Panel / Reflex** | Moderate — more setup than Streamlit, less than a JS frontend | Better than Streamlit — event-driven callbacks rather than full-script rerun | Yes | Good | The fallback if Streamlit's rerun model becomes genuinely painful once the chat panel is built (Step 5) — worth revisiting then with real evidence rather than switching preemptively. Panel (PyData/HoloViz ecosystem) has particularly good matplotlib-embedding support. |
| **FastAPI + HTML/htmx (or Flask)** | Moderate-high — real routes, templates, request validation by hand | Full — can faithfully mirror `pid_app.py`'s tabs/panels exactly | Yes | Good | No framework lock-in, clean OpenAPI docs for free. Real cost: hand-building what Streamlit gives for free (forms, session handling) — not justified for this project's scope. |
| **FastAPI + separate JS frontend (React/Svelte/Vue)** | Highest — two codebases, build tooling, API design discipline | Full, most polished | Yes, best fit for a genuinely public/shareable tool | Good | Overkill for a research-reproducibility tool; the polish this buys isn't the bottleneck here. |

## Non-goals of this document

- Still no code in this document itself — it specs the build, doesn't
  contain it.
- Does not revisit whether to eventually deprecate conda/Tkinter
  entirely — `pid_app.py` stays functional in parallel per the Build
  plan; full retirement is a separate decision.
