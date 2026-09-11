# aituner — multi-provider LLM supervisor

**Status (2026-09-11): Claude/Anthropic implemented and live-verified on
both the GUI and the CLI. OpenAI and Gemini are not started, on either
surface — confirmed by grepping `src/` fresh, the only hits are the
pre-existing "key accepted, not wired" GUI stub and the Anthropic test
files. See "Implementation status" and "Update (2026-09-11)" below for
what's actually built; the sections after that are progressively older
plan/status snapshots, kept for the reasoning trail — read top-down, most
current first, don't take the tail of the doc as current.**

## Update (2026-09-11)

A lot landed between the "Implementation status" section below (written
right after the initial Claude/GUI work) and now — 30 commits, largely
outside a session that had direct visibility into all of them, so treat
this update as a summary of what was checked, not an exhaustive diff:

- **CLI now has Claude too** (`2abf5fd`): `cli_supervisor_pid.py`/
  `cli_supervisor_lqg.py` got `--provider {ollama,anthropic}`, `--model`,
  `--api-key`. The "CLI is still Ollama-only" line in the section below is
  no longer true — superseded by this update, kept there only for history.
- **The 3-tab GUI layout is gone.** `streamlit_app.py` now just calls
  `streamlit_unified_panel.render()` — one page, a Track selector
  (SISO/PID vs MIMO/LQG) and a Mode selector (Manual vs LLM Supervisor)
  instead of tabs. `streamlit_llm_panel.py`/`streamlit_siso_panel.py`/
  `streamlit_mimo_panel.py` still exist and are composed by the unified
  panel, not replaced outright. SISO's Response/Heatmap/Radar also
  changed independently (`8aa1c78`): stacked vertically instead of an
  `st.radio` single-view switch, plus per-view download buttons.
- **`supervisor_llm_anthropic.py` (the shared client both CLI and GUI use)
  gained a real fix, not just features**: parallel tool calls repeating
  the *same* tool name in one turn (e.g. comparing two Q/R weightings)
  were colliding on `tool_use_id` — a name-keyed dict silently resolved
  both to the last id, and Claude's own API then hard-rejects the
  malformed request. Fixed with a per-name FIFO queue instead of a dict.
  This lives in the shared client, so it protects both PID/SISO and
  MIMO/LQG, not just whichever track found it. Regression test:
  `test_duplicate_tool_name_in_one_turn_resolves_each_result_to_its_own_id`
  in `test_supervisor_llm_anthropic.py`.
- **`MAX_TOOL_HOPS` diverged between tracks, deliberately, not by
  accident**: PID/SISO stayed at `6`; LQG was re-derived to `8`
  (`supervisor_session_lqg.py`) after live testing showed the Q/R-sweep
  workflow needed more room. Not a uniform bump — check which file before
  assuming a number.
- **LQG supervisor gained**: a custom A/B/C/D plant path (not preset-only
  anymore, `ea9487e`/`f2c581e`), a `Q_diag_list`/`R_diag_list` sweep
  (compare several weightings in one tool call — the actual fix for the
  hop-budget pressure, not just a bigger ceiling), and a `capture_plots`
  flag on both `Session`/`LQGSession` constructors.
- **Full write-up of the four LQG findings above**:
  `docs/memos/2026-09-07/2026-09-07-supervisor-robustness-memo.{md,html,pdf}`
  — read that before touching the supervisor session classes again, it's
  the authoritative record, more detailed than this bullet list.
- **An open design question surfaced but not resolved**: a demo/smoke-test
  shell script for Anthropic was deliberately not built because cost
  gating (who pays, how much, confirmed how) was still undecided — same
  unresolved question as this doc's own "cost-confirmation UX" gap below,
  now also relevant to whatever OpenAI/Gemini demo tooling gets built.

## Implementation status (as of the initial Claude/GUI build — see update above for what's changed since)

**GUI (`src/streamlit_llm_panel.py` + `src/supervisor_llm_anthropic.py`)**:
hand-rolled `AnthropicClient` (see "Decision: hand-rolled, not aisuite"
below), matching `OllamaClient`'s `.chat(messages, tools=None)` interface
exactly so `Session`/`LQGSession` needed **zero changes**. Model picker
offers Haiku 4.5 (default) and Sonnet 5; Opus deliberately excluded
(most expensive tier, no reason to offer it here). Key resolution: `.env`/
process environment (via `python-dotenv`) → `st.secrets` (Streamlit Cloud's
equivalent, no filesystem `.env` there) → session-only password field,
in that order per provider. Visible cost-confirmation warning once a key
exists. `OPENAI_API_KEY`/`GOOGLE_API_KEY` env vars are recognized and a
key can be entered for those providers, but nothing is wired to them yet —
shows "key accepted, not connected" (see `PROVIDERS`/`WIRED_PROVIDERS` in
that file).

**CLI (`cli_supervisor_pid.py`/`cli_supervisor_lqg.py`)**: unchanged,
still imports only `OllamaClient` from `supervisor_llm.py`. Confirmed by
reading both files directly, not assumed — no `AnthropicClient` reference
exists in either script. This is real, current scope-narrowing (from a
mid-build decision this session to keep Ollama CLI-only for now, to avoid
building host-reachability/container-networking plumbing for a GUI path
that was never going to use it) — "Anthropic over the CLI" is queued as
next-session work, see bottom of this doc.

**Decision: hand-rolled, not aisuite.** The "Candidate approaches" section
below was written before this was resolved; it's now decided, with real
evidence rather than the "not yet verified" hedge the plan originally
flagged. Checked aisuite's actual provider source (not just its README):
- Its Anthropic provider returns `arguments` as a JSON **string**
  (`json.dumps(tool_call.input)`, OpenAI's convention) — `Session._dispatch_tool`
  expects a dict (`fn(**(arguments or {}))`), matching what the native
  `ollama` package already hands over. So aisuite is not the
  zero-`Session`-changes drop-in it looks like either — it needs a
  `json.loads()` shim regardless of provider.
- Its Ollama provider goes through Ollama's **OpenAI-compatible `/v1`
  endpoint**, not the native `ollama` package this project uses today —
  a different code path, not a cleverer translation of the one already in
  use. Switching would risk silently dropping `num_ctx`/`keep_alive`
  (Ollama-native options with no OpenAI-schema equivalent) without
  separately verifying they survive that endpoint.

Net: aisuite isn't poorly built (16.2k stars, active), but adopting it
here has real, non-zero integration cost on both counts — exactly what the
plan's original "verify before trusting" caution was for. Hand-rolled won
for Claude on that basis. This reasoning applies to the OpenAI/Gemini
work too, not just Claude — worth re-checking their SDKs the same way
before assuming a hand-rolled client is still right for them, though the
Ollama-specific finding above is Claude/aisuite-specific and won't
recur.

**Live verification** (both real API calls, Haiku, see session transcript
for full traces): SISO/PID track — `1/(90s+1)`, `L=13`, correctly called
`set_priorities` then `run_whitebox_benchmark`, cited real returned
numbers (Tyreus–Luyben 2.662% overshoot), asked a clarifying question
before finalizing rather than guessing. MIMO/LQG track — `aircraft_hall`
preset, notably called `run_lqg_benchmark` **and** `set_priorities`
together in one turn, exercising the multi-tool-call-in-one-hop batching
path (previously only synthetically tested) against the real API for the
first time. Both tracks: tool_use_id round-trip resolved correctly, real
numbers grounded correctly, system prompts (reused verbatim from the
Ollama-era CLI prompts, not rewritten) produced sensible model behavior on
Claude despite being written/tuned against `qwen3-coder:30b`.

**Tests**: `test_supervisor_llm_anthropic.py` (offline, no network — tool
schema translation, the tool_use_id round-trip including the multi-call
batching rule) and `test_streamlit_llm_panel.py` (`AppTest`-based — key
gating, model list, cost warning, unwired-provider path, all exception
branches). 20 tests, alongside the pre-existing 48 (`test_supervisor_pid.py`,
`test_streamlit_siso_panel.py`) with no regressions.

**Bugs found and fixed along the way** (worth knowing about, not just
historical color — some are genuinely reusable lessons):
- A stale, apt-installed system `Brotli` 1.0.9 (`/usr/lib/python3/dist-packages`,
  unrelated to this project) silently shadowed what `pip install anthropic`
  should have provided, since `Brotli` is an *optional*, never-pip-forced
  dependency of `httpx2` (`try: import brotli`) — broke response
  decompression with a confusing `APIConnectionError` that had nothing to
  do with networking. Fixed with an explicit `Brotli>=1.1` floor in
  `requirements.txt`/`environment.yml`/`Dockerfile` — the first *pinned*
  entry in otherwise-unpinned dependency files, deliberately, as a defensive
  floor against exactly this class of bug recurring on another machine
  with a similarly stale system package.
- `st.chat_input`'s documented auto-pin-to-bottom behavior doesn't
  reliably apply nested inside `st.tabs()` — confirmed with a real
  Playwright screenshot (not just suspected; `AppTest` can't see this
  class of bug at all, matches `docs/gui_plan.md`'s prior nested-tabs
  finding). Fixed by reserving an `st.container()` *before* `chat_input`
  in source order, independent of whatever CSS pinning does or doesn't
  do.
- No project skill existed for actually running/screenshotting this app;
  used Playwright directly (headless Chromium, no `chromium-cli` available
  in this environment). Worth a `/run-skill-generator` pass if this
  becomes a recurring need — not done yet.

**Architecture findings** (from directly answering "how does this
actually work" questions mid-session, not assumptions):
- `cli_pid.py`, the CLI supervisor's `run_whitebox_benchmark` tool, and
  the GUI supervisor's same tool all call the literal same
  `pid_compare.compare_all_methods()` — confirmed via import, not
  inferred. No subprocess/CLI-output-capturing anywhere in the supervisor
  path; tool calls are always direct in-process Python calls returning
  structured JSON.
- `Session`/`LQGSession`, both system prompts, and all `supervisor_tools_*.py`
  functions are the exact same objects imported by both the CLI and GUI
  entry points — the *only* things that differ between them today are
  which LLM client gets plugged in (`OllamaClient` vs `AnthropicClient`)
  and the outer I/O loop (terminal REPL vs Streamlit reruns).
- Both `run_whitebox_benchmark`/`run_blackbox_benchmark` are wired into
  `Session` identically in both CLI and GUI (gated purely on
  `worksheet.tf_known`, not a code-path difference) — but
  `run_blackbox_benchmark` requires `step_signal_path`/`relay_signal_path`
  pointing at a `.npz` file already on the server's filesystem (from
  `cli_pid.py --gen-signal`). Plausible for the CLI (same machine).
  **Effectively unusable through the GUI today** — no file-upload
  mechanism is wired into `streamlit_llm_panel.py` at all, so a browser
  visitor has no way to produce a valid path. Never exercised live in
  either surface (every live test so far opened with `tf_known=true`).
- The SISO/MIMO tabs' already-tuned `ControllerEntry` results are **not**
  available to the chat's tools at all — every benchmark tool call
  recomputes from scratch. Explicitly decided against building a bridge
  for now ("chat tab's independent flow is good enough") — this was the
  original seed idea for this whole feature and is now consciously
  deferred, not forgotten.

**Known gaps / open decisions, updated**:
1. `GOOGLE_API_KEY` vs `GEMINI_API_KEY` — still an unverified guess (it's
   literally in `streamlit_llm_panel.py`'s `ENV_VAR` dict already), moot
   until Gemini is actually wired; confirm against Google's SDK then.
2. `thinking` is never explicitly configured in `AnthropicClient` — Haiku
   (the default) doesn't run adaptive thinking unless asked; Sonnet does
   by default. Left as an emergent per-model default, not a deliberate
   choice either way — fine as-is, not blocking anything.
3. `max_tokens=16000` fixed for every model — not verified against each
   model's real ceiling; no truncation seen in either live test so far.
4. `AnthropicClient.chat()` only special-cases `stop_reason == "refusal"` —
   `pause_turn`/`max_tokens` stop reasons fall through un-specially-handled.
   Low risk (no server tools used here), still genuinely unhandled.
5. Repo is public (made temporarily public to unblock the Streamlit Cloud
   deploy button) — revisit pending, explicitly deferred to "later today"
   as of this session, not decided.
6. Cost-confirmation UX (plan's original open gap #7) — resolved: a
   visible `st.warning` once a key/model exist. Not a spending cap or
   running total, just a disclosure. Fine as a first cut.

## Next up

1. ~~SISO GUI usability~~ — **done** (`8aa1c78`), then superseded by a
   bigger change: the whole 3-tab layout was replaced with
   `streamlit_unified_panel.py`. See "Update (2026-09-11)" above.
2. ~~Anthropic over the CLI~~ — **done** (`2abf5fd`). See "Update
   (2026-09-11)" above.
3. **OpenAI + Gemini**, both CLI and GUI — the two still-unwired
   providers, not started as of 2026-09-11 (confirmed by grepping `src/`
   fresh). Each needs its own current-docs verification pass (no bundled
   skill for either, unlike `claude-api` for Anthropic) before writing a
   hand-rolled client, per the aisuite investigation's lesson above: don't
   assume, check. Two things to inherit from the Anthropic work, not
   rediscover: the per-name FIFO tool_use_id fix (see "Update
   (2026-09-11)" — a hand-rolled OpenAI/Gemini client needs the equivalent
   of whatever each provider's own parallel-tool-call ID scheme requires,
   verified against their docs, not assumed safe by analogy) and the
   robustness memo's findings generally (`docs/memos/2026-09-07/
   2026-09-07-supervisor-robustness-memo.md`) — read before assuming the
   current `supervisor_llm_anthropic.py` is a naive first-draft template.
4. **Anthropic demo shell script** — `src/examples/run_supervisor_demo.sh`/
   `run_supervisor_lqg_demo.sh` only exercise `--provider ollama` (checking
   for a local daemon + pulled model, skipping with instructions if
   either's missing); no `--provider anthropic` equivalent exists yet.
   Not just a copy-paste of the Ollama script's shape, though: gating on
   `ANTHROPIC_API_KEY` being resolvable (mirroring `_resolve_anthropic_key`
   in `cli_supervisor_pid.py`/`cli_supervisor_lqg.py`) is the easy part —
   unlike the free/local Ollama demos, this one would send a real, billed
   request every time someone runs `src/examples/README.md`'s "Run them
   all" loop, which needs a deliberate opt-in decision (separate script
   excluded from that loop by default? explicit `--confirm-cost` flag?)
   before it's added, not just a key check.

## Context

`supervisor_llm.py` is a thin, Ollama-only wrapper (`OllamaClient`) used by
both `cli_supervisor_pid.py` (PID track) and `cli_supervisor_lqg.py`
(LQG track) — two separate scripts/sessions, not a shared mode flag, per
existing project convention. Both already support `--host <url>` for
pointing at a remote/external Ollama instance
(`OllamaClient(host=...)` in `supervisor_llm.py`), which is enough for
"run Ollama somewhere else and point at it" — no new work needed there.

What's actually missing: calling **cloud-hosted models** (OpenAI/ChatGPT,
Anthropic/Claude, Google/Gemini) via API keys, as an alternative to local
Ollama, ideally selectable per run — **and**, per the latest scoping pass,
a real *selection UX* in both the CLI and the future Streamlit GUI
(`docs/gui_plan.md` Step 5), not just "the plumbing exists somewhere."

## Why this isn't a quick swap

`supervisor_llm.py`'s own docstring already flags the crux of the problem:
Ollama's tool calls (as returned by the installed `ollama` package) carry
**no per-call ID** — `supervisor_session_pid.py` threads tool results back
by `tool_name`, not by call ID, specifically because of this. Cloud
providers' tool-calling APIs are *not* shaped the same way — e.g. Claude's
Messages API returns `tool_use` blocks with an `id`, matched by
`tool_result.tool_use_id` on the way back (current as of the `claude-api`
skill consulted for this plan; verify against current SDK docs again when
this is actually implemented, given how fast that API surface moves).
OpenAI's function/tool calling is similarly ID-based. So a second provider
isn't "point the same client at a different URL" — it's a second, actually
different response shape the session/threading logic needs to handle.

## Candidate approaches

**Historical — resolved, see "Decision: hand-rolled, not aisuite" above.**
Kept below for the reasoning trail that led there.

1. **Hand-rolled per-provider clients** — write `OpenAIClient`,
   `AnthropicClient`, `GeminiClient` alongside `OllamaClient`, each
   normalizing into whatever shape `supervisor_session_pid.py` expects
   (or, better, evolve that shape to be ID-aware where available with a
   by-tool-name fallback for Ollama). Full control, but real, ongoing
   per-provider maintenance — three SDKs to track instead of one.

2. **A unified routing library — aisuite (or LiteLLM)** — flagged in
   discussion as the leading candidate. These libraries exist specifically
   to give one call shape across providers (e.g. `model="anthropic:claude-opus-5"`
   vs `model="ollama:qwen3-coder"`), which would absorb the
   tool-call-shape normalization problem instead of us solving it by hand.
   **Not yet verified**: current maturity of aisuite's tool-calling
   support, and specifically whether it handles Ollama the way this
   project already uses it (relay of `qwen3-coder:30b` with the existing
   tool schema). Check this first, before committing — don't take library
   claims on faith, confirm against its current docs/issues.

Recommendation for the follow-up session: start by spiking option 2
against this project's actual tool schema (the PID/LQG benchmark tools in
`supervisor_tools_whitebox_pid.py` / `supervisor_tools_blackbox_pid.py` /
`supervisor_tools_lqg.py`) before writing any hand-rolled adapter — if
aisuite genuinely handles the ID/no-ID split transparently, option 1
mostly disappears. This spike now needs to exercise **both** an Ollama
call and at least one real cloud call with a key, since selection between
them is now in scope, not just "some cloud provider works in the abstract."

## Provider inventory: what "available" means

Two independent families, either or both of which may be configured on a
given machine:

- **Ollama family** — zero or more reachable Ollama endpoints: the local
  daemon (default `http://localhost:11434`), or a remote/LAN instance via
  `--host` (already supported). Which *models* are actually available at
  that host isn't known statically — it depends on what's been `ollama
  pull`ed there.
- **Cloud family** — zero or more of OpenAI/Anthropic/Gemini, gated purely
  on whether that provider's API key is present.

### Suggested Ollama configuration convention

Don't invent a new project-specific config surface for "which Ollama
host." Reuse Ollama's own standard `OLLAMA_HOST` environment variable as
the default host (Ollama's CLI/daemon already respects this name), with
the existing `--host` flag overriding it per-run. This means:

- Same machine / default local daemon: no config needed, current
  zero-arg behavior.
- Same machine, non-default port, or a machine elsewhere on the LAN:
  `export OLLAMA_HOST=http://192.168.1.50:11434` (or the project's own
  `.env`, once `.env` loading exists — see below), or pass `--host`
  per-invocation.
- **Discovering which models are actually available at that host**: call
  `ollama.Client(host=...).list()` (already part of the installed
  `ollama` package) rather than hand-maintaining a static model list.
  This is what both the CLI menu and the GUI selectbox should call to
  populate the Ollama half of the choices — if the call fails (nothing
  listening at that host), Ollama simply contributes no options, silently
  — not an error state, since "no local Ollama" is an expected
  configuration, not a bug.
- Not proposing a multi-named-endpoint registry (e.g. "home" vs. "office"
  Ollama) — out of scope unless someone actually asks for it. One
  configured host per run, same as today.

### Suggested cloud-key convention

- `.env` at the repo root (already gitignored — `.env`/`.envrc` are both
  in `.gitignore` already, confirmed; this was flagged as an open
  prerequisite in the previous version of this plan and is in fact
  already satisfied).
- One var per provider: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GOOGLE_API_KEY` (or `GEMINI_API_KEY` — confirm which name the chosen
  SDK/aisuite actually expects when this is implemented; providers are
  inconsistent here).
- Availability = key presence, checked via `python-dotenv`'s `load_dotenv()`
  + `os.environ.get(...)`. A provider with no key simply doesn't appear
  in the menu — same "silently absent, not an error" treatment as an
  unreachable Ollama host.
- Which *model* per cloud provider: unlike Ollama, don't query this live
  (each provider's model-listing API is shaped differently, and it's a
  second network call per app start for little benefit at this project's
  scale). Instead, a small curated static list per provider, defined in
  code, e.g. 1-3 current model names per provider — accept that this list
  needs occasional manual updating as providers ship new models.

## CLI selection UX

- On startup (after `.env` loading), probe both families: try
  `ollama.Client(host=OLLAMA_HOST or default).list()`, and check which
  cloud env vars are set. Build one merged list of `(provider, model)`
  choices, tagged by family.
- If `--provider`/`--model` are passed explicitly, skip the menu — use
  them directly (this keeps the CLI scriptable/non-interactive for
  batch/CI use, which matters since these CLIs are also used
  head-lessly). If omitted, show a simple numbered menu built from the
  probed list and prompt once.
- If the merged list is empty (no reachable Ollama *and* no cloud keys),
  fail fast with an explicit message naming both fixes ("start Ollama or
  set `--host`", "add a key to `.env`") rather than a generic connection
  error.

## GUI selection UX (`docs/gui_plan.md` Step 5)

Same probing logic as the CLI (this should be a small shared module, not
duplicated between `cli_supervisor_*.py` and `streamlit_app.py`), exposed
as a provider+model selectbox in the LLM chat panel. Two deployment
shapes need different treatment, and it's worth being explicit that
they're different — the phrase "deployed to Streamlit" is ambiguous
between them:

1. **Self-hosted container (this repo's actual current path — `docker
   compose up`, see `docker-compose.yml`/`docs/docker_plan.md`).** A host
   machine's Ollama daemon is *not* reachable from inside the container
   as `localhost:11434` — container networking isolates it, this is a
   real, easy-to-hit gap, not a hypothetical one. Fix: document pointing
   `OLLAMA_HOST` at the host via Docker's special DNS name
   `host.docker.internal` (works out of the box on Docker Desktop; on
   Linux needs `--add-host=host.docker.internal:host-gateway`, or a
   compose `extra_hosts:` entry, or just the host's LAN IP as a
   fallback). This is consistent with `docs/docker_plan.md`'s existing
   "Ollama sidecar deliberately excluded, reached via `--host`" stance —
   it just hasn't had the concrete recipe written down until now.
2. **Public hosted Streamlit (e.g. Streamlit Community Cloud) — no
   user-controlled container/network at all.** No LAN/localhost Ollama is
   reachable from there, period. In this mode the Ollama probe simply
   fails and those options don't appear in the menu (same "silently
   absent" treatment as above — no special-casing needed if the probe is
   already fail-soft). What *does* need adding: a **session-only key
   entry widget** — `st.text_input(type="password")` per provider — so a
   visitor can paste their own OpenAI/Anthropic/Gemini key and use that
   provider through the hosted UI without the operator's `.env` needing
   to hold it. This key lives only in `st.session_state` for that
   browser session, is never written to disk or logged, and is cleared
   on refresh — the same treatment `docs/gui_plan.md` already gives chat
   history ("ephemeral … no persistence layer for v1"). Useful in the
   self-hosted case too, as a per-session override without editing
   `.env` and restarting the container.

## Gaps found in this revision, and suggested fixes

Re-reading the original version of this plan against the CLI+GUI
selection requirement surfaced several real gaps:

1. **No Ollama model-discovery story.** The original plan only covered
   *reaching* an Ollama host (`--host`), not *listing what's available*
   there for a selection menu. Fix: `Client(host=...).list()`, above.
2. **No cloud model-catalog story.** Same issue on the cloud side — which
   concrete model IDs to offer wasn't specified. Fix: small curated
   static list per provider, documented as needing periodic manual
   updates (flag this as a maintenance cost, not a one-time decision).
3. **GUI key entry wasn't in the original plan at all** — it only
   considered `.env`, which doesn't work for a publicly hosted instance
   with no operator-controlled filesystem. Fix: session-only key-entry
   widget, above, with an explicit "never persisted/logged" rule.
4. **Container-network Ollama unreachability wasn't called out** — the
   original plan's "point `--host` at a remote instance" language reads
   as if that trivially covers the container case; it doesn't, container
   networking isolates `localhost` by default. Fix: `host.docker.internal`
   recipe, above, and this should land as an addition to
   `docs/docker_plan.md`'s existing "Ollama sidecar" bullet, not just
   here.
5. **Dockerfile/environment.yml dependency drift.** `src/Dockerfile`
   currently installs `numpy scipy matplotlib ollama streamlit` only —
   no cloud SDKs, no `python-dotenv`, no aisuite/LiteLLM. Once a library
   choice is made, both `src/Dockerfile` and the conda `environment.yml`
   need the new dependency, or the container silently falls back to
   Ollama-only. Easy to forget since it's not "the code," flagging
   explicitly.
6. **No `.env.example`.** Nothing in the repo documents the expected env
   var names. Belongs in implementation (once names are finalized against
   whatever library is chosen), not this planning pass — noted so it
   isn't dropped.
7. **Cost-confirmation UX is still just a TODO, not a design.** Carried
   over from the previous version of this plan, still unresolved: is it a
   one-time banner on provider switch, a per-call confirmation, or a
   running cost counter? Deliberately left open rather than decided here
   — first real design question for the implementation session once the
   selection flow itself exists to hang it on.

## Shape of the work, once started

- **Shared provider-probing module** — one place implementing "what
  Ollama models are reachable" + "which cloud keys are set," imported by
  both the CLI scripts and `streamlit_app.py`'s LLM panel, so the two
  surfaces can't drift out of sync on selection logic.
- **Provider selection + config**: env-var driven (`OLLAMA_HOST`,
  `OPENAI_API_KEY`/`ANTHROPIC_API_KEY`/`GOOGLE_API_KEY`), `.env`-loaded
  via `python-dotenv` locally (free in Docker via `--env-file` / compose's
  `env_file:`, no Python dependency needed there).
- **Cost/safety, not just plumbing**: cloud calls cost real money per
  token (unlike local Ollama), so some kind of "confirm the model/expected
  cost" surfacing belongs in the CLI UX and the GUI, not just silent
  provider swapping — see open gap #7 above.
- **GUI key entry is additive, not a replacement** for `.env` — an
  operator running their own `docker compose up` instance still just uses
  `.env`; the session-only widget only matters for a publicly hosted
  instance with no operator-controlled `.env`.

## Non-goals of this document (historical — describes the state before this session's implementation; see "Implementation status" at top for what's actually true now)

- No implementation, no new dependencies installed, no code changes.
- No decision between hand-rolled vs. aisuite/LiteLLM — that's the first
  real step of the follow-up session (spike aisuite against both an
  Ollama call and a real cloud call, then decide).
- Does not cover the Docker side of running this beyond the
  `host.docker.internal` recipe above (see `docs/docker_plan.md`) —
  Dockerfile/compose edits themselves are implementation, not this plan.
- Does not decide the cost-confirmation UX mechanism — flagged as open,
  not resolved.
