"""Streamlit LLM chat panel — Build plan Step 5 (docs/gui_plan.md).

Cloud-key entry plus the chat itself, for one of two existing,
separately-maintained supervisor conversations: supervisor_session_pid.
Session (SISO/PID) and supervisor_session_lqg.LQGSession (MIMO/LQG).
Deliberately not merged into one session/prompt — supervisor_session_lqg.
py's own docstring documents why these stay separate (mirrors cli_pid.py/
cli_lqg.py staying separate scripts rather than a unified dispatcher).

Which track (`render_controls(track)`'s argument) is streamlit_unified_
panel.py's call to make, not this module's — the Track selector lives
there now, shared with the Manual-mode panels, since Mode=LLM Supervisor
is just one more thing that selector picks between.

Every successful benchmark tool call is drawn automatically, without the
user ever asking for a chart: Session/LQGSession's _wrap_benchmark
captures each call's simulated rows (row["sim"]/row.sim, stripped back
out before the result reaches json.dumps/the model) into
session.plot_calls; _drain_plot_calls() below turns that into
gs.ControllerEntry rows via streamlit_siso_panel.absorb_llm_rows() /
streamlit_mimo_panel.absorb_llm_rows() — the exact same entries a manual
"Compare all methods" click builds, tagged source="llm" — so
streamlit_unified_panel.py's plots column (each track panel's own
render_plots()) draws them without this module needing a plotting
function of its own.

Key resolution, checked in order per provider (see _configured_key):
1. An operator-configured key — `.env`/the process environment (loaded via
   python-dotenv; also how a self-hosted docker-compose env_file: entry
   arrives), or Streamlit's own Secrets manager (st.secrets, the Community
   Cloud equivalent — there's no filesystem `.env` there). Never displayed,
   only its presence matters, and if found the visitor isn't asked for one.
2. Otherwise, a session-only textbox — never written to disk or logged,
   lost on refresh. This is the only source on a public deploy with no
   operator-configured key.

Claude, ChatGPT, and Gemini are all wired to actual clients
(supervisor_llm_anthropic.AnthropicClient, supervisor_llm_openai.
OpenAIClient, supervisor_llm_gemini.GeminiClient) — see docs/aituner_plan.md.
"""

from __future__ import annotations

import datetime
import os
import sys
import traceback

import anthropic
import openai
import streamlit as st
from dotenv import load_dotenv
from google.genai import errors as genai_errors

from supervisor_llm_anthropic import AnthropicClient
from supervisor_llm_openai import OpenAIClient
from supervisor_llm_gemini import GeminiClient
from supervisor_session_pid import Session
from supervisor_session_lqg import LQGSession
from supervisor_tools_blackbox_pid import RUN_BLACKBOX_BENCHMARK_SCHEMA, run_blackbox_benchmark
from supervisor_tools_whitebox_pid import RUN_WHITEBOX_BENCHMARK_SCHEMA, run_whitebox_benchmark
from supervisor_tools_lqg import RUN_LQG_BENCHMARK_SCHEMA, run_lqg_benchmark

import report_html
import streamlit_gui_state as gs
import streamlit_siso_panel as siso_panel
import streamlit_mimo_panel as mimo_panel

WHITEBOX_TOOL = (RUN_WHITEBOX_BENCHMARK_SCHEMA, run_whitebox_benchmark)
BLACKBOX_TOOL = (RUN_BLACKBOX_BENCHMARK_SCHEMA, run_blackbox_benchmark)
LQG_TOOL = (RUN_LQG_BENCHMARK_SCHEMA, run_lqg_benchmark)

PROVIDERS = ["Claude (Anthropic)", "ChatGPT (OpenAI)", "Gemini (Google)"]
WIRED_PROVIDERS = {"Claude (Anthropic)", "ChatGPT (OpenAI)", "Gemini (Google)"}

KEY_STATE = {
    "Claude (Anthropic)": "llm_api_key_anthropic",
    "ChatGPT (OpenAI)": "llm_api_key_openai",
    "Gemini (Google)": "llm_api_key_gemini",
}

ENV_VAR = {
    "Claude (Anthropic)": "ANTHROPIC_API_KEY",
    "ChatGPT (OpenAI)": "OPENAI_API_KEY",
    "Gemini (Google)": "GOOGLE_API_KEY",
}

# (model id, menu label) -- cheapest/most-capable tradeoff made explicit in
# the label since it's not obvious from the name alone. Opus deliberately
# not offered here -- most expensive tier, no reason to offer it in this
# app. Haiku 4.5 is the default: cheapest per-token AND (per the claude-api
# skill's thinking-defaults table) an "older model" tier that does NOT run
# adaptive thinking unless explicitly configured -- unlike Sonnet 5, which
# (like Opus 5) runs adaptive thinking by default, billing extra output
# tokens for invisible reasoning on every call. Correcting an earlier wrong
# assumption here: Sonnet does NOT default to no-thinking the way Haiku does.
ANTHROPIC_MODELS = [
    ("claude-haiku-4-5", "Claude Haiku 4.5 -- fastest, cheapest (default)"),
    ("claude-sonnet-5", "Claude Sonnet 5 -- more capable, more expensive"),
]
ANTHROPIC_DEFAULT_MODEL_INDEX = 0  # Haiku -- see comment above

# Same cheapest/most-capable tradeoff as ANTHROPIC_MODELS above, for
# OpenAI's current (2026-09) tier lineup -- gpt-5.6-sol/gpt-6-astra
# deliberately not offered here, same "most expensive tier, no reason to
# offer it in this app" reasoning as excluding Opus above. See
# supervisor_llm_openai.py's module docstring for where these ids/prices
# were verified.
OPENAI_MODELS = [
    ("gpt-5.6-luna", "GPT-5.6 Luna -- fastest, cheapest (default)"),
    ("gpt-5.6-terra", "GPT-5.6 Terra -- more capable, more expensive"),
]
OPENAI_DEFAULT_MODEL_INDEX = 0  # Luna -- see comment above

# Same cheapest/most-capable tradeoff again, for Gemini's current (2026-09)
# tier lineup -- gemini-3.1-pro-preview deliberately not offered here, same
# "most expensive tier, no reason to offer it in this app" reasoning as
# excluding Opus/gpt-5.6-sol above. See supervisor_llm_gemini.py's module
# docstring for where these ids/prices were verified.
GEMINI_MODELS = [
    ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite -- fastest, cheapest (default)"),
    ("gemini-3.8-flash", "Gemini 3.8 Flash -- more capable, more expensive"),
]
GEMINI_DEFAULT_MODEL_INDEX = 0  # Flash-Lite -- see comment above

# Per-provider (model list, default index) and per-provider model widget
# key -- a second wired provider means the model picker can no longer
# unconditionally use ANTHROPIC_MODELS/a single "llm_model" key: switching
# Provider would otherwise carry the previous provider's selected model id
# into a selectbox whose options no longer include it. A separate widget
# key per provider (rather than resetting a shared one) sidesteps that and
# matches the existing per-provider-key pattern already used for KEY_STATE/
# ENV_VAR above; only wired providers need an entry.
MODELS_BY_PROVIDER = {
    "Claude (Anthropic)": (ANTHROPIC_MODELS, ANTHROPIC_DEFAULT_MODEL_INDEX),
    "ChatGPT (OpenAI)": (OPENAI_MODELS, OPENAI_DEFAULT_MODEL_INDEX),
    "Gemini (Google)": (GEMINI_MODELS, GEMINI_DEFAULT_MODEL_INDEX),
}
MODEL_KEY_STATE = {
    "Claude (Anthropic)": "llm_model_anthropic",
    "ChatGPT (OpenAI)": "llm_model_openai",
    "Gemini (Google)": "llm_model_gemini",
}


def _init_panel_state() -> None:
    """Deliberately does NOT setdefault() KEY_STATE's/MODEL_KEY_STATE's
    entries the way an earlier version of this function did for KEY_STATE --
    st.text_input()/st.selectbox() already default a genuinely-new key to
    ""/their first option with no help needed here, and pre-seeding a
    provider's key *before* its own widget has ever been instantiated (e.g.
    while a different provider is selected) turned out to poison that key's
    Streamlit-internal widget lifecycle: once real openai/gemini providers
    started using KEY_STATE too (not just Claude), a key first created via
    setdefault -- rather than born as a widget -- stopped being cleanly
    removed from st.session_state on a run where its widget doesn't render
    (e.g. Mode=Manual); Streamlit instead left it behind reset to "",
    which is *present*, so preserve_widget_state()'s `key not in
    st.session_state` check never fires and the stale "" silently wins over
    the real value sitting in the shadow copy. Confirmed via a scripted
    AppTest repro (switch provider, type a key, flip to Manual and back --
    the key came back blank) before landing this fix; Claude alone never
    exposed it since it was always the sole default-selected, always-a-
    widget provider."""
    if "llm_dotenv_loaded" not in st.session_state:
        load_dotenv()
        st.session_state["llm_dotenv_loaded"] = True
    st.session_state.setdefault("llm_session_obj", None)
    st.session_state.setdefault("llm_session_fingerprint", None)


def _configured_key(provider: str) -> str | None:
    """An operator-configured key for `provider`, checked ahead of asking
    the visitor for their own — see module docstring for the two sources
    and their precedence. Returns None (not an error) if neither source has
    anything, including when no secrets store is configured at all."""
    env_var = ENV_VAR[provider]
    key = os.environ.get(env_var)
    if key:
        return key
    try:
        return st.secrets.get(env_var)
    except Exception:
        # No secrets.toml / secrets store configured at all -- nothing
        # configured this way, not a real error.
        return None


def _log_exception(exc: Exception) -> None:
    """Full traceback to the server console only -- e.g. the terminal
    running `streamlit run`, or Streamlit Cloud's app-logs viewer for a
    deployed instance. Never shown to the visitor: on a public deploy that
    would leak internals (paths, library versions) to strangers for no
    benefit -- the chat only ever gets the short, friendly message."""
    print(f"\n--- LLM chat error ({type(exc).__name__}) ---", file=sys.stderr)
    traceback.print_exception(exc, file=sys.stderr)


def _new_session(provider: str, api_key: str, track: str, model: str):
    if provider == "ChatGPT (OpenAI)":
        client = OpenAIClient(api_key=api_key, model=model)
    elif provider == "Gemini (Google)":
        client = GeminiClient(api_key=api_key, model=model)
    else:
        client = AnthropicClient(api_key=api_key, model=model)
    if track == "SISO / PID":
        return Session(client, whitebox_tool=WHITEBOX_TOOL, blackbox_tool=BLACKBOX_TOOL,
                        capture_plots=True)
    return LQGSession(client, lqg_tool=LQG_TOOL, capture_plots=True)


def _render_key_entry():
    st.subheader("Model access")
    provider = st.selectbox("Provider", PROVIDERS, key="llm_provider")
    configured_key = _configured_key(provider)
    if configured_key:
        st.success(f"{provider} key configured by the operator — nothing to enter.")
        api_key = configured_key
    else:
        api_key = st.text_input(
            f"{provider} API key", type="password", key=KEY_STATE[provider],
            help="Kept only in this browser session — never written to disk or logged.",
        )
    if provider not in WIRED_PROVIDERS and api_key:
        st.info(f"{provider} key accepted, but chat wiring for this provider isn't built yet "
                 "— Claude is the only provider connected so far.")
        return provider, api_key, None

    model = None
    if provider in WIRED_PROVIDERS:
        models, default_index = MODELS_BY_PROVIDER[provider]
        # setdefault(), not index= -- same "default value but also set via
        # Session State API" collision as streamlit_siso_panel.py's
        # siso_plant_form (see its comment). Safe here despite
        # _init_panel_state()'s own docstring warning against setdefault()
        # on these same MODEL_KEY_STATE keys: that bug came from seeding
        # every provider's key unconditionally, before any of them had ever
        # been born as a widget. This only ever seeds the CURRENTLY
        # selected provider's own key, immediately before that exact
        # widget call in the same branch -- never a key whose widget
        # doesn't also render this run.
        st.session_state.setdefault(MODEL_KEY_STATE[provider], models[default_index][0])
        model = st.selectbox(
            "Model", [m[0] for m in models],
            format_func=lambda m: dict(models)[m],
            key=MODEL_KEY_STATE[provider],
        )
        st.warning(
            "Chatting here sends real, billed requests to the provider using "
            f"the key above -- cost depends on conversation length and the "
            "model picked (the default is the cheapest option offered).",
            icon="💸",
        )
    return provider, api_key, model


_PROTECTED_KEYS = ["llm_provider", "llm_api_key_anthropic", "llm_api_key_openai",
                   "llm_api_key_gemini", "llm_model_anthropic", "llm_model_openai",
                   "llm_model_gemini"]

_TRACK_KIND = {"SISO / PID": "siso", "MIMO / LQG": "mimo"}


def _render_manual_plant_hint(track):
    """"Currently in Manual mode: <plant> — mention this to the
    supervisor to pick up where you left off.", shown only when Manual
    mode currently holds a plant for this Track that no LLM-sourced
    entry already covers. The mirror image of streamlit_siso_panel.
    _render_llm_plant_carryover()'s "Load this plant" affordance, but
    hint-only: st.chat_input has no pre-fill mechanism, and firing a
    real, billed API call from a click was explicitly ruled out (see the
    module docstring's Key resolution note on cost) -- so there's
    nothing to click here, just a nudge to mention it.

    Reads Manual mode's plant via its shadow state rather than requiring
    Manual's own widgets to have rendered this run -- see streamlit_siso_
    panel.current_manual_plant()/streamlit_mimo_panel.current_manual_
    plant_name()."""
    if track == "SISO / PID":
        plant = siso_panel.current_manual_plant()
    else:
        plant = mimo_panel.current_manual_plant_name()
    if plant is None:
        return
    kind = _TRACK_KIND[track]
    if any(e.plant == plant for e in gs.get_by_kind(kind) if e.source == "llm"):
        return
    st.caption(f"Currently in Manual mode: {plant} — mention this to the "
               "supervisor to pick up where you left off.")


def _transcript_html(session) -> str:
    """Every turn in `session.messages` as HTML paragraphs, in order --
    user text, each assistant tool call, each tool result, each assistant
    reply -- the full conversation, not just the current state. Duck-typed
    on the same .content/.tool_calls surface supervisor_session_judge_pid.
    serialize_candidate_trace() uses (see that module's own docstring for
    why that's safe across every client), not a Session-specific type
    check -- this function works unmodified for Ollama/Anthropic/OpenAI/
    Gemini sessions alike."""
    parts = []
    for entry in session.messages:
        if isinstance(entry, dict):
            role = entry.get("role")
            if role == "system":
                continue
            if role == "tool":
                parts.append(f'<p class="section-note">Tool result ({report_html.e(entry["tool_name"])}): '
                              f'<code>{report_html.e(entry["content"])}</code></p>')
            else:
                parts.append(f"<p><strong>{report_html.e(role.capitalize())}:</strong> {report_html.e(entry['content'])}</p>")
            continue
        for tc in getattr(entry, "tool_calls", None) or []:
            args = ", ".join(f"{k}={v!r}" for k, v in (tc.function.arguments or {}).items())
            parts.append(f'<p class="section-note">Called <code>{report_html.e(tc.function.name)}({report_html.e(args)})</code></p>')
        text = getattr(entry, "content", "") or ""
        if text:
            parts.append(f"<p><strong>Assistant:</strong> {report_html.e(text)}</p>")
    return "\n".join(parts)


_ENTRIES_SECTIONS_BY_TRACK = {"SISO / PID": siso_panel.build_entries_report_sections,
                              "MIMO / LQG": mimo_panel.build_entries_report_sections}


def _build_report_html(track, session):
    """The full-history report for Mode="LLM Supervisor": whichever
    session-list entries are currently plotted for this Track
    (build_entries_report_sections() -- the exact same section Manual
    mode's own report shows, not a re-implementation; see that function's
    docstring for why it's "any source," not just source="llm"), followed
    by the entire conversation transcript (_transcript_html(), every
    round, not just the latest). Tables/plots first, transcript last --
    the transcript is usually the longest section by far, and a reader
    opening the report wants the results before a long scroll of prose."""
    entries_subtitle, entries_sections = _ENTRIES_SECTIONS_BY_TRACK[track]()
    sections = entries_sections + [f"<h2>Conversation</h2>{_transcript_html(session)}"]
    return report_html.build_report(f"PIDTuner LLM Supervisor Report ({track})", entries_subtitle, sections)


def _render_download_report_button(track, session):
    if not session.messages[1:]:  # index 0 is always the system prompt
        return
    st.download_button(
        "Download report", data=_build_report_html(track, session),
        file_name=f"pidtuner-supervisor-report-{datetime.date.today().isoformat()}.html",
        mime="text/html", key="llm_download_report",
    )


def reset_clears_track_state(track) -> None:
    """Everything "Reset conversation" clears beyond the conversation
    object itself, for this Track: the plotted source="llm" graph entries
    (see the button's own comment for why), and -- MIMO only -- the
    4-curve/per-channel-step comparison caches (st.session_state["mimo_
    four_curve"]/["mimo_per_channel_step"]), which aren't session-list
    entries at all (see streamlit_mimo_panel.build_entries_report_
    sections()'s own comment) so gs.clear_by_kind_and_source() alone
    never touches them -- a stale one would otherwise still show up in
    the *next* conversation's report. Deliberately NOT called by "Clear
    LLM plots" (streamlit_siso_panel.py/streamlit_mimo_panel.py's session-
    list button), which only ever touches the graph -- these two caches
    are comparisons the user explicitly ran, not something the LLM
    "added" to the graph the way a benchmark-tool call did, so clearing
    them isn't part of what that specific button promises.

    Public (no leading underscore) -- called by both streamlit_llm_
    panel.py's own "Reset conversation" and streamlit_judge_panel.py's,
    so the two rules above stay in exactly one place."""
    gs.clear_by_kind_and_source(_TRACK_KIND[track], "llm")
    if track == "MIMO / LQG":
        st.session_state.pop("mimo_four_curve", None)
        st.session_state.pop("mimo_per_channel_step", None)


def render_controls(track):
    """The left-hand controls half for Mode=LLM Supervisor — called by
    streamlit_unified_panel.py with whichever Track it currently has
    selected. Mirrors streamlit_siso_panel.py/streamlit_mimo_panel.py's
    own render_controls() split, but this module has no render_plots()
    of its own — see the module docstring for why.

    preserve_widget_state()/snapshot_widget_state() bracket _render_key_
    entry() specifically (not the whole function) so the snapshot still
    happens even on this function's own early returns below -- without
    it, switching to Manual mode and back would silently reset a
    session-only API key/model choice to blank/default, since Streamlit
    deletes a widget's state whenever it isn't instantiated on a run.
    See streamlit_gui_state.py's own note on why this is needed at all.

    preserve_widget_state() runs *before* _init_panel_state() -- harmless
    ordering now that _init_panel_state() no longer setdefaults KEY_STATE/
    MODEL_KEY_STATE entries at all (see that function's own docstring for
    why a setdefault there, even ordered after preserve_widget_state(),
    used to permanently break restoration for any provider that wasn't
    the currently-selected one), kept this way regardless since preserve
    must still run before _render_key_entry()'s widgets are instantiated
    either way."""
    gs.preserve_widget_state(_PROTECTED_KEYS)
    _init_panel_state()
    provider, api_key, model = _render_key_entry()
    gs.snapshot_widget_state(_PROTECTED_KEYS)

    if not api_key:
        st.info("Enter an API key above to start chatting.")
        return
    if provider not in WIRED_PROVIDERS:
        return

    fingerprint = (provider, api_key, track, model)
    if st.session_state["llm_session_fingerprint"] != fingerprint:
        st.session_state["llm_session_obj"] = _new_session(provider, api_key, track, model)
        st.session_state["llm_session_fingerprint"] = fingerprint
        gs.clear_chat()

    if st.button("Reset conversation", key="llm_reset"):
        # Also clears this Track's plotted LLM entries (and, for MIMO,
        # the 4-curve/per-channel-step caches) -- see reset_clears_track_
        # state()'s own docstring for why: a fresh conversation's eventual
        # "Download report" must never embed a stale plot from before the
        # reset.
        st.session_state["llm_session_obj"] = _new_session(provider, api_key, track, model)
        gs.clear_chat()
        reset_clears_track_state(track)

    # "Clear LLM plots" lives in the session list itself now (Select all/
    # Deselect all/Clear all/Remove unchecked's own row, in streamlit_
    # siso_panel.py/streamlit_mimo_panel.py's _render_session_list()) --
    # not here. It only ever acts on the graph, not the conversation, so
    # it belongs with its session-list siblings (the same kind of
    # source-scoped bulk action, right next to the ones that aren't
    # source-scoped) rather than buried among chat-management buttons in
    # a different column the graph itself isn't even in. Moving it there
    # also makes it available in every Mode, not just this one.

    _render_manual_plant_hint(track)
    _render_download_report_button(track, st.session_state["llm_session_obj"])

    # Reserving this container before chat_input (below) puts it above
    # chat_input in the DOM regardless of Streamlit's own auto-bottom-pin
    # behavior for chat_input, which doesn't reliably take effect nested
    # inside a column laid out by streamlit_unified_panel.py (same class
    # of live-only rendering quirk noted in docs/gui_plan.md's "Resolved
    # decisions" for the nested-st.tabs case this was originally found
    # in -- confirmed here with a real screenshot, not just suspected).
    # No height/border -- free-floating bubbles in the column's own flow,
    # the column itself scrolls, matching the original look; only the
    # order was ever the bug.
    history_box = st.container()
    user_text = st.chat_input("Tell me about your plant and what matters most to you.")

    if user_text:
        gs.append_chat_message("user", user_text)

    with history_box:
        for message in st.session_state[gs.CHAT_KEY]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_text:
            session = st.session_state["llm_session_obj"]
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        reply = session.handle_user_message(user_text)
                    except (anthropic.AuthenticationError, openai.AuthenticationError) as exc:
                        _log_exception(exc)
                        reply = "That API key was rejected — double-check it and try again."
                    except (anthropic.RateLimitError, openai.RateLimitError) as exc:
                        _log_exception(exc)
                        reply = "Rate limited by the provider — wait a moment and try again."
                    except genai_errors.ClientError as exc:
                        # google-genai has no per-error-type exception classes
                        # the way anthropic/openai do (confirmed from its
                        # errors.py source) -- ClientError covers every 4xx,
                        # so auth/rate-limit are distinguished by status code
                        # instead of exception type.
                        _log_exception(exc)
                        if exc.code in (401, 403):
                            reply = "That API key was rejected — double-check it and try again."
                        elif exc.code == 429:
                            reply = "Rate limited by the provider — wait a moment and try again."
                        else:
                            reply = f"The model provider returned an error: {exc.message}"
                    except (anthropic.APIStatusError, openai.APIStatusError, genai_errors.ServerError) as exc:
                        _log_exception(exc)
                        reply = f"The model provider returned an error: {exc.message}"
                    except (anthropic.APIConnectionError, openai.APIConnectionError) as exc:
                        _log_exception(exc)
                        reply = "Couldn't reach the model provider — check your connection and try again."
                    except Exception as exc:  # noqa: BLE001 - a bad turn must not crash the app
                        _log_exception(exc)
                        reply = "Something went wrong handling that message — check the server logs for details."
                    else:
                        # Own try/except, separate from the one above: a
                        # plotting bug here must not crash the app or throw
                        # away the reply that already computed successfully.
                        try:
                            _drain_plot_calls(session)
                        except Exception as exc:  # noqa: BLE001
                            _log_exception(exc)
                st.markdown(reply)
            gs.append_chat_message("assistant", reply)


def absorb_calls(calls):
    """The per-call dispatch half of _drain_plot_calls() -- split out so a
    caller draining more than one session's plot_calls at once
    (streamlit_judge_panel.py's own drain, which needs to dedupe *across*
    candidates before any of this runs) can reuse the exact same
    kind-dispatch logic rather than re-implementing it. Public (no leading
    underscore) for that reuse. Takes an already-decided list of calls --
    no opinion on where they came from or whether any were filtered out."""
    for call in calls:
        if call["kind"] == "siso":
            # Only Session (SISO/PID) tags calls with "delay" -- see its
            # _wrap_benchmark -- so .get() with a 0.0 default also covers
            # a pre-existing plot_calls entry from before that field.
            siso_panel.absorb_llm_rows(call["plant"] or "?", call["rows"], call.get("delay", 0.0))
        else:
            # Only LQGSession tags calls with "plant_preset"/
            # "custom_plant_literals" -- see its _wrap_benchmark -- so
            # .get() with defaults also covers a pre-existing plot_calls
            # entry from before those fields.
            mimo_panel.absorb_llm_rows(
                call["plant"] or "?", call["rows"],
                plant_preset=call.get("plant_preset", ""),
                custom_plant_literals=call.get("custom_plant_literals"),
                four_curve=call.get("four_curve"))


def _drain_plot_calls(session):
    """Turn every benchmark call captured since the last drain
    (session.plot_calls -- populated by Session/LQGSession's
    _wrap_benchmark, one entry per successful sim-capable tool call)
    into session entries via absorb_calls() -- see its own docstring and
    the module docstring above. Called once per turn, right after
    handle_user_message() returns without raising, so this only ever
    runs after its JSON reply has already been built -- nothing here
    reaches the model.

    Swaps session.plot_calls out for a fresh list *before* processing,
    rather than clearing it after the loop: if absorb_calls() raises
    partway through (the caller wraps this whole function in its own
    try/except), the calls already popped here are gone either way --
    they don't linger to be silently re-added, duplicated, on some later
    turn's successful drain."""
    calls, session.plot_calls = session.plot_calls, []
    absorb_calls(calls)
