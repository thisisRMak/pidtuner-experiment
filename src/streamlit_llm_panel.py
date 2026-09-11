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

Only Claude is wired to an actual client so far (supervisor_llm_anthropic.
AnthropicClient). OpenAI/Gemini keys are accepted here but not yet
connected to anything — see docs/aituner_plan.md.
"""

from __future__ import annotations

import os
import sys
import traceback

import anthropic
import streamlit as st
from dotenv import load_dotenv

from supervisor_llm_anthropic import AnthropicClient
from supervisor_session_pid import Session
from supervisor_session_lqg import LQGSession
from supervisor_tools_blackbox_pid import RUN_BLACKBOX_BENCHMARK_SCHEMA, run_blackbox_benchmark
from supervisor_tools_whitebox_pid import RUN_WHITEBOX_BENCHMARK_SCHEMA, run_whitebox_benchmark
from supervisor_tools_lqg import RUN_LQG_BENCHMARK_SCHEMA, run_lqg_benchmark

import streamlit_gui_state as gs
import streamlit_siso_panel as siso_panel
import streamlit_mimo_panel as mimo_panel

WHITEBOX_TOOL = (RUN_WHITEBOX_BENCHMARK_SCHEMA, run_whitebox_benchmark)
BLACKBOX_TOOL = (RUN_BLACKBOX_BENCHMARK_SCHEMA, run_blackbox_benchmark)
LQG_TOOL = (RUN_LQG_BENCHMARK_SCHEMA, run_lqg_benchmark)

PROVIDERS = ["Claude (Anthropic)", "ChatGPT (OpenAI)", "Gemini (Google)"]
WIRED_PROVIDERS = {"Claude (Anthropic)"}

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


def _init_panel_state() -> None:
    if "llm_dotenv_loaded" not in st.session_state:
        load_dotenv()
        st.session_state["llm_dotenv_loaded"] = True
    for state_key in KEY_STATE.values():
        st.session_state.setdefault(state_key, "")
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
        model = st.selectbox(
            "Model", [m[0] for m in ANTHROPIC_MODELS],
            format_func=lambda m: dict(ANTHROPIC_MODELS)[m],
            index=ANTHROPIC_DEFAULT_MODEL_INDEX, key="llm_model",
        )
        st.warning(
            "Chatting here sends real, billed requests to the provider using "
            f"the key above -- cost depends on conversation length and the "
            "model picked (Haiku is cheapest).",
            icon="💸",
        )
    return provider, api_key, model


_PROTECTED_KEYS = ["llm_provider", "llm_api_key_anthropic", "llm_api_key_openai",
                   "llm_api_key_gemini", "llm_model"]

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

    preserve_widget_state() runs *before* _init_panel_state(): the latter
    does st.session_state.setdefault(state_key, "") for every KEY_STATE
    entry, which -- setdefault only writes when the key is absent -- would
    otherwise itself count as "already set" and block the restore below
    from ever firing, permanently pinning every key to "" the moment it's
    ever garbage-collected."""
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
        st.session_state["llm_session_obj"] = _new_session(provider, api_key, track, model)
        gs.clear_chat()

    if st.button("Clear LLM entries", key="llm_clear_entries"):
        # Only this Track's LLM-tagged entries -- source="you" entries and
        # the other Track's kind are left alone. See gs.clear_by_kind_
        # and_source().
        gs.clear_by_kind_and_source(_TRACK_KIND[track], "llm")

    _render_manual_plant_hint(track)

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
                    except anthropic.AuthenticationError as exc:
                        _log_exception(exc)
                        reply = "That API key was rejected — double-check it and try again."
                    except anthropic.RateLimitError as exc:
                        _log_exception(exc)
                        reply = "Rate limited by the provider — wait a moment and try again."
                    except anthropic.APIStatusError as exc:
                        _log_exception(exc)
                        reply = f"The model provider returned an error: {exc.message}"
                    except anthropic.APIConnectionError as exc:
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


def _drain_plot_calls(session):
    """Turn every benchmark call captured since the last drain
    (session.plot_calls -- populated by Session/LQGSession's
    _wrap_benchmark, one entry per successful sim-capable tool call)
    into session entries via each track panel's own absorb_llm_rows() --
    see the module docstring. Called once per turn, right after
    handle_user_message() returns without raising, so this only ever
    runs after its JSON reply has already been built -- nothing here
    reaches the model.

    Swaps session.plot_calls out for a fresh list *before* processing,
    rather than clearing it after the loop: if absorb_llm_rows raises
    partway through (the caller wraps this whole function in its own
    try/except), the calls already popped here are gone either way --
    they don't linger to be silently re-added, duplicated, on some later
    turn's successful drain."""
    calls, session.plot_calls = session.plot_calls, []
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
