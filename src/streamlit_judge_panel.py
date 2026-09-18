"""Streamlit LLM-as-judge chat panel -- Mode="LLM Judge", the third Mode
streamlit_unified_panel.py's own docstring reserved but never built until
now. One chat thread for the user; every message fans out to N candidate
supervisor_session_pid.Session instances in lockstep (see
supervisor_session_judge_pid.JudgeSession), each running the same
conversation a single-provider Mode="LLM Supervisor" chat would, and a
separate judge model mediates every round -- deciding whether to ask the
user something or arbitrate/synthesize a recommendation from what the
candidates have found so far. The user never sees a candidate's raw reply
directly, only the judge's.

Provider/model/key plumbing is deliberately NOT reinvented here --
PROVIDERS/KEY_STATE/ENV_VAR/MODELS_BY_PROVIDER/_configured_key/_new_session/
_drain_plot_calls/_log_exception are all imported from streamlit_llm_panel
and reused as-is. Reusing the exact same KEY_STATE widget keys means an API
key entered in Mode="LLM Supervisor" carries over to Mode="LLM Judge" (and
back) via streamlit_gui_state.py's existing shadow-state mechanism -- it's
one operator-configured or session-typed key per provider, not per mode.

Role selection is a multiselect (candidates) + a dependent selectbox
(judge), not N separate per-slot provider/model widgets -- avoids a
dynamic number-of-widgets-per-run problem entirely: candidate count is
just "however many options are selected" in one widget, no per-index
keys to preserve across a Mode/Track switch. The same provider can appear
twice as a candidate at two different tiers (e.g. Haiku vs Sonnet), but
the judge's own (provider, model) pair must differ from every candidate's
-- avoids a model favoring its own family's answer. Judge output shape is
"pick + explain + flag disagreement, and optionally synthesize" -- see
supervisor_prompts_judge_pid.JUDGE_SYSTEM_PROMPT for the actual instructions;
this module has no opinion on what the judge says, only on getting it the
right inputs and showing the reply.

A per-turn "what did each candidate actually say" expander is rendered
under every judge reply (_render_candidate_rounds()), in light-grey
st.caption text grouped one block per candidate -- persisted into
gs.JUDGE_CHAT_KEY (via JudgeSession.last_round, see supervisor_session_
judge_pid.py) so it renders identically on history replay, not just for
the turn that was live when it was computed. One candidate's own provider
error (rate limit, a real capacity 503, ...) is caught per-candidate
inside JudgeSession, not here -- it shows up in this same expander rather
than aborting the round; see that module's docstring.

Deliberately NOT built in this pass (see docs/aituner_plan.md's own
scoping notes for the equivalent single-provider gaps): a CLI equivalent,
MIMO/LQG support, and any de-duplication of identical benchmark rows
plotted once per candidate on the same plant (drained as-is, same reuse
as the single-provider path -- see the plot-drain loop in render_controls()).
"""

from __future__ import annotations

import anthropic
import openai
import streamlit as st
from google.genai import errors as genai_errors

import streamlit_llm_panel as llm_panel
from streamlit_llm_panel import (
    KEY_STATE,
    MODELS_BY_PROVIDER,
    _configured_key,
    _init_panel_state,
    _log_exception,
    _new_session,
)
from supervisor_llm_anthropic import AnthropicClient
from supervisor_llm_openai import OpenAIClient
from supervisor_llm_gemini import GeminiClient
from supervisor_session_judge_pid import JudgeSession

import streamlit_gui_state as gs

_PROTECTED_KEYS = ["judge_candidates", "judge_judge_choice", *KEY_STATE.values()]


def _all_candidate_options():
    """Every (provider, model_id) pair a candidate or judge slot could be,
    across every wired provider's model list -- MODELS_BY_PROVIDER only has
    entries for wired providers already (see streamlit_llm_panel.py), so
    no separate WIRED_PROVIDERS filter is needed here."""
    return [
        (provider, model_id)
        for provider, (models, _) in MODELS_BY_PROVIDER.items()
        for model_id, _ in models
    ]


def _option_label(option):
    provider, model_id = option
    models, _ = MODELS_BY_PROVIDER[provider]
    short = dict(models)[model_id].split(" -- ")[0]
    return f"{provider}: {short}"


def _default_candidates(all_options):
    """The two cheapest-tier models from the first two providers
    MODELS_BY_PROVIDER declares (Claude Haiku + GPT-5.6 Luna, as of this
    writing) -- a reasonable out-of-the-box comparison, not a claim that
    these two are the most interesting pair."""
    defaults = [
        (provider, models[default_index][0])
        for provider, (models, default_index) in list(MODELS_BY_PROVIDER.items())[:2]
    ]
    return [c for c in defaults if c in all_options]


def _default_judge_choice(candidates, all_options):
    """The first provider's default model that ISN'T already a candidate --
    with the two defaults above, this lands on the third provider (Gemini),
    a genuinely distinct model family acting as judge over the other two."""
    for provider, (models, default_index) in MODELS_BY_PROVIDER.items():
        pair = (provider, models[default_index][0])
        if pair in all_options and pair not in candidates:
            return pair
    return None


def _render_role_selection():
    all_options = _all_candidate_options()
    candidates = st.multiselect(
        "Candidates being judged",
        options=all_options,
        default=_default_candidates(all_options),
        format_func=_option_label,
        key="judge_candidates",
        help="Pick 2 or more (provider, model) pairs -- the same provider can "
             "appear twice at different tiers.",
    )
    judge_options = [o for o in all_options if o not in candidates]

    # A stale selection (the current judge pick just became a candidate
    # too) would otherwise make st.selectbox raise on the next line --
    # drop it so the widget falls back to its own default instead.
    if st.session_state.get("judge_judge_choice") not in judge_options:
        st.session_state.pop("judge_judge_choice", None)

    if not judge_options:
        st.caption("No model left to act as judge -- remove a candidate above.")
        return candidates, None

    default_judge = _default_judge_choice(candidates, all_options)
    judge_choice = st.selectbox(
        "Judge",
        options=judge_options,
        format_func=_option_label,
        index=judge_options.index(default_judge) if default_judge in judge_options else 0,
        key="judge_judge_choice",
        help="Must differ from every candidate above -- keeps the judge from "
             "favoring its own family's answer.",
    )
    return candidates, judge_choice


def _distinct_providers(candidates, judge_choice):
    in_use = {c[0] for c in candidates}
    if judge_choice:
        in_use.add(judge_choice[0])
    return [p for p in MODELS_BY_PROVIDER if p in in_use]


def _render_provider_keys(providers):
    """One key-entry widget per distinct provider actually in use this
    round, across every candidate slot and the judge slot combined --
    reuses streamlit_llm_panel's own KEY_STATE/ENV_VAR/_configured_key, so
    a key entered here or in Mode="LLM Supervisor" carries over to the
    other (see module docstring)."""
    if not providers:
        return {}
    st.subheader("Model access")
    resolved = {}
    for provider in providers:
        configured_key = _configured_key(provider)
        if configured_key:
            st.success(f"{provider} key configured by the operator — nothing to enter.")
            resolved[provider] = configured_key
        else:
            key = st.text_input(
                f"{provider} API key", type="password", key=KEY_STATE[provider],
                help="Kept only in this browser session — never written to disk or logged.",
            )
            if key:
                resolved[provider] = key
    return resolved


def _render_cost_warning(candidates, judge_choice):
    total = len(candidates) + (1 if judge_choice else 0)
    st.warning(
        f"Judge Mode sends real, billed requests to every candidate PLUS the judge on "
        f"every message you send -- {len(candidates)} candidate call"
        f"{'s' if len(candidates) != 1 else ''} + 1 judge call = **{total} billed calls per turn**. "
        "Cost scales with however many candidates you pick above; there's no cap.",
        icon="💸",
    )


def _build_judge_client(provider, api_key, model):
    """Bare client for the judge role -- no Session wrapper, since the
    judge never calls tools of its own (see supervisor_session_judge_pid.py's
    module docstring). Same 3-way provider dispatch cli_supervisor_pid.py's
    own _build_client already duplicates for the CLI -- kept local rather
    than shared, matching that existing precedent."""
    if provider == "ChatGPT (OpenAI)":
        return OpenAIClient(api_key=api_key, model=model)
    if provider == "Gemini (Google)":
        return GeminiClient(api_key=api_key, model=model)
    return AnthropicClient(api_key=api_key, model=model)


def _build_judge_session(candidates, judge_choice, track, resolved_keys):
    candidate_sessions = [
        (_option_label(c), _new_session(c[0], resolved_keys[c[0]], track, c[1]))
        for c in candidates
    ]
    judge_provider, judge_model = judge_choice
    judge_client = _build_judge_client(judge_provider, resolved_keys[judge_provider], judge_model)
    return JudgeSession(candidate_sessions, judge_client)


def _fingerprint(candidates, judge_choice, track, resolved_keys):
    return (tuple(sorted(candidates)), judge_choice, track, tuple(sorted(resolved_keys.items())))


def _render_candidate_rounds(rounds):
    """The transparency panel: what each candidate actually did this
    round, in light-grey caption text (st.caption's native muted style --
    no custom CSS), one block per candidate. This is the human-facing
    answer to "what is each model saying," distinct from serialize_
    candidate_trace()'s full-history, machine-oriented text the judge
    itself reads (see supervisor_session_judge_pid.py). Always tucked
    under an expander, never the primary reply -- the judge's own message
    is the one voice the main thread is meant to carry (see module
    docstring)."""
    if not rounds:
        return
    with st.expander("What each candidate actually said this round"):
        for item in rounds:
            label = item["label"]
            if item.get("error"):
                st.caption(f"**{label}** — ⚠️ failed to respond: {item['error']}")
                continue
            header = f"**{label}**"
            if item.get("finalized"):
                header += f" → recommends **{item['finalized']}**"
            st.caption(header)
            for name, args in item.get("calls", []):
                arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
                st.caption(f"· called `{name}({arg_text})`")
            if item.get("reply"):
                st.caption(item["reply"])


def render_controls(track):
    """The left-hand controls half for Mode="LLM Judge" -- called by
    streamlit_unified_panel.py with whichever Track it currently has
    selected. Mirrors streamlit_llm_panel.render_controls()'s overall
    shape (preserve/snapshot bracketing _render_role_selection()/
    _render_provider_keys() specifically, for the same reason: those early
    returns below must not skip the snapshot).

    Calls streamlit_llm_panel._init_panel_state() up front -- not just for
    symmetry, but because it's what actually loads .env (guarded, once per
    session) via load_dotenv(). Without this, a user who opens Judge Mode
    without ever visiting Mode="LLM Supervisor" first would only see keys
    already in the real process environment, never a repo-root .env file's
    -- _init_panel_state()'s other seeding (llm_session_obj/fingerprint
    defaults) is namespaced to that other mode's own keys and harmless
    here."""
    _init_panel_state()
    gs.preserve_widget_state(_PROTECTED_KEYS)
    candidates, judge_choice = _render_role_selection()
    resolved_keys = {}
    if len(candidates) >= 2 and judge_choice:
        providers = _distinct_providers(candidates, judge_choice)
        resolved_keys = _render_provider_keys(providers)
        _render_cost_warning(candidates, judge_choice)
    gs.snapshot_widget_state(_PROTECTED_KEYS)

    if len(candidates) < 2:
        st.info("Pick at least 2 candidates above to start Judge Mode.")
        return
    if not judge_choice:
        st.info("Pick a judge above to start Judge Mode.")
        return
    missing = [p for p in _distinct_providers(candidates, judge_choice) if p not in resolved_keys]
    if missing:
        st.info(f"Enter an API key above for {', '.join(missing)} to start chatting.")
        return

    fingerprint = _fingerprint(candidates, judge_choice, track, resolved_keys)
    if st.session_state.get("judge_session_fingerprint") != fingerprint:
        st.session_state["judge_session_obj"] = _build_judge_session(candidates, judge_choice, track, resolved_keys)
        st.session_state["judge_session_fingerprint"] = fingerprint
        gs.clear_judge_chat()

    if st.button("Reset conversation", key="judge_reset"):
        st.session_state["judge_session_obj"] = _build_judge_session(candidates, judge_choice, track, resolved_keys)
        gs.clear_judge_chat()

    history_box = st.container()
    user_text = st.chat_input("Tell me about your plant and what matters most to you.")

    if user_text:
        gs.append_judge_chat_message("user", user_text)

    with history_box:
        for message in st.session_state[gs.JUDGE_CHAT_KEY]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                _render_candidate_rounds(message.get("rounds"))

        if user_text:
            judge_session = st.session_state["judge_session_obj"]
            with st.chat_message("assistant"):
                with st.spinner("Consulting candidates and judging..."):
                    try:
                        reply = judge_session.handle_user_message(user_text)
                    except (anthropic.AuthenticationError, openai.AuthenticationError) as exc:
                        _log_exception(exc)
                        reply = "That API key was rejected — double-check it and try again."
                    except (anthropic.RateLimitError, openai.RateLimitError) as exc:
                        _log_exception(exc)
                        reply = "Rate limited by the provider — wait a moment and try again."
                    except genai_errors.ClientError as exc:
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
                            for _, session in judge_session.candidates:
                                llm_panel._drain_plot_calls(session)
                        except Exception as exc:  # noqa: BLE001
                            _log_exception(exc)
                st.markdown(reply)
                _render_candidate_rounds(getattr(judge_session, "last_round", []))
            gs.append_judge_chat_message("assistant", reply, rounds=getattr(judge_session, "last_round", []))
