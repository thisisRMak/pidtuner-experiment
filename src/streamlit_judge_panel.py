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
absorb_calls/_log_exception are all imported from streamlit_llm_panel
and reused as-is. Reusing the exact same KEY_STATE widget keys means an API
key entered in Mode="LLM Supervisor" carries over to Mode="LLM Judge" (and
back) via streamlit_gui_state.py's existing shadow-state mechanism -- it's
one operator-configured or session-typed key per provider, not per mode.

Role selection is a selectbox (judge, chosen first, no default -- the user
must deliberately pick one before anything else appears) + a dependent
multiselect (candidates, rendered only once a judge exists, seeded with a
default pair the first time but an ordinary widget after that), not N
separate per-slot provider/model widgets -- avoids a dynamic
number-of-widgets-per-run problem entirely: candidate count is just
"however many options are selected" in one widget, no per-index keys to
preserve across a Mode/Track switch. The same provider can appear twice as
a candidate at two different tiers (e.g. Haiku vs Sonnet), but the judge's
own (provider, model) pair must differ from every candidate's -- avoids a
model favoring its own family's answer. Judge output shape is
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

N candidates given the identical prompt routinely call the identical
benchmark tool on the identical plant -- since that benchmark is a
deterministic function of its arguments, draining every candidate's copy
independently would plot the same methods N times over. _drain_judge_
plot_calls() dedupes by (kind, plant, delay/preset/literals) across all
candidates before anything reaches the session list, so the Response
plot's legend, the Heatmap's columns, and the Radar's spokes each show
one entry per method, not N -- see that function's own docstring for the
exact key and its one known gap (two candidates phrasing a
mathematically-equivalent plant as different strings still duplicates).

Both Tracks are wired: `_new_session` (imported from streamlit_llm_panel,
reused as-is) already dispatches on `track` to build either
supervisor_session_pid.Session or supervisor_session_lqg.LQGSession
candidates, and `_build_judge_session` below passes the matching judge
system prompt (`supervisor_prompts_judge_pid.JUDGE_SYSTEM_PROMPT` or
supervisor_prompts_judge_lqg.JUDGE_SYSTEM_PROMPT_LQG, see
_JUDGE_SYSTEM_PROMPT_BY_TRACK) via `JudgeSession`'s `judge_system_prompt=`
kwarg -- everything else about JudgeSession's orchestration is already
track-agnostic (see that module's docstring). A CLI equivalent for both
Tracks also now exists (`cli_supervisor_judge_pid.py`/
`cli_supervisor_judge_lqg.py`) -- this GUI panel and those CLIs share the
same JudgeSession/prompt pieces, built for the CLI first and reused here
unmodified.
"""

from __future__ import annotations

import datetime

import anthropic
import openai
import streamlit as st
from google.genai import errors as genai_errors

import report_html
import streamlit_llm_panel as llm_panel
from streamlit_llm_panel import (
    KEY_STATE,
    MODELS_BY_PROVIDER,
    _configured_key,
    _ENTRIES_SECTIONS_BY_TRACK,
    _init_panel_state,
    _log_exception,
    _new_session,
)
from supervisor_llm_anthropic import AnthropicClient
from supervisor_llm_openai import OpenAIClient
from supervisor_llm_gemini import GeminiClient
from supervisor_session_judge_pid import JudgeSession
from supervisor_prompts_judge_pid import JUDGE_SYSTEM_PROMPT
from supervisor_prompts_judge_lqg import JUDGE_SYSTEM_PROMPT_LQG

import streamlit_gui_state as gs

_PROTECTED_KEYS = ["judge_candidates", "judge_judge_choice", *KEY_STATE.values()]

# Mirrors _ENTRIES_SECTIONS_BY_TRACK's own track-keyed-dict pattern in
# streamlit_llm_panel.py -- JudgeSession's judge_system_prompt= defaults to
# the SISO/PID prompt (see supervisor_session_judge_pid.py), so this is the
# one place that has to know to pass the LQG one instead for that track.
_JUDGE_SYSTEM_PROMPT_BY_TRACK = {
    "SISO / PID": JUDGE_SYSTEM_PROMPT,
    "MIMO / LQG": JUDGE_SYSTEM_PROMPT_LQG,
}


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


def _default_candidates(candidate_options, exclude):
    """The two cheapest-tier models from the first two providers
    MODELS_BY_PROVIDER declares that aren't `exclude` (the judge just
    chosen) -- seeded once, the first time a judge is picked this session
    (see _render_role_selection), not re-applied on every rerun, so a user
    who deliberately clears their candidate selection afterward stays
    cleared. Skips `exclude`'s whole provider, not just its exact model,
    so the seeded pair is two genuinely different vendor families rather
    than one vendor plus a same-family second tier."""
    defaults = []
    for provider, (models, default_index) in MODELS_BY_PROVIDER.items():
        pair = (provider, models[default_index][0])
        if pair == exclude:
            continue
        defaults.append(pair)
        if len(defaults) == 2:
            break
    return [c for c in defaults if c in candidate_options]


def _render_role_selection():
    """Judge first, no default -- a deliberate choice (asked, not assumed):
    nothing runs, not even a default candidate pair, until the user picks
    a judge. Candidates are a dependent multiselect rendered only once a
    judge exists, seeded with a default pair the first time (see
    _default_candidates) but otherwise an ordinary widget from then on."""
    all_options = _all_candidate_options()
    judge_choice = st.selectbox(
        "Judge",
        options=all_options,
        index=None,
        placeholder="Choose a judge...",
        format_func=_option_label,
        key="judge_judge_choice",
        help="The model that arbitrates between candidates -- picked first, "
             "since everything else depends on it.",
    )
    if judge_choice is None:
        st.caption("Choose a judge above to pick candidates.")
        return [], None

    candidate_options = [o for o in all_options if o != judge_choice]

    # A stale selection (a candidate that's now become the judge) would
    # otherwise make st.multiselect raise on the next line -- drop just
    # that one entry so the rest of the selection survives.
    current = st.session_state.get("judge_candidates")
    if current is not None:
        cleaned = [c for c in current if c in candidate_options]
        if cleaned != current:
            st.session_state["judge_candidates"] = cleaned

    if "judge_candidates" not in st.session_state:
        st.session_state["judge_candidates"] = _default_candidates(candidate_options, exclude=judge_choice)

    candidates = st.multiselect(
        "Candidates being judged",
        options=candidate_options,
        format_func=_option_label,
        key="judge_candidates",
        help="Pick 2 or more (provider, model) pairs -- the same provider can "
             "appear twice at different tiers. Must differ from the judge above "
             "-- keeps it from favoring its own family's answer.",
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
    return JudgeSession(candidate_sessions, judge_client,
                         judge_system_prompt=_JUDGE_SYSTEM_PROMPT_BY_TRACK[track])


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


def _rounds_history_html(rounds_history) -> str:
    """JudgeSession.rounds_history -> HTML paragraphs, one block per round:
    the user's message, each candidate's own outcome that round (calls,
    finalized pick, reply, or a failure note), and the judge's verdict.
    Same content _render_candidate_rounds() shows in the live transparency
    expander, just every round instead of only the latest, and rendered as
    static HTML instead of Streamlit elements."""
    parts = []
    for i, round_ in enumerate(rounds_history, start=1):
        parts.append(f"<h3>Round {i}</h3>")
        parts.append(f"<p><strong>User:</strong> {report_html.e(round_['user_text'])}</p>")
        for item in round_["candidates"]:
            label = report_html.e(item["label"])
            if item.get("error"):
                parts.append(f'<p class="section-note">{label} — failed to respond: {report_html.e(item["error"])}</p>')
                continue
            header = f"<strong>{label}</strong>"
            if item.get("finalized"):
                header += f" → recommends <strong>{report_html.e(item['finalized'])}</strong>"
            parts.append(f"<p>{header}</p>")
            for name, args in item.get("calls", []):
                arg_text = ", ".join(f"{k}={v!r}" for k, v in args.items())
                parts.append(f'<p class="section-note">Called <code>{report_html.e(name)}({report_html.e(arg_text)})</code></p>')
            if item.get("reply"):
                parts.append(f"<p>{report_html.e(item['reply'])}</p>")
        parts.append(f'<p class="callout"><strong>Judge:</strong> {report_html.e(round_["judge_reply"])}</p>')
    return "\n".join(parts)


def _build_report_html(track, judge_session):
    """The full-history report for Mode="LLM Judge": whichever
    session-list entries are currently plotted for this Track -- the exact
    same section Manual mode's/Supervisor mode's own reports show, not a
    re-implementation -- followed by every round (_rounds_history_html(),
    not just the latest). Tables/plots first, conversation last -- see
    streamlit_llm_panel._build_report_html()'s identical reasoning, the
    round history is usually the longest section by far."""
    entries_subtitle, entries_sections = _ENTRIES_SECTIONS_BY_TRACK[track]()
    sections = entries_sections + [f"<h2>Conversation</h2>{_rounds_history_html(judge_session.rounds_history)}"]
    return report_html.build_report(f"PIDTuner LLM Judge Report ({track})", entries_subtitle, sections)


def _render_download_report_button(track, judge_session):
    if not judge_session.rounds_history:
        return
    st.download_button(
        "Download report", data=_build_report_html(track, judge_session),
        file_name=f"pidtuner-judge-report-{datetime.date.today().isoformat()}.html",
        mime="text/html", key="judge_download_report",
    )


def _drain_judge_plot_calls(judge_session) -> None:
    """Drains every candidate's plot_calls, but only the first time a
    given (kind, plant, delay/preset/literals) combination is seen this
    call -- N candidates given the identical prompt routinely call the
    identical benchmark tool on the identical plant, and since that
    benchmark is a deterministic function of its arguments, every
    candidate's copy of the result is numerically identical. Draining all
    of them (the naive "for each candidate, drain it" loop this replaces)
    would add the same methods to the session list N times over --
    tripling the Response plot's legend, the Heatmap's columns, and the
    Radar's spokes with entries that are exact duplicates, not distinct
    findings.

    "First" means first in judge_session.candidates' own list order (the
    order the user picked them in the multiselect) -- not thread-
    completion order, which is non-deterministic (_fan_out runs candidates
    concurrently) and would make which candidate's copy "wins" vary
    run to run. Which one wins doesn't change the plotted result (the
    rows are identical either way), only which entry happens to hold it.

    A known gap, not attempted here: two candidates phrasing a
    mathematically-equivalent plant as different strings (e.g.
    "1/(90s+1)" vs "1 / (90*s + 1)") won't be recognized as the same key
    and will still duplicate -- this fixes the reported case (identical
    arguments, the common case since every candidate is given the exact
    same prompt), not full semantic equivalence."""
    seen = set()
    for _, session in judge_session.candidates:
        calls, session.plot_calls = session.plot_calls, []
        deduped = []
        for call in calls:
            key = (call["kind"], call["plant"], call.get("delay", 0.0),
                   call.get("plant_preset", ""), str(call.get("custom_plant_literals")))
            if key in seen:
                continue
            seen.add(key)
            deduped.append(call)
        llm_panel.absorb_calls(deduped)


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

    if not judge_choice:
        st.info("Choose a judge above to start Judge Mode.")
        return
    if len(candidates) < 2:
        st.info("Pick at least 2 candidates above to start Judge Mode.")
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
        # Also clears this Track's plotted LLM entries (and, for MIMO,
        # the 4-curve/per-channel-step caches) -- mirrors streamlit_llm_
        # panel.py's identical change to its own "Reset conversation";
        # see llm_panel.reset_clears_track_state()'s own docstring for
        # the full reasoning, reused here rather than duplicated.
        st.session_state["judge_session_obj"] = _build_judge_session(candidates, judge_choice, track, resolved_keys)
        gs.clear_judge_chat()
        llm_panel.reset_clears_track_state(track)

    # "Clear LLM plots" moved to the session list itself -- see streamlit_
    # llm_panel.py's identical comment on its own former copy of this
    # button for why.

    _render_download_report_button(track, st.session_state["judge_session_obj"])

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
                            _drain_judge_plot_calls(judge_session)
                        except Exception as exc:  # noqa: BLE001
                            _log_exception(exc)
                st.markdown(reply)
                _render_candidate_rounds(getattr(judge_session, "last_round", []))
            gs.append_judge_chat_message("assistant", reply, rounds=getattr(judge_session, "last_round", []))
