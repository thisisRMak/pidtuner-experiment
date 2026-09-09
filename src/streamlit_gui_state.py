"""Session-state schema for streamlit_app.py.

Streamlit reruns the whole script on every interaction, so anything
that must survive a rerun (the tuned-controllers list, chat history)
lives in `st.session_state` under the keys defined here. This module
is the single place that schema is defined and mutated — panels
(Steps 3-5) should call these functions rather than poking
`st.session_state` directly, so the shape stays consistent across the
SISO, MIMO, and chat panels.

Mirrors pid_app.py's TunedEntry/self.tuned pattern (see pid_app.py:72
and :105), generalized to cover both SISO and MIMO entries via `kind`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
import uuid

import streamlit as st

CONTROLLERS_KEY = "controllers"
CHAT_KEY = "chat_history"


@dataclass
class ControllerEntry:
    """One tuned controller, kept in the session for overlay plotting.

    `params`/`result`/`sim` are intentionally left opaque (Any) — SISO
    stores PIDGains/TuningResult/ClosedLoopResult objects, MIMO stores
    its own LQR/LQG equivalents. The session-list machinery (add,
    remove, enable/disable) doesn't need to know which.

    `source`/`plant` are provenance tags — who triggered this run ("you"
    via the manual controls, or "llm" via the chat's supervisor tool
    calls, see streamlit_llm_panel.py's absorb_llm_rows() calls) and
    which plant it ran against. Both entries a manual run and an
    LLM-triggered one land in this same list (streamlit_unified_panel.py
    doesn't separate them), so the session list can show the tag rather
    than silently mixing runs from different origins or plants into one
    overlay. Both optional: existing call sites that don't pass them get
    the "you"/"" defaults, correct for every entry that predates this
    field.

    `plant`, above, is a .pretty()-formatted display string -- readable,
    but not something that can be fed back into TransferFunction.parse()
    or the SISO panel's siso_tf_expr widget. `plant_tf`/`plant_L` are the
    raw, reloadable identity instead: the exact expression string and
    dead-time L that produced this entry's plant, for a caller that wants
    to reconstruct or re-seed a plant from an entry rather than just
    display it (see streamlit_siso_panel.py's "Load this plant"
    affordance). SISO-only for now, empty/0.0 on every MIMO entry and on
    a SISO one that predates this field.
    """

    kind: Literal["siso", "mimo"]
    label: str
    params: Any
    result: Any = None
    sim: Any = None
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    mrow: Any = None    # cached pid_compare.metric_row() (SISO), for heatmap/radar
    checks: Any = None  # cached lqg_checks.checks_for_result() (MIMO)
    source: Literal["you", "llm"] = "you"
    plant: str = ""
    plant_tf: str = ""
    plant_L: float = 0.0


def init_state() -> None:
    """Call once at the top of the app. Idempotent across reruns."""
    if CONTROLLERS_KEY not in st.session_state:
        st.session_state[CONTROLLERS_KEY] = []
    if CHAT_KEY not in st.session_state:
        st.session_state[CHAT_KEY] = []


def add_controller(entry: ControllerEntry) -> None:
    st.session_state[CONTROLLERS_KEY].append(entry)


def remove_controller(entry_id: str) -> None:
    st.session_state[CONTROLLERS_KEY] = [
        e for e in st.session_state[CONTROLLERS_KEY] if e.id != entry_id
    ]


def remove_unchecked() -> None:
    st.session_state[CONTROLLERS_KEY] = [
        e for e in st.session_state[CONTROLLERS_KEY] if e.enabled
    ]


def clear_controllers() -> None:
    st.session_state[CONTROLLERS_KEY] = []


def set_enabled(entry_id: str, value: bool) -> None:
    for e in st.session_state[CONTROLLERS_KEY]:
        if e.id == entry_id:
            e.enabled = value
            return


def set_all_enabled(value: bool) -> None:
    for e in st.session_state[CONTROLLERS_KEY]:
        e.enabled = value


# ── kind-scoped variants ────────────────────────────────────────────────
# A panel (SISO, MIMO) only ever wants to touch its own entries — these
# spare each panel from filtering st.session_state[CONTROLLERS_KEY] by
# hand, which is easy to duplicate once a second panel needs the same
# "select all/deselect all/clear/remove unchecked, but only mine" logic.
def get_by_kind(kind: str) -> list[ControllerEntry]:
    return [e for e in st.session_state[CONTROLLERS_KEY] if e.kind == kind]


def clear_by_kind(kind: str) -> None:
    st.session_state[CONTROLLERS_KEY] = [
        e for e in st.session_state[CONTROLLERS_KEY] if e.kind != kind
    ]


def clear_by_kind_and_source(kind: str, source: str) -> None:
    st.session_state[CONTROLLERS_KEY] = [
        e for e in st.session_state[CONTROLLERS_KEY]
        if e.kind != kind or e.source != source
    ]


def remove_unchecked_by_kind(kind: str) -> None:
    st.session_state[CONTROLLERS_KEY] = [
        e for e in st.session_state[CONTROLLERS_KEY]
        if e.kind != kind or e.enabled
    ]


def set_all_enabled_by_kind(kind: str, value: bool) -> None:
    for e in st.session_state[CONTROLLERS_KEY]:
        if e.kind == kind:
            e.enabled = value


def append_chat_message(role: Literal["user", "assistant"], content: str) -> None:
    st.session_state[CHAT_KEY].append({"role": role, "content": content})


def clear_chat() -> None:
    st.session_state[CHAT_KEY] = []


# ── widget state across a Mode/Track switch ─────────────────────────────
# Streamlit deletes a widget's session_state entry whenever that widget
# isn't instantiated on a script run (confirmed directly via a live
# AppTest repro, not assumed). streamlit_unified_panel.py's
# Track/Mode dispatch means exactly one of siso_panel/mimo_panel/
# llm_panel's render_controls() runs per script run, so every OTHER
# panel's widgets get wiped the moment you're not looking at them --
# without this, switching Track or Mode away and back resets every
# field in the panel you left (a plant you typed in, an API key,
# whatever) back to its hardcoded default. The fix: mirror each key into
# a shadow entry that's never itself a widget's `key=` (so Streamlit
# never reclaims it), then reseed the real key from the shadow right
# before the widget using it is instantiated again -- the same
# "pre-set session_state before creating the widget" trick
# streamlit_siso_panel.py's own "Select all" button already relies on.
_SHADOW_PREFIX = "_shadow__"


def preserve_widget_state(keys) -> None:
    """Call at the very top of a render function, before any of `keys`'
    widgets are instantiated. Pair with snapshot_widget_state(keys) after
    they've all rendered -- see the module note above."""
    for key in keys:
        shadow_key = _SHADOW_PREFIX + key
        if key not in st.session_state and shadow_key in st.session_state:
            st.session_state[key] = st.session_state[shadow_key]


def snapshot_widget_state(keys) -> None:
    """Call once `keys`' widgets have all rendered and hold this turn's
    values -- see preserve_widget_state(). A key that didn't render this
    turn (e.g. a method-specific field for a method that isn't currently
    selected) is simply skipped, not treated as cleared."""
    for key in keys:
        if key in st.session_state:
            st.session_state[_SHADOW_PREFIX + key] = st.session_state[key]


def peek(key: str, default=None):
    """`key`'s live widget value if it rendered this run, else whatever
    snapshot_widget_state() last saved for it -- so a caller can read
    another panel's widget state without that panel's own widgets having
    rendered this run at all (unlike preserve_widget_state(), this never
    writes `key` back into session_state). Used by streamlit_llm_panel.
    py's Manual-mode plant hint to read the SISO/MIMO panels' plant
    widgets while Mode=LLM Supervisor is showing instead. Returns
    `default` if neither is set -- e.g. that panel hasn't rendered even
    once yet this session."""
    if key in st.session_state:
        return st.session_state[key]
    return st.session_state.get(_SHADOW_PREFIX + key, default)
