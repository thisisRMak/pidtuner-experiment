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
    """

    kind: Literal["siso", "mimo"]
    label: str
    params: Any
    result: Any = None
    sim: Any = None
    enabled: bool = True
    id: str = field(default_factory=lambda: uuid.uuid4().hex)


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


def append_chat_message(role: Literal["user", "assistant"], content: str) -> None:
    st.session_state[CHAT_KEY].append({"role": role, "content": content})


def clear_chat() -> None:
    st.session_state[CHAT_KEY] = []
