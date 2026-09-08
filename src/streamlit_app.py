"""Streamlit UI for the PID/LQG tuner.

Build plan: docs/gui_plan.md. Steps 1-5 landed the original three-tab
layout (skeleton, session-state schema in streamlit_gui_state.py, the
SISO PID panel, the MIMO LQR/LQG panel, the LLM chat panel). That's now
superseded by streamlit_unified_panel.py: one page, a Track selector
(SISO/PID vs MIMO/LQG) and a Mode selector (Manual vs LLM Supervisor)
instead of tabs -- see its own module docstring for why. Steps 6-7
(packaging, container check) land as separate turns.

Run: streamlit run streamlit_app.py

Session state lives in streamlit_gui_state.py — panels should use its
add_controller/remove_controller/set_enabled/append_chat_message
functions rather than poking st.session_state directly.
"""

from __future__ import annotations

import streamlit as st

from streamlit_gui_state import init_state
import streamlit_unified_panel as unified_panel

st.set_page_config(page_title="PID/LQG Tuner", layout="wide")
init_state()

unified_panel.render()
