"""Streamlit UI for the PID/LQG tuner.

Build plan: docs/gui_plan.md. Steps 1-4 are done: skeleton, session-
state schema (streamlit_gui_state.py), the SISO PID panel
(streamlit_siso_panel.py, ported from pid_app.py), and the MIMO LQR/LQG
panel (streamlit_mimo_panel.py, ported from cli_lqg.py's design/compare
flow). LLM chat tab is still a placeholder. Steps 5-7 (LLM chat,
packaging, container check) land as separate turns.

Run: streamlit run streamlit_app.py

Session state lives in streamlit_gui_state.py — panels should use its
add_controller/remove_controller/set_enabled/append_chat_message
functions rather than poking st.session_state directly.
"""

from __future__ import annotations

import streamlit as st

from streamlit_gui_state import init_state
import streamlit_siso_panel as siso_panel
import streamlit_mimo_panel as mimo_panel

st.set_page_config(page_title="PID/LQG Tuner", layout="wide")
init_state()

st.title("PID / LQG Tuner")

siso_tab, mimo_tab, chat_tab = st.tabs(["SISO PID", "MIMO LQR/LQG", "LLM Chat"])

with siso_tab:
    siso_panel.render()

with mimo_tab:
    mimo_panel.render()

with chat_tab:
    st.header("LLM Chat")
    st.caption("Placeholder — LLM conversational supervisor panel lands in Build plan Step 5.")
