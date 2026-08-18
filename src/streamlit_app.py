"""Streamlit UI for the PID/LQG tuner.

Build plan: docs/gui_plan.md. Steps 1-3 are done: skeleton, session-
state schema (gui_state.py), and the SISO PID panel (siso_panel.py,
ported from pid_app.py). MIMO and LLM chat tabs are still placeholders.
Steps 4-7 (MIMO panel, LLM chat, packaging, container check) land as
separate turns.

Run: streamlit run streamlit_app.py

Session state lives in gui_state.py — panels should use its
add_controller/remove_controller/set_enabled/append_chat_message
functions rather than poking st.session_state directly.
"""

from __future__ import annotations

import streamlit as st

from gui_state import init_state
import siso_panel

st.set_page_config(page_title="PID/LQG Tuner", layout="wide")
init_state()

st.title("PID / LQG Tuner")

siso_tab, mimo_tab, chat_tab = st.tabs(["SISO PID", "MIMO LQR/LQG", "LLM Chat"])

with siso_tab:
    siso_panel.render()

with mimo_tab:
    st.header("MIMO LQR/LQG")
    st.caption("Placeholder — MIMO LQR/LQG panel lands in Build plan Step 4.")

with chat_tab:
    st.header("LLM Chat")
    st.caption("Placeholder — LLM conversational supervisor panel lands in Build plan Step 5.")
