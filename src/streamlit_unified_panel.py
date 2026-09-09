"""Unified Manual/LLM-Supervisor panel — one page, no tabs.

Supersedes streamlit_app.py's previous three-tab layout (SISO PID / MIMO
LQR/LQG / LLM Chat) with a single left-controls/right-plots split: a
Track selector (SISO/PID vs MIMO/LQG) and a Mode selector (Manual vs
LLM Supervisor -- LLM-as-Judge deliberately not offered yet) pick which
panel's controls render on the left. The right side always shows that
Track's own plots (Response/Heatmap/Radar for SISO, Response/4-curve/
per-channel for MIMO) regardless of which Mode produced the entries in
it, since Manual and LLM Supervisor now write into the same
gs.ControllerEntry list (tagged by entry.source, see
streamlit_gui_state.py) rather than two separate ones -- "the same
output being rendered" either way, not a second plotting path to keep
in sync.

Only one call site ever renders a given Track/Mode's controls per
script run -- this REPLACES st.tabs() rather than sitting alongside it,
so unlike an earlier explored approach (kept on the llm-chat-live-plots
branch, not merged) there's no widget-key collision to design around:
streamlit_siso_panel.py/streamlit_mimo_panel.py/streamlit_llm_panel.py's
own render_controls() are reused as-is, no re-keying needed.
"""

from __future__ import annotations

import streamlit as st

import streamlit_siso_panel as siso_panel
import streamlit_mimo_panel as mimo_panel
import streamlit_llm_panel as llm_panel

TRACKS = ["SISO / PID", "MIMO / LQG"]
MODES = ["Manual", "LLM Supervisor"]


def render():
    st.title("PID / LQG Tuner")

    controls_col, plots_col = st.columns([35, 65])

    with controls_col:
        # required=True + default=<first option>: segmented_control allows
        # deselecting the active pill (down to None) unless required,
        # unlike st.radio, which always has exactly one option selected --
        # required=True keeps that same "always exactly one" guarantee the
        # dispatch below assumes.
        track = st.segmented_control("Track", TRACKS, key="unified_track",
                                     default=TRACKS[0], required=True)
        mode = st.segmented_control("Mode", MODES, key="unified_mode",
                                    default=MODES[0], required=True)
        st.divider()

        if mode == "Manual":
            if track == "SISO / PID":
                siso_panel.render_controls()
            else:
                mimo_panel.render_controls()
        else:
            llm_panel.render_controls(track)

    with plots_col:
        if track == "SISO / PID":
            siso_panel.render_plots()
        else:
            mimo_panel.render_plots()
