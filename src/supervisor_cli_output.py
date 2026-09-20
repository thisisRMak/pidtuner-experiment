"""Shared "save conversation output to disk" plumbing for the four
cli_supervisor_*.py REPL scripts: a per-session output folder, HTML
report building/rewriting, and --log-file transcript writing. The
matplotlib-figure half of this same feature lives in the sibling module
supervisor_cli_plots.py instead (kept separate so a caller that only
needs a --log-file path doesn't pay for a matplotlib+streamlit_mimo_panel
import).

Imports report_html plus the two existing, already track-agnostic HTML
builders this reuses as-is: streamlit_llm_panel._transcript_html (single-
provider) and streamlit_judge_panel._rounds_history_html (judge mode).
Deliberately does NOT import streamlit_llm_panel._build_report_html /
_ENTRIES_SECTIONS_BY_TRACK -- those read gs.get_by_kind() from a live
st.session_state, which a plain CLI process never has. The report is
transcript-text-only (no embedded plot images) in this pass -- a
deliberate scope cut, not an oversight (report_html.fig_to_data_uri
exists and could embed figures later).

Everything saves into one per-session subfolder of cwd,
"controldesign-{YYYYMMDD-HHMMSS}-{tag}/" -- timestamp before the
track/mode tag, so a plain directory listing sorts chronologically
rather than grouped by track/mode. Each write prints one confirmation
line, mirroring the one existing precedent for this in the codebase:
cli_mimo_example.py:109's `print(f"saved {path}")`.

--log-file exists because `| tee` cannot capture a real interactive
session: when stdin is a genuine TTY (not piped/heredoc), the terminal's
own line-editing echo writes what you type directly to the tty, never
through the process's own stdout pipe -- tee only ever sees what this
script itself prints. So the REPL loop must explicitly write both sides
(the text returned by input(), and everything printed) to the log file
itself -- see log_print/log_user_input below.
"""

from __future__ import annotations

import datetime
import os
import re
import sys
from dataclasses import dataclass

import report_html
from streamlit_llm_panel import _transcript_html as transcript_html
from streamlit_judge_panel import _rounds_history_html as rounds_history_html


def slugify(text: str, maxlen: int = 40) -> str:
    """Filesystem-safe stand-in for an arbitrary plant expression or
    candidate label (e.g. "1/((90s+1)(10s+1))", "anthropic:claude-haiku-4-5")
    -- every run of characters outside [A-Za-z0-9._-] becomes one "_",
    trimmed of leading/trailing "_" and truncated to maxlen (trimmed again
    after truncation, in case that cut lands on a "_"). Falls back to "x"
    if nothing printable survives, so a filename never ends up empty."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    slug = slug[:maxlen].strip("_")
    return slug or "x"


@dataclass
class RunPaths:
    """One process invocation's fixed output-file identity -- computed
    once, at the top of main(), never recomputed on /reset (a /reset
    mid-conversation keeps writing into the SAME folder/report path, just
    with fresh content; plot filenames keep incrementing rather than
    restarting at turn 1, so a pre-reset plot can never collide with a
    post-reset one)."""

    dir_path: str
    report_path: str
    turn: int = 0

    @classmethod
    def new(cls, tag: str) -> "RunPaths":
        """tag: "supervisor-pid" | "supervisor-lqg" | "judge-pid" | "judge-lqg"
        -- distinguishes which of the four scripts' output this is. The
        folder is created immediately (exist_ok=True: astronomically
        unlikely to collide within the same second, but harmless if it
        ever does)."""
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        dir_path = f"controldesign-{stamp}-{tag}"
        os.makedirs(dir_path, exist_ok=True)
        return cls(dir_path=dir_path, report_path=os.path.join(dir_path, "report.html"))

    def next_turn(self) -> int:
        """Call exactly once per successfully-completed turn (after
        handle_user_message() returns without raising) -- never on
        /reset, blank input, or a caught provider error."""
        self.turn += 1
        return self.turn

    def plot_stem(self, call_index: int, plant, candidate_label: str | None = None) -> str:
        """Path stem (no extension) for one plot_calls entry's figure(s)
        this turn, inside self.dir_path -- self.turn must already reflect
        this turn (call next_turn() first). call_index is 1-based and
        scoped to *this turn's* calls list only (callers naturally get
        this for free via enumerate(calls, start=1), since self.turn --
        not call_index -- is what's monotonic across the whole session).
        candidate_label (judge CLIs only) disambiguates which candidate
        produced this entry."""
        parts = [f"turn{self.turn:03d}", f"{call_index:02d}"]
        if candidate_label:
            parts.append(slugify(str(candidate_label)))
        parts.append(slugify(str(plant)) if plant else "plant")
        return os.path.join(self.dir_path, "-".join(parts))


def open_session_log(path: str | None):
    """None (the --log-file default) means logging is off -- every
    log_print/log_user_input call becomes a silent no-op. Opened
    append-only, line-buffered (buffering=1) so `tail -f` on a live
    session log shows each turn immediately; append rather than truncate
    since the path is fully user-chosen, unlike the automatic report/plot
    folder -- re-running with the same --log-file path is presumed
    deliberate, not a collision to guard against."""
    if not path:
        return None
    return open(path, "a", buffering=1)


def log_print(log_fh, text: str = "", *, file=None) -> None:
    """print(text) to `file` (unchanged REPL behavior) and also append it
    to the session log, if one is open. Every REPL print() call site
    becomes this instead, one call site at a time.

    `file` defaults to sys.stdout resolved *inside* the call, not bound
    as `file=sys.stdout` in the signature -- a default argument value is
    evaluated once, at def time, so it would otherwise capture whatever
    object sys.stdout was at import time and never see a later
    contextlib.redirect_stdout() reassignment (confirmed the hard way: a
    test using redirect_stdout saw nothing until this was fixed)."""
    print(text, file=file if file is not None else sys.stdout)
    if log_fh is not None:
        print(text, file=log_fh)


def log_user_input(log_fh, prompt: str, text: str) -> None:
    """Appends the user's own typed line to the session log ONLY -- never
    re-printed to stdout, since input()'s own prompt plus the terminal's
    line-editing echo already put `prompt`+`text` on screen as the user
    typed it. This is the one call site `tee` could never have captured
    (see module docstring) and the whole reason this module exists
    instead of `| tee`."""
    if log_fh is not None:
        print(f"{prompt}{text}", file=log_fh)


def write_supervisor_report(run_paths: RunPaths, track: str, session, log_fh=None) -> str:
    """Rewrites run_paths.report_path (overwritten every call, not
    appended -- always reflects current session state) from
    session.messages via transcript_html(session), wrapped in
    report_html.build_report() with the same title format the GUI's own
    "Download report" button uses, for continuity between a GUI-
    downloaded and a CLI-saved report."""
    title = f"PIDTuner LLM Supervisor Report ({track})"
    html = report_html.build_report(title, f"CLI session — {track}", [transcript_html(session)])
    with open(run_paths.report_path, "w") as f:
        f.write(html)
    log_print(log_fh, f"saved report: {run_paths.report_path}")
    return run_paths.report_path


def write_judge_report(run_paths: RunPaths, track: str, judge_session, log_fh=None) -> str:
    """Judge-mode counterpart -- same shape, reusing
    rounds_history_html(judge_session.rounds_history) and the GUI's own
    "PIDTuner LLM Judge Report ({track})" title format."""
    title = f"PIDTuner LLM Judge Report ({track})"
    html = report_html.build_report(title, f"CLI session — {track}",
                                     [rounds_history_html(judge_session.rounds_history)])
    with open(run_paths.report_path, "w") as f:
        f.write(html)
    log_print(log_fh, f"saved report: {run_paths.report_path}")
    return run_paths.report_path
