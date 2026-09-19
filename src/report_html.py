"""Shared HTML report building for every mode's "Download report" button
(Manual/SISO, Manual/MIMO, LLM Supervisor, LLM Judge) -- one place for the
document shell/CSS and the two genuinely reusable primitives (a table, a
base64-embedded plot), so each panel only supplies its own mode-specific
content sections, not HTML boilerplate. Reuses _memo_css.txt, the same
stylesheet the gen_*_memo.py textbook-validation memos already use -- a
downloaded report is the same kind of artifact, just built from live
session state instead of a fixed worked example.

Deliberately HTML only -- no Markdown, no PDF. The only proven PDF path in
this repo is a manual, maintainer-run `google-chrome --headless
--print-to-pdf` step (see gen_pid_worked_example_v2_textbook_memo.py's own
docstring) -- there's no Chrome/Chromium in Dockerfile and no PDF library
in requirements.txt, so it isn't something to shell out to from a live
Streamlit request. A user who wants a PDF can print the downloaded HTML
from their own browser.
"""

from __future__ import annotations

import base64
import datetime
import html
import io
import os

_HERE = os.path.dirname(__file__)
with open(os.path.join(_HERE, "_memo_css.txt")) as _f:
    _CSS = _f.read()


def e(s) -> str:
    """HTML-escape -- short name to keep call sites readable, matching the
    gen_*_memo.py convention (each of those modules has its own `_e`)."""
    return html.escape(str(s))


def fig_to_data_uri(fig) -> str:
    """A matplotlib Figure -> a self-contained data: URI, no temp file --
    the same bytes streamlit_siso_panel._download_fig_button's own PNG
    download produces (fig.savefig into an io.BytesIO buffer), just
    base64-encoded for an <img src=...> instead of offered standalone."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def render_table(headers, rows) -> str:
    """`rows`: iterable of iterables of already display-ready values --
    this only builds markup, it doesn't format numbers. Each mode's own
    report builder decides what a cell says (mirroring gen_*_memo.py's own
    fnum()-then-_e() call sites, kept local to each generator there)."""
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    body_rows = "\n".join(
        "<tr>" + "".join(f"<td>{e(c)}</td>" for c in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>\n{body_rows}\n</tbody></table>"


def build_report(title: str, subtitle: str, sections) -> str:
    """`sections`: a list of already-built HTML strings -- a transcript, a
    table, an <img>, a <h2> heading, whatever the calling mode's report
    needs. This function only wraps them in the shared document shell; it
    has no opinion on what a mode's report contains."""
    date = datetime.date.today().isoformat()
    body = "\n".join(sections)
    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{e(title)}</title>
<style>
{_CSS}
</style>
</head>
<body>
<h1>{e(title)}</h1>
<dl class="memo-header">
  <dt>Date</dt><dd>{e(date)}</dd>
  <dt>Scope</dt><dd>{e(subtitle)}</dd>
</dl>
{body}
</body>
</html>
"""
