"""Unit tests for report_html.py's shared HTML-report primitives.

Run with:
    python test_report_html.py
or:
    python -m unittest test_report_html -v
"""

from __future__ import annotations

import base64
import unittest

from matplotlib.figure import Figure

import report_html


class TestEscape(unittest.TestCase):
    def test_escapes_html_special_characters(self):
        self.assertEqual(report_html.e("<script>&\"'"), "&lt;script&gt;&amp;&quot;&#x27;")

    def test_coerces_non_strings(self):
        self.assertEqual(report_html.e(3.14159), "3.14159")


class TestFigToDataUri(unittest.TestCase):
    def test_produces_a_valid_base64_png_data_uri(self):
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2], [0, 1, 4])
        uri = report_html.fig_to_data_uri(fig)
        self.assertTrue(uri.startswith("data:image/png;base64,"))
        raw = base64.b64decode(uri.split(",", 1)[1])
        self.assertEqual(raw[:8], b"\x89PNG\r\n\x1a\n", "must be a real PNG file signature")


class TestRenderTable(unittest.TestCase):
    def test_builds_headers_and_rows(self):
        out = report_html.render_table(["Method", "OS%"], [["Tyreus-Luyben", 2.662], ["CHR set 0%", 3.986]])
        self.assertIn("<th>Method</th>", out)
        self.assertIn("<th>OS%</th>", out)
        self.assertIn("<td>Tyreus-Luyben</td>", out)
        self.assertIn("<td>2.662</td>", out)

    def test_escapes_cell_content(self):
        out = report_html.render_table(["Name"], [["<b>bold</b>"]])
        self.assertIn("&lt;b&gt;bold&lt;/b&gt;", out)
        self.assertNotIn("<b>bold</b>", out)

    def test_empty_rows_still_renders_a_valid_table(self):
        out = report_html.render_table(["A", "B"], [])
        self.assertIn("<thead>", out)
        self.assertIn("<tbody>", out)


class TestBuildReport(unittest.TestCase):
    def test_wraps_sections_with_title_subtitle_and_css(self):
        out = report_html.build_report("My Report", "SISO / PID, Track scope", ["<h2>Section</h2><p>hello</p>"])
        self.assertIn("<title>My Report</title>", out)
        self.assertIn("<h1>My Report</h1>", out)
        self.assertIn("SISO / PID, Track scope", out)
        self.assertIn("<h2>Section</h2><p>hello</p>", out)
        self.assertIn("<style>", out)
        self.assertIn("--accent", out, "must actually embed _memo_css.txt's content, not just reference it")

    def test_escapes_title_and_subtitle(self):
        out = report_html.build_report("<script>", "<img>", [])
        self.assertNotIn("<script>", out)
        self.assertNotIn("<img>", out)


if __name__ == "__main__":
    unittest.main()
