"""Regression tests for streamlit_siso_comparison_views.py's heatmap
exports (HTML/PNG/CSV) -- specifically that the P0/P1/P2 tier-band
grouping has been removed from all three while the underlying metric
order (Rise, ts, OS%, IAE, IAE_load, ISU, Ms, Mt, u_tv -- pid_compare.py's
METRIC_TIERS flattened) is unchanged.

Previously this module was only exercised indirectly through
test_streamlit_siso_panel.py's AppTest suite, which never inspects the
heatmap's HTML/PNG/CSV structure closely enough to catch a regression
here. These call the pure row->output functions directly.

Run with:
    python3 test_streamlit_siso_comparison_views.py
or:
    python3 -m unittest test_streamlit_siso_comparison_views -v
"""

import csv
import io
import re
import unittest
from unittest.mock import patch

import matplotlib.axes

from pid_compare import TABLE_METRICS
from streamlit_siso_comparison_views import (
    _heatmap_csv_bytes,
    _heatmap_norms,
    _heatmap_png_bytes,
    build_heatmap_html,
)

# The order the task is anchored on -- must match pid_compare.py's
# METRIC_TIERS flattened tier-by-tier (P0, then P1, then P2).
EXPECTED_METRIC_ORDER = ["Rise", "ts", "OS%", "IAE", "IAE_load", "ISU", "Ms", "Mt", "u_tv"]

EXPECTED_LABEL_ORDER = [
    "Rise (10-90%)", "ts", "OS %", "IAE (track)", "IAE (load)",
    "ISU (effort)", "Ms", "Mt", "TV(u)",
]


def _row(name, stable=True, **overrides):
    base = {
        "name": name, "stable": stable, "error": None if stable else "failed",
        "black_box": False, "has_time_delay": False,
        "Rise": 1.0, "ts": 5.0, "OS%": 10.0, "IAE": 2.0, "IAE_load": 1.5,
        "ISU": 3.0, "Ms": 1.5, "Mt": 1.4, "u_tv": 0.5,
    }
    base.update(overrides)
    return base


ROWS = [
    _row("MethodA"),
    _row("MethodB", Rise=1.2, ts=6.0, **{"OS%": 12.0}, IAE=2.5, IAE_load=1.8,
         ISU=3.5, Ms=1.6, Mt=1.5, u_tv=0.6),
    _row("MethodC", stable=False),
]


class MetricOrderSanityTest(unittest.TestCase):
    """Guards the fixture/assumption itself, not the module under test."""

    def test_table_metrics_matches_expected_order(self):
        self.assertEqual(TABLE_METRICS, EXPECTED_METRIC_ORDER)


class HeatmapHtmlNoGroupingTest(unittest.TestCase):
    def setUp(self):
        self.html = build_heatmap_html(ROWS)

    def test_no_tier_label_cells(self):
        # A tier-band divider row previously rendered "P0"/"P1"/"P2" as a
        # colspan cell's entire text content. Assert that exact cell
        # content is gone (not just a substring check, since nothing else
        # in the table should legitimately contain these strings either).
        cell_texts = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", self.html)
        self.assertNotIn("P0", cell_texts)
        self.assertNotIn("P1", cell_texts)
        self.assertNotIn("P2", cell_texts)

    def test_no_colspan_band_row(self):
        self.assertNotIn("colspan", self.html)

    def test_metric_row_order(self):
        row_label_cells = re.findall(
            r'<th style="background:#f7f7f7[^>]*>(.*?)</th>', self.html)
        self.assertEqual(row_label_cells, EXPECTED_LABEL_ORDER)

    def test_exactly_one_row_per_metric(self):
        body_trs = re.findall(r"<tbody>(.*)</tbody>", self.html, re.S)[0]
        self.assertEqual(body_trs.count("<tr>"), len(TABLE_METRICS))


class HeatmapPngNoGroupingTest(unittest.TestCase):
    """ax.table()'s cellText is the PNG's actual row structure, so spy on
    the real Axes.table call rather than parsing pixels."""

    def test_cell_text_has_no_tier_rows(self):
        norm = _heatmap_norms(ROWS)
        captured = {}
        orig_table = matplotlib.axes.Axes.table

        def spy(self, *args, **kwargs):
            captured["cellText"] = kwargs.get("cellText")
            return orig_table(self, *args, **kwargs)

        with patch.object(matplotlib.axes.Axes, "table", spy):
            png_bytes = _heatmap_png_bytes(ROWS, norm)

        self.assertTrue(png_bytes.startswith(b"\x89PNG"))
        cell_text = captured["cellText"]
        self.assertIsNotNone(cell_text)

        first_col = [row[0] for row in cell_text]
        self.assertNotIn("P0", first_col)
        self.assertNotIn("P1", first_col)
        self.assertNotIn("P2", first_col)

        # header row + exactly one row per metric, no extra tier-band rows
        self.assertEqual(len(cell_text), 1 + len(TABLE_METRICS))
        self.assertEqual(first_col[1:], EXPECTED_LABEL_ORDER)


class HeatmapCsvNoGroupingTest(unittest.TestCase):
    def setUp(self):
        csv_bytes = _heatmap_csv_bytes(ROWS)
        self.rows = list(csv.reader(io.StringIO(csv_bytes.decode("utf-8"))))

    def test_no_tier_prefix_on_row_labels(self):
        # Row 0 is the method-name header, row 1 is "stable"; metric rows
        # follow. Previously these were prefixed "P0: Rise", "P1: IAE", etc.
        metric_labels = [row[0] for row in self.rows[2:]]
        for label in metric_labels:
            self.assertNotIn(":", label)
            self.assertFalse(label.startswith("P0"))
            self.assertFalse(label.startswith("P1"))
            self.assertFalse(label.startswith("P2"))

    def test_metric_row_order(self):
        metric_labels = [row[0] for row in self.rows[2:]]
        self.assertEqual(metric_labels, EXPECTED_LABEL_ORDER)

    def test_row_count(self):
        # method-name row + stable row + one row per metric, nothing extra
        self.assertEqual(len(self.rows), 2 + len(TABLE_METRICS))


if __name__ == "__main__":
    unittest.main()
