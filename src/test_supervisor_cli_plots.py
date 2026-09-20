"""Unit tests for supervisor_cli_plots.py -- figure building/saving for
the SISO/PID and MIMO/LQG tracks, driven by synthetic plot_calls-shaped
input (no live API, no Streamlit runtime).

Run with:
    pytest src/test_supervisor_cli_plots.py
"""

from __future__ import annotations

import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

import supervisor_cli_output as clio
import supervisor_cli_plots as clip
from lqg_compare import ComparisonRow
from lqg_simulate import LQGSimResult

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _assert_real_png(test, path):
    with open(path, "rb") as f:
        raw = f.read(8)
    test.assertEqual(raw, _PNG_MAGIC, f"{path} must be a real PNG file")


def _sim(t=None, y=None, u=None):
    t = np.linspace(0, 5, 20) if t is None else t
    return SimpleNamespace(t=t,
                            y=np.ones_like(t) if y is None else y,
                            u=np.ones_like(t) if u is None else u)


class _TmpCwd(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmpdir.name)
        self.addCleanup(self._restore)

    def _restore(self):
        os.chdir(self._old_cwd)
        self._tmpdir.cleanup()

    def _run_paths(self, tag="supervisor-pid"):
        run_paths = clio.RunPaths.new(tag)
        run_paths.next_turn()
        return run_paths


class TestSavePidTurnPlots(_TmpCwd):
    def test_stable_row_with_sim_produces_a_png(self):
        calls = [{"kind": "siso", "plant": "1/(90s+1)", "delay": 0.0, "rows": [
            {"name": "SIMC", "stable": True, "sim": _sim()},
        ]}]
        paths = clip.save_pid_turn_plots(calls, self._run_paths())
        self.assertEqual(len(paths), 1)
        self.assertTrue(os.path.exists(paths[0]))
        _assert_real_png(self, paths[0])

    def test_unstable_row_is_excluded(self):
        calls = [{"kind": "siso", "plant": "1/(90s+1)", "delay": 0.0, "rows": [
            {"name": "Bad", "stable": False, "sim": _sim()},
        ]}]
        self.assertEqual(clip.save_pid_turn_plots(calls, self._run_paths()), [])

    def test_row_with_no_sim_is_excluded(self):
        calls = [{"kind": "siso", "plant": "1/(90s+1)", "delay": 0.0, "rows": [
            {"name": "NoSim", "stable": True, "sim": None},
        ]}]
        self.assertEqual(clip.save_pid_turn_plots(calls, self._run_paths()), [])

    def test_non_siso_call_produces_nothing(self):
        calls = [{"kind": "mimo", "plant": "aircraft_hall", "rows": [
            {"name": "SIMC", "stable": True, "sim": _sim()},
        ]}]
        self.assertEqual(clip.save_pid_turn_plots(calls, self._run_paths()), [])

    def test_candidate_label_appears_in_returned_path(self):
        calls = [{"kind": "siso", "plant": "1/(90s+1)", "delay": 0.0, "rows": [
            {"name": "SIMC", "stable": True, "sim": _sim()},
        ]}]
        paths = clip.save_pid_turn_plots(calls, self._run_paths(), candidate_label="anthropic:claude-haiku-4-5")
        self.assertIn("anthropic", paths[0])


def _comparison_row(name, t=None, x=None, y=None, u=None, tracking_metrics=None):
    t = np.linspace(0, 5, 20) if t is None else t
    n = len(t)
    sim = LQGSimResult(
        t=t,
        x=np.ones((n, 2)) if x is None else x,
        y=np.ones((n, 2)) if y is None else y,
        u=np.ones((n, 1)) if u is None else u,
        stable=True,
        metrics={"unstable": False},
        tracking_metrics=tracking_metrics,
    )
    return ComparisonRow(name=name, result=None, sim=sim, checks={})


class TestSaveLqgTurnPlots(_TmpCwd):
    def test_rows_with_sim_produce_a_response_png(self):
        calls = [{"kind": "mimo", "plant": "aircraft_hall", "rows": [
            _comparison_row("LQR"), _comparison_row("LQG"),
        ], "four_curve": None}]
        paths = clip.save_lqg_turn_plots(calls, self._run_paths("supervisor-lqg"))
        self.assertEqual(len(paths), 1)
        self.assertTrue(paths[0].endswith("-response.png"))
        _assert_real_png(self, paths[0])

    def test_no_rows_with_sim_writes_nothing(self):
        row = _comparison_row("LQR")
        row.sim = None
        calls = [{"kind": "mimo", "plant": "aircraft_hall", "rows": [row], "four_curve": None}]
        self.assertEqual(clip.save_lqg_turn_plots(calls, self._run_paths("supervisor-lqg")), [])

    def test_four_curve_present_also_writes_fourcurve_png(self):
        rows = [_comparison_row(n) for n in
                ("Bryson's rule", "Output-weighted LQR", "Implicit model-following", "Explicit model-following")]
        t = np.linspace(0, 5, 20)
        four_curve = {"rows": rows, "Am": np.diag([-1.0, -1.0]),
                      "t": t, "xm_ref": np.ones((len(t), 2))}
        calls = [{"kind": "mimo", "plant": "aircraft_hall", "rows": rows, "four_curve": four_curve}]
        paths = clip.save_lqg_turn_plots(calls, self._run_paths("judge-lqg"))
        self.assertEqual(len(paths), 2)
        self.assertTrue(any(p.endswith("-fourcurve.png") for p in paths))
        for p in paths:
            _assert_real_png(self, p)

    def test_four_curve_none_writes_no_fourcurve_png(self):
        calls = [{"kind": "mimo", "plant": "aircraft_hall", "rows": [_comparison_row("LQR")],
                  "four_curve": None}]
        paths = clip.save_lqg_turn_plots(calls, self._run_paths("supervisor-lqg"))
        self.assertFalse(any(p.endswith("-fourcurve.png") for p in paths))

    def test_non_mimo_call_produces_nothing(self):
        calls = [{"kind": "siso", "plant": "1/(90s+1)", "rows": [], "four_curve": None}]
        self.assertEqual(clip.save_lqg_turn_plots(calls, self._run_paths("supervisor-lqg")), [])

    def test_candidate_label_appears_in_returned_path(self):
        calls = [{"kind": "mimo", "plant": "aircraft_hall", "rows": [_comparison_row("LQR")],
                  "four_curve": None}]
        paths = clip.save_lqg_turn_plots(calls, self._run_paths("judge-lqg"),
                                          candidate_label="openai:gpt-5.6-luna")
        self.assertIn("openai", paths[0])


if __name__ == "__main__":
    unittest.main()
