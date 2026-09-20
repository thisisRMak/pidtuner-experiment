"""Unit tests for supervisor_cli_output.py -- filename/folder generation,
report building, and --log-file writing. No live API, no Streamlit
runtime, no real CLI REPL loop involved (matches every existing
test_cli_supervisor_*.py's own "the REPL loop stays untested, only
deterministic plumbing gets tests" convention).

Run with:
    pytest src/test_supervisor_cli_output.py
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import tempfile
import unittest
from types import SimpleNamespace

import supervisor_cli_output as clio


class TestSlugify(unittest.TestCase):
    def test_special_chars_collapsed_to_underscore(self):
        self.assertEqual(clio.slugify("1/((90s+1)(10s+1))"), "1_90s_1_10s_1")

    def test_trims_leading_trailing_underscores(self):
        self.assertEqual(clio.slugify("***hello***"), "hello")

    def test_truncates_to_maxlen_and_trims_again(self):
        out = clio.slugify("a" * 50, maxlen=10)
        self.assertEqual(len(out), 10)
        self.assertFalse(out.startswith("_"))
        self.assertFalse(out.endswith("_"))

    def test_falls_back_to_x_when_nothing_survives(self):
        self.assertEqual(clio.slugify("***"), "x")
        self.assertEqual(clio.slugify(""), "x")

    def test_preserves_safe_characters(self):
        self.assertEqual(clio.slugify("anthropic:claude-haiku-4-5"), "anthropic_claude-haiku-4-5")


class _TmpCwd(unittest.TestCase):
    """Base class: run each test inside a fresh temp directory as cwd, so
    RunPaths.new()'s real os.makedirs() call never touches the actual
    working tree."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmpdir.name)
        self.addCleanup(self._restore)

    def _restore(self):
        os.chdir(self._old_cwd)
        self._tmpdir.cleanup()


class TestRunPathsNew(_TmpCwd):
    def test_creates_directory_with_timestamp_before_tag(self):
        run_paths = clio.RunPaths.new("supervisor-pid")
        self.assertTrue(os.path.isdir(run_paths.dir_path))
        self.assertRegex(os.path.basename(run_paths.dir_path),
                          r"^controldesign-\d{8}-\d{6}-supervisor-pid$")

    def test_report_path_lives_inside_the_folder(self):
        run_paths = clio.RunPaths.new("judge-lqg")
        self.assertEqual(run_paths.report_path, os.path.join(run_paths.dir_path, "report.html"))

    def test_starts_at_turn_zero(self):
        self.assertEqual(clio.RunPaths.new("supervisor-lqg").turn, 0)

    def test_different_tags_are_distinguishable(self):
        a = clio.RunPaths.new("supervisor-pid")
        b = clio.RunPaths.new("judge-pid")
        self.assertNotEqual(a.dir_path, b.dir_path)
        self.assertTrue(a.dir_path.endswith("-supervisor-pid"))
        self.assertTrue(b.dir_path.endswith("-judge-pid"))


class TestRunPathsNextTurn(unittest.TestCase):
    def test_increments_monotonically_and_never_resets(self):
        run_paths = clio.RunPaths(dir_path="/tmp/irrelevant", report_path="/tmp/irrelevant/report.html")
        self.assertEqual(run_paths.next_turn(), 1)
        self.assertEqual(run_paths.next_turn(), 2)
        self.assertEqual(run_paths.next_turn(), 3)
        self.assertEqual(run_paths.turn, 3)


class TestRunPathsPlotStem(unittest.TestCase):
    def setUp(self):
        self.run_paths = clio.RunPaths(dir_path="out", report_path="out/report.html")
        self.run_paths.turn = 2

    def test_includes_turn_and_call_index(self):
        stem = self.run_paths.plot_stem(1, "1/(90s+1)")
        self.assertEqual(os.path.basename(stem), "turn002-01-1_90s_1")

    def test_omits_candidate_label_when_none(self):
        stem = self.run_paths.plot_stem(3, "aircraft_hall")
        self.assertNotIn("anthropic", stem)
        self.assertEqual(os.path.basename(stem), "turn002-03-aircraft_hall")

    def test_includes_candidate_label_before_plant_when_given(self):
        stem = self.run_paths.plot_stem(1, "aircraft_hall", candidate_label="anthropic:claude-haiku-4-5")
        self.assertEqual(os.path.basename(stem),
                          "turn002-01-anthropic_claude-haiku-4-5-aircraft_hall")

    def test_falls_back_when_plant_is_falsy(self):
        stem = self.run_paths.plot_stem(1, None)
        self.assertTrue(os.path.basename(stem).endswith("-plant"))

    def test_stem_is_inside_dir_path(self):
        stem = self.run_paths.plot_stem(1, "x")
        self.assertTrue(stem.startswith("out" + os.sep))


class TestOpenSessionLog(unittest.TestCase):
    def test_none_path_returns_none(self):
        self.assertIsNone(clio.open_session_log(None))

    def test_real_path_returns_a_writable_handle(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.log")
            fh = clio.open_session_log(path)
            try:
                fh.write("hello\n")
                fh.flush()
                with open(path) as f:
                    self.assertEqual(f.read(), "hello\n")
            finally:
                fh.close()

    def test_appends_rather_than_truncates_an_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.log")
            with open(path, "w") as f:
                f.write("first\n")
            fh = clio.open_session_log(path)
            try:
                fh.write("second\n")
                fh.flush()
                with open(path) as f:
                    self.assertEqual(f.read(), "first\nsecond\n")
            finally:
                fh.close()


class TestLogPrint(unittest.TestCase):
    def test_no_log_fh_prints_only_to_stdout(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            clio.log_print(None, "hello")
        self.assertIn("hello", stdout.getvalue())

    def test_with_log_fh_writes_to_both_stdout_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.log")
            fh = clio.open_session_log(path)
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    clio.log_print(fh, "hello")
                fh.flush()
                with open(path) as f:
                    logged = f.read()
            finally:
                fh.close()
        self.assertIn("hello", stdout.getvalue())
        self.assertIn("hello", logged)


class TestLogUserInput(unittest.TestCase):
    def test_no_log_fh_is_a_noop(self):
        clio.log_user_input(None, "you> ", "hello")  # must not raise

    def test_with_log_fh_writes_prompt_and_text_but_never_to_stdout(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "session.log")
            fh = clio.open_session_log(path)
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    clio.log_user_input(fh, "you> ", "1/(90s+1) L=13")
                fh.flush()
                with open(path) as f:
                    logged = f.read()
            finally:
                fh.close()
        self.assertEqual(stdout.getvalue(), "", "must never echo the user's own input to stdout")
        self.assertIn("you> 1/(90s+1) L=13", logged)


def _fake_session(reply_text="the final reply"):
    return SimpleNamespace(messages=[
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "my plant is 1/(90s+1)"},
        SimpleNamespace(tool_calls=[], content=reply_text),
    ])


def _fake_judge_session():
    return SimpleNamespace(rounds_history=[{
        "user_text": "my plant is 1/(90s+1)",
        "candidates": [{"label": "anthropic:claude-haiku-4-5", "calls": [],
                        "finalized": "SIMC", "reply": "I recommend SIMC"}],
        "judge_reply": "Both candidates agree on SIMC.",
    }])


class TestWriteSupervisorReport(unittest.TestCase):
    def test_writes_a_report_with_title_and_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_paths = clio.RunPaths(dir_path=tmp, report_path=os.path.join(tmp, "report.html"))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                path = clio.write_supervisor_report(run_paths, "SISO / PID", _fake_session())
            self.assertEqual(path, run_paths.report_path)
            with open(path) as f:
                content = f.read()
        self.assertIn("PIDTuner LLM Supervisor Report (SISO / PID)", content)
        self.assertIn("the final reply", content)
        self.assertIn(f"saved report: {run_paths.report_path}", stdout.getvalue())

    def test_rewrites_rather_than_appends(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_paths = clio.RunPaths(dir_path=tmp, report_path=os.path.join(tmp, "report.html"))
            clio.write_supervisor_report(run_paths, "SISO / PID", _fake_session("first reply"))
            clio.write_supervisor_report(run_paths, "SISO / PID", _fake_session("second reply"))
            with open(run_paths.report_path) as f:
                content = f.read()
        self.assertNotIn("first reply", content)
        self.assertIn("second reply", content)


class TestWriteJudgeReport(unittest.TestCase):
    def test_writes_a_report_with_title_and_rounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_paths = clio.RunPaths(dir_path=tmp, report_path=os.path.join(tmp, "report.html"))
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                path = clio.write_judge_report(run_paths, "MIMO / LQG", _fake_judge_session())
            with open(path) as f:
                content = f.read()
        self.assertIn("PIDTuner LLM Judge Report (MIMO / LQG)", content)
        self.assertIn("Both candidates agree on SIMC.", content)
        self.assertIn(f"saved report: {run_paths.report_path}", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
