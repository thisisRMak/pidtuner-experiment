"""Unit tests for cli_supervisor_judge_pid.py's provider/model/key-
resolution plumbing and cross-flag validation -- no live API required.
Mirrors test_cli_supervisor_pid.py's style/conventions.

Run with:
    pytest src/test_cli_supervisor_judge_pid.py
or, from the repo root:
    pytest

Covers everything deterministic: --candidate/--judge spec parsing,
--api-key-<provider>/.env key resolution precedence (identical logic to
cli_supervisor_pid.py's own, just duplicated per that module's own
convention), the "at least 2 candidates" / "judge must differ from every
candidate" cross-flag invariants, and client dispatch. The REPL loop in
main() itself is untested here, same as cli_supervisor_pid.py's own
test file -- see test_supervisor_session_judge_pid.py for JudgeSession's
own orchestration coverage, reused as-is by this CLI.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

from supervisor_llm_anthropic import AnthropicClient, DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from supervisor_llm_openai import OpenAIClient, DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from supervisor_llm_gemini import GeminiClient, DEFAULT_MODEL as GEMINI_DEFAULT_MODEL
from supervisor_session_judge_pid import JudgeSession

import cli_supervisor_judge_pid as cli


def _args(api_key_anthropic=None, api_key_openai=None, api_key_gemini=None, log_file=None):
    return argparse.Namespace(api_key_anthropic=api_key_anthropic, api_key_openai=api_key_openai,
                               api_key_gemini=api_key_gemini, log_file=log_file)


# ─────────────────────────────────────────────────────────────────────────────
# --candidate/--judge spec parsing
# ─────────────────────────────────────────────────────────────────────────────

class TestParseProviderModel(unittest.TestCase):
    def test_provider_and_model(self):
        self.assertEqual(cli._parse_provider_model("anthropic:claude-haiku-4-5"),
                          ("anthropic", "claude-haiku-4-5"))

    def test_bare_provider_has_no_model(self):
        self.assertEqual(cli._parse_provider_model("openai"), ("openai", None))

    def test_unknown_provider_raises(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            cli._parse_provider_model("ollama:qwen3-coder")

    def test_model_with_colon_in_it_is_kept_whole(self):
        # partition() splits on the FIRST colon only -- a model id that
        # itself contains a colon (unlikely for these providers, but not
        # this function's business to reject) must survive intact.
        self.assertEqual(cli._parse_provider_model("anthropic:foo:bar"), ("anthropic", "foo:bar"))


# ─────────────────────────────────────────────────────────────────────────────
# API key resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveAnthropicKey(unittest.TestCase):
    def test_explicit_key_wins_without_consulting_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-env"}), \
             patch("cli_supervisor_judge_pid.load_dotenv") as mock_load_dotenv:
            key = cli._resolve_anthropic_key("sk-explicit")
        self.assertEqual(key, "sk-explicit")
        mock_load_dotenv.assert_not_called()

    def test_falls_back_to_env_var_when_no_explicit_key(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-env"}), \
             patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_anthropic_key(None)
        self.assertEqual(key, "sk-env")

    def test_no_explicit_key_and_no_env_var_returns_none(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_anthropic_key(None)
        self.assertIsNone(key)


class TestResolveOpenAIKey(unittest.TestCase):
    def test_explicit_key_wins_without_consulting_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}), \
             patch("cli_supervisor_judge_pid.load_dotenv") as mock_load_dotenv:
            key = cli._resolve_openai_key("sk-explicit")
        self.assertEqual(key, "sk-explicit")
        mock_load_dotenv.assert_not_called()

    def test_falls_back_to_env_var_when_no_explicit_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}), \
             patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_openai_key(None)
        self.assertEqual(key, "sk-env")

    def test_no_explicit_key_and_no_env_var_returns_none(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_openai_key(None)
        self.assertIsNone(key)


class TestResolveGeminiKey(unittest.TestCase):
    def test_explicit_key_wins_without_consulting_env(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "sk-env"}), \
             patch("cli_supervisor_judge_pid.load_dotenv") as mock_load_dotenv:
            key = cli._resolve_gemini_key("sk-explicit")
        self.assertEqual(key, "sk-explicit")
        mock_load_dotenv.assert_not_called()

    def test_falls_back_to_google_api_key_when_no_explicit_key(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "sk-env"}), \
             patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_gemini_key(None)
        self.assertEqual(key, "sk-env")

    def test_falls_back_to_gemini_api_key_when_no_google_api_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "sk-gemini-env"}, clear=True), \
             patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_gemini_key(None)
        self.assertEqual(key, "sk-gemini-env")

    def test_google_api_key_wins_when_both_are_set(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "sk-google", "GEMINI_API_KEY": "sk-gemini"}), \
             patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_gemini_key(None)
        self.assertEqual(key, "sk-google")

    def test_no_explicit_key_and_no_env_var_returns_none(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_gemini_key(None)
        self.assertIsNone(key)


class TestResolveKeyDispatch(unittest.TestCase):
    """_resolve_key() dispatches to the right per-provider resolver and
    hard-exits with a provider-specific message when nothing is found --
    the "missing key is a hard error, not an interactive prompt" rule."""

    def test_dispatches_to_anthropic_resolver(self):
        with patch.dict("os.environ", {}, clear=True), patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_key("anthropic", _args(api_key_anthropic="sk-explicit"))
        self.assertEqual(key, "sk-explicit")

    def test_dispatches_to_openai_resolver(self):
        with patch.dict("os.environ", {}, clear=True), patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_key("openai", _args(api_key_openai="sk-explicit"))
        self.assertEqual(key, "sk-explicit")

    def test_dispatches_to_gemini_resolver(self):
        with patch.dict("os.environ", {}, clear=True), patch("cli_supervisor_judge_pid.load_dotenv"):
            key = cli._resolve_key("gemini", _args(api_key_gemini="sk-explicit"))
        self.assertEqual(key, "sk-explicit")

    def test_missing_key_exits_with_provider_specific_error(self):
        stderr = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_judge_pid.load_dotenv"), \
             contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._resolve_key("openai", _args())
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("no OpenAI API key found", stderr.getvalue())
        self.assertIn("--api-key-openai", stderr.getvalue())


# ─────────────────────────────────────────────────────────────────────────────
# Client dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestClientFor(unittest.TestCase):
    def test_anthropic(self):
        client = cli._client_for("anthropic", ANTHROPIC_DEFAULT_MODEL, "sk-explicit")
        self.assertIsInstance(client, AnthropicClient)
        self.assertEqual(client.model, ANTHROPIC_DEFAULT_MODEL)

    def test_openai(self):
        client = cli._client_for("openai", OPENAI_DEFAULT_MODEL, "sk-explicit")
        self.assertIsInstance(client, OpenAIClient)
        self.assertEqual(client.model, OPENAI_DEFAULT_MODEL)

    def test_gemini(self):
        client = cli._client_for("gemini", GEMINI_DEFAULT_MODEL, "sk-explicit")
        self.assertIsInstance(client, GeminiClient)
        self.assertEqual(client.model, GEMINI_DEFAULT_MODEL)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-flag validation: candidate count + judge/candidate distinctness
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateSelection(unittest.TestCase):
    def test_at_least_two_candidates_required(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._validate_selection([("anthropic", "claude-haiku-4-5")], ("openai", "gpt-5.6-luna"))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("at least 2", stderr.getvalue())

    def test_zero_candidates_required(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._validate_selection(None, ("openai", "gpt-5.6-luna"))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("got 0", stderr.getvalue())

    def test_duplicate_candidates_rejected(self):
        stderr = io.StringIO()
        candidates = [("anthropic", "claude-haiku-4-5"), ("anthropic", "claude-haiku-4-5")]
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._validate_selection(candidates, ("openai", "gpt-5.6-luna"))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("duplicate candidate", stderr.getvalue())

    def test_duplicate_candidates_rejected_after_default_model_resolution(self):
        """A bare "anthropic" and an explicit "anthropic:<its own default>"
        resolve to the identical pair -- must be caught as a duplicate too,
        not just a byte-for-byte repeated spec."""
        stderr = io.StringIO()
        candidates = [("anthropic", None), ("anthropic", ANTHROPIC_DEFAULT_MODEL)]
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._validate_selection(candidates, ("openai", "gpt-5.6-luna"))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("duplicate candidate", stderr.getvalue())

    def test_judge_matching_a_candidate_rejected(self):
        stderr = io.StringIO()
        candidates = [("anthropic", "claude-haiku-4-5"), ("openai", "gpt-5.6-luna")]
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._validate_selection(candidates, ("anthropic", "claude-haiku-4-5"))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("must differ from every candidate", stderr.getvalue())

    def test_judge_matching_a_candidate_via_default_model_rejected(self):
        """Same default-model-resolution wrinkle as the duplicate-candidate
        case above, but between the judge and a candidate."""
        stderr = io.StringIO()
        candidates = [("anthropic", ANTHROPIC_DEFAULT_MODEL), ("openai", "gpt-5.6-luna")]
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._validate_selection(candidates, ("anthropic", None))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("must differ from every candidate", stderr.getvalue())

    def test_same_provider_different_model_is_allowed_for_candidates(self):
        candidates = [("anthropic", "claude-haiku-4-5"), ("anthropic", "claude-sonnet-5")]
        resolved_candidates, resolved_judge = cli._validate_selection(candidates, ("openai", "gpt-5.6-luna"))
        self.assertEqual(resolved_candidates, candidates)
        self.assertEqual(resolved_judge, ("openai", "gpt-5.6-luna"))

    def test_valid_selection_resolves_default_models(self):
        candidates = [("anthropic", None), ("openai", "gpt-5.6-terra")]
        resolved_candidates, resolved_judge = cli._validate_selection(candidates, ("gemini", None))
        self.assertEqual(resolved_candidates, [("anthropic", ANTHROPIC_DEFAULT_MODEL),
                                                ("openai", "gpt-5.6-terra")])
        self.assertEqual(resolved_judge, ("gemini", GEMINI_DEFAULT_MODEL))


# ─────────────────────────────────────────────────────────────────────────────
# Judge session construction
# ─────────────────────────────────────────────────────────────────────────────

class TestNewJudgeSession(unittest.TestCase):
    def test_builds_a_fresh_judge_session_from_prebuilt_clients(self):
        candidate_clients = [
            ("anthropic:claude-haiku-4-5", AnthropicClient(api_key="sk-a", model=ANTHROPIC_DEFAULT_MODEL)),
            ("openai:gpt-5.6-luna", OpenAIClient(api_key="sk-b", model=OPENAI_DEFAULT_MODEL)),
        ]
        judge_client = GeminiClient(api_key="sk-c", model=GEMINI_DEFAULT_MODEL)
        session = cli._new_judge_session(candidate_clients, judge_client)
        self.assertIsInstance(session, JudgeSession)
        self.assertEqual([label for label, _ in session.candidates],
                          ["anthropic:claude-haiku-4-5", "openai:gpt-5.6-luna"])
        self.assertIs(session.judge_client, judge_client)
        self.assertEqual(session.last_round, [])
        self.assertEqual(session.rounds_history, [])

    def test_reset_produces_a_session_with_no_carried_over_history(self):
        """/reset in the REPL calls _new_judge_session again with the same
        already-built clients -- the point is a clean JudgeSession each
        time, not reused clients being a problem (they're stateless
        provider wrappers, no conversation state lives on them)."""
        candidate_clients = [
            ("anthropic:claude-haiku-4-5", AnthropicClient(api_key="sk-a", model=ANTHROPIC_DEFAULT_MODEL)),
            ("openai:gpt-5.6-luna", OpenAIClient(api_key="sk-b", model=OPENAI_DEFAULT_MODEL)),
        ]
        judge_client = GeminiClient(api_key="sk-c", model=GEMINI_DEFAULT_MODEL)
        first = cli._new_judge_session(candidate_clients, judge_client)
        first.dialogue.append({"role": "user", "content": "hello"})
        second = cli._new_judge_session(candidate_clients, judge_client)
        self.assertEqual(second.dialogue, [])
        self.assertIsNot(first, second)


if __name__ == "__main__":
    unittest.main()
