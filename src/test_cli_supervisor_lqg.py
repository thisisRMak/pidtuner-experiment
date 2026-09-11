"""Unit tests for cli_supervisor_lqg.py's provider/model/key-resolution
plumbing -- no live Ollama or Anthropic API required. Mirrors
test_cli_supervisor_pid.py (and, further back, test_supervisor_pid.py's
style/conventions) -- see that file's docstring; this one is the LQG-track
twin, kept as a separate script/test module per existing project convention
(cli_supervisor_pid.py/cli_supervisor_lqg.py stay separate scripts, not a
shared mode flag).

Run with:
    python test_cli_supervisor_lqg.py
or:
    python -m unittest test_cli_supervisor_lqg -v
"""

from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from unittest.mock import patch

from supervisor_llm import DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL, OllamaClient
from supervisor_llm_anthropic import AnthropicClient, DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from supervisor_llm_openai import OpenAIClient, DEFAULT_MODEL as OPENAI_DEFAULT_MODEL
from supervisor_llm_gemini import GeminiClient, DEFAULT_MODEL as GEMINI_DEFAULT_MODEL

import cli_supervisor_lqg as cli


def _args(provider="ollama", model=None, api_key=None, host=None, num_ctx=8192, keep_alive="30m"):
    return argparse.Namespace(provider=provider, model=model, api_key=api_key,
                               host=host, num_ctx=num_ctx, keep_alive=keep_alive)


# ─────────────────────────────────────────────────────────────────────────────
# API key resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveAnthropicKey(unittest.TestCase):
    def test_explicit_key_wins_without_consulting_env(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-env"}), \
             patch("cli_supervisor_lqg.load_dotenv") as mock_load_dotenv:
            key = cli._resolve_anthropic_key("sk-explicit")
        self.assertEqual(key, "sk-explicit")
        mock_load_dotenv.assert_not_called()

    def test_falls_back_to_env_var_when_no_explicit_key(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-env"}), \
             patch("cli_supervisor_lqg.load_dotenv"):
            key = cli._resolve_anthropic_key(None)
        self.assertEqual(key, "sk-env")

    def test_no_explicit_key_and_no_env_var_returns_none(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_lqg.load_dotenv"):
            key = cli._resolve_anthropic_key(None)
        self.assertIsNone(key)


class TestResolveOpenAIKey(unittest.TestCase):
    def test_explicit_key_wins_without_consulting_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}), \
             patch("cli_supervisor_lqg.load_dotenv") as mock_load_dotenv:
            key = cli._resolve_openai_key("sk-explicit")
        self.assertEqual(key, "sk-explicit")
        mock_load_dotenv.assert_not_called()

    def test_falls_back_to_env_var_when_no_explicit_key(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}), \
             patch("cli_supervisor_lqg.load_dotenv"):
            key = cli._resolve_openai_key(None)
        self.assertEqual(key, "sk-env")

    def test_no_explicit_key_and_no_env_var_returns_none(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_lqg.load_dotenv"):
            key = cli._resolve_openai_key(None)
        self.assertIsNone(key)


class TestResolveGeminiKey(unittest.TestCase):
    def test_explicit_key_wins_without_consulting_env(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "sk-env"}), \
             patch("cli_supervisor_lqg.load_dotenv") as mock_load_dotenv:
            key = cli._resolve_gemini_key("sk-explicit")
        self.assertEqual(key, "sk-explicit")
        mock_load_dotenv.assert_not_called()

    def test_falls_back_to_google_api_key_when_no_explicit_key(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "sk-env"}), \
             patch("cli_supervisor_lqg.load_dotenv"):
            key = cli._resolve_gemini_key(None)
        self.assertEqual(key, "sk-env")

    def test_falls_back_to_gemini_api_key_when_no_google_api_key(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "sk-gemini-env"}, clear=True), \
             patch("cli_supervisor_lqg.load_dotenv"):
            key = cli._resolve_gemini_key(None)
        self.assertEqual(key, "sk-gemini-env")

    def test_google_api_key_wins_when_both_are_set(self):
        """Matches the google-genai SDK's own precedence -- confirmed from
        its source, not assumed."""
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "sk-google", "GEMINI_API_KEY": "sk-gemini"}), \
             patch("cli_supervisor_lqg.load_dotenv"):
            key = cli._resolve_gemini_key(None)
        self.assertEqual(key, "sk-google")

    def test_no_explicit_key_and_no_env_var_returns_none(self):
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_lqg.load_dotenv"):
            key = cli._resolve_gemini_key(None)
        self.assertIsNone(key)


# ─────────────────────────────────────────────────────────────────────────────
# Client dispatch
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildClient(unittest.TestCase):
    def test_default_provider_is_ollama_with_ollama_default_model(self):
        client = cli._build_client(_args(provider="ollama"))
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client.model, OLLAMA_DEFAULT_MODEL)

    def test_ollama_explicit_model_overrides_default(self):
        client = cli._build_client(_args(provider="ollama", model="qwen3-coder:30b"))
        self.assertEqual(client.model, "qwen3-coder:30b")

    def test_ollama_host_and_options_are_passed_through(self):
        client = cli._build_client(_args(provider="ollama", host="http://example:11434",
                                          num_ctx=4096, keep_alive="5m"))
        self.assertEqual(client.num_ctx, 4096)
        self.assertEqual(client.keep_alive, "5m")

    def test_anthropic_uses_explicit_api_key_and_default_model(self):
        client = cli._build_client(_args(provider="anthropic", api_key="sk-explicit"))
        self.assertIsInstance(client, AnthropicClient)
        self.assertEqual(client.model, ANTHROPIC_DEFAULT_MODEL)

    def test_anthropic_explicit_model_overrides_default(self):
        client = cli._build_client(_args(provider="anthropic", api_key="sk-explicit", model="claude-sonnet-5"))
        self.assertEqual(client.model, "claude-sonnet-5")

    def test_anthropic_falls_back_to_env_key_when_no_flag(self):
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-env"}), \
             patch("cli_supervisor_lqg.load_dotenv"):
            client = cli._build_client(_args(provider="anthropic"))
        self.assertIsInstance(client, AnthropicClient)

    def test_anthropic_missing_key_exits_with_error(self):
        stderr = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_lqg.load_dotenv"), \
             contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._build_client(_args(provider="anthropic"))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("no Anthropic API key found", stderr.getvalue())

    def test_openai_uses_explicit_api_key_and_default_model(self):
        client = cli._build_client(_args(provider="openai", api_key="sk-explicit"))
        self.assertIsInstance(client, OpenAIClient)
        self.assertEqual(client.model, OPENAI_DEFAULT_MODEL)

    def test_openai_explicit_model_overrides_default(self):
        client = cli._build_client(_args(provider="openai", api_key="sk-explicit", model="gpt-5.6-terra"))
        self.assertEqual(client.model, "gpt-5.6-terra")

    def test_openai_falls_back_to_env_key_when_no_flag(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-env"}), \
             patch("cli_supervisor_lqg.load_dotenv"):
            client = cli._build_client(_args(provider="openai"))
        self.assertIsInstance(client, OpenAIClient)

    def test_openai_missing_key_exits_with_error(self):
        stderr = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_lqg.load_dotenv"), \
             contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._build_client(_args(provider="openai"))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("no OpenAI API key found", stderr.getvalue())

    def test_gemini_uses_explicit_api_key_and_default_model(self):
        client = cli._build_client(_args(provider="gemini", api_key="sk-explicit"))
        self.assertIsInstance(client, GeminiClient)
        self.assertEqual(client.model, GEMINI_DEFAULT_MODEL)

    def test_gemini_explicit_model_overrides_default(self):
        client = cli._build_client(_args(provider="gemini", api_key="sk-explicit", model="gemini-3.8-flash"))
        self.assertEqual(client.model, "gemini-3.8-flash")

    def test_gemini_falls_back_to_env_key_when_no_flag(self):
        with patch.dict("os.environ", {"GOOGLE_API_KEY": "sk-env"}), \
             patch("cli_supervisor_lqg.load_dotenv"):
            client = cli._build_client(_args(provider="gemini"))
        self.assertIsInstance(client, GeminiClient)

    def test_gemini_missing_key_exits_with_error(self):
        stderr = io.StringIO()
        with patch.dict("os.environ", {}, clear=True), \
             patch("cli_supervisor_lqg.load_dotenv"), \
             contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                cli._build_client(_args(provider="gemini"))
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("no Gemini API key found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
