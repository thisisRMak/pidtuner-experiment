"""Regression tests for the Streamlit LLM chat panel — no browser, no live
API key, just Streamlit's AppTest harness driving streamlit_app.py's actual
widget tree. Mirrors test_streamlit_siso_panel.py's style/conventions.

Run with:
    python test_streamlit_llm_panel.py
or:
    python -m unittest test_streamlit_llm_panel -v

Covers: key-source gating (env/secrets-configured vs. session textbox),
the model picker's contents/default, the cost warning, the unwired-provider
path, and each exception branch's friendly message -- via a monkeypatched
Session.handle_user_message so no real network call happens.

What this does NOT cover (see docs/gui_plan.md "Testing debt" for the
general caveat, and this session's live Playwright check for the one thing
AppTest genuinely cannot see): the real chat_input/history layout order --
that was a rendering-only bug AppTest's element-tree inspection missed
entirely; a real browser check found and confirmed the fix. Also not
covered: an actual live conversation against the real Anthropic API (see
docs/aituner_plan.md and the scripted live check run alongside this file).
"""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = __file__.replace("test_streamlit_llm_panel.py", "streamlit_app.py")


def _run_app(env=None):
    """clear=True so these tests never depend on whatever the machine's own
    .env/shell happens to have set (e.g. a real ANTHROPIC_API_KEY someone
    added locally) -- load_dotenv() only skips vars that are *already* set,
    so with nothing pre-set it would otherwise happily load the real file."""
    with patch.dict(os.environ, env or {}, clear=True), \
         patch("streamlit_llm_panel.load_dotenv"):
        return AppTest.from_file(APP_PATH).run(timeout=30)


class TestKeyGating(unittest.TestCase):
    def test_no_key_anywhere_shows_textbox_and_locks_chat(self):
        at = _run_app()
        self.assertEqual(at.exception[:], [])
        self.assertIn("Claude (Anthropic) API key", [ti.label for ti in at.text_input])
        self.assertEqual(len(at.chat_input), 0)

    def test_env_configured_key_skips_textbox_and_unlocks_chat(self):
        at = _run_app({"ANTHROPIC_API_KEY": "sk-ant-test-from-env"})
        self.assertNotIn("Claude (Anthropic) API key", [ti.label for ti in at.text_input])
        self.assertIn("Claude (Anthropic) key configured by the operator — nothing to enter.",
                       [s.value for s in at.success])
        self.assertEqual(len(at.chat_input), 1)


class TestModelPicker(unittest.TestCase):
    def test_offers_only_haiku_and_sonnet_defaulting_to_haiku(self):
        at = _run_app({"ANTHROPIC_API_KEY": "sk-ant-test"})
        sb = at.selectbox(key="llm_model")
        self.assertEqual(len(sb.options), 2)
        self.assertTrue(any("Haiku" in o for o in sb.options))
        self.assertTrue(any("Sonnet" in o for o in sb.options))
        self.assertFalse(any("Opus" in o for o in sb.options), "Opus must not be offered")
        self.assertEqual(sb.value, "claude-haiku-4-5")

    def test_cost_warning_shown_once_a_model_is_selectable(self):
        # Shown whenever Claude (the default provider) is selected, even
        # before a key is entered -- _render_key_entry gates the model
        # picker/warning on provider only, not on api_key. Harmless (no
        # call can happen without a key regardless), just noting the actual
        # gate here rather than assuming key-presence also gates it.
        at = _run_app({"ANTHROPIC_API_KEY": "sk-ant-test"})
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("billed" in w for w in warnings))


class TestUnwiredProvider(unittest.TestCase):
    def test_openai_key_accepted_but_not_unlocked(self):
        at = _run_app()
        at.selectbox(key="llm_provider").set_value("ChatGPT (OpenAI)").run(timeout=30)
        at.text_input(key="llm_api_key_openai").set_value("sk-fake-openai-key").run(timeout=30)
        infos = [i.value for i in at.info]
        self.assertTrue(any("isn't built yet" in i for i in infos))
        self.assertEqual(len(at.chat_input), 0, "chat must stay locked for an unwired provider")


class TestChatErrorHandling(unittest.TestCase):
    """Each exception branch, via a monkeypatched Session so no network call
    happens. Patches supervisor_session_pid.Session.handle_user_message
    directly (not the client) -- streamlit_llm_panel imports the Session
    class by reference, so patching the method on the class patches what
    the panel's already-constructed session instance calls too."""

    def _send_and_get_reply(self, raise_exc):
        from supervisor_session_pid import Session

        # ANTHROPIC_API_KEY must stay patched (and the real .env blocked --
        # see _run_app) across *both* reruns -- the follow-up chat_input
        # interaction below reruns the whole script too, and
        # _configured_key() re-reads os.environ fresh every time.
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True), \
             patch("streamlit_llm_panel.load_dotenv"):
            at = AppTest.from_file(APP_PATH).run(timeout=30)
            with patch.object(Session, "handle_user_message", side_effect=raise_exc):
                at.chat_input[0].set_value("hello").run(timeout=30)
        self.assertEqual(at.exception[:], [], "a bad turn must not crash the app")
        messages = [m.markdown[0].value for m in at.chat_message if m.markdown]
        return messages[-1]

    def test_authentication_error(self):
        import anthropic
        reply = self._send_and_get_reply(
            anthropic.AuthenticationError("bad key", response=_fake_response(401), body=None)
        )
        self.assertIn("rejected", reply)

    def test_rate_limit_error(self):
        import anthropic
        reply = self._send_and_get_reply(
            anthropic.RateLimitError("slow down", response=_fake_response(429), body=None)
        )
        self.assertIn("Rate limited", reply)

    def test_api_connection_error(self):
        import anthropic
        import httpx2
        reply = self._send_and_get_reply(
            anthropic.APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
        )
        self.assertIn("Couldn't reach", reply)

    def test_unexpected_exception_falls_back_to_generic_message(self):
        reply = self._send_and_get_reply(ValueError("something unrelated broke"))
        self.assertIn("check the server logs", reply)


def _fake_response(status_code):
    import httpx2
    request = httpx2.Request("POST", "https://api.anthropic.com")
    return httpx2.Response(status_code, request=request)


if __name__ == "__main__":
    unittest.main()
