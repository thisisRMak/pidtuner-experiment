"""Regression tests for Mode="LLM Judge" -- no browser, no live API key,
just Streamlit's AppTest harness driving streamlit_app.py's actual widget
tree. Mirrors test_streamlit_llm_panel.py's style/conventions.

Run with:
    python test_streamlit_judge_panel.py
or:
    python -m unittest test_streamlit_judge_panel -v

Covers: no default judge (the user must choose one before anything else
appears), candidates seeded once a judge is chosen, a judge choice that
collides with a current candidate dropping just that stale candidate
without crashing, per-provider key gating/dedup across roles, the scaling
cost warning, and a full round-trip conversation -- via monkeypatched
Session.handle_user_message (all candidates) and GeminiClient.chat (the
default test judge, being the 3rd provider) so no real network call
happens -- confirming only the judge's reply lands in the visible thread.

What this does NOT cover: an actual live conversation against any real
provider API (same posture as test_streamlit_llm_panel.py), and the
per-round trace content itself (covered directly, without any Streamlit
involved, by test_supervisor_session_judge_pid.py).
"""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

APP_PATH = __file__.replace("test_streamlit_judge_panel.py", "streamlit_app.py")

ALL_KEYS_ENV = {
    "ANTHROPIC_API_KEY": "sk-ant-test",
    "OPENAI_API_KEY": "sk-openai-test",
    "GOOGLE_API_KEY": "fake-gemini-test",
}

DEFAULT_JUDGE = ("Gemini (Google)", "gemini-3.5-flash-lite")


def _select_judge(at, judge):
    return at.selectbox(key="judge_judge_choice").set_value(judge).run(timeout=30)


def _run_app(env=None, judge=DEFAULT_JUDGE):
    """`judge`, when given (the default), also picks a judge -- there's no
    default judge selection anymore (asked, not assumed: the user must
    deliberately choose one), so most tests need one explicitly picked to
    get anywhere near a usable chat. Pass judge=None to test the
    before-any-judge-is-chosen state itself."""
    with patch.dict(os.environ, env or {}, clear=True), \
         patch("streamlit_llm_panel.load_dotenv"):
        at = AppTest.from_file(APP_PATH).run(timeout=30)
        at.segmented_control(key="unified_mode").set_value("LLM Judge").run(timeout=30)
        if judge is not None:
            at = _select_judge(at, judge)
        return at


class TestDefaultState(unittest.TestCase):
    def test_no_judge_chosen_yet_shows_no_candidates_widget(self):
        at = _run_app(judge=None)
        self.assertEqual(at.exception[:], [])
        self.assertIsNone(at.selectbox(key="judge_judge_choice").value)
        self.assertEqual(len(at.multiselect), 0,
                          "the candidates widget isn't rendered until a judge is chosen")
        self.assertTrue(any("Choose a judge" in c.value for c in at.caption))
        self.assertEqual(len(at.chat_input), 0)

    def test_choosing_a_judge_seeds_default_candidates_from_other_providers(self):
        at = _run_app()  # default test judge = Gemini
        candidates = at.multiselect(key="judge_candidates").value
        self.assertIn(("Claude (Anthropic)", "claude-haiku-4-5"), candidates)
        self.assertIn(("ChatGPT (OpenAI)", "gpt-5.6-luna"), candidates)
        self.assertNotIn(("Gemini (Google)", "gemini-3.5-flash-lite"), candidates,
                          "the judge's own pair must not also be a seeded candidate")

    def test_no_keys_shows_info_and_no_chat(self):
        at = _run_app()
        self.assertEqual(len(at.chat_input), 0)
        infos = [i.value for i in at.info]
        self.assertTrue(any("API key" in i for i in infos))


class TestRoleSelection(unittest.TestCase):
    def test_picking_the_judge_to_match_a_current_candidate_drops_just_that_candidate(self):
        at = _run_app()  # judge=Gemini, candidates seeded to Claude+ChatGPT
        self.assertIn(("Claude (Anthropic)", "claude-haiku-4-5"), at.multiselect(key="judge_candidates").value)

        # Re-point the judge at Claude's default model -- the same pair
        # currently sitting in the candidates multiselect.
        at = _select_judge(at, ("Claude (Anthropic)", "claude-haiku-4-5"))
        self.assertEqual(at.exception[:], [], "a stale candidate selection must not crash the widget")
        candidates_after = at.multiselect(key="judge_candidates").value
        self.assertNotIn(("Claude (Anthropic)", "claude-haiku-4-5"), candidates_after)
        self.assertIn(("ChatGPT (OpenAI)", "gpt-5.6-luna"), candidates_after, "the other candidate survives")

    def test_fewer_than_two_candidates_blocks_start(self):
        at = _run_app()
        ms = at.multiselect(key="judge_candidates")
        at = ms.set_value([ms.value[0]]).run(timeout=30)
        self.assertEqual(at.exception[:], [])
        self.assertTrue(any("at least 2 candidates" in i.value for i in at.info))
        self.assertEqual(len(at.chat_input), 0)


class TestKeyGating(unittest.TestCase):
    def test_partial_keys_names_the_missing_provider(self):
        at = _run_app({"ANTHROPIC_API_KEY": "sk-ant-test", "OPENAI_API_KEY": "sk-openai-test"})
        infos = [i.value for i in at.info]
        self.assertTrue(any("Gemini (Google)" in i for i in infos), infos)
        self.assertEqual(len(at.chat_input), 0)

    def test_all_keys_present_unlocks_chat_and_dedupes_key_widgets(self):
        at = _run_app(ALL_KEYS_ENV)
        self.assertEqual(len(at.chat_input), 1)
        # One success message per distinct provider in use (2 candidates +
        # 1 judge, all different providers here) -- not one per role.
        successes = [s.value for s in at.success]
        self.assertEqual(len(successes), 3)
        self.assertEqual(len({s for s in successes}), 3, "no duplicate key widget per provider")


class TestCostWarning(unittest.TestCase):
    def test_warning_scales_with_candidate_count(self):
        at = _run_app(ALL_KEYS_ENV)
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("2 candidate calls + 1 judge call = **3 billed calls per turn**" in w for w in warnings), warnings)

        ms = at.multiselect(key="judge_candidates")
        at = ms.set_value(list(ms.value) + [("Claude (Anthropic)", "claude-sonnet-5")]).run(timeout=30)
        warnings = [w.value for w in at.warning]
        self.assertTrue(any("3 candidate calls + 1 judge call = **4 billed calls per turn**" in w for w in warnings), warnings)


class TestJudgeConversation(unittest.TestCase):
    def test_only_the_judges_reply_appears_in_the_thread(self):
        from supervisor_session_pid import Session
        from supervisor_llm_gemini import GeminiClient

        judge_reply = SimpleNamespace(message=SimpleNamespace(content="Tyreus-Luyben is the better pick.", tool_calls=None))
        with patch.dict(os.environ, ALL_KEYS_ENV, clear=True), \
             patch("streamlit_llm_panel.load_dotenv"), \
             patch.object(Session, "handle_user_message", return_value="a candidate's raw reply, never shown directly"), \
             patch.object(GeminiClient, "chat", return_value=judge_reply) as judge_chat:
            at = AppTest.from_file(APP_PATH).run(timeout=30)
            at.segmented_control(key="unified_mode").set_value("LLM Judge").run(timeout=30)
            at = _select_judge(at, DEFAULT_JUDGE)
            at.chat_input[0].set_value("1/(90s+1), delay 13, minimize overshoot.").run(timeout=30)

        self.assertEqual(at.exception[:], [], "a bad turn must not crash the app")
        markdown_values = [m.markdown[0].value for m in at.chat_message if m.markdown]
        self.assertIn("Tyreus-Luyben is the better pick.", markdown_values)
        self.assertNotIn("a candidate's raw reply, never shown directly", markdown_values)
        judge_chat.assert_called_once()
        self.assertIsNone(judge_chat.call_args.kwargs.get("tools"))

    def test_one_failing_candidate_does_not_abort_the_round(self):
        """A real provider error (e.g. Gemini's actual capacity 503) hits
        exactly one of the two default candidates (Claude). The round must
        still complete: the judge still replies, using ChatGPT's answer,
        and the failure shows up in the transparency expander rather than
        the whole turn falling back to a generic top-level error."""
        from supervisor_llm_anthropic import AnthropicClient
        from supervisor_llm_openai import OpenAIClient
        from supervisor_llm_gemini import GeminiClient

        openai_reply = SimpleNamespace(message=SimpleNamespace(content="I recommend Tyreus-Luyben.", tool_calls=None))
        judge_reply = SimpleNamespace(message=SimpleNamespace(content="Only ChatGPT responded; going with its pick.", tool_calls=None))
        with patch.dict(os.environ, ALL_KEYS_ENV, clear=True), \
             patch("streamlit_llm_panel.load_dotenv"), \
             patch.object(AnthropicClient, "chat", side_effect=RuntimeError("This model is currently experiencing high demand.")), \
             patch.object(OpenAIClient, "chat", return_value=openai_reply), \
             patch.object(GeminiClient, "chat", return_value=judge_reply):
            at = AppTest.from_file(APP_PATH).run(timeout=30)
            at.segmented_control(key="unified_mode").set_value("LLM Judge").run(timeout=30)
            at = _select_judge(at, DEFAULT_JUDGE)
            at.chat_input[0].set_value("1/(90s+1), delay 13, minimize overshoot.").run(timeout=30)

        self.assertEqual(at.exception[:], [], "one candidate's provider error must not crash the app")
        markdown_values = [m.markdown[0].value for m in at.chat_message if m.markdown]
        self.assertIn("Only ChatGPT responded; going with its pick.", markdown_values)

        captions = [c.value for c in at.caption]
        self.assertTrue(
            any("failed to respond" in c and "high demand" in c for c in captions), captions)

    def test_transparency_expander_groups_calls_and_finalized_pick_per_candidate(self):
        from supervisor_session_pid import Session
        from supervisor_llm_gemini import GeminiClient

        judge_reply = SimpleNamespace(message=SimpleNamespace(content="Tyreus-Luyben is the better pick.", tool_calls=None))

        def fake_handle_user_message(self, text):
            # Mimic what a real Session.handle_user_message run leaves
            # behind -- a finalize_recommendation tool call plus a final
            # reply -- so summarize_candidate_round() has something real
            # to report per candidate.
            from types import SimpleNamespace as NS
            call = NS(function=NS(name="finalize_recommendation",
                                   arguments={"method_name": "Tyreus-Luyben", "rationale": "lowest overshoot"}))
            self.messages.append(NS(content="", tool_calls=[call]))
            reply_text = "I recommend Tyreus-Luyben."
            self.messages.append(NS(content=reply_text, tool_calls=None))
            return reply_text

        with patch.dict(os.environ, ALL_KEYS_ENV, clear=True), \
             patch("streamlit_llm_panel.load_dotenv"), \
             patch.object(Session, "handle_user_message", fake_handle_user_message), \
             patch.object(GeminiClient, "chat", return_value=judge_reply):
            at = AppTest.from_file(APP_PATH).run(timeout=30)
            at.segmented_control(key="unified_mode").set_value("LLM Judge").run(timeout=30)
            at = _select_judge(at, DEFAULT_JUDGE)
            at.chat_input[0].set_value("1/(90s+1), delay 13, minimize overshoot.").run(timeout=30)

        self.assertEqual(at.exception[:], [])
        captions = [c.value for c in at.caption]
        self.assertTrue(any("recommends" in c and "Tyreus-Luyben" in c for c in captions), captions)
        self.assertTrue(any("finalize_recommendation" in c for c in captions), captions)
        # Grouped one block per candidate -- both default candidates'
        # labels appear as their own caption headers.
        self.assertTrue(any("Claude (Anthropic)" in c for c in captions), captions)
        self.assertTrue(any("ChatGPT (OpenAI)" in c for c in captions), captions)

    def test_a_broken_plot_drain_does_not_crash_or_lose_the_reply(self):
        from supervisor_session_pid import Session
        from supervisor_llm_gemini import GeminiClient

        judge_reply = SimpleNamespace(message=SimpleNamespace(content="Final answer.", tool_calls=None))
        with patch.dict(os.environ, ALL_KEYS_ENV, clear=True), \
             patch("streamlit_llm_panel.load_dotenv"), \
             patch.object(Session, "handle_user_message", return_value="stub"), \
             patch.object(GeminiClient, "chat", return_value=judge_reply), \
             patch("streamlit_llm_panel._drain_plot_calls", side_effect=RuntimeError("boom")):
            at = AppTest.from_file(APP_PATH).run(timeout=30)
            at.segmented_control(key="unified_mode").set_value("LLM Judge").run(timeout=30)
            at = _select_judge(at, DEFAULT_JUDGE)
            at.chat_input[0].set_value("hello").run(timeout=30)

        self.assertEqual(at.exception[:], [])
        markdown_values = [m.markdown[0].value for m in at.chat_message if m.markdown]
        self.assertIn("Final answer.", markdown_values)


if __name__ == "__main__":
    unittest.main()
