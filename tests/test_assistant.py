from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from assistant import (
    RouteDecision,
    ToolOutcome,
    _parse_days,
    _parse_settime_argument,
    handle_follow_up_message,
    tool_update_summary_preferences,
    route_chat_message,
)


class AssistantRoutingTests(unittest.IsolatedAsyncioTestCase):
    def test_parse_days_normalizes_input(self) -> None:
        self.assertEqual(_parse_days("fri, mon, tue"), "Mon,Tue,Fri")

    def test_parse_settime_argument_validates_timezone(self) -> None:
        self.assertEqual(_parse_settime_argument("/settime 08:30 Asia/Manila"), (8, 30, "Asia/Manila"))
        self.assertIsNone(_parse_settime_argument("/settime 25:30 Asia/Manila"))
        self.assertIsNone(_parse_settime_argument("/settime 08:30 Not/A_Zone"))

    async def test_route_chat_message_executes_a_tool(self) -> None:
        user = {"id": 1, "gmail_token_json": "{\"token\": true}", "summary_hour": 8}
        with (
            patch("assistant.get_chat_conversation_state", new=AsyncMock(return_value=None)),
            patch("assistant.decide_route", new=AsyncMock(return_value=RouteDecision(kind="tool_call", message="", tool_name="summary", tool_args={}))),
            patch("assistant.execute_tool", new=AsyncMock(return_value=ToolOutcome(text="summary text", queue_audio=True))),
            patch("assistant.clear_chat_conversation_state", new=AsyncMock()),
            patch("assistant.save_chat_conversation_state", new=AsyncMock()),
        ):
            outcome = await route_chat_message(
                user=user,
                gchat_user_id="user-1",
                gchat_space_id="space-1",
                scheduler=object(),
                message_text="summarize my unread mail",
            )

        self.assertEqual(outcome.text, "summary text")
        self.assertTrue(outcome.queue_audio)

    async def test_route_chat_message_stores_confirmation(self) -> None:
        user = {"id": 1, "summary_hour": 8}
        with (
            patch("assistant.get_chat_conversation_state", new=AsyncMock(return_value=None)),
            patch(
                "assistant.decide_route",
                new=AsyncMock(
                    return_value=RouteDecision(
                        kind="confirm",
                        message="Update your summary time to 08:00 Asia/Manila?",
                        tool_name="settime",
                        tool_args={"hour": 8, "minute": 0, "timezone": "Asia/Manila"},
                    )
                ),
            ),
            patch("assistant.clear_chat_conversation_state", new=AsyncMock()),
            patch("assistant.save_chat_conversation_state", new=AsyncMock()),
            patch("assistant.execute_tool", new=AsyncMock()),
        ):
            outcome = await route_chat_message(
                user=user,
                gchat_user_id="user-1",
                gchat_space_id="space-1",
                scheduler=object(),
                message_text="move my summary to 8am",
            )

        self.assertEqual(outcome.text, "Update your summary time to 08:00 Asia/Manila?")
        self.assertFalse(outcome.queue_audio)

    async def test_follow_up_yes_executes_pending_action(self) -> None:
        user = {"id": 1}
        pending_state = {
            "kind": "confirm",
            "tool_name": "pause",
            "tool_args": {},
            "message": "Pause scheduled summaries?",
        }
        with patch("assistant.execute_tool", new=AsyncMock(return_value=ToolOutcome(text="Scheduled summaries paused."))):
            outcome, next_state, should_clear = await handle_follow_up_message(
                user=user,
                gchat_user_id="user-1",
                gchat_space_id="space-1",
                scheduler=object(),
                message_text="yes",
                pending_state=pending_state,
            )

        self.assertEqual(outcome.text, "Scheduled summaries paused.")
        self.assertIsNone(next_state)
        self.assertTrue(should_clear)

    async def test_combined_summary_preferences_update(self) -> None:
        user = {"id": 1}
        with patch("assistant.update_user_preferences", new=AsyncMock()) as update_mock:
            outcome = await tool_update_summary_preferences(
                type("Ctx", (), {"user": user})(),
                {"reply_tone": "casual", "draft_scope": "high,medium", "include_reminders": "off"},
            )

        self.assertIn("Updated reply tone to casual", outcome.text)
        update_mock.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
