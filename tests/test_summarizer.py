from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from summarizer import (
    BucketItem,
    BucketSection,
    SummaryDigest,
    SummaryPreferences,
    build_reminder_candidates,
    render_summary_digest,
    summarize_emails,
)


class SummarizerTests(unittest.IsolatedAsyncioTestCase):
    def test_build_reminder_candidates_detects_same_day_meeting_language(self) -> None:
        emails = [
            {
                "sender": "Alice <alice@example.com>",
                "subject": "Team meeting today at 3pm",
                "snippet": "As discussed, let's sync later today.",
            },
            {
                "sender": "Bob <bob@example.com>",
                "subject": "Weekly newsletter",
                "snippet": "No action needed.",
            },
        ]

        candidates = build_reminder_candidates(emails)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].subject, "Team meeting today at 3pm")

    def test_render_summary_digest_includes_only_high_and_medium_drafts(self) -> None:
        digest = SummaryDigest(
            high=BucketSection(
                summary="Two urgent emails need attention.",
                items=[
                    BucketItem(
                        sender="Alice <alice@example.com>",
                        subject="Approve the invoice",
                        summary="Needs approval today.",
                        draft_reply="Hi Alice, I reviewed it and will approve it shortly.",
                    )
                ],
            ),
            medium=BucketSection(
                summary="One medium-priority follow-up is pending.",
                items=[
                    BucketItem(
                        sender="Ben <ben@example.com>",
                        subject="Project update",
                        summary="Reply with the revised timeline.",
                        draft_reply="Thanks Ben, I’ll send the revised timeline today.",
                    )
                ],
            ),
            low=BucketSection(
                summary="Low-priority FYIs only.",
                items=[
                    BucketItem(
                        sender="Nina <nina@example.com>",
                        subject="FYI newsletter",
                        summary="For awareness only.",
                        draft_reply="This should not be shown.",
                    )
                ],
            ),
            reminders=["Team meeting today at 3 PM."],
        )
        preferences = SummaryPreferences(
            reply_tone="friendly, concise, and professional",
            draft_replies_high=True,
            draft_replies_medium=True,
            draft_replies_low=False,
            include_reminders=True,
        )

        rendered = render_summary_digest(digest, preferences)

        self.assertIn("🔴 *Urgent*", rendered.text)
        self.assertIn("🟡 *Action needed*", rendered.text)
        self.assertIn("⚪ *Low priority*", rendered.text)
        self.assertIn("✍️ Draft reply:", rendered.text)
        self.assertIn("Hi Alice, I reviewed it and will approve it shortly.", rendered.text)
        self.assertIn("Thanks Ben, I’ll send the revised timeline today.", rendered.text)
        self.assertNotIn("This should not be shown.", rendered.text)
        self.assertIn("↳ Send this:", rendered.text)
        self.assertIn('💬 Reply "tweak 1: [your instruction]"', rendered.text)
        self.assertIn("*Reminders*", rendered.text)

    async def test_summarize_emails_uses_agent_output_and_user_preferences(self) -> None:
        digest = SummaryDigest(
            high=BucketSection(
                summary="Urgent follow-ups.",
                items=[
                    BucketItem(
                        sender="Alice <alice@example.com>",
                        subject="Approve invoice",
                        summary="Needs action today.",
                        draft_reply="Hi Alice, I’ll approve this shortly.",
                    )
                ],
            ),
            medium=BucketSection(
                summary="Medium-priority requests.",
                items=[
                    BucketItem(
                        sender="Ben <ben@example.com>",
                        subject="Send timeline",
                        summary="Reply with the updated plan.",
                        draft_reply="Thanks Ben, I’ll send the updated plan by EOD.",
                    )
                ],
            ),
            low=BucketSection(summary="FYIs.", items=[]),
            reminders=["Design review today at 2 PM."],
        )
        fake_agent = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(output=digest)))
        user = {
            "summary_style": "brief",
            "summary_length": "medium",
            "summary_focus": "all",
            "summary_prompt_mode": "structured",
            "reply_tone": "warm",
            "draft_replies_high": 1,
            "draft_replies_medium": 1,
            "draft_replies_low": 0,
            "include_reminders": 1,
        }

        with patch("summarizer._build_agent", return_value=fake_agent):
            rendered = await summarize_emails(
                [
                    {
                        "sender": "Alice <alice@example.com>",
                        "subject": "Approve invoice",
                        "snippet": "Needs action today.",
                        "received_at": "2026-04-09T08:00:00+00:00",
                    }
                ],
                user=user,
            )

        self.assertIn("Urgent follow-ups.", rendered.text)
        self.assertIn("✍️ Draft reply:", rendered.text)
        self.assertIn("Hi Alice, I’ll approve this shortly.", rendered.text)
        self.assertIn("*Reminders*", rendered.text)
        self.assertTrue(rendered.drafts)
        self.assertEqual(rendered.drafts[0].number, 1)


if __name__ == "__main__":
    unittest.main()
