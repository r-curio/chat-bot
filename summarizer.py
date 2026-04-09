from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from config import get_settings

SYSTEM_PROMPT = (
    "You are an email assistant for Google Chat. Summarize unread email into high, medium, and low "
    "urgency buckets. Draft replies only for the buckets the user has enabled. Use the user's saved "
    "reply tone when writing drafts. Include reminders only when the user has enabled reminders and "
    "only for Gmail-derived same-day meetings, deadlines, or similarly time-sensitive obligations. "
    "Be concise, practical, and action-oriented."
)

MEETING_HINTS = (
    "meeting",
    "call",
    "sync",
    "standup",
    "catch up",
    "catch-up",
    "appointment",
    "interview",
    "review",
    "briefing",
    "demo",
    "check-in",
    "check in",
)
TIME_HINTS = (
    "today",
    "this morning",
    "this afternoon",
    "tonight",
    "tomorrow",
    "later today",
    "before noon",
    "after lunch",
    "at ",
    "am",
    "pm",
)


class EmailSummaryInput(BaseModel):
    sender: str
    subject: str
    snippet: str
    received_at: str | None = None


class ReminderCandidate(BaseModel):
    sender: str
    subject: str
    snippet: str
    reason: str


class BucketItem(BaseModel):
    sender: str
    subject: str
    summary: str
    draft_reply: str | None = None


class BucketSection(BaseModel):
    summary: str
    items: list[BucketItem] = Field(default_factory=list)


class SummaryDigest(BaseModel):
    high: BucketSection
    medium: BucketSection
    low: BucketSection
    reminders: list[str] = Field(default_factory=list)


class SummaryPreferences(BaseModel):
    summary_style: str = "brief"
    summary_length: str = "medium"
    summary_focus: str = "all"
    summary_prompt_mode: str = "structured"
    reply_tone: str = "friendly, concise, and professional"
    draft_replies_high: bool = True
    draft_replies_medium: bool = True
    draft_replies_low: bool = False
    include_reminders: bool = True


def _build_agent() -> Agent:
    settings = get_settings()
    return Agent(
        f"google-gla:{settings.gemini_model}",
        system_prompt=SYSTEM_PROMPT,
        output_type=SummaryDigest,
    )


def _preferences_from_user(user: dict[str, Any]) -> SummaryPreferences:
    return SummaryPreferences(
        summary_style=str(user.get("summary_style", "brief")),
        summary_length=str(user.get("summary_length", "medium")),
        summary_focus=str(user.get("summary_focus", "all")),
        summary_prompt_mode=str(user.get("summary_prompt_mode", "structured")),
        reply_tone=str(user.get("reply_tone", "friendly, concise, and professional")),
        draft_replies_high=bool(user.get("draft_replies_high", 1)),
        draft_replies_medium=bool(user.get("draft_replies_medium", 1)),
        draft_replies_low=bool(user.get("draft_replies_low", 0)),
        include_reminders=bool(user.get("include_reminders", 1)),
    )


def _preference_prompt(preferences: SummaryPreferences) -> str:
    style_instructions = {
        "brief": "Keep the summary compact and skimmable.",
        "detailed": "Include a bit more context for each email.",
        "executive": "Write for a busy executive who wants the key decision or action fast.",
        "bullets": "Favor terse bullet points over narrative phrasing.",
    }
    length_instructions = {
        "short": "Use the minimum detail needed to understand what matters.",
        "medium": "Balance coverage and brevity.",
        "long": "Include fuller detail where useful, but keep it readable in Chat.",
    }
    focus_instructions = {
        "all": "Cover all relevant unread emails.",
        "urgent": "Emphasize urgent items, deadlines, and reply-needed emails.",
        "action-items": "Emphasize concrete actions the user likely needs to take.",
        "meetings": "Emphasize meeting-related emails, scheduling, and timing-sensitive items.",
    }
    draft_scope = ", ".join(
        scope
        for scope, enabled in (
            ("high", preferences.draft_replies_high),
            ("medium", preferences.draft_replies_medium),
        )
        if enabled
    )
    if not draft_scope:
        draft_scope = "none"

    reminder_text = (
        "Include a reminders section for same-day Gmail-derived meetings or deadlines."
        if preferences.include_reminders
        else "Do not include a reminders section."
    )

    return " ".join(
        [
            style_instructions.get(preferences.summary_style, style_instructions["brief"]),
            length_instructions.get(preferences.summary_length, length_instructions["medium"]),
            focus_instructions.get(preferences.summary_focus, focus_instructions["all"]),
            f"Prompt mode: {preferences.summary_prompt_mode}.",
            f"Draft replies using this tone: {preferences.reply_tone}.",
            f"Draft replies are enabled for: {draft_scope}.",
            reminder_text,
        ]
    )


def _received_today(received_at: str | None, timezone_name: str) -> bool:
    if not received_at:
        return False
    try:
        received = datetime.fromisoformat(received_at)
    except ValueError:
        return False
    local_now = datetime.now(ZoneInfo(timezone_name))
    return received.astimezone(ZoneInfo(timezone_name)).date() == local_now.date()


def _is_reminder_candidate(email: dict[str, str], timezone_name: str) -> bool:
    text = f"{email.get('subject', '')} {email.get('snippet', '')}".lower()
    if not any(hint in text for hint in MEETING_HINTS):
        return False
    return (
        any(hint in text for hint in TIME_HINTS)
        or bool(re.search(r"\b\d{1,2}(:\d{2})?\s?(am|pm)\b", text))
        or _received_today(email.get("received_at"), timezone_name)
    )


def build_reminder_candidates(
    emails: list[dict[str, str]],
    *,
    timezone_name: str = "UTC",
) -> list[ReminderCandidate]:
    candidates: list[ReminderCandidate] = []
    for email in emails:
        if not _is_reminder_candidate(email, timezone_name):
            continue
        text = f"{email.get('subject', '')} {email.get('snippet', '')}".strip()
        candidates.append(
            ReminderCandidate(
                sender=email.get("sender", "Unknown sender"),
                subject=email.get("subject", "(No subject)"),
                snippet=email.get("snippet", ""),
                reason=text,
            )
        )
    return candidates


def build_summary_prompt(
    emails: list[dict[str, str]],
    preferences: SummaryPreferences,
    reminder_candidates: list[ReminderCandidate],
) -> str:
    normalized_emails = [EmailSummaryInput.model_validate(email).model_dump() for email in emails]
    reminder_json = [candidate.model_dump() for candidate in reminder_candidates]
    return (
        "Classify these unread emails into three buckets: high, medium, and low urgency. "
        "For each bucket, write one concise summary sentence and itemize the most relevant emails. "
        "For high and medium urgency, include a short draft reply for each item when draft replies are enabled. "
        "Do not include draft replies in low urgency. "
        "For reminders, convert only the provided reminder candidates into short same-day reminder lines if reminders are enabled. "
        "Keep the output practical and specific for Google Chat.\n\n"
        f"User preferences:\n{_preference_prompt(preferences)}\n\n"
        f"Emails:\n{json.dumps(normalized_emails, indent=2)}\n\n"
        f"Reminder candidates:\n{json.dumps(reminder_json, indent=2)}"
    )


def _render_bucket(title: str, section: BucketSection, *, show_drafts: bool) -> list[str]:
    lines = [f"*{title} urgency*", f"- {section.summary.strip()}"]
    if not section.items:
        lines.append("- None")
        return lines

    for item in section.items:
        lines.append(f"- *{item.subject}* from {item.sender}: {item.summary.strip()}")
        if show_drafts and item.draft_reply:
            lines.append(f"  - Draft reply: {item.draft_reply.strip()}")
    return lines


def render_summary_digest(digest: SummaryDigest, preferences: SummaryPreferences) -> str:
    sections: list[str] = []
    sections.append("\n".join(_render_bucket("High", digest.high, show_drafts=preferences.draft_replies_high)))
    sections.append("\n".join(_render_bucket("Medium", digest.medium, show_drafts=preferences.draft_replies_medium)))
    sections.append("\n".join(_render_bucket("Low", digest.low, show_drafts=False)))

    if preferences.include_reminders and digest.reminders:
        reminder_lines = ["*Reminders*"]
        reminder_lines.extend(f"- {reminder.strip()}" for reminder in digest.reminders)
        sections.append("\n".join(reminder_lines))

    return "\n\n".join(section for section in sections if section).strip()


async def summarize_emails(
    emails: list[dict[str, str]],
    *,
    user: dict[str, Any],
) -> str:
    if not emails:
        return "No unread emails right now."

    preferences = _preferences_from_user(user)
    reminder_candidates = (
        build_reminder_candidates(emails, timezone_name=str(user.get("summary_timezone", "UTC")))
        if preferences.include_reminders
        else []
    )
    prompt = build_summary_prompt(emails, preferences, reminder_candidates)

    result = await _build_agent().run(prompt)
    return render_summary_digest(result.output, preferences)
