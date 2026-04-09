from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from config import get_settings
from gmail import upsert_thread_draft

SYSTEM_PROMPT = (
    "You are an email assistant for Google Chat. Summarize recent Gmail from the last 24 hours into "
    "high, medium, and low urgency buckets. Draft replies only for clearly reply-worthy emails such as "
    "direct asks, meeting invites, approval requests, or urgent follow-ups. Never draft replies for "
    "status updates, FYIs, newsletters, or other informational mail. Use the user's saved reply tone "
    "when writing drafts. Include reminders only when the user has enabled reminders and only for "
    "Gmail-derived same-day meetings, deadlines, or similarly time-sensitive obligations. "
    "Be concise, practical, and action-oriented."
)

MAX_THREAD_DRAFTS_PER_SUMMARY = 3
THREAD_DRAFT_TIMEOUT_SECONDS = 2.5

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
    gmail_message_id: str | None = None
    thread_id: str | None = None
    sender: str
    reply_to: str | None = None
    subject: str
    snippet: str
    received_at: str | None = None
    message_id_header: str | None = None
    references: str | None = None
    in_reply_to: str | None = None


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
    time_note: str | None = None
    reply_needed: bool = True


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
    draft_writing_style: str = "friendly, concise, and professional"
    draft_replies_high: bool = True
    draft_replies_medium: bool = True
    draft_replies_low: bool = False
    include_reminders: bool = True


class RenderedDraftItem(BaseModel):
    number: int
    urgency: str
    subject: str
    sender: str
    sender_email: str | None = None
    reply_to: str | None = None
    summary: str
    time_note: str | None = None
    reply_needed: bool
    draft_reply: str | None = None
    compose_url: str | None = None
    thread_url: str | None = None
    thread_id: str | None = None
    gmail_message_id: str | None = None
    message_id_header: str | None = None
    references: str | None = None
    draft_id: str | None = None
    tweak_hint: str | None = None


class SummaryResult(BaseModel):
    text: str
    drafts: list[RenderedDraftItem] = Field(default_factory=list)


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
        draft_writing_style=str(user.get("draft_writing_style", user.get("reply_tone", "friendly, concise, and professional"))),
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
        "all": "Cover all relevant recent emails from the last 24 hours.",
        "urgent": "Emphasize urgent items, deadlines, and reply-needed emails from the last 24 hours.",
        "action-items": "Emphasize concrete actions the user likely needs to take from the last 24 hours.",
        "meetings": "Emphasize meeting-related emails, scheduling, and timing-sensitive items from the last 24 hours.",
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
            f"Draft writing style: {preferences.draft_writing_style}.",
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


def _extract_email_address(sender: str) -> str | None:
    match = re.search(r"<([^>]+)>", sender)
    if match:
        return match.group(1).strip().lower()
    sender = sender.strip().lower()
    return sender if "@" in sender else None


def _compose_url(to: str | None, subject: str, body: str) -> str | None:
    if not to:
        return None
    encoded_to = quote_plus(to)
    encoded_subject = quote_plus(subject)
    encoded_body = quote_plus(body)
    return f"https://mail.google.com/mail/?view=cm&fs=1&tf=cm&to={encoded_to}&su={encoded_subject}&body={encoded_body}"


def _normalize_match_value(value: str | None) -> str:
    return (value or "").strip().casefold()


def _find_source_email(
    available_emails: list[dict[str, Any]],
    *,
    sender: str,
    subject: str,
) -> dict[str, Any] | None:
    normalized_sender = _normalize_match_value(sender)
    normalized_subject = _normalize_match_value(subject)
    for index, email in enumerate(available_emails):
        if (
            _normalize_match_value(str(email.get("sender"))) == normalized_sender
            and _normalize_match_value(str(email.get("subject"))) == normalized_subject
        ):
            return available_emails.pop(index)
    return None


def _display_sender(sender: str) -> str:
    match = re.search(r"^(.+?)\s*<([^>]+)>$", sender.strip())
    if match:
        name = match.group(1).strip()
        email = match.group(2).strip()
        domain = email.split("@", 1)[1] if "@" in email else email
        org = domain.split(".", 1)[0].replace("-", " ").title()
        if name:
            return f"{name} @ {org}"
        return email
    return sender.strip()


def _format_date_heading(user: dict[str, Any] | None) -> str:
    timezone_name = str((user or {}).get("summary_timezone", "UTC"))
    now = datetime.now(ZoneInfo(timezone_name))
    return f"{now:%b} {now.day}, {now:%Y}"


def _merge_style_note(current: str, instruction: str) -> str:
    current = current.strip()
    instruction = instruction.strip()
    if not current:
        return instruction
    if not instruction:
        return current
    if instruction.lower() in current.lower():
        return current
    return f"{current}; {instruction}"


def apply_draft_style_tweak(current_style: str, instruction: str) -> str:
    instruction = instruction.strip()
    normalized = instruction.lower()
    if any(keyword in normalized for keyword in ("warmer", "warm", "friendlier", "friendly")):
        return _merge_style_note(current_style, "warmer and more friendly")
    if any(keyword in normalized for keyword in ("direct", "shorter", "concise", "brief")):
        return _merge_style_note(current_style, "more direct and concise")
    if any(keyword in normalized for keyword in ("professional", "formal")):
        return _merge_style_note(current_style, "more professional and polished")
    return _merge_style_note(current_style, instruction)


def _looks_like_update_or_fyi(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "status update",
            "weekly update",
            "project update",
            "update",
            "fyi",
            "for your information",
            "newsletter",
            "digest",
            "announcement",
            "recap",
            "summary",
            "no action needed",
            "no reply needed",
            "informational",
            "keeping you posted",
            "keeping you in the loop",
        )
    )


def _looks_like_reply_worthy(text: str) -> bool:
    lowered = text.lower()
    return any(
        phrase in lowered
        for phrase in (
            "meeting invite",
            "meeting",
            "invite",
            "invitation",
            "can you",
            "could you",
            "please",
            "need your",
            "your input",
            "approval",
            "approve",
            "confirm",
            "response needed",
            "action required",
            "reply",
            "follow up",
            "follow-up",
            "schedule",
            "availability",
            "reschedule",
            "urgent",
            "asap",
            "by eod",
            "decision",
            "sign off",
            "review",
            "deadline",
        )
    )


def _is_reply_worthy_item(item: BucketItem) -> bool:
    text = " ".join(
        part
        for part in (
            item.sender,
            item.subject,
            item.summary,
            item.time_note or "",
            item.draft_reply or "",
        )
        if part
    )
    if not item.reply_needed or not item.draft_reply:
        return False
    if _looks_like_update_or_fyi(text):
        return False
    return _looks_like_reply_worthy(text)


def _bucket_header(urgency: str) -> str:
    return {
        "high": "🔴 *Urgent*",
        "medium": "🟡 *Action needed*",
        "low": "⚪ *Low priority*",
    }.get(urgency, f"*{urgency.title()}*")


def build_summary_prompt(
    emails: list[dict[str, str]],
    preferences: SummaryPreferences,
    reminder_candidates: list[ReminderCandidate],
) -> str:
    normalized_emails = [EmailSummaryInput.model_validate(email).model_dump() for email in emails]
    reminder_json = [candidate.model_dump() for candidate in reminder_candidates]
    return (
        "Classify these recent emails from the last 24 hours into three buckets: high, medium, and low urgency. "
        "For each bucket, write one concise summary sentence and itemize the most relevant emails in the same order as the input. "
        "For each item, provide sender, subject, summary, a short time_note if the email mentions a meeting time or deadline, "
        "and a draft_reply only when the item is clearly reply-worthy. Use null for draft_reply and reply_needed=false when no reply is needed. "
        "Draft replies should be limited to direct asks, meeting invites, approval requests, or urgent follow-ups. "
        "Do not include draft replies for updates, FYIs, newsletters, or similar informational mail. "
        "For high and medium urgency, include a short draft reply only when the item is clearly reply-worthy and draft replies are enabled. "
        "Do not include draft replies in low urgency. "
        "For reminders, convert only the provided reminder candidates into short same-day reminder lines if reminders are enabled. "
        "Keep the output practical and specific for Google Chat.\n\n"
        f"User preferences:\n{_preference_prompt(preferences)}\n\n"
        f"Emails:\n{json.dumps(normalized_emails, indent=2)}\n\n"
        f"Reminder candidates:\n{json.dumps(reminder_json, indent=2)}"
    )


def render_summary_digest(
    digest: SummaryDigest,
    preferences: SummaryPreferences,
    *,
    user: dict[str, Any] | None = None,
) -> SummaryResult:
    sections: list[str] = [f"📬 *Email Summary — {_format_date_heading(user)}*"]
    rendered_drafts: list[RenderedDraftItem] = []
    item_number = 1
    urgency_order = [
        ("high", digest.high, preferences.draft_replies_high),
        ("medium", digest.medium, preferences.draft_replies_medium),
        ("low", digest.low, False),
    ]

    for urgency, section, show_drafts in urgency_order:
        sections.append(_bucket_header(urgency))
        sections.append(f"- {section.summary.strip()}" if section.summary.strip() else "- None")
        if not section.items:
            continue
        for item in section.items:
            title = f"*{item_number}. {item.subject.strip()} — {_display_sender(item.sender)}*"
            if item.time_note:
                title += f"\n{item.time_note.strip()}"
            sections.append(title)

            email_address = _extract_email_address(item.sender)
            reply_needed = _is_reply_worthy_item(item)
            if reply_needed and show_drafts:
                sections.append("✍️ Draft reply:")
                sections.append(f"> {item.draft_reply.strip()}")
                compose_url = _compose_url(email_address, f"Re: {item.subject.strip()}", item.draft_reply.strip())
                if compose_url:
                    sections.append(f"↳ Send this: {compose_url}")
                sections.append(f'💬 Reply "tweak {item_number}: [your instruction]" to update this draft')
            else:
                sections.append(f"> No reply needed — {item.summary.strip()}")

            rendered_drafts.append(
                RenderedDraftItem(
                    number=item_number,
                    urgency=urgency,
                    subject=item.subject,
                    sender=item.sender,
                    sender_email=email_address,
                    reply_to=None,
                    summary=item.summary,
                    time_note=item.time_note,
                    reply_needed=reply_needed,
                    draft_reply=item.draft_reply if reply_needed else None,
                    compose_url=_compose_url(
                        email_address,
                        f"Re: {item.subject.strip()}",
                        item.draft_reply.strip() if reply_needed and item.draft_reply else item.summary.strip(),
                    )
                    if reply_needed
                    else None,
                    thread_url=None,
                    thread_id=None,
                    gmail_message_id=None,
                    message_id_header=None,
                    references=None,
                    draft_id=None,
                    tweak_hint=f'tweak {item_number}: [your instruction]' if reply_needed else None,
                )
            )
            item_number += 1

    if preferences.include_reminders and digest.reminders:
        reminder_lines = ["⏰ *Reminders*"]
        reminder_lines.extend(f"- {reminder.strip()}" for reminder in digest.reminders)
        sections.append("\n".join(reminder_lines))

    return SummaryResult(text="\n\n".join(section for section in sections if section).strip(), drafts=rendered_drafts)


async def summarize_emails(
    emails: list[dict[str, str]],
    *,
    user: dict[str, Any],
) -> SummaryResult:
    if not emails:
        return SummaryResult(text="No recent emails from the last 24 hours.", drafts=[])

    preferences = _preferences_from_user(user)
    reminder_candidates = (
        build_reminder_candidates(emails, timezone_name=str(user.get("summary_timezone", "UTC")))
        if preferences.include_reminders
        else []
    )
    prompt = build_summary_prompt(emails, preferences, reminder_candidates)

    result = await _build_agent().run(prompt)
    rendered = render_summary_digest(result.output, preferences, user=user)
    return await attach_thread_draft_links(rendered, emails=emails, user=user)


async def attach_thread_draft_links(
    summary: SummaryResult,
    *,
    emails: list[dict[str, Any]],
    user: dict[str, Any],
) -> SummaryResult:
    available_emails = [dict(email) for email in emails]
    updated_text = summary.text
    updated_drafts: list[RenderedDraftItem] = []
    attempted_drafts = 0

    for draft in summary.drafts:
        updated_draft = draft.model_copy(deep=True)
        if not updated_draft.reply_needed or not updated_draft.draft_reply:
            updated_drafts.append(updated_draft)
            continue

        source_email = _find_source_email(
            available_emails,
            sender=updated_draft.sender,
            subject=updated_draft.subject,
        )
        if not source_email:
            updated_drafts.append(updated_draft)
            continue

        updated_draft.reply_to = str(source_email.get("reply_to") or source_email.get("sender") or "")
        updated_draft.thread_id = source_email.get("thread_id")
        updated_draft.gmail_message_id = source_email.get("gmail_message_id")
        updated_draft.message_id_header = source_email.get("message_id_header")
        updated_draft.references = source_email.get("references")
        if updated_draft.thread_id:
            updated_draft.thread_url = f"https://mail.google.com/mail/u/0/#all/{updated_draft.thread_id}"

        if attempted_drafts >= MAX_THREAD_DRAFTS_PER_SUMMARY:
            if updated_draft.thread_url:
                fallback_link = _compose_url(
                    updated_draft.sender_email,
                    f"Re: {updated_draft.subject.strip()}",
                    updated_draft.draft_reply,
                )
                if fallback_link:
                    updated_text = updated_text.replace(fallback_link, updated_draft.thread_url, 1)
                updated_text = updated_text.replace("↳ Send this:", "↳ Open thread:", 1)
                updated_draft.compose_url = updated_draft.thread_url
            updated_drafts.append(updated_draft)
            continue

        attempted_drafts += 1
        try:
            draft_link = await asyncio.wait_for(
                upsert_thread_draft(
                    user,
                    email=source_email,
                    draft_reply=updated_draft.draft_reply,
                ),
                timeout=THREAD_DRAFT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            draft_link = None

        if not draft_link:
            if updated_draft.thread_url:
                fallback_link = _compose_url(
                    updated_draft.sender_email,
                    f"Re: {updated_draft.subject.strip()}",
                    updated_draft.draft_reply,
                )
                if fallback_link:
                    updated_text = updated_text.replace(fallback_link, updated_draft.thread_url, 1)
                updated_text = updated_text.replace("↳ Send this:", "↳ Open thread:", 1)
                updated_draft.compose_url = updated_draft.thread_url
            updated_drafts.append(updated_draft)
            continue

        updated_draft.draft_id = draft_link.get("draft_id")
        updated_draft.thread_id = draft_link.get("thread_id") or updated_draft.thread_id
        updated_draft.thread_url = draft_link.get("thread_url")
        updated_draft.compose_url = draft_link.get("thread_url") or updated_draft.compose_url

        fallback_link = _compose_url(
            updated_draft.sender_email,
            f"Re: {updated_draft.subject.strip()}",
            updated_draft.draft_reply,
        )
        if fallback_link and updated_draft.thread_url:
            updated_text = updated_text.replace(fallback_link, updated_draft.thread_url, 1)
        updated_text = updated_text.replace("↳ Send this:", "↳ Open thread:", 1)
        updated_drafts.append(updated_draft)

    return SummaryResult(text=updated_text, drafts=updated_drafts)


async def rewrite_draft_reply(
    *,
    current_reply: str,
    sender: str,
    subject: str,
    instruction: str,
    writing_style: str,
    reply_tone: str,
) -> str:
    settings = get_settings()
    agent = Agent(
        f"google-gla:{settings.gemini_model}",
        system_prompt=(
            "You rewrite email reply drafts for Google Chat users. Keep the reply natural, helpful, and concise. "
            "Preserve the intent of the original reply, but apply the requested tweak and the user's saved writing style. "
            "Return only the rewritten reply text."
        ),
    )
    prompt = (
        f"Original sender: {sender}\n"
        f"Original subject: {subject}\n"
        f"Current reply tone: {reply_tone}\n"
        f"Current writing style: {writing_style}\n"
        f"User instruction: {instruction}\n"
        f"Current draft reply:\n{current_reply}\n"
    )
    result = await agent.run(prompt)
    return result.output.strip()
