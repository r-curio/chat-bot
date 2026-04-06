from __future__ import annotations

import json

from pydantic import BaseModel
from pydantic_ai import Agent

from config import get_settings

SYSTEM_PROMPT = (
    "You are an email assistant. Summarize the user's unread emails grouped by urgency. "
    "Be concise. Format output for Google Chat text messages only: use *bold* section "
    "titles and dash bullets, and do not use Markdown headings, tables, or **bold** syntax."
)


class EmailSummaryInput(BaseModel):
    sender: str
    subject: str
    snippet: str


def _build_agent() -> Agent:
    settings = get_settings()
    return Agent(
        f"google-gla:{settings.gemini_model}",
        system_prompt=SYSTEM_PROMPT,
    )


def _preference_prompt(style: str, length: str, focus: str) -> str:
    style_instructions = {
        "brief": "Keep the summary compact and skimmable.",
        "detailed": "Include a bit more context for each email.",
        "executive": "Write for a busy executive who wants the key decision or action fast.",
        "bullets": "Favor terse bullet points over narrative phrasing.",
    }
    length_instructions = {
        "short": "Use the minimum detail needed to understand what matters.",
        "medium": "Balance coverage and brevity.",
        "long": "Include fuller detail where useful, but stay readable in Chat.",
    }
    focus_instructions = {
        "all": "Cover all relevant unread emails.",
        "urgent": "Emphasize urgent items, deadlines, and reply-needed emails.",
        "action-items": "Emphasize concrete actions the user likely needs to take.",
        "meetings": "Emphasize meeting-related emails, scheduling, and calendar actions.",
    }
    return " ".join(
        [
            style_instructions.get(style, style_instructions["brief"]),
            length_instructions.get(length, length_instructions["medium"]),
            focus_instructions.get(focus, focus_instructions["all"]),
        ]
    )


async def summarize_emails(
    emails: list[dict[str, str]],
    *,
    style: str = "brief",
    length: str = "medium",
    focus: str = "all",
) -> str:
    if not emails:
        return "No unread emails right now."

    normalized_emails = [EmailSummaryInput.model_validate(email).model_dump() for email in emails]
    prompt = (
        "Summarize these unread emails. Group them by urgency using short headings, "
        "then give concise bullet-style plain text under each group. Format strictly "
        "for Google Chat using *bold* headings and '-' bullets. "
        f"{_preference_prompt(style, length, focus)}\n\n"
        f"{json.dumps(normalized_emails, indent=2)}"
    )

    result = await _build_agent().run(prompt)
    return result.output.strip()
