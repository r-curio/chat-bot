from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Awaitable, Callable, Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from config import get_settings
from db import (
    clear_chat_conversation_state,
    get_chat_conversation_state,
    save_chat_conversation_state,
    update_summary_schedule,
    update_user_preferences,
    add_user_exclusion,
)
from gmail import upsert_thread_draft
from scheduler import reschedule_summary_for_user, run_summary_for_user
from summarizer import (
    SummaryResult,
    apply_draft_style_tweak,
    rewrite_draft_reply,
)

VALID_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
VALID_STYLES = {"brief", "detailed", "executive", "bullets"}
VALID_LENGTHS = {"short", "medium", "long"}
VALID_FOCUS = {"urgent", "action-items", "meetings", "all"}
VALID_REPLY_TONE_EXAMPLES = {
    "casual",
    "friendly",
    "formal",
    "direct",
    "warm",
    "concise",
    "professional",
}

TWEAK_PATTERN = re.compile(r"^\s*tweak\s+(\d+)\s*:\s*(.+?)\s*$", re.IGNORECASE)
DRAFT_STATE_KIND = "summary_drafts"

AFFIRMATIVE_RESPONSES = {
    "yes",
    "y",
    "yeah",
    "yep",
    "ok",
    "okay",
    "sure",
    "do it",
    "confirm",
    "please do",
    "go ahead",
    "proceed",
}
NEGATIVE_RESPONSES = {
    "no",
    "n",
    "nope",
    "cancel",
    "stop",
    "don't",
    "do not",
    "never mind",
    "not now",
}


class RouteDecision(BaseModel):
    kind: Literal["reply", "tool_call", "clarify", "confirm"]
    message: str
    tool_name: str | None = None
    tool_args: dict[str, Any] = Field(default_factory=dict)


class ToolOutcome(BaseModel):
    text: str
    queue_audio: bool = False
    drafts: list[dict[str, Any]] | None = None


@dataclass(slots=True)
class ToolContext:
    user: dict[str, Any]
    scheduler: Any


ToolHandler = Callable[[ToolContext, dict[str, Any]], Awaitable[ToolOutcome]]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    mutating: bool
    handler: ToolHandler


def _help_text() -> str:
    return (
        "*Available Commands*\n\n"
        "*Summaries*\n"
        "- `/summary` Get your Gmail summary from the last 24 hours now.\n"
        "- `/testsummary` Preview a summary using your saved preferences.\n\n"
        "*Schedule*\n"
        "- `/settime HH:MM Area/City` Set your daily summary time and timezone.\n"
        "  Example: `/settime 08:00 Asia/Manila`\n"
        "- `/setdays Mon,Tue,Fri` Choose delivery days.\n"
        "- `/pause` Pause scheduled summaries.\n"
        "- `/resume` Resume scheduled summaries.\n\n"
        "*Preferences*\n"
        "- `/settings` Show your current preferences.\n"
        "- `/style brief|detailed|executive|bullets` Set summary style.\n"
        "- `/length short|medium|long` Set summary length.\n"
        "- `/focus urgent|action-items|meetings|all` Set summary focus.\n"
        "- `/replytone casual|friendly|formal|direct|warm|concise` Set draft reply tone.\n"
        "- `/drafts high|high,medium|off` Set which urgency buckets get draft replies.\n"
        "- `/reminders on|off` Include Gmail-based same-day reminders.\n"
        "- `/exclude sender@company.com` Exclude a sender or domain.\n\n"
        "*Utilities*\n"
        "- `/timezones [filter]` List timezone names, optionally filtered.\n"
        "  Example: `/timezones Asia`\n"
        "- `/help` Show this command list."
    )


def _parse_days(argument: str) -> str | None:
    if not argument:
        return None
    days = [part.strip().title() for part in argument.split(",") if part.strip()]
    if not days or any(day not in VALID_DAYS for day in days):
        return None

    ordered = [day for day in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun") if day in set(days)]
    return ",".join(ordered)


def _format_settings(user: dict[str, Any]) -> str:
    exclusions = user.get("exclusions", [])
    exclusion_text = ", ".join(exclusions) if exclusions else "None"
    paused_text = "Yes" if user.get("is_paused") else "No"
    prompt_mode = user.get("summary_prompt_mode", "structured")
    reply_tone = user.get("reply_tone", "friendly, concise, and professional")
    draft_writing_style = user.get(
        "draft_writing_style",
        user.get("reply_tone", "friendly, concise, and professional"),
    )
    drafts = [
        bucket
        for bucket, enabled in (
            ("high", user.get("draft_replies_high", 1)),
            ("medium", user.get("draft_replies_medium", 1)),
            ("low", user.get("draft_replies_low", 0)),
        )
        if enabled
    ]
    draft_text = ", ".join(drafts) if drafts else "None"
    reminder_text = "On" if user.get("include_reminders", 1) else "Off"
    return (
        "Your current settings:\n"
        f"- Time: {int(user.get('summary_hour', 8)):02d}:{int(user.get('summary_minute', 0)):02d} "
        f"{user.get('summary_timezone', get_settings().timezone)}\n"
        f"- Paused: {paused_text}\n"
        f"- Days: {user.get('summary_days', 'Mon,Tue,Wed,Thu,Fri,Sat,Sun')}\n"
        f"- Style: {user.get('summary_style', 'brief')}\n"
        f"- Length: {user.get('summary_length', 'medium')}\n"
        f"- Focus: {user.get('summary_focus', 'all')}\n"
        f"- Prompt mode: {prompt_mode}\n"
        f"- Reply tone: {reply_tone}\n"
        f"- Draft writing style: {draft_writing_style}\n"
        f"- Draft replies: {draft_text}\n"
        f"- Reminders: {reminder_text}\n"
        f"- Exclusions: {exclusion_text}"
    )


def _timezones_text(filter_text: str = "") -> str:
    from zoneinfo import available_timezones

    matching = sorted(
        timezone
        for timezone in available_timezones()
        if not filter_text or filter_text.lower() in timezone.lower()
    )
    if not matching:
        return f"No timezones matched '{filter_text}'.\nTry /timezones Asia or /timezones America."

    if filter_text:
        return f"Matching timezones for '{filter_text}' ({len(matching)}):\n" + "\n".join(matching)
    return f"Available timezones ({len(matching)}):\n" + "\n".join(matching)


def _oauth_link(install_id: int) -> str:
    from urllib.parse import urlencode, urlsplit, urlunsplit

    settings = get_settings()
    callback_parts = urlsplit(settings.callback_url)
    base_path = callback_parts.path.removesuffix("/auth/callback").rstrip("/")
    start_path = f"{base_path}/auth/start" if base_path else "/auth/start"
    query = urlencode({"install_id": install_id})
    return urlunsplit((callback_parts.scheme, callback_parts.netloc, start_path, query, ""))


def _parse_settime_argument(argument_text: str) -> tuple[int, int, str] | None:
    import re
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    pattern = re.compile(r"^/settime\s+(\d{1,2}):(\d{2})\s+([A-Za-z0-9_\-+]+(?:/[A-Za-z0-9_\-+]+)+)$")
    match = pattern.fullmatch(argument_text)
    if not match:
        return None

    hour = int(match.group(1))
    minute = int(match.group(2))
    timezone = match.group(3)
    if hour > 23 or minute > 59:
        return None

    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        return None

    return hour, minute, timezone


def _is_affirmative(message_text: str) -> bool:
    normalized = message_text.strip().lower()
    return normalized in AFFIRMATIVE_RESPONSES


def _is_negative(message_text: str) -> bool:
    normalized = message_text.strip().lower()
    return normalized in NEGATIVE_RESPONSES


def _normalize_reply_tone(argument: str) -> str | None:
    tone = argument.strip()
    if not tone:
        return None
    lowered = tone.lower()
    if lowered in VALID_REPLY_TONE_EXAMPLES:
        return lowered
    return tone


def _parse_draft_scope(argument: str) -> tuple[bool, bool, bool] | None:
    normalized = argument.strip().lower()
    if not normalized:
        return None
    if normalized in {"off", "none", "no", "disable"}:
        return False, False, False
    if normalized in {"high,medium", "high, medium", "high medium", "default"}:
        return True, True, False
    if normalized in {"high"}:
        return True, False, False
    if normalized in {"medium"}:
        return False, True, False

    tokens = {part for part in re.split(r"[,\s/]+", normalized) if part}
    valid_tokens = {"high", "medium"}
    if not tokens or not tokens <= valid_tokens:
        return None
    return ("high" in tokens, "medium" in tokens, False)


def _parse_bool_argument(argument: str) -> bool | None:
    normalized = argument.strip().lower()
    if normalized in {"on", "yes", "true", "enable", "enabled"}:
        return True
    if normalized in {"off", "no", "false", "disable", "disabled"}:
        return False
    return None


def _summary_draft_state(summary: SummaryResult) -> dict[str, Any]:
    return {
        "kind": DRAFT_STATE_KIND,
        "drafts": [draft.model_dump() for draft in summary.drafts],
    }


async def save_summary_draft_state(
    *,
    user: dict[str, Any],
    summary: SummaryResult,
) -> None:
    if not summary.drafts:
        await clear_chat_conversation_state(user["gchat_user_id"], user["gchat_space_id"])
        return
    await save_chat_conversation_state(
        user["gchat_user_id"],
        user["gchat_space_id"],
        _summary_draft_state(summary),
    )


async def _get_summary_draft_state(user: dict[str, Any]) -> dict[str, Any] | None:
    state = await get_chat_conversation_state(user["gchat_user_id"], user["gchat_space_id"])
    if not state or state.get("kind") != DRAFT_STATE_KIND:
        return None
    return state


def _find_tweak_target(state: dict[str, Any], draft_number: int) -> dict[str, Any] | None:
    for draft in state.get("drafts", []):
        if int(draft.get("number", 0)) == draft_number:
            return draft
    return None


async def tool_summary(context: ToolContext, _: dict[str, Any]) -> ToolOutcome:
    if not context.user.get("gmail_token_json"):
        return ToolOutcome(
            text=(
                "Connect Gmail first to enable summaries:\n"
                f"{_oauth_link(int(context.user['id']))}"
            ),
        )

    summary = await run_summary_for_user(context.user)
    return ToolOutcome(
        text=summary.text,
        queue_audio=True,
        drafts=[draft.model_dump() for draft in summary.drafts],
    )


async def tool_test_summary(context: ToolContext, _: dict[str, Any]) -> ToolOutcome:
    if not context.user.get("gmail_token_json"):
        return ToolOutcome(
            text=(
                "Connect Gmail first to enable summaries:\n"
                f"{_oauth_link(int(context.user['id']))}"
            ),
        )

    summary = await run_summary_for_user(context.user)
    return ToolOutcome(
        text=f"*Preview*\n{summary.text}",
        drafts=[draft.model_dump() for draft in summary.drafts],
    )


async def tool_settings(context: ToolContext, _: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(text=_format_settings(context.user))


async def tool_set_time(context: ToolContext, args: dict[str, Any]) -> ToolOutcome:
    hour = int(args["hour"])
    minute = int(args["minute"])
    timezone = str(args["timezone"])
    await update_summary_schedule(
        int(context.user["id"]),
        summary_hour=hour,
        summary_minute=minute,
        summary_timezone=timezone,
    )
    await reschedule_summary_for_user(context.scheduler, int(context.user["id"]))
    return ToolOutcome(
        text=(
            f"Daily summary time updated to {hour:02d}:{minute:02d} {timezone}.\n"
            "Use /settings to review your preferences."
        )
    )


async def tool_pause(context: ToolContext, _: dict[str, Any]) -> ToolOutcome:
    await update_user_preferences(int(context.user["id"]), is_paused=True)
    return ToolOutcome(text="Scheduled summaries paused.")


async def tool_resume(context: ToolContext, _: dict[str, Any]) -> ToolOutcome:
    await update_user_preferences(int(context.user["id"]), is_paused=False)
    return ToolOutcome(text="Scheduled summaries resumed.")


async def tool_set_days(context: ToolContext, args: dict[str, Any]) -> ToolOutcome:
    parsed_days = _parse_days(str(args["days"]))
    if not parsed_days:
        return ToolOutcome(text="Use: /setdays Mon,Tue,Wed,Fri")
    await update_user_preferences(int(context.user["id"]), summary_days=parsed_days)
    return ToolOutcome(text=f"Delivery days updated to {parsed_days}.")


async def tool_set_style(context: ToolContext, args: dict[str, Any]) -> ToolOutcome:
    style = str(args["style"]).lower()
    if style not in VALID_STYLES:
        return ToolOutcome(text="Use: /style brief|detailed|executive|bullets")
    await update_user_preferences(int(context.user["id"]), summary_style=style)
    return ToolOutcome(text=f"Summary style updated to {style}.")


async def tool_set_length(context: ToolContext, args: dict[str, Any]) -> ToolOutcome:
    length = str(args["length"]).lower()
    if length not in VALID_LENGTHS:
        return ToolOutcome(text="Use: /length short|medium|long")
    await update_user_preferences(int(context.user["id"]), summary_length=length)
    return ToolOutcome(text=f"Summary length updated to {length}.")


async def tool_set_focus(context: ToolContext, args: dict[str, Any]) -> ToolOutcome:
    focus = str(args["focus"]).lower()
    if focus not in VALID_FOCUS:
        return ToolOutcome(text="Use: /focus urgent|action-items|meetings|all")
    await update_user_preferences(int(context.user["id"]), summary_focus=focus)
    return ToolOutcome(text=f"Summary focus updated to {focus}.")


async def tool_update_summary_preferences(context: ToolContext, args: dict[str, Any]) -> ToolOutcome:
    reply_tone = args.get("reply_tone")
    draft_scope = args.get("draft_scope")
    include_reminders = args.get("include_reminders")
    draft_writing_style = args.get("draft_writing_style")

    updates: dict[str, Any] = {}
    confirmation_bits: list[str] = []

    if reply_tone is not None:
        normalized_tone = _normalize_reply_tone(str(reply_tone))
        if not normalized_tone:
            return ToolOutcome(text="Use: /replytone casual|friendly|formal|direct|warm|concise")
        updates["reply_tone"] = normalized_tone
        confirmation_bits.append(f"reply tone to {normalized_tone}")

    if draft_scope is not None:
        parsed_scope = _parse_draft_scope(str(draft_scope))
        if parsed_scope is None:
            return ToolOutcome(text="Use: /drafts high|high,medium|off")
        draft_high, draft_medium, draft_low = parsed_scope
        updates["draft_replies_high"] = draft_high
        updates["draft_replies_medium"] = draft_medium
        updates["draft_replies_low"] = draft_low
        scope_text = [
            bucket
            for bucket, enabled in (
                ("high", draft_high),
                ("medium", draft_medium),
            )
            if enabled
        ]
        confirmation_bits.append(f"draft replies for {', '.join(scope_text) if scope_text else 'none'}")

    if draft_writing_style is not None:
        updates["draft_writing_style"] = str(draft_writing_style).strip()
        confirmation_bits.append("draft writing style updated")

    if include_reminders is not None:
        parsed_bool = _parse_bool_argument(str(include_reminders))
        if parsed_bool is None:
            return ToolOutcome(text="Use: /reminders on|off")
        updates["include_reminders"] = parsed_bool
        confirmation_bits.append("reminders on" if parsed_bool else "reminders off")

    if not updates:
        return ToolOutcome(text="Tell me what to change, like reply tone, drafts, or reminders.")

    await update_user_preferences(int(context.user["id"]), **updates)
    changed_text = ", ".join(confirmation_bits)
    return ToolOutcome(text=f"Updated {changed_text}.")


async def tool_add_exclusion(context: ToolContext, args: dict[str, Any]) -> ToolOutcome:
    exclusion_value = str(args["exclusion"]).strip().lower()
    if not exclusion_value:
        return ToolOutcome(text="Use: /exclude sender@company.com or /exclude company.com")
    await add_user_exclusion(int(context.user["id"]), exclusion_value)
    return ToolOutcome(text=f"Added exclusion: {exclusion_value}")


async def tool_timezones(_: ToolContext, args: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(text=_timezones_text(str(args.get("filter", ""))))


async def tool_help(_: ToolContext, __: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(text=_help_text())


async def tool_connect_gmail(context: ToolContext, _: dict[str, Any]) -> ToolOutcome:
    return ToolOutcome(
        text=(
            "Connect Gmail first to enable summaries:\n"
            f"{_oauth_link(int(context.user['id']))}"
        )
    )


def _parse_tweak_message(message_text: str) -> tuple[int, str] | None:
    match = TWEAK_PATTERN.fullmatch(message_text)
    if not match:
        return None
    return int(match.group(1)), match.group(2).strip()


def _build_tweak_response(
    *,
    updated_draft: dict[str, Any],
    style_note: str,
) -> str:
    lines = [
        f"✍️ Updated draft for *{updated_draft.get('number', '?')}. {updated_draft.get('subject', '').strip()}*",
    ]
    time_note = updated_draft.get("time_note")
    if time_note:
        lines.append(str(time_note).strip())
    draft_reply = updated_draft.get("draft_reply") or ""
    lines.append("✍️ Draft reply:")
    lines.append(f"> {draft_reply.strip()}")
    compose_url = updated_draft.get("compose_url")
    if compose_url:
        lines.append(f"{'↳ Open draft' if updated_draft.get('draft_status') == 'saved' else '↳ Open thread'}: {compose_url}")
    draft_status = updated_draft.get("draft_status")
    if draft_status == "saved":
        lines.append("Saved as a Gmail draft on this thread.")
    elif draft_status == "missing_compose_scope":
        lines.append("Could not save the Gmail draft yet. Reconnect Gmail to enable in-thread drafts.")
    elif draft_status:
        lines.append("Could not save the Gmail draft yet.")
    lines.append(f"Saved writing style for future drafts: {style_note}")
    return "\n".join(lines).strip()


async def handle_tweak_request(
    *,
    user: dict[str, Any],
    gchat_user_id: str,
    gchat_space_id: str,
    draft_number: int,
    instruction: str,
) -> ToolOutcome:
    state = await _get_summary_draft_state(user)
    if not state:
        return ToolOutcome(text="I couldn’t find a recent draft to tweak. Ask me for a summary first.")

    target = _find_tweak_target(state, draft_number)
    if not target:
        return ToolOutcome(text=f"I couldn’t find draft {draft_number}. Try one of the recent numbered items.")

    current_style = str(user.get("draft_writing_style") or user.get("reply_tone") or "friendly, concise, and professional")
    updated_style = apply_draft_style_tweak(current_style, instruction)
    await update_user_preferences(
        int(user["id"]),
        draft_writing_style=updated_style,
    )

    rewritten_reply = await rewrite_draft_reply(
        current_reply=str(target.get("draft_reply") or target.get("summary") or ""),
        sender=str(target.get("sender") or ""),
        subject=str(target.get("subject") or ""),
        instruction=instruction,
        writing_style=updated_style,
        reply_tone=str(user.get("reply_tone") or "friendly, concise, and professional"),
    )

    updated_target = dict(target)
    updated_target["draft_reply"] = rewritten_reply
    draft_link = await upsert_thread_draft(
        user,
        email={
            "sender": target.get("sender"),
            "reply_to": target.get("reply_to") or target.get("sender"),
            "subject": target.get("subject"),
            "thread_id": target.get("thread_id"),
            "gmail_message_id": target.get("gmail_message_id"),
            "message_id_header": target.get("message_id_header"),
            "references": target.get("references"),
        },
        draft_reply=rewritten_reply,
        draft_id=str(target.get("draft_id")) if target.get("draft_id") else None,
    )
    if draft_link:
        updated_target["draft_status"] = draft_link.get("status")
        updated_target["draft_id"] = draft_link.get("draft_id")
        updated_target["thread_id"] = draft_link.get("thread_id") or target.get("thread_id")
        updated_target["thread_url"] = draft_link.get("thread_url")
        updated_target["draft_url"] = draft_link.get("draft_url")
        updated_target["compose_url"] = draft_link.get("draft_url") or draft_link.get("thread_url") or target.get("compose_url")
    else:
        updated_target["draft_status"] = "not_saved"
        updated_target["compose_url"] = updated_target.get("compose_url") or target.get("compose_url")
    target_index = None
    for idx, draft in enumerate(state.get("drafts", [])):
        if int(draft.get("number", 0)) == draft_number:
            target_index = idx
            break
    if target_index is not None:
        state["drafts"][target_index] = updated_target
        await save_chat_conversation_state(gchat_user_id, gchat_space_id, state)

    return ToolOutcome(
        text=_build_tweak_response(
            updated_draft=updated_target,
            style_note=updated_style,
        )
    )


TOOL_REGISTRY: dict[str, ToolSpec] = {
    "summary": ToolSpec("summary", "Fetch the Gmail summary from the last 24 hours now.", False, tool_summary),
    "testsummary": ToolSpec("testsummary", "Preview a summary using saved preferences.", False, tool_test_summary),
    "settings": ToolSpec("settings", "Show the user's current summary preferences.", False, tool_settings),
    "settime": ToolSpec("settime", "Change the daily summary time and timezone.", True, tool_set_time),
    "pause": ToolSpec("pause", "Pause scheduled summaries.", True, tool_pause),
    "resume": ToolSpec("resume", "Resume scheduled summaries.", True, tool_resume),
    "setdays": ToolSpec("setdays", "Update which weekdays summaries are delivered.", True, tool_set_days),
    "style": ToolSpec("style", "Set summary style.", True, tool_set_style),
    "length": ToolSpec("length", "Set summary length.", True, tool_set_length),
    "focus": ToolSpec("focus", "Set summary focus.", True, tool_set_focus),
    "summary_preferences": ToolSpec(
        "summary_preferences",
        "Update reply tone, draft reply scope, and reminders for this user's summary prompt.",
        True,
        tool_update_summary_preferences,
    ),
    "exclude": ToolSpec("exclude", "Exclude a sender or domain.", True, tool_add_exclusion),
    "timezones": ToolSpec("timezones", "List supported timezone names.", False, tool_timezones),
    "help": ToolSpec("help", "Show the command list.", False, tool_help),
    "connect_gmail": ToolSpec("connect_gmail", "Generate a Gmail OAuth link.", False, tool_connect_gmail),
}


def build_tool_catalog_text() -> str:
    lines = [
        "Available tools:",
        "- summary: fetch the Gmail summary from the last 24 hours now",
        "- testsummary: preview a summary using saved preferences",
        "- settings: show the user's current summary preferences",
        "- settime: change the daily summary time and timezone; args: hour, minute, timezone",
        "- pause: pause scheduled summaries",
        "- resume: resume scheduled summaries",
        "- setdays: update weekdays summaries are delivered; args: days",
        "- style: set summary style; args: style",
        "- length: set summary length; args: length",
        "- focus: set summary focus; args: focus",
        "- summary_preferences: update reply tone, draft scope, or reminders; args: reply_tone, draft_scope, include_reminders",
        "- exclude: exclude a sender or domain; args: exclusion",
        "- timezones: list supported timezone names; args: filter",
        "- help: show the command list",
        "- connect_gmail: return a Gmail OAuth link",
    ]
    return "\n".join(lines)


@lru_cache
def build_router_agent() -> Agent:
    settings = get_settings()
    system_prompt = (
        "You are a conversational router for a Google Chat bot that manages Gmail summaries and user preferences. "
        "Pick the most appropriate tool or ask a short clarifying question. "
        "Use tool_call for straightforward requests, clarify when a required detail is missing, "
        "and confirm before mutating actions when the user's intent is not explicit enough. "
        "Use summary_preferences when the user wants to change draft reply tone, which urgency buckets get draft replies, "
        "or whether same-day reminders are included. "
        "If the user is asking for a general help message or command list, use reply or the help tool. "
        "Keep responses concise and chat-friendly.\n\n"
        f"{build_tool_catalog_text()}"
    )
    return Agent(
        f"google-gla:{settings.gemini_model}",
        system_prompt=system_prompt,
        output_type=RouteDecision,
    )


async def decide_route(
    *,
    user: dict[str, Any],
    message_text: str,
    pending_state: dict[str, Any] | None = None,
) -> RouteDecision:
    user_context = _format_settings(user)
    pending_text = ""
    if pending_state:
        pending_text = f"\nPending state:\n{pending_state}"

    prompt = (
        f"Current user context:\n{user_context}\n\n"
        f"Incoming message:\n{message_text}{pending_text}\n\n"
        "Return exactly one structured decision."
    )
    result = await build_router_agent().run(prompt)
    return result.output


async def execute_tool(
    tool_name: str,
    context: ToolContext,
    tool_args: dict[str, Any],
) -> ToolOutcome:
    spec = TOOL_REGISTRY.get(tool_name)
    if not spec:
        return ToolOutcome(text="I couldn't find a matching action for that request. Try /help.")
    return await spec.handler(context, tool_args)


async def handle_follow_up_message(
    *,
    user: dict[str, Any],
    gchat_user_id: str,
    gchat_space_id: str,
    scheduler: Any,
    message_text: str,
    pending_state: dict[str, Any],
) -> tuple[ToolOutcome | None, dict[str, Any] | None, bool]:
    """Resolve a follow-up turn against a stored pending state."""

    if pending_state.get("kind") == "confirm":
        if _is_affirmative(message_text):
            tool_name = str(pending_state.get("tool_name") or "")
            tool_args = dict(pending_state.get("tool_args") or {})
            outcome = await execute_tool(tool_name, ToolContext(user=user, scheduler=scheduler), tool_args)
            return outcome, None, True
        if _is_negative(message_text):
            return ToolOutcome(text="Cancelled."), None, True
        return ToolOutcome(text="Please reply yes or no."), pending_state, False

    if pending_state.get("kind") == "clarify":
        decision = await decide_route(user=user, message_text=message_text, pending_state=pending_state)
        return await _apply_decision(
            decision=decision,
            user=user,
            scheduler=scheduler,
            pending_state=pending_state,
        )

    return None, None, False


async def _apply_decision(
    *,
    decision: RouteDecision,
    user: dict[str, Any],
    scheduler: Any,
    pending_state: dict[str, Any] | None = None,
) -> tuple[ToolOutcome | None, dict[str, Any] | None, bool]:
    context = ToolContext(user=user, scheduler=scheduler)

    if decision.kind == "reply":
        return ToolOutcome(text=decision.message), None, True

    if decision.kind in {"clarify", "confirm"}:
        state = {
            "kind": decision.kind,
            "tool_name": decision.tool_name,
            "tool_args": decision.tool_args,
            "message": decision.message,
        }
        if pending_state and pending_state.get("kind") == "clarify":
            state["previous_state"] = pending_state
        return ToolOutcome(text=decision.message), state, False

    if decision.kind == "tool_call":
        tool_name = decision.tool_name or ""
        tool_args = decision.tool_args or {}
        outcome = await execute_tool(tool_name, context, tool_args)
        return outcome, None, True

    return ToolOutcome(text="I’m not sure how to help with that."), None, True


async def route_chat_message(
    *,
    user: dict[str, Any],
    gchat_user_id: str,
    gchat_space_id: str,
    scheduler: Any,
    message_text: str,
) -> ToolOutcome:
    pending_state = await get_chat_conversation_state(gchat_user_id, gchat_space_id)
    if pending_state and pending_state.get("kind") == DRAFT_STATE_KIND:
        tweak_target = _parse_tweak_message(message_text)
        if tweak_target:
            draft_number, instruction = tweak_target
            return await handle_tweak_request(
                user=user,
                gchat_user_id=gchat_user_id,
                gchat_space_id=gchat_space_id,
                draft_number=draft_number,
                instruction=instruction,
            )
        pending_state = None

    if pending_state:
        outcome, next_state, should_clear = await handle_follow_up_message(
            user=user,
            gchat_user_id=gchat_user_id,
            gchat_space_id=gchat_space_id,
            scheduler=scheduler,
            message_text=message_text,
            pending_state=pending_state,
        )
        if should_clear:
            await clear_chat_conversation_state(gchat_user_id, gchat_space_id)
        elif next_state is not None:
            await save_chat_conversation_state(gchat_user_id, gchat_space_id, next_state)
        if outcome is not None:
            return outcome

    decision = await decide_route(user=user, message_text=message_text, pending_state=pending_state)
    outcome, next_state, should_clear = await _apply_decision(
        decision=decision,
        user=user,
        scheduler=scheduler,
        pending_state=pending_state,
    )
    if should_clear:
        await clear_chat_conversation_state(gchat_user_id, gchat_space_id)
    elif next_state is not None:
        await save_chat_conversation_state(gchat_user_id, gchat_space_id, next_state)
    return outcome
