from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from fastapi import APIRouter, Request

from config import get_settings
from assistant import route_chat_message
from db import (
    add_user_exclusion,
    clear_chat_conversation_state,
    delete_user,
    get_user_by_chat_ids,
    save_chat_conversation_state,
    save_user,
    update_summary_schedule,
    update_user_preferences,
)
from scheduler import (
    remove_summary_schedule_for_user,
    reschedule_summary_for_user,
    run_summary_for_user,
    send_gchat_message,
)
from storage import get_storage_backend, store_audio
from tts import generate_audio

router = APIRouter(prefix="/gchat", tags=["gchat"])
logger = logging.getLogger(__name__)

SETTIME_PATTERN = re.compile(
    r"^/settime\s+(\d{1,2}):(\d{2})\s+([A-Za-z0-9_\-+]+(?:/[A-Za-z0-9_\-+]+)+)$"
)
VALID_DAYS = {"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"}
VALID_STYLES = {"brief", "detailed", "executive", "bullets"}
VALID_LENGTHS = {"short", "medium", "long"}
VALID_FOCUS = {"urgent", "action-items", "meetings", "all"}


def _chat_response(message: dict[str, str], payload: dict) -> dict:
    if payload.get("chat"):
        return {
            "hostAppDataAction": {
                "chatDataAction": {
                    "createMessageAction": {
                        "message": message,
                    }
                }
            }
        }
    return message


def _oauth_link(install_id: int) -> str:
    settings = get_settings()
    callback_parts = urlsplit(settings.callback_url)
    base_path = callback_parts.path.removesuffix("/auth/callback").rstrip("/")
    start_path = f"{base_path}/auth/start" if base_path else "/auth/start"
    query = urlencode({"install_id": install_id})
    return urlunsplit((callback_parts.scheme, callback_parts.netloc, start_path, query, ""))


def _event_user_id(payload: dict) -> str:
    return (
        payload.get("user", {}).get("name")
        or payload.get("message", {}).get("sender", {}).get("name")
        or payload.get("chat", {}).get("user", {}).get("name")
        or payload.get("chat", {}).get("messagePayload", {}).get("message", {}).get("sender", {}).get("name", "")
    )


def _event_space_id(payload: dict) -> str:
    return (
        payload.get("space", {}).get("name")
        or payload.get("message", {}).get("space", {}).get("name")
        or payload.get("chat", {}).get("messagePayload", {}).get("space", {}).get("name")
        or payload.get("chat", {}).get("messagePayload", {}).get("message", {}).get("space", {}).get("name", "")
    )


def _event_message_text(payload: dict) -> str:
    return (
        payload.get("message", {}).get("argumentText")
        or payload.get("message", {}).get("text")
        or payload.get("chat", {}).get("messagePayload", {}).get("message", {}).get("argumentText")
        or payload.get("chat", {}).get("messagePayload", {}).get("message", {}).get("text", "")
    ).strip()


def _event_type(payload: dict) -> str:
    explicit_type = payload.get("type", "")
    if explicit_type:
        return explicit_type

    chat_payload = payload.get("chat", {}).get("messagePayload", {})
    if chat_payload.get("message"):
        return "MESSAGE"
    if chat_payload.get("space") and not chat_payload.get("message"):
        return "ADDED_TO_SPACE"
    return ""


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


def _parse_settime_command(message_text: str) -> tuple[int, int, str] | None:
    match = SETTIME_PATTERN.fullmatch(message_text)
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


def _timezones_text(filter_text: str = "") -> str:
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


def _command_name(message_text: str) -> str:
    if not message_text:
        return ""
    return message_text.split(maxsplit=1)[0].lower()


def _command_argument(message_text: str) -> str:
    parts = message_text.split(maxsplit=1)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def _parse_days(argument: str) -> str | None:
    if not argument:
        return None
    days = [part.strip().title() for part in argument.split(",") if part.strip()]
    if not days or any(day not in VALID_DAYS for day in days):
        return None

    ordered = [day for day in ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun") if day in set(days)]
    return ",".join(ordered)


def _normalize_reply_tone(argument: str) -> str | None:
    tone = argument.strip()
    if not tone:
        return None
    lowered = tone.lower()
    if lowered in {"casual", "friendly", "formal", "direct", "warm", "concise", "professional"}:
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


def _format_settings(user: dict) -> str:
    exclusions = user.get("exclusions", [])
    exclusion_text = ", ".join(exclusions) if exclusions else "None"
    paused_text = "Yes" if user.get("is_paused") else "No"
    prompt_mode = user.get("summary_prompt_mode", "structured")
    reply_tone = user.get("reply_tone", "friendly, concise, and professional")
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
        f"- Draft replies: {draft_text}\n"
        f"- Reminders: {reminder_text}\n"
        f"- Exclusions: {exclusion_text}"
    )


async def _get_or_create_user(gchat_user_id: str, gchat_space_id: str) -> dict:
    user = await get_user_by_chat_ids(gchat_user_id, gchat_space_id)
    if user:
        return user
    return await save_user(gchat_user_id, gchat_space_id)


async def _send_on_demand_audio_summary(user: dict, summary: str) -> None:
    try:
        mp3_bytes = await generate_audio(summary)
        audio_location = await store_audio(str(user["gchat_user_id"]), mp3_bytes)

        if get_storage_backend() == "local":
            logger.info(
                "Saved on-demand audio summary for user %s in %s to %s.",
                user["gchat_user_id"],
                user["gchat_space_id"],
                audio_location,
            )
        else:
            await send_gchat_message(
                user["gchat_space_id"],
                f"🔊 Audio summary: {audio_location}",
            )
    except Exception:
        logger.exception(
            "Failed to generate or store on-demand audio summary for user %s in %s.",
            user["gchat_user_id"],
            user["gchat_space_id"],
        )


async def _persist_summary_drafts(user: dict, drafts: list[dict] | None) -> None:
    if drafts is None:
        return
    if not drafts:
        await clear_chat_conversation_state(user["gchat_user_id"], user["gchat_space_id"])
        return

    await save_chat_conversation_state(
        user["gchat_user_id"],
        user["gchat_space_id"],
        {
            "kind": "summary_drafts",
            "drafts": drafts,
        },
    )


@router.post("/webhook")
async def gchat_webhook(request: Request) -> dict:
    payload = await request.json()
    event_type = _event_type(payload)
    gchat_user_id = _event_user_id(payload)
    gchat_space_id = _event_space_id(payload)
    message_text = _event_message_text(payload)
    command_name = _command_name(message_text)
    command_argument = _command_argument(message_text)

    if event_type == "ADDED_TO_SPACE":
        user = await save_user(gchat_user_id, gchat_space_id)
        return _chat_response(
            {
                "text": (
                    "Thanks for adding me. Connect your Gmail to enable summaries:\n"
                    f"{_oauth_link(user['id'])}\n\n"
                    "Use /help to see personalization commands."
                )
            },
            payload,
        )

    if event_type == "MESSAGE":
        if command_name.startswith("/"):
            await clear_chat_conversation_state(gchat_user_id, gchat_space_id)

        if command_name == "/help":
            return _chat_response({"text": _help_text()}, payload)

        if command_name == "/timezones":
            return _chat_response({"text": _timezones_text(command_argument)}, payload)

        user = await _get_or_create_user(gchat_user_id, gchat_space_id)

        if command_name in {"/summary", "/testsummary"}:
            if not user.get("gmail_token_json"):
                return _chat_response(
                    {
                        "text": (
                            "Connect Gmail first to enable summaries:\n"
                            f"{_oauth_link(user['id'])}"
                        )
                    },
                    payload,
                )

            summary = await run_summary_for_user(user)
            await _persist_summary_drafts(user, [draft.model_dump() for draft in summary.drafts])
            if command_name == "/testsummary":
                summary_text = f"*Preview*\n{summary.text}"
            elif command_name == "/summary":
                asyncio.create_task(_send_on_demand_audio_summary(user, summary.text))
                summary_text = summary.text
            return _chat_response({"text": summary_text}, payload)

        if command_name == "/settings":
            refreshed_user = await get_user_by_chat_ids(gchat_user_id, gchat_space_id) or user
            return _chat_response({"text": _format_settings(refreshed_user)}, payload)

        if command_name == "/settime":
            parsed = _parse_settime_command(message_text)
            if not parsed:
                return _chat_response(
                    {
                        "text": (
                            "Invalid format.\n"
                            "Use: /settime HH:MM Area/City\n"
                            "Example: /settime 08:00 Asia/Manila"
                        )
                    },
                    payload,
                )

            hour, minute, timezone = parsed
            await update_summary_schedule(
                int(user["id"]),
                summary_hour=hour,
                summary_minute=minute,
                summary_timezone=timezone,
            )
            await reschedule_summary_for_user(request.app.state.scheduler, int(user["id"]))
            return _chat_response(
                {
                    "text": (
                        f"Daily summary time updated to {hour:02d}:{minute:02d} {timezone}.\n"
                        "Use /settings to review your preferences."
                    )
                },
                payload,
            )

        if command_name == "/pause":
            await update_user_preferences(int(user["id"]), is_paused=True)
            return _chat_response({"text": "Scheduled summaries paused."}, payload)

        if command_name == "/resume":
            await update_user_preferences(int(user["id"]), is_paused=False)
            return _chat_response({"text": "Scheduled summaries resumed."}, payload)

        if command_name == "/setdays":
            parsed_days = _parse_days(command_argument)
            if not parsed_days:
                return _chat_response(
                    {
                        "text": (
                            "Invalid format.\n"
                            "Use: /setdays Mon,Tue,Wed,Fri"
                        )
                    },
                    payload,
                )
            await update_user_preferences(int(user["id"]), summary_days=parsed_days)
            return _chat_response({"text": f"Delivery days updated to {parsed_days}."}, payload)

        if command_name == "/style":
            style = command_argument.lower()
            if style not in VALID_STYLES:
                return _chat_response(
                    {"text": "Use: /style brief|detailed|executive|bullets"},
                    payload,
                )
            await update_user_preferences(int(user["id"]), summary_style=style)
            return _chat_response({"text": f"Summary style updated to {style}."}, payload)

        if command_name == "/length":
            length = command_argument.lower()
            if length not in VALID_LENGTHS:
                return _chat_response(
                    {"text": "Use: /length short|medium|long"},
                    payload,
                )
            await update_user_preferences(int(user["id"]), summary_length=length)
            return _chat_response({"text": f"Summary length updated to {length}."}, payload)

        if command_name == "/focus":
            focus = command_argument.lower()
            if focus not in VALID_FOCUS:
                return _chat_response(
                    {"text": "Use: /focus urgent|action-items|meetings|all"},
                    payload,
            )
            await update_user_preferences(int(user["id"]), summary_focus=focus)
            return _chat_response({"text": f"Summary focus updated to {focus}."}, payload)

        if command_name == "/replytone":
            tone = _normalize_reply_tone(command_argument)
            if not tone:
                return _chat_response(
                    {"text": "Use: /replytone casual|friendly|formal|direct|warm|concise"},
                    payload,
                )
            await update_user_preferences(int(user["id"]), reply_tone=tone)
            return _chat_response({"text": f"Reply tone updated to {tone}."}, payload)

        if command_name == "/drafts":
            parsed_scope = _parse_draft_scope(command_argument)
            if not parsed_scope:
                return _chat_response(
                    {"text": "Use: /drafts high|high,medium|off"},
                    payload,
                )
            draft_high, draft_medium, draft_low = parsed_scope
            scope_text = [
                bucket
                for bucket, enabled in (
                    ("high", draft_high),
                    ("medium", draft_medium),
                )
                if enabled
            ]
            await update_user_preferences(
                int(user["id"]),
                draft_replies_high=draft_high,
                draft_replies_medium=draft_medium,
                draft_replies_low=draft_low,
            )
            return _chat_response(
                {"text": f"Draft replies updated to {', '.join(scope_text) if scope_text else 'none'}."},
                payload,
            )

        if command_name == "/reminders":
            include_reminders = _parse_bool_argument(command_argument)
            if include_reminders is None:
                return _chat_response({"text": "Use: /reminders on|off"}, payload)
            await update_user_preferences(int(user["id"]), include_reminders=include_reminders)
            return _chat_response(
                {"text": f"Reminders turned {'on' if include_reminders else 'off'}."},
                payload,
            )

        if command_name == "/exclude":
            exclusion_value = command_argument.lower()
            if not exclusion_value:
                return _chat_response(
                    {"text": "Use: /exclude sender@company.com or /exclude company.com"},
                    payload,
                )
            await add_user_exclusion(int(user["id"]), exclusion_value)
            return _chat_response({"text": f"Added exclusion: {exclusion_value}"}, payload)

        if command_name.startswith("/"):
            return _chat_response({"text": _help_text()}, payload)

        routed_result = await route_chat_message(
            user=user,
            gchat_user_id=gchat_user_id,
            gchat_space_id=gchat_space_id,
            scheduler=request.app.state.scheduler,
            message_text=message_text,
        )
        await _persist_summary_drafts(user, routed_result.drafts)
        if routed_result.queue_audio:
            asyncio.create_task(_send_on_demand_audio_summary(user, routed_result.text))
        return _chat_response({"text": routed_result.text}, payload)

    if event_type == "REMOVED_FROM_SPACE":
        user = await get_user_by_chat_ids(gchat_user_id, gchat_space_id)
        if user:
            remove_summary_schedule_for_user(request.app.state.scheduler, int(user["id"]))
        await delete_user(gchat_user_id, gchat_space_id)
        return {}

    return _chat_response({"text": "Event ignored."}, payload)
