from __future__ import annotations

import re
from urllib.parse import urlencode, urlsplit, urlunsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

from fastapi import APIRouter, Request

from config import get_settings
from db import (
    add_user_exclusion,
    delete_user,
    get_user_by_chat_ids,
    save_user,
    update_summary_schedule,
    update_user_preferences,
)
from scheduler import remove_summary_schedule_for_user, reschedule_summary_for_user, run_summary_for_user

router = APIRouter(prefix="/gchat", tags=["gchat"])

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
        "- `/summary` Get your unread Gmail summary now.\n"
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


def _format_settings(user: dict) -> str:
    exclusions = user.get("exclusions", [])
    exclusion_text = ", ".join(exclusions) if exclusions else "None"
    paused_text = "Yes" if user.get("is_paused") else "No"
    return (
        "Your current settings:\n"
        f"- Time: {int(user.get('summary_hour', 8)):02d}:{int(user.get('summary_minute', 0)):02d} "
        f"{user.get('summary_timezone', get_settings().timezone)}\n"
        f"- Paused: {paused_text}\n"
        f"- Days: {user.get('summary_days', 'Mon,Tue,Wed,Thu,Fri,Sat,Sun')}\n"
        f"- Style: {user.get('summary_style', 'brief')}\n"
        f"- Length: {user.get('summary_length', 'medium')}\n"
        f"- Focus: {user.get('summary_focus', 'all')}\n"
        f"- Exclusions: {exclusion_text}"
    )


async def _get_or_create_user(gchat_user_id: str, gchat_space_id: str) -> dict:
    user = await get_user_by_chat_ids(gchat_user_id, gchat_space_id)
    if user:
        return user
    return await save_user(gchat_user_id, gchat_space_id)


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
            if command_name == "/testsummary":
                summary = f"*Preview*\n{summary}"
            return _chat_response({"text": summary}, payload)

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

        if command_name == "/exclude":
            exclusion_value = command_argument.lower()
            if not exclusion_value:
                return _chat_response(
                    {"text": "Use: /exclude sender@company.com or /exclude company.com"},
                    payload,
                )
            await add_user_exclusion(int(user["id"]), exclusion_value)
            return _chat_response({"text": f"Added exclusion: {exclusion_value}"}, payload)

        return _chat_response({"text": _help_text()}, payload)

    if event_type == "REMOVED_FROM_SPACE":
        user = await get_user_by_chat_ids(gchat_user_id, gchat_space_id)
        if user:
            remove_summary_schedule_for_user(request.app.state.scheduler, int(user["id"]))
        await delete_user(gchat_user_id, gchat_space_id)
        return {}

    return _chat_response({"text": "Event ignored."}, payload)
