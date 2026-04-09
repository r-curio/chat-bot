from __future__ import annotations

import asyncio
import logging
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from apscheduler.jobstores.base import JobLookupError
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from config import Settings, get_settings
from db import clear_chat_conversation_state, get_all_users, get_user_by_install_id, save_chat_conversation_state
from gmail import fetch_recent_emails
from storage import get_storage_backend, store_audio
from summarizer import SummaryResult, summarize_emails
from tts import generate_audio

logger = logging.getLogger(__name__)

CHAT_BOT_SCOPE = ["https://www.googleapis.com/auth/chat.bot"]
CHAT_API_BASE_URL = "https://chat.googleapis.com/v1"


def build_scheduler() -> AsyncIOScheduler:
    settings = get_settings()
    return AsyncIOScheduler(timezone=settings.timezone)


def _get_chat_access_token_sync() -> str:
    settings = get_settings()
    credentials = service_account.Credentials.from_service_account_file(
        settings.google_chat_service_account_file,
        scopes=CHAT_BOT_SCOPE,
    )
    credentials.refresh(Request())
    if not credentials.token:
        raise RuntimeError("Failed to acquire a Google Chat access token.")
    return credentials.token


async def _get_chat_access_token() -> str:
    return await asyncio.to_thread(_get_chat_access_token_sync)


async def send_gchat_message(space_id: str, text: str) -> None:
    access_token = await _get_chat_access_token()
    url = f"{CHAT_API_BASE_URL}/{space_id}/messages"

    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            url,
            headers={"Authorization": f"Bearer {access_token}"},
            json={"text": text},
        )
        response.raise_for_status()


async def build_summary_text_for_user(user: dict) -> SummaryResult:
    emails = await fetch_recent_emails(user)
    return await summarize_emails(emails, user=user)


async def run_summary_for_user(user: dict) -> SummaryResult:
    if not user.get("gmail_token_json"):
        return SummaryResult(text="Connect Gmail first to receive summaries.", drafts=[])
    return await build_summary_text_for_user(user)


def _job_id_for_user(install_id: int) -> str:
    return f"daily-gmail-summary-{install_id}"


def _normalize_schedule(user: dict[str, Any], settings: Settings) -> tuple[int, int, str]:
    return (
        int(user.get("summary_hour", 8)),
        int(user.get("summary_minute", 0)),
        str(user.get("summary_timezone") or settings.timezone),
    )


def _user_accepts_today(user: dict[str, Any]) -> bool:
    if user.get("is_paused"):
        return False

    timezone_name = str(user.get("summary_timezone") or get_settings().timezone)
    allowed_days = {
        part.strip()
        for part in str(user.get("summary_days") or "").split(",")
        if part.strip()
    }
    if not allowed_days:
        return True

    today_name = datetime_now_in_timezone(timezone_name).strftime("%a")
    return today_name in allowed_days


def datetime_now_in_timezone(timezone_name: str):
    from datetime import datetime

    return datetime.now(ZoneInfo(timezone_name))


def schedule_summary_for_user(scheduler: AsyncIOScheduler, user: dict[str, Any]) -> None:
    settings = get_settings()
    hour, minute, timezone = _normalize_schedule(user, settings)
    scheduler.add_job(
        send_scheduled_summary,
        CronTrigger(hour=hour, minute=minute, timezone=timezone),
        id=_job_id_for_user(int(user["id"])),
        replace_existing=True,
        max_instances=1,
        kwargs={"install_id": int(user["id"])},
    )


def remove_summary_schedule_for_user(scheduler: AsyncIOScheduler, install_id: int) -> None:
    try:
        scheduler.remove_job(_job_id_for_user(install_id))
    except JobLookupError:
        return


async def reschedule_summary_for_user(scheduler: AsyncIOScheduler, install_id: int) -> None:
    user = await get_user_by_install_id(install_id)
    if not user:
        remove_summary_schedule_for_user(scheduler, install_id)
        return
    schedule_summary_for_user(scheduler, user)


async def schedule_all_user_summaries(scheduler: AsyncIOScheduler) -> None:
    users = await get_all_users()
    for user in users:
        schedule_summary_for_user(scheduler, user)


async def send_scheduled_summary(install_id: int) -> None:
    user = await get_user_by_install_id(install_id)
    if not user:
        logger.info("Skipping scheduled summary for install %s: user not found.", install_id)
        return
    if not user.get("gmail_token_json"):
        logger.info("Skipping user %s in %s: no Gmail token.", user["gchat_user_id"], user["gchat_space_id"])
        return
    if not _user_accepts_today(user):
        logger.info(
            "Skipping user %s in %s: paused or not scheduled for today.",
            user["gchat_user_id"],
            user["gchat_space_id"],
        )
        return

    try:
        summary = await build_summary_text_for_user(user)
        if summary.drafts:
            await save_chat_conversation_state(
                user["gchat_user_id"],
                user["gchat_space_id"],
                {
                    "kind": "summary_drafts",
                    "drafts": [draft.model_dump() for draft in summary.drafts],
                },
            )
        else:
            await clear_chat_conversation_state(user["gchat_user_id"], user["gchat_space_id"])

        await send_gchat_message(user["gchat_space_id"], summary.text)
        try:
            mp3_bytes = await generate_audio(summary.text)
            audio_location = await store_audio(str(user["gchat_user_id"]), mp3_bytes)

            if get_storage_backend() == "local":
                logger.info(
                    "Saved audio summary for user %s in %s to %s.",
                    user["gchat_user_id"],
                    user["gchat_space_id"],
                    audio_location,
                )
            else:
                await send_gchat_message(user["gchat_space_id"], f"🔊 Audio summary: {audio_location}")
        except Exception:
            logger.exception(
                "Failed to generate or store audio summary for user %s in %s.",
                user["gchat_user_id"],
                user["gchat_space_id"],
            )
    except Exception:
        logger.exception(
            "Failed to process summary for user %s in %s.",
            user["gchat_user_id"],
            user["gchat_space_id"],
        )
