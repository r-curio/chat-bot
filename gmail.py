from __future__ import annotations

import asyncio
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from db import update_token

GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

EMAIL_PATTERN = re.compile(r"<([^>]+)>")


def _fetch_unread_emails_sync(token_json: str) -> tuple[list[dict[str, str]], str]:
    token_info = json.loads(token_json)
    credentials = Credentials.from_authorized_user_info(token_info, scopes=GMAIL_SCOPES)

    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())

    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
    recent_after = int((datetime.now(UTC) - timedelta(hours=48)).timestamp())
    response = (
        service.users()
        .messages()
        .list(userId="me", q=f"is:unread after:{recent_after}")
        .execute()
    )

    messages = response.get("messages", [])
    emails: list[dict[str, str]] = []

    for message in messages:
        details = (
            service.users()
            .messages()
            .get(
                userId="me",
                id=message["id"],
                format="metadata",
                metadataHeaders=["From", "Subject"],
            )
            .execute()
        )
        headers = {header["name"]: header["value"] for header in details.get("payload", {}).get("headers", [])}
        internal_date = details.get("internalDate")
        received_at = None
        if internal_date:
            try:
                received_at = datetime.fromtimestamp(int(internal_date) / 1000, tz=UTC).isoformat()
            except (TypeError, ValueError, OSError):
                received_at = None
        emails.append(
            {
                "sender": headers.get("From", "Unknown sender"),
                "subject": headers.get("Subject", "(No subject)"),
                "snippet": details.get("snippet", ""),
                "received_at": received_at,
            }
        )

    return emails, credentials.to_json()


async def fetch_unread_emails(user: dict[str, Any]) -> list[dict[str, str]]:
    token_json = user.get("gmail_token_json")
    if not token_json:
        raise ValueError("User does not have a Gmail token.")

    emails, refreshed_token_json = await asyncio.to_thread(_fetch_unread_emails_sync, token_json)
    if refreshed_token_json != token_json:
        await update_token(user["id"], refreshed_token_json)
    exclusions = {value.lower() for value in user.get("exclusions", [])}
    return [email for email in emails if not _is_excluded(email, exclusions)]


def _extract_sender_address(sender: str) -> str:
    match = EMAIL_PATTERN.search(sender)
    if match:
        return match.group(1).strip().lower()
    return sender.strip().lower()


def _is_excluded(email: dict[str, str], exclusions: set[str]) -> bool:
    if not exclusions:
        return False

    sender = email.get("sender", "")
    sender_address = _extract_sender_address(sender)
    sender_domain = sender_address.split("@", 1)[1] if "@" in sender_address else ""

    for exclusion in exclusions:
        if exclusion == sender_address:
            return True
        if sender_domain and exclusion == sender_domain:
            return True
        if exclusion and exclusion in sender.lower():
            return True
    return False
