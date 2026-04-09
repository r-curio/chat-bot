from __future__ import annotations

import asyncio
import base64
import json
import re
from email.message import EmailMessage
from datetime import UTC, datetime, timedelta
from typing import Any

from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from db import update_token

GMAIL_READ_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
GMAIL_COMPOSE_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]

EMAIL_PATTERN = re.compile(r"<([^>]+)>")
RE_PREFIX_PATTERN = re.compile(r"^\s*re\s*:\s*", re.IGNORECASE)


def _build_credentials(token_json: str, scopes: list[str]) -> Credentials:
    token_info = json.loads(token_json)
    return Credentials.from_authorized_user_info(token_info, scopes=scopes)


def _refresh_credentials(credentials: Credentials) -> str:
    if credentials.expired and credentials.refresh_token:
        credentials.refresh(Request())
    return credentials.to_json()


def _build_service(token_json: str, scopes: list[str]):
    credentials = _build_credentials(token_json, scopes)
    refreshed_token_json = _refresh_credentials(credentials)
    service = build("gmail", "v1", credentials=credentials, cache_discovery=False)
    return service, refreshed_token_json


def _fetch_recent_emails_sync(token_json: str) -> tuple[list[dict[str, str]], str]:
    service, refreshed_token_json = _build_service(token_json, GMAIL_READ_SCOPES)
    recent_after = int((datetime.now(UTC) - timedelta(hours=24)).timestamp())
    response = (
        service.users()
        .messages()
        .list(userId="me", q=f"after:{recent_after} -in:spam -in:trash")
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
                metadataHeaders=["From", "Reply-To", "Subject", "Message-ID", "References", "In-Reply-To"],
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
                "gmail_message_id": details.get("id"),
                "thread_id": details.get("threadId"),
                "sender": headers.get("From", "Unknown sender"),
                "reply_to": headers.get("Reply-To") or headers.get("From", "Unknown sender"),
                "subject": headers.get("Subject", "(No subject)"),
                "snippet": details.get("snippet", ""),
                "received_at": received_at,
                "message_id_header": headers.get("Message-ID"),
                "references": headers.get("References"),
                "in_reply_to": headers.get("In-Reply-To"),
            }
        )

    return emails, refreshed_token_json


async def fetch_recent_emails(user: dict[str, Any]) -> list[dict[str, str]]:
    token_json = user.get("gmail_token_json")
    if not token_json:
        raise ValueError("User does not have a Gmail token.")

    emails, refreshed_token_json = await asyncio.to_thread(_fetch_recent_emails_sync, token_json)
    if refreshed_token_json != token_json:
        await update_token(user["id"], refreshed_token_json)
    exclusions = {value.lower() for value in user.get("exclusions", [])}
    return [email for email in emails if not _is_excluded(email, exclusions)]


async def fetch_unread_emails(user: dict[str, Any]) -> list[dict[str, str]]:
    return await fetch_recent_emails(user)


def _normalize_subject_for_reply(subject: str) -> str:
    subject = subject.strip() or "(No subject)"
    if RE_PREFIX_PATTERN.match(subject):
        return subject
    return f"Re: {subject}"


def _thread_url(thread_id: str | None) -> str | None:
    if not thread_id:
        return None
    return f"https://mail.google.com/mail/u/0/#all/{thread_id}"


def _draft_url(draft_id: str | None) -> str | None:
    if not draft_id:
        return None
    return f"https://mail.google.com/mail/u/0/#drafts/{draft_id}"


def _build_reply_raw_message(email: dict[str, Any], draft_reply: str) -> str:
    message = EmailMessage()
    reply_to = str(email.get("reply_to") or email.get("sender") or "").strip()
    subject = _normalize_subject_for_reply(str(email.get("subject") or "(No subject)"))
    references = str(email.get("references") or "").strip()
    message_id_header = str(email.get("message_id_header") or "").strip()

    if reply_to:
        message["To"] = reply_to
    message["Subject"] = subject
    if message_id_header:
        message["In-Reply-To"] = message_id_header
    if references and message_id_header:
        message["References"] = f"{references} {message_id_header}".strip()
    elif message_id_header:
        message["References"] = message_id_header
    elif references:
        message["References"] = references

    message.set_content(draft_reply.strip())
    return base64.urlsafe_b64encode(message.as_bytes()).decode().rstrip("=")


def _upsert_thread_draft_sync(
    token_json: str,
    *,
    email: dict[str, Any],
    draft_reply: str,
    draft_id: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    try:
        service, refreshed_token_json = _build_service(token_json, GMAIL_COMPOSE_SCOPES)
    except RefreshError:
        return {"status": "missing_compose_scope"}, token_json
    raw_message = _build_reply_raw_message(email, draft_reply)
    message_body = {
        "raw": raw_message,
        "threadId": email.get("thread_id"),
    }

    try:
        if draft_id:
            draft = (
                service.users()
                .drafts()
                .update(
                    userId="me",
                    id=draft_id,
                    body={"id": draft_id, "message": message_body},
                )
                .execute()
            )
        else:
            draft = (
                service.users()
                .drafts()
                .create(
                    userId="me",
                    body={"message": message_body},
                )
                .execute()
            )
    except (HttpError, RefreshError):
        return {"status": "error"}, refreshed_token_json

    return draft, refreshed_token_json


async def upsert_thread_draft(
    user: dict[str, Any],
    *,
    email: dict[str, Any],
    draft_reply: str,
    draft_id: str | None = None,
) -> dict[str, Any] | None:
    token_json = user.get("gmail_token_json")
    if not token_json:
        raise ValueError("User does not have a Gmail token.")

    draft, refreshed_token_json = await asyncio.to_thread(
        _upsert_thread_draft_sync,
        token_json,
        email=email,
        draft_reply=draft_reply,
        draft_id=draft_id,
    )
    if refreshed_token_json != token_json:
        await update_token(user["id"], refreshed_token_json)
    if not draft:
        return None
    if "status" in draft:
        return {
            "status": draft["status"],
            "draft_id": None,
            "thread_id": email.get("thread_id"),
            "thread_url": _thread_url(str(email.get("thread_id") or "")),
            "draft_url": None,
        }

    thread_id = str(email.get("thread_id") or draft.get("message", {}).get("threadId") or "")
    draft_id_value = draft.get("id")
    return {
        "status": "saved",
        "draft_id": draft_id_value,
        "thread_id": thread_id or None,
        "thread_url": _thread_url(thread_id),
        "draft_url": _draft_url(str(draft_id_value or "")),
    }


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
