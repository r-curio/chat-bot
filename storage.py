from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from dotenv import load_dotenv

from config import get_settings
from db import create_audio_link, get_audio_link

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def get_storage_backend() -> str:
    backend = os.getenv("STORAGE_BACKEND", "local").strip().lower()
    if backend == "gcs":
        return "firebase"
    if backend not in {"local", "firebase"}:
        raise RuntimeError("STORAGE_BACKEND must be either 'local' or 'firebase'.")
    return backend


async def store_audio(user_id: str, mp3_bytes: bytes) -> str:
    if not mp3_bytes:
        raise ValueError("Cannot store empty audio bytes.")

    backend = get_storage_backend()
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    if backend == "local":
        return await asyncio.to_thread(_store_audio_local, user_id, timestamp, mp3_bytes)

    return await _store_audio_firebase(user_id, timestamp, mp3_bytes)


def _store_audio_local(user_id: str, timestamp: str, mp3_bytes: bytes) -> str:
    audio_dir = Path("/tmp/audio")
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"{_safe_user_id(user_id)}_{timestamp}.mp3"
    audio_path.write_bytes(mp3_bytes)
    return str(audio_path)


async def _store_audio_firebase(user_id: str, timestamp: str, mp3_bytes: bytes) -> str:
    object_name = await asyncio.to_thread(_upload_audio_to_firebase, user_id, timestamp, mp3_bytes)
    token = await create_audio_link(user_id, object_name)
    return _build_short_audio_url(token)


def _upload_audio_to_firebase(user_id: str, timestamp: str, mp3_bytes: bytes) -> str:
    from google.cloud import storage

    bucket_name = _firebase_bucket_name()
    if not bucket_name:
        raise RuntimeError("FIREBASE_STORAGE_BUCKET is required when STORAGE_BACKEND=firebase.")

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    object_name = f"audio/{_safe_user_id(user_id)}_{timestamp}.mp3"
    blob = bucket.blob(object_name)
    blob.upload_from_string(mp3_bytes, content_type="audio/mpeg")
    return object_name


async def resolve_audio_redirect_url(token: str) -> str | None:
    audio_link = await get_audio_link(token)
    if not audio_link:
        return None

    return await asyncio.to_thread(_generate_firebase_signed_url, str(audio_link["object_name"]))


def _safe_user_id(user_id: str) -> str:
    sanitized = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in user_id)
    return sanitized or "user"


def _firebase_bucket_name() -> str:
    raw_bucket = os.getenv("FIREBASE_STORAGE_BUCKET", "landinc-signal-folio.firebasestorage.app").strip()
    if raw_bucket.startswith("https://"):
        raw_bucket = raw_bucket.removeprefix("https://")
    if raw_bucket.startswith("http://"):
        raw_bucket = raw_bucket.removeprefix("http://")
    return raw_bucket.rstrip("/")


def _generate_firebase_signed_url(object_name: str) -> str:
    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(_firebase_bucket_name())
    blob = bucket.blob(object_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=1),
        method="GET",
    )


def _build_short_audio_url(token: str) -> str:
    callback_parts = urlsplit(get_settings().callback_url)
    base_path = callback_parts.path.removesuffix("/auth/callback").rstrip("/")
    short_path = f"{base_path}/a/{token}" if base_path else f"/a/{token}"
    return urlunsplit((callback_parts.scheme, callback_parts.netloc, short_path, "", ""))
