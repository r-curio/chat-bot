from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    google_client_id: str
    google_client_secret: str
    callback_url: str
    database_path: Path
    gemini_model: str
    google_chat_service_account_file: Path
    timezone: str = "Asia/Manila"

    @property
    def credentials_json_path(self) -> Path:
        return BASE_DIR / "credentials.json"


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache
def get_settings() -> Settings:
    gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not gemini_api_key:
        raise RuntimeError("Missing required environment variable: GEMINI_API_KEY")

    # PydanticAI's direct Gemini provider looks for GOOGLE_API_KEY.
    os.environ.setdefault("GOOGLE_API_KEY", gemini_api_key)

    database_path = Path(os.getenv("DATABASE_PATH", BASE_DIR / "gmail_gchat_bot.db"))
    if not database_path.is_absolute():
        database_path = BASE_DIR / database_path

    service_account_path = Path(
        os.getenv("GOOGLE_CHAT_SERVICE_ACCOUNT_FILE", BASE_DIR / "chat-service-account.json")
    )
    if not service_account_path.is_absolute():
        service_account_path = BASE_DIR / service_account_path

    return Settings(
        gemini_api_key=gemini_api_key,
        google_client_id=_require_env("GOOGLE_CLIENT_ID"),
        google_client_secret=_require_env("GOOGLE_CLIENT_SECRET"),
        callback_url=_require_env("CALLBACK_URL"),
        database_path=database_path,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        google_chat_service_account_file=service_account_path,
    )
