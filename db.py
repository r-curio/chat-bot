from __future__ import annotations

import json
from typing import Any

import aiosqlite

from config import get_settings

CREATE_USERS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gchat_user_id TEXT NOT NULL,
    gchat_space_id TEXT NOT NULL,
    gmail_token_json TEXT,
    summary_hour INTEGER NOT NULL DEFAULT 8,
    summary_minute INTEGER NOT NULL DEFAULT 0,
    summary_timezone TEXT NOT NULL DEFAULT 'Asia/Manila',
    is_paused INTEGER NOT NULL DEFAULT 0,
    summary_days TEXT NOT NULL DEFAULT 'Mon,Tue,Wed,Thu,Fri,Sat,Sun',
    summary_style TEXT NOT NULL DEFAULT 'brief',
    summary_length TEXT NOT NULL DEFAULT 'medium',
    summary_focus TEXT NOT NULL DEFAULT 'all',
    summary_prompt_mode TEXT NOT NULL DEFAULT 'structured',
    reply_tone TEXT NOT NULL DEFAULT 'friendly, concise, and professional',
    draft_replies_high INTEGER NOT NULL DEFAULT 1,
    draft_replies_medium INTEGER NOT NULL DEFAULT 1,
    draft_replies_low INTEGER NOT NULL DEFAULT 0,
    include_reminders INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(gchat_user_id, gchat_space_id)
)
"""

CREATE_USER_EXCLUSIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS user_exclusions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    exclusion_value TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, exclusion_value),
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
)
"""

CREATE_AUDIO_LINKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audio_links (
    token TEXT PRIMARY KEY,
    gchat_user_id TEXT NOT NULL,
    object_name TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
)
"""

CREATE_CHAT_STATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS chat_conversation_state (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gchat_user_id TEXT NOT NULL,
    gchat_space_id TEXT NOT NULL,
    state_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(gchat_user_id, gchat_space_id)
)
"""

USER_SELECT_COLUMNS = """
    id,
    gchat_user_id,
    gchat_space_id,
    gmail_token_json,
    summary_hour,
    summary_minute,
    summary_timezone,
    is_paused,
    summary_days,
    summary_style,
    summary_length,
    summary_focus,
    summary_prompt_mode,
    reply_tone,
    draft_replies_high,
    draft_replies_medium,
    draft_replies_low,
    include_reminders,
    created_at
"""


async def _connect() -> aiosqlite.Connection:
    connection = await aiosqlite.connect(get_settings().database_path)
    connection.row_factory = aiosqlite.Row
    await connection.execute("PRAGMA foreign_keys = ON")
    return connection


async def init_db() -> None:
    db = await _connect()
    try:
        await db.execute(CREATE_USERS_TABLE_SQL)
        await db.execute(CREATE_USER_EXCLUSIONS_TABLE_SQL)
        await db.execute(CREATE_AUDIO_LINKS_TABLE_SQL)
        await db.execute(CREATE_CHAT_STATE_TABLE_SQL)
        await _ensure_users_schema(db)
        await db.commit()
    finally:
        await db.close()


async def _ensure_users_schema(db: aiosqlite.Connection) -> None:
    cursor = await db.execute("PRAGMA table_info(users)")
    columns = {row["name"] for row in await cursor.fetchall()}

    column_definitions = {
        "summary_hour": "INTEGER NOT NULL DEFAULT 8",
        "summary_minute": "INTEGER NOT NULL DEFAULT 0",
        "summary_timezone": "TEXT NOT NULL DEFAULT 'Asia/Manila'",
        "is_paused": "INTEGER NOT NULL DEFAULT 0",
        "summary_days": "TEXT NOT NULL DEFAULT 'Mon,Tue,Wed,Thu,Fri,Sat,Sun'",
        "summary_style": "TEXT NOT NULL DEFAULT 'brief'",
        "summary_length": "TEXT NOT NULL DEFAULT 'medium'",
        "summary_focus": "TEXT NOT NULL DEFAULT 'all'",
        "summary_prompt_mode": "TEXT NOT NULL DEFAULT 'structured'",
        "reply_tone": "TEXT NOT NULL DEFAULT 'friendly, concise, and professional'",
        "draft_replies_high": "INTEGER NOT NULL DEFAULT 1",
        "draft_replies_medium": "INTEGER NOT NULL DEFAULT 1",
        "draft_replies_low": "INTEGER NOT NULL DEFAULT 0",
        "include_reminders": "INTEGER NOT NULL DEFAULT 1",
    }
    for column_name, column_sql in column_definitions.items():
        if column_name not in columns:
            await db.execute(f"ALTER TABLE users ADD COLUMN {column_name} {column_sql}")

    settings = get_settings()
    await db.execute(
        """
        UPDATE users
        SET summary_hour = COALESCE(summary_hour, 8),
            summary_minute = COALESCE(summary_minute, 0),
            summary_timezone = COALESCE(summary_timezone, ?),
            is_paused = COALESCE(is_paused, 0),
            summary_days = COALESCE(summary_days, 'Mon,Tue,Wed,Thu,Fri,Sat,Sun'),
            summary_style = COALESCE(summary_style, 'brief'),
            summary_length = COALESCE(summary_length, 'medium'),
            summary_focus = COALESCE(summary_focus, 'all'),
            summary_prompt_mode = COALESCE(summary_prompt_mode, 'structured'),
            reply_tone = COALESCE(reply_tone, 'friendly, concise, and professional'),
            draft_replies_high = COALESCE(draft_replies_high, 1),
            draft_replies_medium = COALESCE(draft_replies_medium, 1),
            draft_replies_low = COALESCE(draft_replies_low, 0),
            include_reminders = COALESCE(include_reminders, 1)
        """,
        (settings.timezone,),
    )


async def _get_user_exclusions(db: aiosqlite.Connection, user_id: int) -> list[str]:
    cursor = await db.execute(
        """
        SELECT exclusion_value
        FROM user_exclusions
        WHERE user_id = ?
        ORDER BY exclusion_value ASC
        """,
        (user_id,),
    )
    rows = await cursor.fetchall()
    return [str(row["exclusion_value"]) for row in rows]


async def _hydrate_user(db: aiosqlite.Connection, row: aiosqlite.Row | None) -> dict[str, Any] | None:
    if not row:
        return None
    user = dict(row)
    user["is_paused"] = bool(user.get("is_paused"))
    user["exclusions"] = await _get_user_exclusions(db, int(user["id"]))
    return user


async def save_user(gchat_user_id: str, gchat_space_id: str) -> dict[str, Any]:
    db = await _connect()
    try:
        await db.execute(
            """
            INSERT INTO users (gchat_user_id, gchat_space_id)
            VALUES (?, ?)
            ON CONFLICT(gchat_user_id, gchat_space_id)
            DO UPDATE SET gchat_user_id = excluded.gchat_user_id
            """,
            (gchat_user_id, gchat_space_id),
        )
        await db.commit()
        cursor = await db.execute(
            f"""
            SELECT {USER_SELECT_COLUMNS}
            FROM users
            WHERE gchat_user_id = ? AND gchat_space_id = ?
            """,
            (gchat_user_id, gchat_space_id),
        )
        row = await cursor.fetchone()
        return await _hydrate_user(db, row) or {}
    finally:
        await db.close()


async def get_user_by_install_id(install_id: int) -> dict[str, Any] | None:
    db = await _connect()
    try:
        cursor = await db.execute(
            f"""
            SELECT {USER_SELECT_COLUMNS}
            FROM users
            WHERE id = ?
            """,
            (install_id,),
        )
        row = await cursor.fetchone()
        return await _hydrate_user(db, row)
    finally:
        await db.close()


async def get_user_by_chat_ids(gchat_user_id: str, gchat_space_id: str) -> dict[str, Any] | None:
    db = await _connect()
    try:
        cursor = await db.execute(
            f"""
            SELECT {USER_SELECT_COLUMNS}
            FROM users
            WHERE gchat_user_id = ? AND gchat_space_id = ?
            """,
            (gchat_user_id, gchat_space_id),
        )
        row = await cursor.fetchone()
        return await _hydrate_user(db, row)
    finally:
        await db.close()


async def get_all_users() -> list[dict[str, Any]]:
    db = await _connect()
    try:
        cursor = await db.execute(
            f"""
            SELECT {USER_SELECT_COLUMNS}
            FROM users
            ORDER BY id ASC
            """
        )
        rows = await cursor.fetchall()
        return [user for row in rows if (user := await _hydrate_user(db, row))]
    finally:
        await db.close()


async def update_token(install_id: int, gmail_token_json: str | dict[str, Any]) -> None:
    token_payload = gmail_token_json if isinstance(gmail_token_json, str) else json.dumps(gmail_token_json)
    db = await _connect()
    try:
        await db.execute(
            "UPDATE users SET gmail_token_json = ? WHERE id = ?",
            (token_payload, install_id),
        )
        await db.commit()
    finally:
        await db.close()


async def update_summary_schedule(
    install_id: int,
    *,
    summary_hour: int,
    summary_minute: int,
    summary_timezone: str,
) -> None:
    db = await _connect()
    try:
        await db.execute(
            """
            UPDATE users
            SET summary_hour = ?, summary_minute = ?, summary_timezone = ?
            WHERE id = ?
            """,
            (summary_hour, summary_minute, summary_timezone, install_id),
        )
        await db.commit()
    finally:
        await db.close()


async def update_user_preferences(
    install_id: int,
    *,
    is_paused: bool | None = None,
    summary_days: str | None = None,
    summary_style: str | None = None,
    summary_length: str | None = None,
    summary_focus: str | None = None,
    summary_prompt_mode: str | None = None,
    reply_tone: str | None = None,
    draft_replies_high: bool | None = None,
    draft_replies_medium: bool | None = None,
    draft_replies_low: bool | None = None,
    include_reminders: bool | None = None,
) -> None:
    updates: list[str] = []
    values: list[Any] = []

    if is_paused is not None:
        updates.append("is_paused = ?")
        values.append(int(is_paused))
    if summary_days is not None:
        updates.append("summary_days = ?")
        values.append(summary_days)
    if summary_style is not None:
        updates.append("summary_style = ?")
        values.append(summary_style)
    if summary_length is not None:
        updates.append("summary_length = ?")
        values.append(summary_length)
    if summary_focus is not None:
        updates.append("summary_focus = ?")
        values.append(summary_focus)
    if summary_prompt_mode is not None:
        updates.append("summary_prompt_mode = ?")
        values.append(summary_prompt_mode)
    if reply_tone is not None:
        updates.append("reply_tone = ?")
        values.append(reply_tone)
    if draft_replies_high is not None:
        updates.append("draft_replies_high = ?")
        values.append(int(draft_replies_high))
    if draft_replies_medium is not None:
        updates.append("draft_replies_medium = ?")
        values.append(int(draft_replies_medium))
    if draft_replies_low is not None:
        updates.append("draft_replies_low = ?")
        values.append(int(draft_replies_low))
    if include_reminders is not None:
        updates.append("include_reminders = ?")
        values.append(int(include_reminders))

    if not updates:
        return

    db = await _connect()
    try:
        values.append(install_id)
        await db.execute(
            f"UPDATE users SET {', '.join(updates)} WHERE id = ?",
            values,
        )
        await db.commit()
    finally:
        await db.close()


async def add_user_exclusion(install_id: int, exclusion_value: str) -> None:
    db = await _connect()
    try:
        await db.execute(
            """
            INSERT INTO user_exclusions (user_id, exclusion_value)
            VALUES (?, ?)
            ON CONFLICT(user_id, exclusion_value)
            DO NOTHING
            """,
            (install_id, exclusion_value.lower()),
        )
        await db.commit()
    finally:
        await db.close()


async def delete_user(gchat_user_id: str, gchat_space_id: str) -> None:
    db = await _connect()
    try:
        await db.execute(
            "DELETE FROM users WHERE gchat_user_id = ? AND gchat_space_id = ?",
            (gchat_user_id, gchat_space_id),
        )
        await db.commit()
    finally:
        await db.close()


async def create_audio_link(gchat_user_id: str, object_name: str) -> str:
    import secrets

    db = await _connect()
    try:
        for _ in range(5):
            token = secrets.token_urlsafe(6)
            try:
                await db.execute(
                    """
                    INSERT INTO audio_links (token, gchat_user_id, object_name)
                    VALUES (?, ?, ?)
                    """,
                    (token, gchat_user_id, object_name),
                )
                await db.commit()
                return token
            except aiosqlite.IntegrityError:
                continue
    finally:
        await db.close()

    raise RuntimeError("Failed to create a unique audio link token.")


async def get_chat_conversation_state(gchat_user_id: str, gchat_space_id: str) -> dict[str, Any] | None:
    db = await _connect()
    try:
        cursor = await db.execute(
            """
            SELECT state_json, created_at, updated_at
            FROM chat_conversation_state
            WHERE gchat_user_id = ? AND gchat_space_id = ?
            """,
            (gchat_user_id, gchat_space_id),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        state = json.loads(str(row["state_json"]))
        state["_meta"] = {
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        return state
    finally:
        await db.close()


async def save_chat_conversation_state(
    gchat_user_id: str,
    gchat_space_id: str,
    state: dict[str, Any],
) -> None:
    db = await _connect()
    try:
        payload = json.dumps(state)
        await db.execute(
            """
            INSERT INTO chat_conversation_state (gchat_user_id, gchat_space_id, state_json)
            VALUES (?, ?, ?)
            ON CONFLICT(gchat_user_id, gchat_space_id)
            DO UPDATE SET state_json = excluded.state_json,
                          updated_at = CURRENT_TIMESTAMP
            """,
            (gchat_user_id, gchat_space_id, payload),
        )
        await db.commit()
    finally:
        await db.close()


async def clear_chat_conversation_state(gchat_user_id: str, gchat_space_id: str) -> None:
    db = await _connect()
    try:
        await db.execute(
            """
            DELETE FROM chat_conversation_state
            WHERE gchat_user_id = ? AND gchat_space_id = ?
            """,
            (gchat_user_id, gchat_space_id),
        )
        await db.commit()
    finally:
        await db.close()


async def get_audio_link(token: str) -> dict[str, Any] | None:
    db = await _connect()
    try:
        cursor = await db.execute(
            """
            SELECT token, gchat_user_id, object_name, created_at
            FROM audio_links
            WHERE token = ?
            """,
            (token,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None
    finally:
        await db.close()
