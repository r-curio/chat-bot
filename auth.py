from __future__ import annotations

import asyncio
import base64
import json
import secrets

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from google_auth_oauthlib.flow import Flow

from config import get_settings
from db import get_user_by_install_id, update_token
from scheduler import reschedule_summary_for_user

router = APIRouter(prefix="/auth", tags=["auth"])

GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]


def _encode_state(install_id: int, code_verifier: str) -> str:
    raw = json.dumps({"install_id": install_id, "code_verifier": code_verifier}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_state(state: str) -> dict:
    padded_state = state + "=" * (-len(state) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(padded_state.encode()).decode())
    except (ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid OAuth state.") from exc


def _build_flow(state: str | None = None, code_verifier: str | None = None) -> Flow:
    settings = get_settings()
    flow = Flow.from_client_secrets_file(
        settings.credentials_json_path,
        scopes=GMAIL_SCOPES,
        state=state,
        redirect_uri=settings.callback_url,
    )
    if code_verifier:
        flow.code_verifier = code_verifier
    return flow


@router.get("/start")
async def auth_start(
    install_id: int = Query(..., description="Internal installation id for the Chat user/space pair."),
) -> RedirectResponse:
    user = await get_user_by_install_id(install_id)
    if not user:
        raise HTTPException(status_code=404, detail="Installation not found.")

    code_verifier = secrets.token_urlsafe(64)
    state = _encode_state(install_id, code_verifier)
    flow = _build_flow(state=state, code_verifier=code_verifier)
    authorization_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
        code_challenge_method="S256",
    )
    return RedirectResponse(url=authorization_url)


@router.get("/callback")
async def auth_callback(request: Request, code: str, state: str) -> HTMLResponse:
    state_payload = _decode_state(state)
    install_id = state_payload.get("install_id")
    code_verifier = state_payload.get("code_verifier")
    if not isinstance(install_id, int) or not isinstance(code_verifier, str):
        raise HTTPException(status_code=400, detail="Invalid OAuth state.")

    user = await get_user_by_install_id(install_id)
    if not user:
        raise HTTPException(status_code=404, detail="Installation not found.")

    flow = _build_flow(state=state, code_verifier=code_verifier)
    await asyncio.to_thread(flow.fetch_token, code=code)
    await update_token(install_id, flow.credentials.to_json())
    await reschedule_summary_for_user(request.app.state.scheduler, install_id)

    return HTMLResponse(
        "<h2>Gmail connected.</h2><p>You can return to Google Chat and use /summary, /settime, or /help.</p>"
    )
