from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse

from storage import resolve_audio_redirect_url

router = APIRouter(tags=["audio"])


@router.get("/a/{token}")
async def audio_redirect(token: str) -> RedirectResponse:
    redirect_url = await resolve_audio_redirect_url(token)
    if not redirect_url:
        raise HTTPException(status_code=404, detail="Audio link not found.")
    return RedirectResponse(url=redirect_url, status_code=302)
