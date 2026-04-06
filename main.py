from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from auth import router as auth_router
from db import init_db
from gchat import router as gchat_router
from scheduler import build_scheduler, schedule_all_user_summaries


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    scheduler = build_scheduler()
    scheduler.start()
    await schedule_all_user_summaries(scheduler)
    app.state.scheduler = scheduler
    try:
        yield
    finally:
        scheduler.shutdown(wait=False)


app = FastAPI(title="Gmail GChat Bot", lifespan=lifespan)
app.include_router(auth_router)
app.include_router(gchat_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
