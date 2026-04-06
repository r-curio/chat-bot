# Gmail GChat Bot

## Run with uv

```bash
uv sync
cp .env.example .env
uv run uvicorn main:app --reload
```

## Required files

- `credentials.json` at the repo root for the Gmail OAuth client. Keep it local and do not commit it.
- `chat-service-account.json` at the repo root, or another path referenced by `GOOGLE_CHAT_SERVICE_ACCOUNT_FILE`, for Google Chat app authentication. Keep it local and do not commit it.

## Google Chat commands

- `/summary` to fetch a summary immediately.
- `/testsummary` to preview a summary using your saved preferences.
- `/settime HH:MM Area/City` to set the daily summary time and timezone, for example `/settime 08:00 Asia/Manila`.
- `/timezones` to list the available timezone names.
- `/timezones Asia` to filter the timezone list by a term.
- `/settings` to view your current preferences.
- `/pause` and `/resume` to stop or restart scheduled summaries.
- `/setdays Mon,Tue,Fri` to choose delivery days.
- `/style brief|detailed|executive|bullets` to change summary style.
- `/length short|medium|long` to change summary length.
- `/focus urgent|action-items|meetings|all` to change summary focus.
- `/exclude sender@company.com` to exclude a sender or domain.
- `/help` to show the available commands.
