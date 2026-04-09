from __future__ import annotations

from io import BytesIO

import edge_tts

DEFAULT_TTS_VOICE = "en-US-GuyNeural"


async def generate_audio(text: str) -> bytes:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Cannot generate audio for empty text.")

    buffer = BytesIO()
    communicate = edge_tts.Communicate(cleaned_text, DEFAULT_TTS_VOICE)

    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            buffer.write(chunk["data"])

    audio_bytes = buffer.getvalue()
    if not audio_bytes:
        raise RuntimeError("Edge TTS returned no audio data.")

    return audio_bytes
