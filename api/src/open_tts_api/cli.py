"""Run the local API with Uvicorn."""

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "open_tts_api.main:app",
        host=os.getenv("OPEN_TTS_API_HOST", "127.0.0.1"),
        port=int(os.getenv("OPEN_TTS_API_PORT", "8000")),
    )
