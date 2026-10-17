"""Stable interface between the API and a speech generation backend."""

from pathlib import Path
from threading import Event
from typing import Iterator, Protocol

from .audio import AudioChunk

from .options import GenerationOptions


class SpeechEngine(Protocol):
    def synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
        stop_event: Event | None = None,
    ) -> bytes:
        """Generate speech and return it as WAV bytes."""

    def stream_synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
    ) -> Iterator[AudioChunk]:
        """Yield mono signed 16-bit PCM blocks at the provider sample rate."""
