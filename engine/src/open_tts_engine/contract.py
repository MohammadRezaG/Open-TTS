"""Stable interface between the API and a speech generation backend."""

from pathlib import Path
from typing import Protocol

from .options import GenerationOptions


class SpeechEngine(Protocol):
    def synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
    ) -> bytes:
        """Generate speech and return it as WAV bytes."""
