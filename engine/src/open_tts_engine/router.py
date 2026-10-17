"""Route language chunks to already-loaded compatible speech engines."""

from __future__ import annotations

from io import BytesIO
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Iterator, Mapping
import wave

import numpy as np

from .audio import AudioChunk
from .language import english_numbers_to_words, split_language_chunks
from .options import GenerationOptions


class LanguageRoutedEngine:
    def __init__(self, engines: Mapping[str, object]) -> None:
        self._engines = dict(engines)

    def set_engine(self, language: str, engine: object) -> None:
        self._engines[language] = engine

    def stream_synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
        stop_event: Event | None = None,
    ) -> Iterator[AudioChunk]:
        stop_event = stop_event or Event()
        previous_language = None
        for language, part in split_language_chunks(text, options.language):
            if stop_event.is_set():
                return
            engine = self._engines.get(language)
            if engine is None:
                raise ValueError(f"Install a {language.upper()} speech model to synthesize this text.")
            if previous_language and previous_language != language:
                yield AudioChunk(24000, np.zeros(1920, dtype="<i2").tobytes(), pause_boundary=True)
            routed_options = replace(options, language=language)
            if language == "en":
                part = english_numbers_to_words(part)
            else:
                routed_options = replace(routed_options, model_speed=1.0)
            yield from engine.stream_synthesize(
                text=part,
                options=routed_options,
                voice_path=voice_path if language == "fa" else None,
                stop_event=stop_event,
            )
            previous_language = language

    def synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
        stop_event: Event | None = None,
    ) -> bytes:
        output = BytesIO()
        sample_rate = 24000
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wrote_audio = False
            for chunk in self.stream_synthesize(
                text=text, options=options, voice_path=voice_path, stop_event=stop_event
            ):
                if chunk.pcm_s16le:
                    if chunk.sample_rate != sample_rate:
                        raise ValueError("Routed models must use the same sample rate.")
                    wav.writeframesraw(chunk.pcm_s16le)
                    wrote_audio = True
            if not wrote_audio:
                raise ValueError("No speech audio was generated.")
        return output.getvalue()
