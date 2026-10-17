"""Local ONNX Runtime adapter for Kitten TTS Mini."""

from __future__ import annotations

from io import BytesIO
import json
from pathlib import Path
from threading import Event, Lock
from typing import Iterator
import wave

import numpy as np

from .audio import AudioChunk, change_tempo, pause_duration
from .chunking import TextChunker
from .options import GenerationOptions


def _asset_path(model_dir: Path, name: object) -> Path:
    if not isinstance(name, str) or not name:
        raise ValueError("Kitten config contains an invalid asset path.")
    root = model_dir.resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Kitten config refers to an asset outside the model folder.")
    return path


class KittenTtsEngine:
    sample_rate = 24000

    def __init__(self, model: object, voices: tuple[str, ...], chunker: TextChunker | None = None):
        self._model = model
        self.voices = voices
        self._chunker = chunker or TextChunker()
        self._lock = Lock()

    @classmethod
    def from_pretrained(cls, model_dir: Path, device: str = "auto") -> "KittenTtsEngine":
        config_path = model_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        model_path = _asset_path(model_dir, config["model_file"])
        voices_path = _asset_path(model_dir, config["voices"])
        if not model_path.is_file() or not voices_path.is_file():
            raise ValueError("Kitten model installation is incomplete; install the model again.")
        try:
            from kittentts.onnx_model import KittenTTS_1_Onnx
        except ImportError as exc:
            raise RuntimeError("Install the Kitten TTS backend before selecting this model.") from exc
        backend = "cuda" if device == "cuda" else None
        model = KittenTTS_1_Onnx(
            model_path=str(model_path),
            voices_path=str(voices_path),
            speed_priors=config.get("speed_priors", {}),
            voice_aliases=config.get("voice_aliases", {}),
            backend=backend,
        )
        voices = tuple(getattr(model, "all_voice_names", ()))
        if not voices:
            raise ValueError("Kitten Mini did not expose any built-in voices.")
        return cls(model, voices)

    def stream_synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
        stop_event: Event | None = None,
    ) -> Iterator[AudioChunk]:
        del voice_path  # Kitten Mini uses its built-in named voices.
        if not text.strip():
            raise ValueError("Enter text to synthesize.")
        chunks = self._chunker.split_with_context(
            text,
            max_chars=options.max_chars,
            min_chars=options.min_chars,
            split_on_comma=options.split_on_comma,
        )
        if not chunks:
            raise ValueError("No speakable text was found.")
        stop_event = stop_event or Event()
        voice = self.voices[0] if options.voice_name == "Default" and self.voices else options.voice_name
        if self.voices and voice not in self.voices:
            raise ValueError(f"Voice {voice!r} is not available for Kitten Mini.")
        with self._lock:
            for chunk in chunks:
                if stop_event.is_set():
                    return
                buffered: list[np.ndarray] = []
                for frame in self._model.generate_stream(
                    chunk.text, voice=voice, speed=options.model_speed
                ):
                    if stop_event.is_set():
                        break
                    samples = np.asarray(frame, dtype=np.float32).reshape(-1)
                    if options.post_speed == 1.0:
                        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2", copy=False)
                        if pcm.size:
                            yield AudioChunk(self.sample_rate, pcm.tobytes())
                    else:
                        buffered.append(samples)
                if buffered and not stop_event.is_set():
                    samples = change_tempo(
                        np.concatenate(buffered), self.sample_rate, options.post_speed
                    )
                    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2", copy=False)
                    if pcm.size:
                        yield AudioChunk(self.sample_rate, pcm.tobytes())
                if stop_event.is_set():
                    return
                gap = pause_duration(chunk, options.pause_multiplier)
                if gap and not stop_event.is_set():
                    silence = np.zeros(int(self.sample_rate * gap), dtype="<i2")
                    yield AudioChunk(self.sample_rate, silence.tobytes())
                if chunk.boundary in {"comma", "semicolon", "period", "question", "exclamation"}:
                    yield AudioChunk(self.sample_rate, b"", pause_boundary=True)

    def synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
        stop_event: Event | None = None,
    ) -> bytes:
        output = BytesIO()
        with wave.open(output, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.sample_rate)
            wrote_audio = False
            for chunk in self.stream_synthesize(
                text=text, options=options, voice_path=voice_path, stop_event=stop_event
            ):
                if chunk.pcm_s16le:
                    wav.writeframesraw(chunk.pcm_s16le)
                    wrote_audio = True
            if not wrote_audio:
                raise ValueError("No speech audio was generated.")
        return output.getvalue()
