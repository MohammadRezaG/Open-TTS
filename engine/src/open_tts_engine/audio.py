"""Small audio helpers shared by streaming and WAV generation."""

from __future__ import annotations

from dataclasses import dataclass
import shutil
import subprocess
from typing import Iterator

import numpy as np

from .chunking import TextChunk

_PUNCTUATION_PAUSES = {
    "comma": 0.05,
    "semicolon": 0.09,
    "period": 0.12,
    "question": 0.16,
    "exclamation": 0.16,
}


@dataclass(frozen=True, slots=True)
class AudioChunk:
    sample_rate: int
    pcm_s16le: bytes
    pause_boundary: bool = False


def change_tempo(samples: np.ndarray, sample_rate: int, speed: float) -> np.ndarray:
    """Change speech tempo without changing pitch; requires FFmpeg for non-neutral speed."""
    if speed == 1.0 or not samples.size:
        return samples
    executable = shutil.which("ffmpeg")
    if executable is None:
        raise ValueError("Post-processing speed needs FFmpeg on PATH.")
    result = subprocess.run(
        [
            executable, "-hide_banner", "-loglevel", "error", "-f", "f32le",
            "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0", "-af",
            f"atempo={speed}", "-f", "f32le", "pipe:1",
        ],
        input=np.asarray(samples, dtype="<f4").tobytes(),
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"FFmpeg could not adjust speech speed: {result.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(result.stdout, dtype="<f4").copy()


def pause_duration(chunk: TextChunk, multiplier: float) -> float:
    """Estimate a sentence or phrase pause from punctuation and both sides."""
    base = _PUNCTUATION_PAUSES.get(chunk.boundary, 0.0)
    if not base:
        return 0.0
    if chunk.boundary == "comma" and not chunk.next_phrase_words:
        return 0.0
    if chunk.boundary != "comma" and not chunk.next_sentence_words:
        return 0.0

    if chunk.boundary == "comma":
        before = min(chunk.phrase_words, 30) * 0.002
        after = min(chunk.next_phrase_words, 30) * 0.0015
    else:
        before = min(chunk.sentence_words, 30) * 0.004
        after = min(chunk.next_sentence_words, 30) * 0.003
    return min(0.5, base + before + after) * multiplier


class SilenceTrimmer:
    """Drop leading and excessive trailing silence from one generated text chunk."""

    def __init__(self, sample_rate: int) -> None:
        self._window = max(1, sample_rate // 100)
        self._guard = max(1, sample_rate // 25)
        self._buffer = np.empty(0, dtype=np.float32)
        self._quiet: list[np.ndarray] = []
        self._seen_voice = False

    def feed(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        self._buffer = np.concatenate((self._buffer, samples))
        while self._buffer.size >= self._window:
            window, self._buffer = self._buffer[: self._window], self._buffer[self._window :]
            if np.max(np.abs(window), initial=0.0) < 0.0002:
                if self._seen_voice:
                    self._quiet.append(window)
                continue
            if self._seen_voice:
                yield from self._quiet
            self._quiet.clear()
            if not self._seen_voice:
                window = self._fade_in(window)
            self._seen_voice = True
            yield window

    def finish(self) -> Iterator[np.ndarray]:
        if self._buffer.size:
            remainder = np.pad(self._buffer, (0, self._window - self._buffer.size))
            self._buffer = np.empty(0, dtype=np.float32)
            yield from self.feed(remainder)
        if self._quiet:
            tail = np.concatenate(self._quiet)[-self._guard :].copy()
            if tail.size:
                fade = min(self._window // 2, tail.size)
                tail[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
                yield tail
        self._quiet.clear()

    @staticmethod
    def _fade_in(samples: np.ndarray) -> np.ndarray:
        samples = samples.copy()
        fade = min(samples.size, max(1, len(samples) // 2))
        samples[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        return samples



