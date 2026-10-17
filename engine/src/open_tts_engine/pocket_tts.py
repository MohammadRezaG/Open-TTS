"""Pocket TTS implementation of the speech engine contract."""

from __future__ import annotations

from io import BytesIO
from itertools import chain
from pathlib import Path
from threading import Event, Lock
from typing import Iterator
import wave

import numpy as np
from pocket_tts import TTSModel

from .audio import AudioChunk, SilenceTrimmer, change_tempo, pause_duration
from .chunking import TextChunker
from .options import GenerationOptions
from .persian import PersianPhonemizer, normalize_persian
from .settings import EngineSettings, resolve_device

_POCKET_PHONEME_MAP = str.maketrans(
    {"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"}
)


def normalize_pocket_phonemes(phonemes: str) -> str:
    """Format G2P output for the Pocket TTS Farsi phoneme vocabulary."""
    return phonemes.strip().replace("1", "").translate(_POCKET_PHONEME_MAP)


class PocketTtsEngine:
    def __init__(
        self,
        model: TTSModel,
        phonemizer: PersianPhonemizer | None,
        default_voice_state: dict,
        chunker: TextChunker | None = None,
    ) -> None:
        self._model = model
        self._phonemizer = phonemizer
        self._default_voice_state = default_voice_state
        self._chunker = chunker or TextChunker()
        self._generation_lock = Lock()

    @classmethod
    def from_pretrained(cls, settings: EngineSettings | None = None) -> "PocketTtsEngine":
        settings = settings or EngineSettings.from_environment()
        config_path = (
            settings.model_path / "model.yaml"
            if settings.model_path is not None
            else settings.tts_config
        )
        model = TTSModel.load_model(config=str(config_path), temp=settings.temperature)
        model.to(resolve_device(settings.device))
        phonemizer = (
            PersianPhonemizer.from_pretrained(str(settings.g2p_path or settings.g2p_model))
            if settings.input_mode == "persian_phonemes"
            else None
        )
        default_voice_state = model.get_state_for_audio_prompt(settings.default_voice)
        return cls(model, phonemizer, default_voice_state)

    def synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
    ) -> bytes:
        output = BytesIO()
        with wave.open(output, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            sample_rate = self._model.sample_rate
            wav_file.setframerate(sample_rate)
            wrote_audio = False
            for audio_chunk in self.stream_synthesize(
                text=text, options=options, voice_path=voice_path
            ):
                if not audio_chunk.pcm_s16le:
                    continue
                if audio_chunk.sample_rate != sample_rate:
                    raise RuntimeError("The speech engine changed sample rate while streaming.")
                wav_file.writeframesraw(audio_chunk.pcm_s16le)
                wrote_audio = True
            if not wrote_audio:
                raise ValueError("No speech audio was generated.")
        return output.getvalue()

    def stream_synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
        stop_event: Event | None = None,
    ) -> Iterator[AudioChunk]:
        if not text.strip():
            raise ValueError("Enter text to synthesize.")

        normalized = normalize_persian(text)
        chunks = self._chunker.split_with_context(
            normalized,
            max_chars=options.max_chars,
            min_chars=options.min_chars,
            split_on_comma=options.split_on_comma,
        )
        if not chunks:
            raise ValueError("No speakable text was found.")

        stop_event = stop_event or Event()
        acquired = False
        while not acquired and not stop_event.is_set():
            acquired = self._generation_lock.acquire(timeout=0.1)
        if not acquired:
            return
        try:
            self._model.eos_threshold = options.eos_threshold
            voice_state = (
                self._model.get_state_for_audio_prompt(voice_path)
                if voice_path
                else self._default_voice_state
            )
            sample_rate = self._model.sample_rate
            index = 0
            while index < len(chunks) and not stop_event.is_set():
                current = chunks[index]
                text_to_generate = current.text
                while True:
                    try:
                        frames = iter(
                            self._generate_one(
                                text_to_generate, voice_state, options, stop_event
                            )
                        )
                        first_frame = next(frames)
                        if not first_frame.size:
                            raise RuntimeError("The model returned an empty audio chunk.")
                        break
                    except StopIteration as exc:
                        if stop_event.is_set():
                            return
                        error: Exception = RuntimeError("The model returned an empty audio chunk.")
                        error.__cause__ = exc
                    except Exception as exc:
                        if stop_event.is_set():
                            return
                        error = exc

                    if (
                        not options.retry_silent_chunks
                        or current.boundary in {"period", "question", "exclamation", "semicolon"}
                        or index + 1 >= len(chunks)
                    ):
                        raise error
                    following = chunks[index + 1]
                    text_to_generate = f"{text_to_generate} {following.text}"
                    current = following
                    index += 1

                trimmer = SilenceTrimmer(sample_rate)
                speech_parts: list[np.ndarray] = []
                for frame in chain((first_frame,), frames):
                    if stop_event.is_set():
                        break
                    for speech in trimmer.feed(frame):
                        if options.post_speed == 1.0:
                            yield self._as_pcm_chunk(speech, sample_rate)
                        else:
                            speech_parts.append(speech)

                if not stop_event.is_set():
                    for speech in trimmer.finish():
                        if options.post_speed == 1.0:
                            yield self._as_pcm_chunk(speech, sample_rate)
                        else:
                            speech_parts.append(speech)

                if options.post_speed != 1.0 and speech_parts and not stop_event.is_set():
                    stretched = change_tempo(
                        np.concatenate(speech_parts), sample_rate, options.post_speed
                    )
                    yield self._as_pcm_chunk(stretched, sample_rate)

                if stop_event.is_set():
                    break
                gap = pause_duration(current, options.pause_multiplier)
                if gap:
                    silence = np.zeros(int(sample_rate * gap), dtype=np.float32)
                    yield self._as_pcm_chunk(silence, sample_rate)
                if (
                    current.boundary
                    in {"comma", "semicolon", "period", "question", "exclamation"}
                    and (
                        current.next_phrase_words
                        if current.boundary == "comma"
                        else current.next_sentence_words
                    )
                ):
                    yield AudioChunk(sample_rate, b"", pause_boundary=True)
                index += 1

        finally:
            self._generation_lock.release()

    def _generate_one(
        self,
        text: str,
        voice_state: dict,
        options: GenerationOptions,
        stop_event: Event,
    ) -> Iterator[np.ndarray]:
        model_input = (
            normalize_pocket_phonemes(self._phonemizer.phonemize(text))
            if self._phonemizer is not None
            else text
        )
        if stop_event.is_set():
            return
        if not model_input:
            raise ValueError("The Persian text could not be converted to phonemes.")

        for audio in self._model.generate_audio_stream(
            voice_state,
            model_input,
            frames_after_eos=options.frames_after_eos,
            stop=stop_event,
        ):
            yield audio.detach().cpu().numpy().astype(np.float32).reshape(-1)

    @staticmethod
    def _as_pcm_chunk(samples: np.ndarray, sample_rate: int) -> AudioChunk:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2", copy=False)
        return AudioChunk(sample_rate, pcm.tobytes())
