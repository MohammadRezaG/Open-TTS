"""Pocket TTS implementation of the speech engine contract."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from threading import Lock

import numpy as np
import scipy.io.wavfile
from pocket_tts import TTSModel

from .chunking import TextChunker
from .options import GenerationOptions
from .persian import PersianPhonemizer, normalize_persian
from .settings import EngineSettings


class PocketTtsEngine:
    def __init__(
        self,
        model: TTSModel,
        phonemizer: PersianPhonemizer,
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
        model = TTSModel.load_model(config=settings.tts_config, temp=settings.temperature)
        phonemizer = PersianPhonemizer.from_pretrained(settings.g2p_model)
        default_voice_state = model.get_state_for_audio_prompt(settings.default_voice)
        return cls(model, phonemizer, default_voice_state)

    def synthesize(
        self,
        *,
        text: str,
        options: GenerationOptions,
        voice_path: Path | None = None,
    ) -> bytes:
        if not text.strip():
            raise ValueError("Enter text to synthesize.")

        normalized = normalize_persian(text)
        chunks = self._chunker.split(
            normalized,
            max_chars=options.max_chars,
            min_chars=options.min_chars,
            split_on_comma=options.split_on_comma,
        )
        if not chunks:
            raise ValueError("No speakable text was found.")

        with self._generation_lock:
            self._model.eos_threshold = options.eos_threshold
            voice_state = (
                self._model.get_state_for_audio_prompt(voice_path)
                if voice_path
                else self._default_voice_state
            )
            samples = self._generate_chunks(chunks, voice_state, options)

        return self._to_wav(samples)

    def _generate_chunks(
        self,
        chunks: list[str],
        voice_state: dict,
        options: GenerationOptions,
    ) -> np.ndarray:
        audio_chunks: list[np.ndarray] = []
        index = 0
        silence = np.zeros(int(self._model.sample_rate * 0.25), dtype=np.float32)

        while index < len(chunks):
            text = chunks[index]
            try:
                audio = self._generate_one(text, voice_state, options)
                if not audio.size:
                    raise RuntimeError("The model returned an empty audio chunk.")
            except Exception:
                sentence_end = text.rstrip().endswith((".", "!", "?", "؟", "؛"))
                if (
                    not options.retry_silent_chunks
                    or sentence_end
                    or index + 1 >= len(chunks)
                ):
                    raise
                text = f"{text} {chunks[index + 1]}"
                audio = self._generate_one(text, voice_state, options)
                index += 1

            audio_chunks.append(audio)
            if index < len(chunks) - 1:
                audio_chunks.append(silence)
            index += 1

        if not audio_chunks:
            raise RuntimeError("The model did not produce audio.")
        return np.concatenate(audio_chunks)

    def _generate_one(
        self,
        text: str,
        voice_state: dict,
        options: GenerationOptions,
    ) -> np.ndarray:
        phonemes = self._phonemizer.convert(text)
        if not phonemes:
            raise ValueError("The Persian text could not be converted to phonemes.")

        audio = self._model.generate_audio(
            voice_state,
            phonemes,
            frames_after_eos=options.frames_after_eos or None,
        )
        return audio.detach().cpu().numpy().astype(np.float32).reshape(-1)

    def _to_wav(self, samples: np.ndarray) -> bytes:
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        output = BytesIO()
        scipy.io.wavfile.write(output, self._model.sample_rate, pcm)
        return output.getvalue()
