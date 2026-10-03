"""Generation settings shared by the API and model engine."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_chars: int = 100
    min_chars: int = 30
    split_on_comma: bool = True
    retry_silent_chunks: bool = True
    frames_after_eos: int = 2
    eos_threshold: float = -4.0
