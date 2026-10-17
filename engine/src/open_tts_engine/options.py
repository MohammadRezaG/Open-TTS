"""Generation settings shared by the API and model engine."""

from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class GenerationOptions:
    max_chars: int = 100
    min_chars: int = 30
    split_on_comma: bool = True
    retry_silent_chunks: bool = True
    frames_after_eos: int = 2
    eos_threshold: float = -4.0
    pause_multiplier: float = 1.0
    model_speed: float = 1.0
    post_speed: float = 1.0
    voice_name: str = "Default"
    language: str = "fa"

    def __post_init__(self) -> None:
        if not math.isfinite(self.pause_multiplier) or not 0 <= self.pause_multiplier <= 2:
            raise ValueError("Pause length must be between 0 and 2.")
        if not math.isfinite(self.model_speed) or not 0.5 <= self.model_speed <= 2:
            raise ValueError("Model speed must be between 0.5 and 2.")
        if not math.isfinite(self.post_speed) or not 0.5 <= self.post_speed <= 2:
            raise ValueError("Post-processing speed must be between 0.5 and 2.")
        if self.language not in {"fa", "en", "auto"}:
            raise ValueError("Language must be Persian, English, or Auto.")
