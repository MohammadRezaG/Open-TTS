"""Model identifiers and local engine settings."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class EngineSettings:
    tts_config: str = "hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml"
    g2p_model: str = "mehdi-hf/Homo-GE2PE-Persian-HF"
    default_voice: Path = Path(__file__).parent / "assets" / "example_voice.wav"
    temperature: float = 0.3

    @classmethod
    def from_environment(cls) -> "EngineSettings":
        defaults = cls()
        return cls(
            tts_config=os.getenv("OPEN_TTS_TTS_CONFIG", defaults.tts_config),
            g2p_model=os.getenv("OPEN_TTS_G2P_MODEL", defaults.g2p_model),
            default_voice=Path(
                os.getenv("OPEN_TTS_DEFAULT_VOICE", str(defaults.default_voice))
            ),
            temperature=float(os.getenv("OPEN_TTS_TEMPERATURE", str(defaults.temperature))),
        )
