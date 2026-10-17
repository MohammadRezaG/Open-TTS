"""Model identifiers and local engine settings."""

import os
from dataclasses import dataclass
from pathlib import Path

_INPUT_MODES = {"persian_phonemes", "persian_text"}
_MODEL_RUNTIMES = {
    "pocket_farsi_v2": {"pytorch"},
    "kitten_mini_08": {"onnxruntime"},
}
_DEVICES = {"auto", "cpu", "cuda"}


def resolve_device(device: str) -> str:
    """Resolve the requested device and reject unavailable CUDA explicitly."""
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable. Install a CUDA-enabled PyTorch build and restart the API.")
    return device


@dataclass(frozen=True, slots=True)
class EngineSettings:
    tts_config: str = "hf://mehdi-hf/pocket-tts-farsi-v2/model.yaml"
    g2p_model: str = "mehdi-hf/Homo-GE2PE-Persian-HF"
    default_voice: Path = Path(__file__).parent / "assets" / "example_voice.wav"
    temperature: float = 0.3
    input_mode: str = "persian_phonemes"
    model_id: str = "pocket_farsi_v2"
    runtime: str = "pytorch"
    device: str = "auto"
    model_path: Path | None = None
    g2p_path: Path | None = None

    def __post_init__(self) -> None:
        if self.input_mode not in _INPUT_MODES:
            choices = ", ".join(sorted(_INPUT_MODES))
            raise ValueError(f"OPEN_TTS_INPUT_MODE must be one of: {choices}.")
        if self.model_id not in _MODEL_RUNTIMES:
            raise ValueError(f"Unsupported TTS model: {self.model_id}.")
        if self.runtime not in _MODEL_RUNTIMES[self.model_id]:
            raise ValueError(f"Unsupported inference runtime: {self.runtime}.")
        if self.device not in _DEVICES:
            choices = ", ".join(sorted(_DEVICES))
            raise ValueError(f"OPEN_TTS_DEVICE must be one of: {choices}.")

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
            input_mode=os.getenv("OPEN_TTS_INPUT_MODE", defaults.input_mode).strip().lower(),
            model_id=os.getenv("OPEN_TTS_MODEL", defaults.model_id).strip().lower(),
            runtime=os.getenv("OPEN_TTS_RUNTIME", defaults.runtime).strip().lower(),
            device=os.getenv("OPEN_TTS_DEVICE", defaults.device).strip().lower(),
        )
