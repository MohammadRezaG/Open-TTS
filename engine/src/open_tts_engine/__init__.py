"""Persian speech engine for Open TTS, loaded on demand."""

from importlib import import_module

__all__ = [
    "EngineSettings", "GenerationOptions", "KittenTtsEngine",
    "LanguageRoutedEngine", "PocketTtsEngine", "SpeechEngine",
]

_EXPORTS = {
    "EngineSettings": ("settings", "EngineSettings"),
    "GenerationOptions": ("options", "GenerationOptions"),
    "KittenTtsEngine": ("kitten_tts", "KittenTtsEngine"),
    "LanguageRoutedEngine": ("router", "LanguageRoutedEngine"),
    "PocketTtsEngine": ("pocket_tts", "PocketTtsEngine"),
    "SpeechEngine": ("contract", "SpeechEngine"),
}


def __getattr__(name: str):
    try:
        module_name, member_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    return getattr(import_module(f".{module_name}", __name__), member_name)
