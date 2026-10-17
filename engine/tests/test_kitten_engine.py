import unittest

import numpy as np

from open_tts_engine.audio import AudioChunk
from open_tts_engine.kitten_tts import KittenTtsEngine
from open_tts_engine.options import GenerationOptions
from open_tts_engine.settings import EngineSettings


class FakeKittenModel:
    all_voice_names = ["Bella", "Jasper"]

    def __init__(self):
        self.calls = []

    def generate_stream(self, text, *, voice, speed):
        self.calls.append((text, voice, speed))
        yield np.ones(8, dtype=np.float32) * 0.25


class KittenEngineTests(unittest.TestCase):
    def test_native_speed_and_named_voice_are_passed_to_model(self):
        model = FakeKittenModel()
        engine = KittenTtsEngine(model, tuple(model.all_voice_names))
        chunks = list(engine.stream_synthesize(
            text="hello world.", options=GenerationOptions(model_speed=1.4, voice_name="Jasper")
        ))
        self.assertEqual(model.calls[0], ("hello world.", "Jasper", 1.4))
        self.assertIsInstance(chunks[0], AudioChunk)
        self.assertEqual(len(chunks[0].pcm_s16le), 16)

    def test_unknown_named_voice_is_rejected(self):
        engine = KittenTtsEngine(FakeKittenModel(), ("Bella", "Jasper"))
        with self.assertRaisesRegex(ValueError, "not available"):
            list(engine.stream_synthesize(
                text="hello.", options=GenerationOptions(voice_name="Unknown")
            ))

    def test_model_runtime_pairs_are_validated(self):
        settings = EngineSettings(model_id="kitten_mini_08", runtime="onnxruntime")
        self.assertEqual(settings.model_id, "kitten_mini_08")
        with self.assertRaisesRegex(ValueError, "Unsupported inference runtime"):
            EngineSettings(model_id="kitten_mini_08", runtime="pytorch")


if __name__ == "__main__":
    unittest.main()
