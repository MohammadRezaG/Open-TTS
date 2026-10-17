import unittest

from open_tts_engine.language import english_numbers_to_words, split_language_chunks
from open_tts_engine.options import GenerationOptions
from open_tts_engine.router import LanguageRoutedEngine
from open_tts_engine.audio import AudioChunk


class LanguageRoutingTests(unittest.TestCase):
    def test_auto_splits_mixed_language_and_keeps_punctuation(self):
        chunks = split_language_chunks("سلام, hello 12 دنیا.", "auto")
        self.assertEqual(chunks, [("fa", "سلام,"), ("en", "hello 12"), ("fa", "دنیا.")])

    def test_language_can_be_selected_without_detection(self):
        self.assertEqual(split_language_chunks("hello 12", "en"), [("en", "hello 12")])

    def test_english_number_expansion_integers_and_ids(self):
        self.assertEqual(english_numbers_to_words("I have 21 books and 007."), "I have twenty-one books and zero zero seven.")

    def test_router_expands_numbers_and_uses_matching_engine(self):
        class FakeEngine:
            def __init__(self):
                self.text = None

            def stream_synthesize(self, *, text, options, voice_path, stop_event):
                self.text = text
                yield AudioChunk(24000, b"\x00\x00")

        english = FakeEngine()
        router = LanguageRoutedEngine({"en": english})
        list(router.stream_synthesize(text="hello 1", options=GenerationOptions(language="auto")))
        self.assertEqual(english.text, "hello one")


if __name__ == "__main__":
    unittest.main()
