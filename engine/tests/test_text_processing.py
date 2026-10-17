import io
import wave

import numpy as np
import torch

from open_tts_engine.audio import SilenceTrimmer, pause_duration
from open_tts_engine.chunking import TextChunker
from open_tts_engine.options import GenerationOptions
from open_tts_engine.persian import normalize_persian
from open_tts_engine.pocket_tts import PocketTtsEngine
from open_tts_engine.pocket_tts import normalize_pocket_phonemes


assert normalize_persian("1") == "یک"
assert normalize_persian("۱۲۳") == "صد و بیست و سه"
assert normalize_persian("قیمت 1,250 تومان است.") == "قیمت هزار و دویست و پنجاه تومان است."
assert normalize_persian("شماره من 0912 است.") == "شماره من صفر نه یک دو است."
assert normalize_persian("سلام   دنیا  ؟") == "سلام دنیا؟"
assert normalize_pocket_phonemes("/a@$c1") == "aA?SC"

assert TextChunker().split(
    normalize_persian("١. ٢."), max_chars=100, min_chars=30
) == ["یک.", "دو."]

sentences = TextChunker().split_with_context("یک دو سه. چهار پنج شش هفت.", max_chars=100)
assert sentences[0].boundary == "period"
assert abs(pause_duration(sentences[0], 1.0) - 0.144) < 1e-9
assert abs(pause_duration(sentences[0], 0.5) - 0.072) < 1e-9
assert abs(pause_duration(sentences[0], 2.0) - 0.288) < 1e-9
assert pause_duration(sentences[-1], 1.0) == 0

phrases = TextChunker().split_with_context("یک دو، سه چهار پنج.", max_chars=100)
assert phrases[0].boundary == "comma"
assert abs(pause_duration(phrases[0], 1.0) - 0.0585) < 1e-9

forced_split = TextChunker().split_with_context("یک دو سه چهار.", max_chars=3)
assert forced_split[0].boundary == "split"
assert pause_duration(forced_split[0], 1.0) == 0

trimmer = SilenceTrimmer(sample_rate=1000)
signal = np.concatenate((np.zeros(20), np.ones(50) * 0.1, np.zeros(80))).astype(np.float32)
trimmed = [*trimmer.feed(signal), *trimmer.finish()]
assert sum(len(block) for block in trimmed) == 90

class FakeModel:
    sample_rate = 24000
    eos_threshold = -4.0

    def __init__(self):
        self.frames_yielded = 0
        self.frames_after_eos = None

    def get_state_for_audio_prompt(self, _voice_path):
        return {}

    def generate_audio_stream(self, _voice_state, _text, *, frames_after_eos, stop):
        self.frames_after_eos = frames_after_eos
        for _ in range(4):
            if stop.is_set():
                return
            self.frames_yielded += 1
            yield torch.full((2400,), 0.1)


model = FakeModel()
engine = PocketTtsEngine(model, None, {})
stream = engine.stream_synthesize(
    text="سلام.", options=GenerationOptions(frames_after_eos=0)
)
first_audio = next(stream)
assert first_audio.sample_rate == 24000
assert first_audio.pcm_s16le
assert model.frames_yielded == 1
stream.close()
assert model.frames_after_eos == 0

wav_data = engine.synthesize(text="سلام.", options=GenerationOptions())
with wave.open(io.BytesIO(wav_data), "rb") as wav:
    assert wav.getframerate() == 24000
    assert wav.getnchannels() == 1
    assert wav.getnframes() > 0
