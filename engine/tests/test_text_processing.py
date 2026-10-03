from open_tts_engine.chunking import TextChunker
from open_tts_engine.persian import normalize_persian


assert normalize_persian("1") == "یک"
assert normalize_persian("۱۲۳") == "صد و بیست و سه"

assert TextChunker().split("١. ٢.", max_chars=100, min_chars=30) == ["یک.", "دو."]
