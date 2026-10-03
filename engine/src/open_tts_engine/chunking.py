"""Small, predictable text chunking for the phoneme-based Persian model."""

import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟؛])\s+|(?<=[.!?؟؛])(?=\S)")
_COMMA_BOUNDARY = re.compile(r"(?<=[,،])\s+|(?<=[,،])(?=\S)")
_SENTENCE_END = re.compile(r"[.!?؟؛]$")


class TextChunker:
    def split(
        self,
        text: str,
        *,
        max_chars: int,
        min_chars: int = 0,
        split_on_comma: bool = True,
    ) -> list[str]:
        """Split on punctuation, then words, and merge tiny adjacent pieces."""
        separators = [_SENTENCE_BOUNDARY]
        if split_on_comma:
            separators.append(_COMMA_BOUNDARY)

        parts = [text.strip()]
        for separator in separators:
            parts = [piece for part in parts for piece in separator.split(part) if piece.strip()]

        chunks = [
            word_chunk
            for part in parts
            for word_chunk in self._split_words(part, max_chars)
        ]
        return self._merge_short(chunks, min_chars, max_chars)

    @staticmethod
    def _split_words(text: str, max_chars: int) -> list[str]:
        chunks: list[str] = []
        current = ""
        for word in text.split():
            if current and len(current) + len(word) + 1 > max_chars:
                chunks.append(current)
                current = word
            else:
                current = f"{current} {word}".strip()
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _merge_short(chunks: list[str], min_chars: int, max_chars: int) -> list[str]:
        if min_chars <= 0:
            return chunks

        merged: list[str] = []
        for chunk in chunks:
            if merged and len(merged[-1]) < min_chars and not _SENTENCE_END.search(merged[-1]):
                combined = f"{merged[-1]} {chunk}"
                if len(combined) <= max_chars:
                    merged[-1] = combined
                    continue
            merged.append(chunk)

        if (
            len(merged) > 1
            and len(merged[-1]) < min_chars
            and not _SENTENCE_END.search(merged[-2])
        ):
            combined = f"{merged[-2]} {merged[-1]}"
            if len(combined) <= max_chars:
                merged[-2:] = [combined]
        return merged
