"""Sentence-aware text chunking before punctuation is removed by G2P."""

from dataclasses import dataclass
import re

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟؛])\s+|(?<=[.!?؟؛])(?=\S)")
_COMMA_BOUNDARY = re.compile(r"(?<=[,،])\s+|(?<=[,،])(?=\S)")
_SENTENCE_END = re.compile(r"[.!?؟؛]$")


@dataclass(frozen=True, slots=True)
class TextChunk:
    text: str
    boundary: str
    sentence_words: int
    next_sentence_words: int = 0
    phrase_words: int = 0
    next_phrase_words: int = 0


class TextChunker:
    def split(
        self,
        text: str,
        *,
        max_chars: int,
        min_chars: int = 0,
        split_on_comma: bool = True,
    ) -> list[str]:
        return [
            chunk.text
            for chunk in self.split_with_context(
                text,
                max_chars=max_chars,
                min_chars=min_chars,
                split_on_comma=split_on_comma,
            )
        ]

    def split_with_context(
        self,
        text: str,
        *,
        max_chars: int,
        min_chars: int = 0,
        split_on_comma: bool = True,
    ) -> list[TextChunk]:
        sentences = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
        sentence_word_counts = [self._word_count(sentence) for sentence in sentences]
        chunks: list[TextChunk] = []

        for sentence_index, sentence in enumerate(sentences):
            sentence_words = sentence_word_counts[sentence_index]
            next_sentence_words = (
                sentence_word_counts[sentence_index + 1]
                if sentence_index + 1 < len(sentences)
                else 0
            )
            boundary = self._sentence_boundary(sentence)
            phrases = (
                [part.strip() for part in _COMMA_BOUNDARY.split(sentence) if part.strip()]
                if split_on_comma
                else [sentence]
            )
            phrase_word_counts = [self._word_count(phrase) for phrase in phrases]

            for phrase_index, phrase in enumerate(phrases):
                is_last_phrase = phrase_index == len(phrases) - 1
                phrase_boundary = boundary if is_last_phrase else "comma"
                next_phrase_words = (
                    phrase_word_counts[phrase_index + 1] if not is_last_phrase else 0
                )
                parts = self._split_words(phrase, max_chars)
                for part_index, part in enumerate(parts):
                    last_part = part_index == len(parts) - 1
                    chunks.append(
                        TextChunk(
                            text=part,
                            boundary=phrase_boundary if last_part else "split",
                            sentence_words=sentence_words,
                            next_sentence_words=next_sentence_words,
                            phrase_words=phrase_word_counts[phrase_index],
                            next_phrase_words=next_phrase_words,
                        )
                    )

        return self._merge_short(chunks, min_chars, max_chars)

    @staticmethod
    def _word_count(text: str) -> int:
        return len(text.rstrip(".,،;؛!?؟:").split())

    @staticmethod
    def _sentence_boundary(text: str) -> str:
        if not _SENTENCE_END.search(text):
            return "end"
        mark = text.rstrip()[-1]
        return {
            "?": "question",
            "؟": "question",
            "!": "exclamation",
            ";": "semicolon",
            "؛": "semicolon",
        }.get(mark, "period")

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
    def _merge_short(chunks: list[TextChunk], min_chars: int, max_chars: int) -> list[TextChunk]:
        if min_chars <= 0:
            return chunks

        merged: list[TextChunk] = []
        for chunk in chunks:
            if (
                merged
                and merged[-1].sentence_words == chunk.sentence_words
                and merged[-1].phrase_words == chunk.phrase_words
                and merged[-1].boundary == "split"
                and len(merged[-1].text) < min_chars
            ):
                previous = merged[-1]
                combined = f"{previous.text} {chunk.text}"
                if len(combined) <= max_chars:
                    merged[-1] = TextChunk(
                        text=combined,
                        boundary=chunk.boundary,
                        sentence_words=chunk.sentence_words,
                        next_sentence_words=chunk.next_sentence_words,
                        phrase_words=chunk.phrase_words,
                        next_phrase_words=chunk.next_phrase_words,
                    )
                    continue
            merged.append(chunk)
        return merged
