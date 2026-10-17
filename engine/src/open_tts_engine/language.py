"""Language selection and simple integer expansion for speech chunks."""

from __future__ import annotations

import re

_PERSIAN = re.compile(r"[\u0600-\u06ff]")
_LATIN = re.compile(r"[A-Za-z]")
_TOKENS = re.compile(r"\s+|[^\s]+")
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")


def split_language_chunks(text: str, language: str) -> list[tuple[str, str]]:
    """Split adjacent Persian and Latin text while retaining spaces and punctuation."""
    if language in {"fa", "en"}:
        return [(language, text)]
    chunks: list[tuple[str, str]] = []
    current_language = "fa"
    current_text = ""
    tokens = _TOKENS.findall(text)
    for index, token in enumerate(tokens):
        detected = _token_language(token)
        if detected is None and any(char.isdigit() for char in token) and not current_text.strip():
            detected = next(
                (_token_language(following) for following in tokens[index + 1:] if _token_language(following)),
                None,
            )
        if detected is not None and detected != current_language:
            if current_text.strip():
                chunks.append((current_language, current_text.strip()))
            current_text = ""
            current_language = detected
        current_text += token
    if current_text.strip():
        chunks.append((current_language, current_text.strip()))
    return chunks


def _token_language(token: str) -> str | None:
    if _PERSIAN.search(token):
        return "fa"
    if _LATIN.search(token):
        return "en"
    return None


def english_numbers_to_words(text: str) -> str:
    """Expand ordinary integers; long and zero-padded IDs are read digit by digit."""
    def replace_number(match: re.Match[str]) -> str:
        digits = match.group().replace(",", "")
        if len(digits) > 6 or (len(digits) > 1 and digits.startswith("0")):
            return " ".join(_ONES[int(digit)] for digit in digits)
        return _under_million(int(digits))

    return re.sub(r"(?<!\w)\d+(?:,\d{3})*(?!\w|\.\d)", replace_number, text)


def _under_million(value: int) -> str:
    if value < 10:
        return _ONES[value]
    if value < 20:
        return _TEENS[value - 10]
    if value < 100:
        return _TENS[value // 10] + (f"-{_ONES[value % 10]}" if value % 10 else "")
    if value < 1000:
        rest = value % 100
        return f"{_ONES[value // 100]} hundred" + (f" {_under_million(rest)}" if rest else "")
    rest = value % 1000
    return f"{_under_million(value // 1000)} thousand" + (f" {_under_million(rest)}" if rest else "")
