"""Persian text normalization and grapheme-to-phoneme conversion."""

import re
import unicodedata

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase, T5ForConditionalGeneration

_LETTERS = "آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیئ"
_ZWNJ = "\u200c"
_KEPT_PUNCTUATION = ".،؛؟!:"
_LETTER_MAP = str.maketrans(
    {
        "ي": "ی", "ى": "ی", "ۍ": "ی", "ې": "ی",
        "ك": "ک", "ڪ": "ک", "ة": "ه", "ۀ": "ه",
        "ۂ": "ه", "ۃ": "ه", "أ": "ا", "إ": "ا",
        "ٱ": "ا", "ٲ": "ا", "ٳ": "ا", "ؤ": "و",
        "ء": "", "ـ": "",
    }
)
_DIGIT_MAP = str.maketrans(
    {chr(start + i): str(i) for start in (0x06F0, 0x0660) for i in range(10)}
)
_PUNCTUATION_MAP = str.maketrans(
    {
        ",": "،", "?": "؟", ";": "؛", "…": ".",
        "٫": ".", "٬": "",
        "%": " درصد ", "٪": " درصد ",
        "«": " ", "»": " ", "“": " ", "”": " ",
        "‘": " ", "’": " ",
        '"': " ", "'": " ", "`": " ", "(": " ", ")": " ", "[": " ",
        "]": " ", "{": " ", "}": " ", "–": " ", "—": " ", "−": " ",
        "-": " ", "‐": " ", "/": " ", "\\": " ", "*": " ", "_": " ",
        "|": " ", "×": " ", "=": " ", "+": " ",
    }
)
_DIACRITICS = re.compile(r"[ً-ٰٕۖ-ۭٖ-ٟ]")
_INVISIBLES = re.compile(r"[\u200b\u200d-\u200f\u202a-\u202e\u2066-\u2069\ufeff\u00ad]")
_NUMBER_RE = re.compile(r"(-?\d+)(?:\.(\d+))?")
_TO_PHONEMES = str.maketrans({"/": "a", "a": "A", "@": "?", "$": "S", "c": "C"})

_ONES = ["", "یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه"]
_TEENS = [
    "ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده",
    "شانزده", "هفده", "هجده", "نوزده",
]
_TENS = ["", "", "بیست", "سی", "چهل", "پنجاه", "شصت", "هفتاد", "هشتاد", "نود"]
_HUNDREDS = ["", "صد", "دویست", "سیصد", "چهارصد", "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد"]
_SCALES = (
    (10**12, "تریلیون"), (10**9, "میلیارد"),
    (10**6, "میلیون"), (10**3, "هزار"),
)
_FRACTION_UNITS = {1: "دهم", 2: "صدم", 3: "هزارم"}
# ponytail: read longer digit strings individually; extend scales if natural readings above 12 digits are needed.
_MAX_NUMBER_DIGITS = 12


def _under_thousand(number: int) -> str:
    parts = []
    if number >= 100:
        parts.append(_HUNDREDS[number // 100])
        number %= 100
    if number >= 20:
        parts.append(_TENS[number // 10])
        number %= 10
    if 10 <= number < 20:
        parts.append(_TEENS[number - 10])
        number = 0
    if number:
        parts.append(_ONES[number])
    return " و ".join(parts)


def number_to_words(number: int) -> str:
    """Read a non-negative integer in Persian."""
    if number == 0:
        return "صفر"
    parts = []
    for value, name in _SCALES:
        if number >= value:
            count, number = divmod(number, value)
            head = "" if value == 1000 and count == 1 else _under_thousand(count) + " "
            parts.append(f"{head}{name}".strip())
    if number:
        parts.append(_under_thousand(number))
    return " و ".join(parts)


def _read_digits(digits: str) -> str:
    return " ".join("صفر" if digit == "0" else _ONES[int(digit)] for digit in digits)


def _number_words(match: re.Match[str]) -> str:
    whole, fraction = match.groups()
    negative = whole.startswith("-")
    whole = whole.lstrip("-")
    if len(whole) > _MAX_NUMBER_DIGITS or (whole.startswith("0") and len(whole) > 1):
        words = _read_digits(whole)
        if fraction:
            words += " " + _read_digits(fraction)
    else:
        words = number_to_words(int(whole))
        if fraction:
            fraction = fraction.rstrip("0")
            if fraction:
                unit = _FRACTION_UNITS.get(len(fraction))
                words += " ممیز " + (
                    number_to_words(int(fraction)) + " " + unit
                    if unit else _read_digits(fraction)
                )
    return f" {'منفی ' if negative else ''}{words} "


def numbers_to_words(text: str) -> str:
    """Replace Persian, Arabic, and Western numeric forms with spoken words."""
    text = re.sub(r"(?<=\d)[,٬](?=\d{3}(?:\D|$))", "", text)
    return _NUMBER_RE.sub(_number_words, text)


def normalize_persian(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLES.sub("", text)
    text = _DIACRITICS.sub("", text).translate(_LETTER_MAP).translate(_DIGIT_MAP)
    text = text.replace("٬", "").replace("٫", ".")
    text = re.sub(r"[٪%]", " درصد ", text)
    text = numbers_to_words(text)
    text = text.translate(_PUNCTUATION_MAP)
    allowed = set(_LETTERS + _ZWNJ + _KEPT_PUNCTUATION + " ")
    text = "".join(char if char in allowed else " " for char in text)
    text = re.sub(rf"{_ZWNJ}+", _ZWNJ, text)
    text = re.sub(
        rf"\s*{_ZWNJ}\s*",
        lambda match: _ZWNJ if " " not in match.group() else " ",
        text,
    )
    text = re.sub(
        rf"(?<![{_LETTERS}]){_ZWNJ}|{_ZWNJ}(?![{_LETTERS}])", "", text
    )
    text = re.sub(r"\s*([.،؛؟!:])", r"\1", text)
    text = re.sub(r"([.،؛؟!:])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lstrip(_KEPT_PUNCTUATION + " ")


class PersianPhonemizer:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: T5ForConditionalGeneration,
    ) -> None:
        self._tokenizer = tokenizer
        self._model = model

    @classmethod
    def from_pretrained(cls, model_id: str) -> "PersianPhonemizer":
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = T5ForConditionalGeneration.from_pretrained(model_id).eval()
        return cls(tokenizer, model)

    def convert(self, text: str) -> str:
        normalized = normalize_persian(text)
        if not normalized:
            raise ValueError("Enter Persian text to synthesize.")

        inputs = self._tokenizer(
            [normalized], add_special_tokens=False, return_tensors="pt"
        )
        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                num_beams=5,
                max_length=512,
                early_stopping=True,
            )
        phonemes = self._tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        return phonemes.strip().replace("1", "").translate(_TO_PHONEMES)
