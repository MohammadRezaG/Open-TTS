# Open TTS

Local-first Persian text-to-speech with a browser UI, HTTP API, and Python
engine. The UI sends requests to the API; the engine normalizes Persian text,
converts it to phonemes, and generates WAV audio with Pocket TTS.

This is an early Persian-only project. It does not yet route English or mixed
language text, offer a named voice catalog, or provide speed, pitch, or breath
controls.

## What it includes

- An English-language Gradio interface for Persian text.
- A FastAPI endpoint that returns synthesized WAV audio.
- Persian and Arabic character normalization and number-to-spoken-words
  preprocessing before phonemization.
- Sentence and optional comma chunking, with pauses between generated chunks.
- An optional WAV voice prompt, up to five seconds. This is prompt-based voice
  conditioning, not a preset voice selector.

## How it works

```text
Persian text
  -> normalize characters and numbers
  -> split into chunks before G2P
  -> Persian grapheme-to-phoneme model
  -> Pocket TTS Farsi model
  -> WAV audio
```

The G2P step can discard punctuation, so sentence boundaries are handled before
phonemization. The current pause between chunks is fixed; timing is not yet
adjustable.

## Requirements

- Python 3.12 or newer.
- Internet access on first API startup to download the model files from Hugging
  Face. The files are cached outside the source tree.

The included PowerShell commands use Python 3.13; use `py -3.12` if that is the
Python version installed on your system.

## Install

From the repository root in PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .\engine -e .\api -e .\webui
```

## Run

Start the API and UI in separate terminals from the repository root:

```powershell
open-tts-api
```

```powershell
open-tts-webui
```

Open <http://127.0.0.1:7862>. The API listens on
<http://127.0.0.1:8000>; its interactive API docs are at <http://127.0.0.1:8000/docs>.
In VS Code, you can instead select **Open TTS (local)** in Run and Debug and
press F5.

## API

- `GET /health` reports whether the models have loaded.
- `POST /v1/speech` accepts Persian text and generation settings as form fields
  and returns `audio/wav`.

The endpoint supports chunk size, comma splitting, silent-chunk retry, and
end-of-speech settings. It also accepts an optional WAV voice prompt (maximum
10 MB and five seconds). See the API schema at `/docs` for exact field limits.

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OPEN_TTS_API_HOST` | `127.0.0.1` | API bind address |
| `OPEN_TTS_API_PORT` | `8000` | API port |
| `OPEN_TTS_API_URL` | `http://127.0.0.1:8000` | API address used by the UI |
| `OPEN_TTS_WEBUI_PORT` | `7862` | UI port |
| `OPEN_TTS_TTS_CONFIG` | Persian Pocket TTS Farsi v2 config | TTS model/config source |
| `OPEN_TTS_G2P_MODEL` | `mehdi-hf/Homo-GE2PE-Persian-HF` | Persian phonemizer model |
| `OPEN_TTS_DEFAULT_VOICE` | Bundled example WAV | Default voice prompt |
| `OPEN_TTS_TEMPERATURE` | `0.3` | TTS sampling temperature |

`HF_HOME` can be set to choose a different Hugging Face cache directory.

## Models and attribution

- Speech model: [Pocket TTS Farsi v2](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2).
  Its model card specifies **CC BY-NC 4.0** terms; review them before using the
  model, especially for commercial purposes.
- Persian phonemizer: [Homo-GE2PE-Persian-HF](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF),
  listed as MIT-licensed and crediting the original model authors.
- The Persian normalizer is adapted from
  [Fouladi-K/pocket-tts-Persian-webui](https://github.com/Fouladi-K/pocket-tts-Persian-webui/blob/main/webui-v2/normalize_fa.py).
- The bundled example voice is copied from that project's `webui-v2` folder;
  see its [license](https://github.com/Fouladi-K/pocket-tts-Persian-webui/blob/main/LICENSE).

Model weights are downloaded at runtime and are not included in this repository.
The repository itself does not yet have a project-level license. Add one before
granting others permission to reuse or redistribute this project's code.

## Development check

Run the current focused text-processing check from the repository root:

```powershell
$env:PYTHONPATH = "engine/src"
.\.venv\Scripts\python.exe .\engine\tests\test_text_processing.py
```

This is a small normalization/chunking check, not a full API or audio-quality
test suite.
