# Open TTS

Local-first text-to-speech with a browser UI, HTTP API, and Python engine. It
routes Persian to Pocket TTS Farsi v2 and English to Kitten TTS Mini 0.8. Auto
mode routes language chunks separately; it does not translate text.

## What it includes

- An English-language Gradio interface with Persian, English, and Auto modes.
- FastAPI endpoints for a downloadable WAV and incremental PCM audio streaming.
- Persian and Arabic character normalization and spoken-number expansion before
  phonemization, plus simple English integer expansion.
- Separate model, runtime, and device selectors, named Kitten voices, and an
  optional Persian reference recording.
- Sentence and optional comma chunking, with pauses based on punctuation and
  both neighboring phrase lengths.
- Independent native model speed and pitch-preserving post-processing speed.
  Post-processing speed requires FFmpeg; pauses remain separate.
- A local-only model installer and settings screen protected by a one-time code
  printed in the API terminal.
- TOML configuration, local model storage, rotating API logs, and an optional
  speech-client CIDR allowlist.

## How it works

```text
Text
  -> normalize Persian characters and expand numbers by language chunk
  -> split at sentence and language boundaries
  -> route Persian to G2P + Pocket TTS, English to Kitten Mini
  -> apply native model speed, then optional pitch-preserving tempo
  -> insert punctuation pauses and return WAV or streamed PCM
```

The Persian G2P step discards punctuation, so sentence boundaries and pause
metadata are captured first. Kitten's speed control uses its model-native
parameter. The separate post-speed control uses FFmpeg's pitch-preserving
`atempo` filter. The pause multiplier scales calculated silence, not audible
breathing.

Pocket TTS uses PyTorch. Kitten Mini uses ONNX Runtime, with CPU or CUDA shown
when that provider is installed. These models have separate voices and may sound
different when Auto switches languages.

## Requirements

- Python 3.12 or 3.13.
- Internet access while downloading models from the local model manager. Once
  installed, model weights and required assets are stored locally for offline
  startup and synthesis.

The included PowerShell commands use Python 3.13; use `py -3.12` if that is the
Python version installed on your system.

## Install

From the repository root in PowerShell, run the interactive installer:

```powershell
.\install.ps1
```

The installer creates `.venv` with Python 3.12 or 3.13 if needed, installs the
app, then offers CPU or NVIDIA CUDA 13.0/12.6 PyTorch packages and a CPU or
NVIDIA ONNX Runtime option for Kitten. It reuses an existing compatible `.venv`;
it does not remove the environment. For other Python or driver setups, use the
[official PyTorch selector](https://pytorch.org/get-started/locally/). Restart
the API after changing a backend. Confirm PyTorch GPU support with:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CUDA unavailable')"
```

## Run

Start the API and UI in separate terminals from the repository root:

```powershell
open-tts-api
```

```powershell
open-tts-webui
```

Open <http://127.0.0.1:7863>. For first setup, open **Local model and settings
manager**, click **Print owner code in API terminal**, enter that code in the UI,
and download the models you want. This manager appears only when you open the UI
from `localhost` or `127.0.0.1`; the API also enforces loopback-only management.
Python backends are installed separately by `install.ps1`; changing the
runtime/device package selection requires stopping and restarting the API.
The model selector reports whether model files and the required Python backend
are both ready. Refresh model choices after a download, then select a
model/runtime/device. Each model has a fixed supported runtime: Pocket uses
PyTorch and Kitten uses ONNX Runtime.

The API listens on
<http://127.0.0.1:8000>; its interactive API docs are at <http://127.0.0.1:8000/docs>.
In VS Code, you can instead select **Open TTS (local)** in Run and Debug and
press F5.

## API

- `GET /health` reports readiness or whether model files, a Python backend, or
  model loading still needs attention.
- `GET /v1/runtime-options` lists the available model, runtime, and devices.
- `POST /v1/runtime-options` changes the loaded selection from this computer;
  model selection does not require the owner session.
- `POST /v1/speech` accepts text, language, voice, speed, and generation settings
  as form fields
  and returns `audio/wav`.
- `POST /v1/speech/stream` accepts the same fields and streams newline-delimited
  JSON: an audio format record, base64 encoded PCM blocks, then a completion
  record or an error record.

Both endpoints support chunk size, comma splitting, silent-chunk retry,
end-of-speech settings, and a pause multiplier
(`0`–`2x`). The automatic pause uses punctuation and the lengths of both
neighboring sentences. They also accept an optional WAV voice prompt (maximum
10 MB and five seconds). Kitten's named voices are selected with `voice_name`.
See the API schema at `/docs` for field limits.

Local management endpoints are `POST /v1/admin/request-key`,
`POST /v1/admin/session`, `GET /v1/models`, `POST /v1/models/{id}/install`, and
`GET`/`POST /v1/settings`. Download jobs can be checked or cancelled under
`/v1/models/install/{job_id}`. These actions require an in-memory owner session.

## Configuration

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `OPEN_TTS_CONFIG` | `%LOCALAPPDATA%\OpenTTS\config.toml` | TOML config path override |
| `OPEN_TTS_API_HOST` | `127.0.0.1` | API bind address |
| `OPEN_TTS_API_PORT` | `8000` | API port |
| `OPEN_TTS_API_URL` | `http://127.0.0.1:8000` | API address used by the UI |
| `OPEN_TTS_WEBUI_HOST` | `127.0.0.1` | UI bind address |
| `OPEN_TTS_WEBUI_PORT` | `7863` | UI port |
| `OPEN_TTS_MODELS_DIR` | `%LOCALAPPDATA%\OpenTTS\models` | Model storage override |
| `OPEN_TTS_LOGS_DIR` | `%LOCALAPPDATA%\OpenTTS\logs` | Log storage override |

The TOML file also stores active model/runtime/device, allowed speech client
CIDRs, log level and retention, and the admin-session timeout. Both services
bind to loopback by default. Management stays loopback-only when speech access
is enabled for a configured network.

## Models and attribution

- Speech model: [Pocket TTS Farsi v2](https://huggingface.co/mehdi-hf/pocket-tts-farsi-v2).
  Its model card specifies **CC BY-NC 4.0** terms; review them before using the
  model, especially for commercial purposes.
- Persian phonemizer: [Homo-GE2PE-Persian-HF](https://huggingface.co/mehdi-hf/Homo-GE2PE-Persian-HF),
  listed as MIT-licensed and crediting the original model authors.
- English speech model: [Kitten TTS Mini 0.8](https://huggingface.co/KittenML/kitten-tts-mini-0.8),
  listed under Apache-2.0 terms.
- Persian text normalization: [Hazm](https://github.com/roshan-research/hazm),
  used for spacing and punctuation cleanup before the model-specific frontend.
- The Persian normalizer is adapted from
  [Fouladi-K/pocket-tts-Persian-webui](https://github.com/Fouladi-K/pocket-tts-Persian-webui/blob/main/webui-v2/normalize_fa.py).
- The bundled example voice is copied from that project's `webui-v2` folder;
  see its [license](https://github.com/Fouladi-K/pocket-tts-Persian-webui/blob/main/LICENSE).

Model assets are installed in the configured local model folder and are not
included in this repository. Kitten Mini 0.8 currently exposes ONNX inference
assets, not a published Mini fine-tuning recipe; see
[training feasibility](docs/kitten-training-feasibility.md). The Persian model
license is non-commercial; review its terms before use.
The repository itself does not yet have a project-level license. Add one before
granting others permission to reuse or redistribute this project's code.

## Development check

Run the focused engine checks and API streaming checks from the repository root:

```powershell
$env:PYTHONPATH = "engine/src"
.\.venv\Scripts\python.exe .\engine\tests\test_text_processing.py
$env:PYTHONPATH = "engine/src;api/src"
.\.venv\Scripts\python.exe -m unittest discover -s .\api\tests -v
$env:PYTHONPATH = "engine/src;api/src;webui/src"
.\.venv\Scripts\python.exe -m unittest discover -s .\webui\tests -v
```

These checks cover text processing, pause calculations, silence handling,
WAV output, API stream framing, and cancellation. Listening tests are
still needed for real Pocket TTS timing, pronunciation, and audio quality.
