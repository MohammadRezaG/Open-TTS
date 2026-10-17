# API

FastAPI application served by Uvicorn. It owns HTTP validation, model
management, access checks, settings, and audio responses. Installed models are
loaded at startup; model management remains available when none are installed.

- `GET /health` reports model readiness or whether model files, the Python
  backend, or model loading needs attention.
- `GET /v1/runtime-options` reports the supported model, runtime, and device choices.
- `POST /v1/runtime-options` loads a supported model/runtime/device selection
  for later requests from a loopback client. It does not require the owner session.
- `POST /v1/speech` accepts language, generation settings, and an optional WAV
  prompt, then returns generated WAV audio.
- `POST /v1/speech/stream` returns PCM blocks as newline-delimited JSON for
  incremental playback. The UI uses this endpoint and saves the same received
  samples as its downloadable WAV.

Persian uses Pocket TTS + PyTorch; English uses Kitten Mini + ONNX Runtime. The
API exposes terminal owner-code exchange, local model download jobs, and
settings endpoints. Management routes require a loopback client; speech routes
also support configured CIDR allowlists. Both speech endpoints accept
`language`, model/post speed, voice, and `pause_multiplier` options.

Model files are downloaded through the local manager. Python backends are
installed with the repository's `install.ps1` while the API is stopped, then
loaded after restart. Each model uses its supported runtime: Pocket/PyTorch or
Kitten/ONNX Runtime.

Run from the workspace root after installing the packages:

```powershell
open-tts-api
```
