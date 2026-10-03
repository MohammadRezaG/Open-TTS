# API

FastAPI application served by Uvicorn. It owns HTTP validation and audio
responses; the model is loaded once during application startup by `engine/`.

- `GET /health` reports whether the models finished loading.
- `POST /v1/speech` accepts Persian text, generation settings, and an optional
  WAV voice prompt, then returns generated WAV audio.

Run from the workspace root after installing the packages:

```powershell
open-tts-api
```
