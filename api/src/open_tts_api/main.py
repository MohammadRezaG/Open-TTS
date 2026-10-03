"""FastAPI application for local speech generation."""

from contextlib import asynccontextmanager
from logging import getLogger
from pathlib import Path
from shutil import copyfileobj
from tempfile import NamedTemporaryFile
from typing import Annotated

import scipy.io.wavfile
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response

from open_tts_engine import EngineSettings, GenerationOptions, PocketTtsEngine, SpeechEngine

logger = getLogger(__name__)
_MAX_VOICE_BYTES = 10 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models once before the API starts accepting requests."""
    app.state.speech_engine = PocketTtsEngine.from_pretrained(EngineSettings.from_environment())
    yield
    app.state.speech_engine = None


app = FastAPI(title="Open TTS API", lifespan=lifespan)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    engine = getattr(request.app.state, "speech_engine", None)
    return {"status": "ready" if engine is not None else "loading"}


@app.post("/v1/speech", response_class=Response)
def synthesize(
    request: Request,
    text: Annotated[str, Form(min_length=1)],
    max_chars: Annotated[int, Form(ge=20, le=500)] = 100,
    min_chars: Annotated[int, Form(ge=0, le=100)] = 30,
    split_on_comma: Annotated[bool, Form()] = True,
    retry_silent_chunks: Annotated[bool, Form()] = True,
    frames_after_eos: Annotated[int, Form(ge=0, le=16)] = 2,
    eos_threshold: Annotated[float, Form(ge=-8, le=-1)] = -4.0,
    voice: Annotated[UploadFile | None, File()] = None,
) -> Response:
    """Synthesize Persian text and return a WAV file."""
    if min_chars > max_chars:
        raise HTTPException(422, "Minimum chunk length cannot exceed maximum chunk length.")

    engine: SpeechEngine | None = getattr(request.app.state, "speech_engine", None)
    if engine is None:
        raise HTTPException(503, "The speech models are still loading.")

    voice_path: Path | None = None
    try:
        if voice is not None:
            if Path(voice.filename or "voice.wav").suffix.lower() != ".wav":
                raise HTTPException(415, "Upload a WAV reference recording.")
            voice.file.seek(0, 2)
            if voice.file.tell() > _MAX_VOICE_BYTES:
                raise HTTPException(413, "Reference recording must be smaller than 10 MB.")
            voice.file.seek(0)
            with NamedTemporaryFile(suffix=".wav", delete=False) as temporary_voice:
                copyfileobj(voice.file, temporary_voice)
                voice_path = Path(temporary_voice.name)

            sample_rate, prompt = scipy.io.wavfile.read(voice_path)
            if prompt.shape[0] / sample_rate > 5:
                raise HTTPException(422, "Reference recording must be 5 seconds or shorter.")

        options = GenerationOptions(
            max_chars=max_chars,
            min_chars=min_chars,
            split_on_comma=split_on_comma,
            retry_silent_chunks=retry_silent_chunks,
            frames_after_eos=frames_after_eos,
            eos_threshold=eos_threshold,
        )
        audio = engine.synthesize(text=text, options=options, voice_path=voice_path)
        return Response(
            content=audio,
            media_type="audio/wav",
            headers={"Content-Disposition": 'attachment; filename="speech.wav"'},
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logger.exception("Speech generation failed")
        raise HTTPException(500, "Speech generation failed. Check the API log.") from exc
    finally:
        if voice_path is not None:
            voice_path.unlink(missing_ok=True)
        if voice is not None:
            voice.file.close()