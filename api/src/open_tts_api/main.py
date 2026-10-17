"""FastAPI application for local speech generation."""

from contextlib import asynccontextmanager
import asyncio
import base64
from dataclasses import dataclass, field, replace
from importlib.util import find_spec
import json
import logging
from logging import getLogger
from logging.handlers import RotatingFileHandler
from pathlib import Path
import queue
from shutil import copyfileobj
import shutil
from uuid import uuid4
from threading import Event, Lock, Thread
from tempfile import NamedTemporaryFile
from typing import Annotated

import scipy.io.wavfile
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from starlette.background import BackgroundTask

from open_tts_engine.options import GenerationOptions
from open_tts_engine.contract import SpeechEngine
from open_tts_engine.router import LanguageRoutedEngine
from open_tts_engine.settings import EngineSettings, resolve_device
from open_tts_common import AppConfig, OwnerSessions, is_loopback, is_network_allowed, load_config, save_config
from .models import catalog as model_catalog, install_model
from .models import MODEL_CATALOG, installed_models

logger = getLogger(__name__)
_MAX_VOICE_BYTES = 10 * 1024 * 1024
_STREAM_BLOCK_SECONDS = 0.5


@dataclass(slots=True)
class InstallJob:
    cancel: Event
    status: str = "queued"
    message: str = "Waiting to start."
    result: dict[str, object] | None = None
    lock: Lock = field(default_factory=Lock)


class RuntimeSelection(BaseModel):
    model: str
    runtime: str
    device: str


def _save_voice(voice: UploadFile | None) -> Path | None:
    if voice is None:
        return None
    if Path(voice.filename or "voice.wav").suffix.lower() != ".wav":
        raise HTTPException(415, "Upload a WAV reference recording.")
    voice.file.seek(0, 2)
    if voice.file.tell() > _MAX_VOICE_BYTES:
        raise HTTPException(413, "Reference recording must be smaller than 10 MB.")
    voice.file.seek(0)
    with NamedTemporaryFile(suffix=".wav", delete=False) as temporary_voice:
        copyfileobj(voice.file, temporary_voice)
        voice_path = Path(temporary_voice.name)

    try:
        sample_rate, prompt = scipy.io.wavfile.read(voice_path)
    except Exception as exc:
        voice_path.unlink(missing_ok=True)
        raise HTTPException(422, "The reference recording is not a valid WAV file.") from exc
    if sample_rate <= 0 or prompt.shape[0] / sample_rate > 5:
        voice_path.unlink(missing_ok=True)
        raise HTTPException(422, "Reference recording must be 5 seconds or shorter.")
    return voice_path


def _json_line(event: str, **data: object) -> bytes:
    return json.dumps({"event": event, **data}, separators=(",", ":")).encode() + b"\n"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the selected model once before the API starts accepting requests."""
    config = load_config()
    _configure_logging(config)
    settings = replace(
        EngineSettings.from_environment(),
        model_id=config.active_model,
        runtime=config.runtime,
        device=config.device,
    )
    engines = {}
    statuses = installed_models(config)
    candidates = [settings.model_id, "pocket_farsi_v2", "kitten_mini_08"]
    for model_id in dict.fromkeys(candidates):
        if not statuses.get(model_id, {}).get("installed"):
            continue
        runtime = "pytorch" if model_id == "pocket_farsi_v2" else "onnxruntime"
        # The saved device choice belongs to the active model. Resolve the
        # other language backend independently so an ONNX-only CUDA setup does
        # not prevent the Persian PyTorch model from loading on CPU (or vice
        # versa).
        candidate_settings = replace(
            settings,
            model_id=model_id,
            runtime=runtime,
            device=settings.device if model_id == config.active_model else "auto",
        )
        try:
            engine = _load_engine(candidate_settings, config)
            language = "fa" if model_id == "pocket_farsi_v2" else "en"
            engines[language] = engine
            if model_id == config.active_model:
                settings = candidate_settings
        except Exception:
            logger.exception("Could not initialize installed model %s", model_id)
    app.state.speech_engine = LanguageRoutedEngine(engines) if engines else None
    app.state.loaded_speech_engines = engines
    app.state.engine_settings = settings
    app.state.app_config = config
    app.state.owner_sessions = OwnerSessions(config.admin_session_minutes)
    app.state.install_jobs = {}
    app.state.engine_update_lock = Lock()
    yield
    app.state.speech_engine = None


app = FastAPI(title="Open TTS API", lifespan=lifespan)


def _configure_logging(config: AppConfig) -> None:
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(config.log_level.upper())
    if not any(getattr(handler, "_open_tts_file", False) for handler in root.handlers):
        handler = RotatingFileHandler(
            config.logs_dir / "open-tts-api.log", maxBytes=5 * 1024 * 1024,
            backupCount=config.log_retention, encoding="utf-8",
        )
        handler._open_tts_file = True
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)


@app.get("/health")
def health(request: Request) -> dict[str, str]:
    engine = getattr(request.app.state, "speech_engine", None)
    if engine is not None:
        return {"status": "ready"}
    config: AppConfig = getattr(request.app.state, "app_config", AppConfig())
    statuses = installed_models(config)
    if any(
        status.get("installed") and _backend_available(model_id)
        for model_id, status in statuses.items()
    ):
        return {"status": "models_loading"}
    if any(status.get("installed") for status in statuses.values()):
        return {"status": "backend_required"}
    return {"status": "models_required"}


def _require_loopback(request: Request) -> None:
    host = request.client.host if request.client else None
    if not is_loopback(host):
        raise HTTPException(403, "Management is available only from this computer.")


def _require_owner(request: Request) -> None:
    _require_loopback(request)
    token = request.headers.get("x-open-tts-session")
    sessions: OwnerSessions = request.app.state.owner_sessions
    if not sessions.is_valid(token):
        raise HTTPException(401, "Request a terminal owner code and unlock local management.")


def _require_speech_network(request: Request) -> None:
    host = request.client.host if request.client else None
    config: AppConfig = getattr(request.app.state, "app_config", AppConfig())
    if not is_network_allowed(host, config.allowed_networks):
        raise HTTPException(403, "This client network is not allowed to use speech endpoints.")


@app.post("/v1/admin/request-key")
def request_owner_code(request: Request) -> dict[str, str]:
    """Print a single-use local owner code to the API terminal."""
    _require_loopback(request)
    sessions: OwnerSessions = request.app.state.owner_sessions
    code = sessions.issue_code()
    print(f"Open-TTS one-time owner code (expires in 5 minutes): {code}", flush=True)
    return {"status": "code_printed", "message": "Enter the code shown in the API terminal."}


class OwnerCode(BaseModel):
    code: str


@app.post("/v1/admin/session")
def exchange_owner_code(body: OwnerCode, request: Request) -> dict[str, str | int]:
    _require_loopback(request)
    sessions: OwnerSessions = request.app.state.owner_sessions
    token = sessions.exchange(body.code)
    if token is None:
        raise HTTPException(401, "The owner code is invalid, expired, or already used.")
    config: AppConfig = request.app.state.app_config
    return {"session_key": token, "expires_in": config.admin_session_minutes * 60}


@app.get("/v1/models")
def list_models(request: Request) -> dict[str, object]:
    _require_loopback(request)
    config: AppConfig = request.app.state.app_config
    return {"models": model_catalog(config)}


@app.post("/v1/models/{model_id}/install")
def install_model_endpoint(model_id: str, request: Request) -> dict[str, object]:
    _require_owner(request)
    config: AppConfig = request.app.state.app_config
    if model_id not in {item.model_id for item in MODEL_CATALOG}:
        raise HTTPException(422, f"Unsupported model: {model_id}.")
    job_id = uuid4().hex
    job = InstallJob(cancel=Event())
    request.app.state.install_jobs[job_id] = job

    def run_install() -> None:
        with job.lock:
            job.status, job.message = "downloading", "Downloading required model files."
        try:
            result = install_model(model_id, config, job.cancel)
            with job.lock:
                job.status, job.message, job.result = "complete", "Model installed.", result
        except InterruptedError:
            with job.lock:
                job.status, job.message = "cancelled", "Installation cancelled; temporary files removed."
        except Exception as exc:
            logger.exception("Model installation failed")
            with job.lock:
                job.status, job.message = "failed", str(exc)

    Thread(target=run_install, name=f"model-install-{model_id}", daemon=True).start()
    return {"job_id": job_id, "status": job.status}


@app.get("/v1/models/install/{job_id}")
def get_model_install_job(job_id: str, request: Request) -> dict[str, object]:
    _require_owner(request)
    job = request.app.state.install_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Model installation job not found.")
    with job.lock:
        return {"job_id": job_id, "status": job.status, "message": job.message, "result": job.result}


@app.delete("/v1/models/install/{job_id}")
def cancel_model_install_job(job_id: str, request: Request) -> dict[str, str]:
    _require_owner(request)
    job = request.app.state.install_jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Model installation job not found.")
    job.cancel.set()
    return {"status": "cancellation_requested"}


class SettingsUpdate(BaseModel):
    models_dir: str | None = None
    logs_dir: str | None = None
    log_level: str | None = None
    log_retention: int | None = None
    allowed_networks: list[str] | None = None


@app.get("/v1/settings")
def get_settings(request: Request) -> dict[str, object]:
    _require_loopback(request)
    config: AppConfig = request.app.state.app_config
    return {
        "models_dir": str(config.models_dir),
        "logs_dir": str(config.logs_dir),
        "log_level": config.log_level,
        "log_retention": config.log_retention,
        "allowed_networks": list(config.allowed_networks),
    }


@app.post("/v1/settings")
def update_settings(body: SettingsUpdate, request: Request) -> dict[str, object]:
    _require_owner(request)
    current: AppConfig = request.app.state.app_config
    changes = body.model_dump(exclude_none=True)
    if "models_dir" in changes:
        changes["models_dir"] = Path(changes["models_dir"]).expanduser()
    if "logs_dir" in changes:
        changes["logs_dir"] = Path(changes["logs_dir"]).expanduser()
    if "allowed_networks" in changes:
        changes["allowed_networks"] = tuple(changes["allowed_networks"])
    try:
        updated = replace(current, **changes)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    save_config(updated)
    request.app.state.app_config = updated
    return {"status": "saved", "settings": {"models_dir": str(updated.models_dir), "logs_dir": str(updated.logs_dir), "log_level": updated.log_level, "log_retention": updated.log_retention, "allowed_networks": updated.allowed_networks}}


def _runtime_options(settings: EngineSettings, config: AppConfig | None = None) -> dict[str, object]:
    import torch

    spec = next((item for item in MODEL_CATALOG if item.model_id == settings.model_id), None)
    if spec is None:
        raise ValueError(f"Unsupported TTS model: {settings.model_id}.")
    cuda_available = torch.cuda.is_available() if settings.runtime == "pytorch" else _onnx_cuda_available()
    devices = [
        {"value": "auto", "label": "✨ Auto (best available)"},
        {"value": "cpu", "label": "🖥 CPU"},
    ]
    if cuda_available:
        # ONNX CUDA can be available even when this installation's PyTorch is
        # CPU-only, so only ask PyTorch for the device name on its own runtime.
        gpu_name = (
            torch.cuda.get_device_name(0)
            if settings.runtime == "pytorch"
            else "CUDA"
        )
        devices.append({"value": "cuda", "label": f"🎮 GPU — {gpu_name}"})
    if settings.runtime == "pytorch":
        try:
            active_device = resolve_device(settings.device)
        except ValueError:
            active_device = "unavailable"
    else:
        active_device = (
            "cuda" if settings.device == "cuda" or (settings.device == "auto" and cuda_available)
            else "cpu"
        )
    config = config or AppConfig()
    statuses = installed_models(config)
    return {
        "models": [
            {
                "value": item.model_id,
                "label": item.label,
                "installed": bool(statuses[item.model_id]["installed"]),
                "backend_available": _backend_available(item.model_id),
                "ready": bool(statuses[item.model_id]["installed"])
                and _backend_available(item.model_id),
            }
            for item in MODEL_CATALOG
        ],
        "runtimes": [{"value": spec.runtime, "label": "PyTorch" if spec.runtime == "pytorch" else "ONNX Runtime"}],
        "devices": devices,
        "selection": {
            "model": settings.model_id,
            "runtime": settings.runtime,
            "device": settings.device,
            "active_device": active_device,
        },
        "cuda_available": cuda_available,
        "device_note": (
            f"CUDA GPU is available for {settings.runtime}."
            if cuda_available
            else (
                "CUDA GPU is unavailable in this ONNX Runtime install."
                if settings.runtime == "onnxruntime"
                else "CUDA GPU is unavailable: this PyTorch install has no CUDA support."
            )
        ),
    }


def _onnx_cuda_available() -> bool:
    try:
        import onnxruntime

        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except ImportError:
        return False


def _backend_available(model_id: str) -> bool:
    """Return whether the optional Python runtime for a model is installed."""
    if model_id == "pocket_farsi_v2":
        return find_spec("torch") is not None
    if model_id == "kitten_mini_08":
        return (
            find_spec("onnxruntime") is not None
            and find_spec("kittentts") is not None
        )
    return False


def _load_engine(settings: EngineSettings, config: AppConfig):
    status = installed_models(config).get(settings.model_id, {})
    if not status.get("installed"):
        raise ValueError(f"Install {settings.model_id} before selecting it.")
    if not _backend_available(settings.model_id):
        if settings.model_id == "kitten_mini_08":
            raise ValueError(
                "Kitten's Python backend is missing. Stop the API, run .\\install.ps1, "
                "choose the Kitten CPU or GPU backend, then restart the API."
            )
        raise ValueError(
            "The PyTorch backend is missing. Stop the API, run .\\install.ps1, "
            "then restart the API."
        )
    model_path = config.models_dir / settings.model_id
    if settings.model_id == "pocket_farsi_v2":
        from open_tts_engine.pocket_tts import PocketTtsEngine
        local_settings = replace(
            settings,
            model_path=model_path,
            g2p_path=model_path / "g2p",
        )
        return PocketTtsEngine.from_pretrained(local_settings)
    if settings.model_id == "kitten_mini_08":
        from open_tts_engine.kitten_tts import KittenTtsEngine
        if settings.device == "cuda" and not _onnx_cuda_available():
            raise ValueError("CUDA ONNX Runtime is not installed or unavailable.")
        device = (
            "cuda"
            if settings.device == "cuda" or (settings.device == "auto" and _onnx_cuda_available())
            else "cpu"
        )
        return KittenTtsEngine.from_pretrained(model_path, device)
    raise ValueError(f"Unsupported TTS model: {settings.model_id}.")


@app.get("/v1/runtime-options")
def runtime_options(request: Request) -> dict[str, object]:
    _require_loopback(request)
    """List models, runtimes, and devices supported by the running API."""
    settings: EngineSettings | None = getattr(request.app.state, "engine_settings", None)
    if settings is None:
        raise HTTPException(503, "The speech models are still loading.")
    return _runtime_options(settings, request.app.state.app_config)


@app.post("/v1/runtime-options")
def set_runtime_options(
    selection: RuntimeSelection, request: Request
) -> dict[str, object]:
    """Load the selected supported model, runtime, and device for future requests."""
    # Model selection only affects this local speech service. Keep it usable
    # without unlocking the settings and model-download manager.
    _require_loopback(request)
    lock: Lock = request.app.state.engine_update_lock
    with lock:
        current: EngineSettings = request.app.state.engine_settings
        try:
            spec = next((item for item in MODEL_CATALOG if item.model_id == selection.model), None)
            if spec is None:
                raise ValueError(f"Unsupported TTS model: {selection.model}.")
            if selection.model != current.model_id:
                selection = selection.model_copy(update={"runtime": spec.runtime, "device": "auto"})
            elif selection.runtime != spec.runtime:
                selection = selection.model_copy(update={"runtime": spec.runtime})
            updated = replace(
                current,
                model_id=selection.model,
                runtime=selection.runtime,
                device=selection.device,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        try:
            # Build first, then swap: if loading fails, the current engine stays usable.
            engine = _load_engine(
                updated, request.app.state.app_config,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.exception("Could not load the selected speech runtime")
            raise HTTPException(500, "Could not load the selected speech model or device.") from exc
        config: AppConfig = request.app.state.app_config
        updated_config = replace(
            config, active_model=updated.model_id, runtime=updated.runtime, device=updated.device
        )
        try:
            save_config(updated_config)
        except OSError as exc:
            logger.exception("Could not persist the selected speech runtime")
            raise HTTPException(500, "Could not save the selected model settings.") from exc
        language = "fa" if updated.model_id == "pocket_farsi_v2" else "en"
        router: LanguageRoutedEngine | None = request.app.state.speech_engine
        if router is None:
            router = LanguageRoutedEngine({})
        router.set_engine(language, engine)
        request.app.state.speech_engine = router
        request.app.state.loaded_speech_engines[language] = engine
        request.app.state.engine_settings = updated
        request.app.state.app_config = updated_config
    return _runtime_options(updated, request.app.state.app_config)


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
    pause_multiplier: Annotated[float, Form(ge=0, le=2)] = 1.0,
    language: Annotated[str, Form(pattern="^(fa|en|auto)$")] = "fa",
    model_speed: Annotated[float, Form(ge=0.5, le=2)] = 1.0,
    post_speed: Annotated[float, Form(ge=0.5, le=2)] = 1.0,
    voice_name: Annotated[str, Form(max_length=40)] = "Default",
    voice: Annotated[UploadFile | None, File()] = None,
) -> Response:
    """Synthesize Persian text and return a WAV file."""
    _require_speech_network(request)
    if min_chars > max_chars:
        raise HTTPException(422, "Minimum chunk length cannot exceed maximum chunk length.")
    if post_speed != 1.0 and shutil.which("ffmpeg") is None:
        raise HTTPException(422, "Post-processing speed needs FFmpeg on PATH.")
    engine: SpeechEngine | None = getattr(request.app.state, "speech_engine", None)
    if engine is None:
        raise HTTPException(503, "The speech models are still loading.")

    voice_path: Path | None = None
    try:
        voice_path = _save_voice(voice)

        options = GenerationOptions(
            max_chars=max_chars,
            min_chars=min_chars,
            split_on_comma=split_on_comma,
            retry_silent_chunks=retry_silent_chunks,
            frames_after_eos=frames_after_eos,
            eos_threshold=eos_threshold,
            pause_multiplier=pause_multiplier,
            language=language,
            model_speed=model_speed,
            post_speed=post_speed,
            voice_name=voice_name,
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


@app.post("/v1/speech/stream")
def synthesize_stream(
    request: Request,
    text: Annotated[str, Form(min_length=1)],
    max_chars: Annotated[int, Form(ge=20, le=500)] = 100,
    min_chars: Annotated[int, Form(ge=0, le=100)] = 30,
    split_on_comma: Annotated[bool, Form()] = True,
    retry_silent_chunks: Annotated[bool, Form()] = True,
    frames_after_eos: Annotated[int, Form(ge=0, le=16)] = 2,
    eos_threshold: Annotated[float, Form(ge=-8, le=-1)] = -4.0,
    pause_multiplier: Annotated[float, Form(ge=0, le=2)] = 1.0,
    language: Annotated[str, Form(pattern="^(fa|en|auto)$")] = "fa",
    model_speed: Annotated[float, Form(ge=0.5, le=2)] = 1.0,
    post_speed: Annotated[float, Form(ge=0.5, le=2)] = 1.0,
    voice_name: Annotated[str, Form(max_length=40)] = "Default",
    voice: Annotated[UploadFile | None, File()] = None,
) -> StreamingResponse:
    """Stream generated PCM as newline-delimited JSON records."""
    _require_speech_network(request)
    if min_chars > max_chars:
        raise HTTPException(422, "Minimum chunk length cannot exceed maximum chunk length.")
    if post_speed != 1.0 and shutil.which("ffmpeg") is None:
        raise HTTPException(422, "Post-processing speed needs FFmpeg on PATH.")
    engine: SpeechEngine | None = getattr(request.app.state, "speech_engine", None)
    if engine is None:
        raise HTTPException(503, "The speech models are still loading.")

    voice_path = _save_voice(voice)
    try:
        options = GenerationOptions(
            max_chars=max_chars,
            min_chars=min_chars,
            split_on_comma=split_on_comma,
            retry_silent_chunks=retry_silent_chunks,
            frames_after_eos=frames_after_eos,
            eos_threshold=eos_threshold,
            pause_multiplier=pause_multiplier,
            language=language,
            model_speed=model_speed,
            post_speed=post_speed,
            voice_name=voice_name,
        )
    except Exception:
        if voice_path is not None:
            voice_path.unlink(missing_ok=True)
        if voice is not None:
            voice.file.close()
        raise

    voice_cleanup_lock = Lock()
    voice_cleaned = False

    def cleanup_voice() -> None:
        nonlocal voice_cleaned
        with voice_cleanup_lock:
            if voice_cleaned:
                return
            voice_cleaned = True
            if voice_path is not None:
                voice_path.unlink(missing_ok=True)
            if voice is not None:
                voice.file.close()

    async def events():
        stop_event = Event()
        audio_queue: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=12)

        def send(message: tuple[str, object]) -> bool:
            while not stop_event.is_set():
                try:
                    audio_queue.put(message, timeout=0.1)
                    return True
                except queue.Full:
                    continue
            return False

        def produce_audio() -> None:
            try:
                for chunk in engine.stream_synthesize(
                    text=text,
                    options=options,
                    voice_path=voice_path,
                    stop_event=stop_event,
                ):
                    if not send(("audio", chunk)):
                        break
            except Exception as exc:
                send(("error", str(exc)))
            finally:
                cleanup_voice()
                send(("done", None))

        worker = Thread(target=produce_audio, daemon=True)
        worker.start()
        buffered = bytearray()
        sample_rate: int | None = None
        format_sent = False
        try:
            while True:
                try:
                    kind, value = await asyncio.to_thread(audio_queue.get, True, 0.1)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue
                if kind == "audio":
                    chunk = value
                    if not chunk.pcm_s16le and not chunk.pause_boundary:
                        continue
                    sample_rate = chunk.sample_rate
                    if not format_sent:
                        if not chunk.pcm_s16le:
                            continue
                        yield _json_line(
                            "format",
                            sample_rate=sample_rate,
                            encoding="pcm_s16le",
                            channels=1,
                        )
                        format_sent = True
                    buffered.extend(chunk.pcm_s16le)
                    if buffered and (
                        len(buffered) >= int(sample_rate * 2 * _STREAM_BLOCK_SECONDS)
                        or chunk.pause_boundary
                    ):
                        yield _json_line(
                            "audio",
                            pcm=base64.b64encode(buffered).decode("ascii"),
                        )
                        buffered.clear()
                elif kind == "error":
                    yield _json_line("error", message=value)
                elif kind == "done":
                    break

            if buffered:
                yield _json_line("audio", pcm=base64.b64encode(buffered).decode("ascii"))
            yield _json_line("done")
        finally:
            stop_event.set()

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        background=BackgroundTask(cleanup_voice),
    )
