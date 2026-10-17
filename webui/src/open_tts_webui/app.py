"""Gradio interface for the local Persian speech API."""

from __future__ import annotations

import base64
from contextlib import ExitStack
import io
import json
import os
import ipaddress
import time
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterator
import wave
from open_tts_common import load_config

import gradio as gr
import httpx

_APP_CONFIG = load_config()
_API_HOST = _APP_CONFIG.api_host if _APP_CONFIG.api_host not in {"0.0.0.0", "::"} else "127.0.0.1"
API_URL = os.getenv(
    "OPEN_TTS_API_URL", f"http://{_API_HOST}:{_APP_CONFIG.api_port}"
).rstrip("/")

_CSS = """
body, .gradio-container { background: #11191e !important; color: #e9ece8 !important; }
.gradio-container { max-width: 1280px !important; padding: 32px 24px 56px !important; }
header, footer { display: none !important; }
#studio-heading { padding: 12px 4px 26px; border-bottom: 1px solid #2d383d; margin-bottom: 22px; }
#studio-heading h1 { color: #f1eee7; font-size: clamp(2rem, 4vw, 3.1rem); letter-spacing: -0.045em; margin-bottom: 8px; }
#studio-heading p { color: #aab5b3; font-size: 1rem; max-width: 650px; }
.studio-panel { background: #192328; border: 1px solid #304046; border-radius: 18px; padding: 22px !important; }
.studio-panel h2, .studio-panel h3 { color: #f1eee7; }
#script-input textarea { direction: rtl !important; text-align: right !important; line-height: 2 !important; font-size: 1.15rem !important; min-height: 250px !important; background: #10181d !important; border-color: #405157 !important; }
#generate-btn button { background: #e59c68 !important; color: #1c2224 !important; border: 0 !important; font-weight: 700 !important; min-height: 48px; }
#generate-btn button:hover { background: #f0ad7a !important; }
#listen-panel { min-height: 375px; }
#out-audio { margin-top: 12px; }
.gradio-container details > summary { color: #e9ece8 !important; font-weight: 600; }
.gradio-container input[type=range] { accent-color: #e59c68; }
.gradio-container button:focus-visible, .gradio-container textarea:focus-visible { outline: 2px solid #e59c68 !important; outline-offset: 2px; }
@media (max-width: 760px) {
  .gradio-container { padding: 18px 12px 36px !important; }
  .studio-panel { padding: 16px !important; }
  #script-input textarea { min-height: 190px !important; }
  #listen-panel { min-height: 240px; }
}
"""

_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.orange,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
    font=["Inter", "Segoe UI", "sans-serif"],
)


def generate_speech(
    text: str,
    max_chars: int,
    min_chars: int,
    split_on_comma: bool,
    retry_silent_chunks: bool,
    frames_after_eos: int,
    eos_threshold: float,
    pause_multiplier: float,
    language: str,
    model_speed: float,
    post_speed: float,
    voice_name: str,
    voice_path: str | None,
) -> Iterator[tuple[bytes | None, str | None, str, str | None]]:
    if not text or not text.strip():
        yield None, None, "Enter Persian text first.", None
        return

    form = {
        "text": text,
        "max_chars": str(int(max_chars)),
        "min_chars": str(int(min_chars)),
        "split_on_comma": str(split_on_comma).lower(),
        "retry_silent_chunks": str(retry_silent_chunks).lower(),
        "frames_after_eos": str(int(frames_after_eos)),
        "eos_threshold": str(float(eos_threshold)),
        "pause_multiplier": str(float(pause_multiplier)),
        "language": language,
        "model_speed": str(float(model_speed)),
        "post_speed": str(float(post_speed)),
        "voice_name": voice_name,
    }
    output_path: Path | None = None
    wav_file: wave.Wave_write | None = None
    received_audio = False
    completed = False
    try:
        with ExitStack() as stack:
            files = None
            if voice_path:
                voice_file = Path(voice_path)
                voice_stream = stack.enter_context(voice_file.open("rb"))
                files = {"voice": (voice_file.name, voice_stream, "audio/wav")}
            with httpx.stream(
                "POST",
                f"{API_URL}/v1/speech/stream",
                data=form,
                files=files,
                timeout=httpx.Timeout(600, connect=10),
            ) as response:
                if response.is_error:
                    try:
                        detail = response.json().get("detail", response.text)
                    except ValueError:
                        detail = response.text
                    yield None, None, f"Speech generation failed: {detail}", None
                    return

                sample_rate = None
                with NamedTemporaryFile(prefix="open-tts-", suffix=".wav", delete=False) as output:
                    output_path = Path(output.name)
                for line in response.iter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event["event"] == "format":
                        if event["encoding"] != "pcm_s16le" or event["channels"] != 1:
                            raise RuntimeError("The API returned an unsupported audio format.")
                        sample_rate = int(event["sample_rate"])
                        wav_file = wave.open(str(output_path), "wb")
                        wav_file.setnchannels(1)
                        wav_file.setsampwidth(2)
                        wav_file.setframerate(sample_rate)
                    elif event["event"] == "audio":
                        if wav_file is None or sample_rate is None:
                            raise RuntimeError("The API sent audio before its format record.")
                        pcm = base64.b64decode(event["pcm"], validate=True)
                        wav_file.writeframesraw(pcm)
                        received_audio = True
                        yield _wav_block(pcm, sample_rate), None, "Generating speech…", None
                    elif event["event"] == "error":
                        raise RuntimeError(event.get("message", "Speech generation failed."))
                    elif event["event"] == "done":
                        completed = True
                        break
                else:
                    raise RuntimeError("The API stream ended before speech was complete.")
    except httpx.RequestError as exc:
        yield None, None, f"Cannot reach the API at {API_URL}: {exc}", None
        return
    except (ValueError, RuntimeError) as exc:
        yield None, None, f"Speech generation failed: {exc}", None
        return
    finally:
        if wav_file is not None:
            wav_file.close()
        if output_path is not None and (not received_audio or not completed):
            output_path.unlink(missing_ok=True)

    if output_path is None or not received_audio or not completed:
        yield None, None, "The API returned no speech audio.", None
        return
    # A final None tells Gradio to flush the encoder and close the live stream.
    yield None, str(output_path), "Speech ready. Listen above or download the WAV.", str(output_path)


def _wav_block(pcm: bytes, sample_rate: int) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return output.getvalue()


def _runtime_updates(options: dict) -> tuple[object, object, object, str]:
    selection = options["selection"]
    selected_model = next(
        item for item in options["models"] if item["value"] == selection["model"]
    )
    active_device = selection["active_device"]
    device_label = {"cpu": "🖥 CPU", "cuda": "🎮 GPU", "unavailable": "Unavailable"}.get(
        active_device, "Auto"
    )
    status = f"Runtime: **{device_label}** · {options['device_note']}"
    if not selected_model["installed"]:
        status += " Model files are missing; download this model in Local model and settings manager."
    elif not selected_model["backend_available"]:
        status += " Python backend is missing; stop the API, run `install.ps1`, then restart it."
    return (
        gr.update(
            choices=[
                (
                    f"{item['label']} · "
                    + (
                        "ready" if item.get("ready")
                        else "backend missing" if item.get("installed")
                        else "model download required"
                    ),
                    item["value"],
                )
                for item in options["models"]
            ],
            value=selection["model"],
        ),
        gr.update(
            choices=[(item["label"], item["value"]) for item in options["runtimes"]],
            value=selection["runtime"],
        ),
        gr.update(
            choices=[(item["label"], item["value"]) for item in options["devices"]],
            value=selection["device"],
        ),
        status,
    )


def load_runtime_settings() -> tuple[object, object, object, str]:
    try:
        response = httpx.get(f"{API_URL}/v1/runtime-options", timeout=5)
        response.raise_for_status()
        return _runtime_updates(response.json())
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            f"Cannot load model/runtime choices from the API: {exc}",
        )


def _is_local_request(request: gr.Request) -> bool:
    host = request.client.host if request.client else ""
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def show_local_management(request: gr.Request) -> object:
    return gr.update(visible=_is_local_request(request))


def load_local_settings(request: gr.Request) -> tuple[str, str, str, int, str]:
    if not _is_local_request(request):
        return "", "", "INFO", 5, ""
    try:
        response = httpx.get(f"{API_URL}/v1/settings", timeout=5)
        response.raise_for_status()
        settings = response.json()
        return (
            settings["models_dir"], settings["logs_dir"], settings["log_level"],
            settings["log_retention"], "\n".join(settings["allowed_networks"]),
        )
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError, KeyError):
        return "", "", "INFO", 5, ""


def request_owner_code() -> str:
    try:
        response = httpx.post(f"{API_URL}/v1/admin/request-key", timeout=5)
        response.raise_for_status()
        return "Owner code printed in the API terminal."
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        return f"Could not request an owner code: {exc}"


def exchange_owner_code(code: str) -> tuple[str | None, str]:
    try:
        response = httpx.post(
            f"{API_URL}/v1/admin/session", json={"code": code}, timeout=5
        )
        response.raise_for_status()
        payload = response.json()
        minutes = int(payload["expires_in"]) // 60
        return payload["session_key"], f"Local management unlocked for {minutes} minutes."
    except httpx.HTTPStatusError as exc:
        return None, f"Could not unlock management: {exc.response.text}"
    except (httpx.RequestError, ValueError, KeyError) as exc:
        return None, f"Could not unlock management: {exc}"


def management_controls(session_key: str | None) -> tuple[object, ...]:
    enabled = bool(session_key)
    return tuple(gr.update(interactive=enabled) for _ in range(4))


def install_model_from_ui(model_id: str, session_key: str | None) -> Iterator[tuple[str, str | None]]:
    try:
        response = httpx.post(
            f"{API_URL}/v1/models/{model_id}/install",
            headers={"x-open-tts-session": session_key or ""},
            timeout=httpx.Timeout(10, connect=10),
        )
        response.raise_for_status()
        job_id = response.json()["job_id"]
        yield "Download started. You can cancel it below.", job_id
        while True:
            status_response = httpx.get(
                f"{API_URL}/v1/models/install/{job_id}",
                headers={"x-open-tts-session": session_key or ""}, timeout=10,
            )
            status_response.raise_for_status()
            job = status_response.json()
            if job["status"] in {"complete", "failed", "cancelled"}:
                result = job.get("result") or {}
                message = job["message"]
                if job["status"] == "complete":
                    message += f" Installed in `{result.get('directory', '')}`. Restart the API to load it."
                yield message, None
                return
            yield job["message"], job_id
            time.sleep(1)
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        yield f"Model installation failed: {detail}", None
    except httpx.RequestError as exc:
        yield f"Cannot reach the API at {API_URL}: {exc}", None


def cancel_model_install(job_id: str | None, session_key: str | None) -> str:
    if not job_id:
        return "No active download to cancel."
    try:
        response = httpx.delete(
            f"{API_URL}/v1/models/install/{job_id}",
            headers={"x-open-tts-session": session_key or ""}, timeout=5,
        )
        response.raise_for_status()
        return "Cancellation requested. The current file download will be cleaned up when it returns."
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        return f"Could not cancel the download: {exc}"


def save_local_settings(
    models_dir: str, logs_dir: str, log_level: str, retention: int,
    networks: str, session_key: str | None,
) -> str:
    allowed_networks = [item.strip() for item in networks.splitlines() if item.strip()]
    try:
        response = httpx.post(
            f"{API_URL}/v1/settings",
            json={
                "models_dir": models_dir,
                "logs_dir": logs_dir,
                "log_level": log_level,
                "log_retention": int(retention),
                "allowed_networks": allowed_networks,
            },
            headers={"x-open-tts-session": session_key or ""},
            timeout=10,
        )
        response.raise_for_status()
        return "Settings saved. Restart the API to apply path and logging changes."
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError) as exc:
        return f"Could not save settings: {exc}"


def _voice_choices(language: str) -> tuple[object, object, object, object]:
    voices = ["Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"]
    choices = [("Default voice / upload a Persian prompt", "Default")] if language != "en" else []
    if language in {"en", "auto"}:
        choices += [(name, name) for name in voices]
    label = {"fa": "Persian text", "en": "English text", "auto": "Text"}.get(language, "Text")
    placeholder = {
        "fa": "Type or paste Persian text here...",
        "en": "Type or paste English text here...",
        "auto": "Type or paste Persian, English, or mixed text here...",
    }.get(language, "Type text here...")
    return (
        gr.update(choices=choices, value=choices[0][1]),
        gr.update(interactive=language != "fa"),
        gr.update(label=label, placeholder=placeholder, rtl=language != "en"),
        gr.update(interactive=language != "en"),
    )


def apply_runtime_settings(
    model: str, runtime: str, device: str, request: gr.Request
) -> tuple[object, object, object, str]:
    if not _is_local_request(request):
        return gr.update(), gr.update(), gr.update(), "Model selection is available from localhost only."
    try:
        response = httpx.post(
            f"{API_URL}/v1/runtime-options",
            json={"model": model, "runtime": runtime, "device": device},
            timeout=httpx.Timeout(600, connect=10),
        )
        response.raise_for_status()
        return _runtime_updates(response.json())
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        message = f"Could not apply selection: {detail}"
    except httpx.RequestError as exc:
        message = f"Cannot reach the API at {API_URL}: {exc}"

    try:
        current = httpx.get(f"{API_URL}/v1/runtime-options", timeout=5)
        current.raise_for_status()
        updates = _runtime_updates(current.json())
        return (*updates[:3], message)
    except (httpx.RequestError, httpx.HTTPStatusError, ValueError):
        return gr.update(), gr.update(), gr.update(), message


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Persian Voice Studio") as interface:
        admin_session = gr.State(None)
        gr.Markdown(
            """# Persian Voice Studio
Write in Persian and listen to natural speech. Generation runs locally and
playback starts while audio is being produced.""",
            elem_id="studio-heading",
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=6, elem_classes="studio-panel"):
                gr.Markdown("### Your script")
                language = gr.Dropdown(
                    label="Language", choices=[("Persian", "fa"), ("English", "en"), ("Auto detect per chunk", "auto")], value="fa"
                )
                text = gr.Textbox(
                    label="Persian text",
                    lines=8,
                    placeholder="Type or paste Persian text here...",
                    rtl=True,
                    elem_id="script-input",
                )
                with gr.Row():
                    generate = gr.Button(
                        "Generate speech", variant="primary", elem_id="generate-btn", scale=3
                    )
                    stop = gr.Button("Stop", scale=1)
                    clear = gr.Button("Clear", scale=1)

                with gr.Accordion("Generation settings", open=False):
                    split_on_comma = gr.Checkbox(
                        label="Split at commas",
                        value=True,
                        info="More, shorter chunks. Turn off for longer phrases.",
                    )
                    max_chars = gr.Slider(
                        label="Maximum characters per chunk",
                        minimum=20,
                        maximum=500,
                        step=10,
                        value=100,
                    )
                    min_chars = gr.Slider(
                        label="Merge chunks shorter than",
                        minimum=0,
                        maximum=100,
                        step=1,
                        value=30,
                    )
                    retry_silent = gr.Checkbox(
                        label="Retry failed chunks by merging with the next",
                        value=True,
                    )
                    frames_after_eos = gr.Slider(
                        label="Frames after speech ends",
                        minimum=0,
                        maximum=16,
                        step=1,
                        value=2,
                    )
                    eos_threshold = gr.Slider(
                        label="End-of-speech threshold",
                        minimum=-8,
                        maximum=-1,
                        step=0.5,
                        value=-4,
                    )
                    pause_multiplier = gr.Slider(
                        label="Breath / pause length",
                        minimum=0,
                        maximum=2,
                        step=0.1,
                        value=1.0,
                        info="Scales the automatic pause based on both neighboring sentences.",
                    )

                    model_speed = gr.Slider(
                        label="Model speech speed", minimum=0.5, maximum=2.0, step=0.05,
                        value=1.0, interactive=False,
                        info="Native speed control for Kitten English. Persian model has no native speed control.",
                    )
                    post_speed = gr.Slider(
                        label="Pitch-preserving post speed", minimum=0.5, maximum=2.0, step=0.05,
                        value=1.0, info="Requires FFmpeg on PATH. Sentence pauses are added after this step.",
                    )

                with gr.Accordion("Voice", open=False):
                    voice_name = gr.Dropdown(
                        label="Named voice", choices=[("Default voice / upload a Persian prompt", "Default")], value="Default"
                    )
                    voice = gr.Audio(
                        label="Optional WAV reference (up to 5 seconds)",
                        sources=["upload"],
                        type="filepath",
                        format="wav",
                    )
                    clear_voice = gr.Button("Use default voice", size="sm")

                with gr.Accordion("Model, runtime, and device", open=True):
                    gr.Markdown(
                        "Choose an installed model and a supported device. The runtime is selected "
                        "automatically because each model uses its own inference backend."
                    )
                    with gr.Row():
                        model = gr.Dropdown(label="Model", choices=[], value=None)
                        runtime = gr.Dropdown(label="Runtime", choices=[], value=None)
                        device = gr.Dropdown(label="Device", choices=[], value=None)
                    runtime_status = gr.Markdown("Loading model/runtime choices…")
                    for selector in (model, runtime, device):
                        selector.change(
                            fn=apply_runtime_settings,
                            inputs=[model, runtime, device],
                            outputs=[model, runtime, device, runtime_status],
                            queue=True,
                        )

                with gr.Accordion("Local model and settings manager", open=False, visible=False) as management:
                    gr.Markdown(
                        "Model downloads and settings changes require a one-time code from the API terminal. "
                        "Model files download here. To install or change Python backends, stop the API, "
                        "run `install.ps1` in the project folder, choose CPU or GPU, and restart the API."
                    )
                    request_code = gr.Button("Print owner code in API terminal")
                    owner_code = gr.Textbox(label="Owner code", type="password")
                    unlock = gr.Button("Unlock local management")
                    admin_status = gr.Markdown()
                    request_code.click(fn=request_owner_code, outputs=admin_status)
                    with gr.Row():
                        install_job_id = gr.State(None)
                        install_model = gr.Dropdown(
                            label="Model to install",
                            choices=[("Pocket TTS Persian v2", "pocket_farsi_v2"), ("Kitten TTS Mini 0.8", "kitten_mini_08")],
                            value="pocket_farsi_v2",
                            interactive=False,
                        )
                        install_button = gr.Button("Download model", interactive=False)
                        cancel_install = gr.Button("Cancel download", interactive=False)
                    install_status = gr.Markdown()
                    install_event = install_button.click(
                        fn=install_model_from_ui,
                        inputs=[install_model, admin_session],
                        outputs=[install_status, install_job_id],
                    )
                    cancel_install.click(
                        fn=cancel_model_install,
                        inputs=[install_job_id, admin_session],
                        outputs=install_status,
                    )
                    refresh_models = gr.Button("Refresh model choices")
                    refresh_models.click(
                        fn=load_runtime_settings, outputs=[model, runtime, device, runtime_status]
                    )
                    with gr.Accordion("Settings", open=False):
                        models_dir = gr.Textbox(label="Model folder", value="")
                        logs_dir = gr.Textbox(label="Log folder", value="")
                        log_level = gr.Dropdown(label="Log level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], value="INFO")
                        log_retention = gr.Slider(label="Log files to retain", minimum=1, maximum=30, step=1, value=5)
                        allowed_networks = gr.Textbox(label="Allowed speech client CIDRs", lines=3, placeholder="One CIDR per line. Loopback is always allowed.")
                        save_button = gr.Button("Save settings", interactive=False)
                        settings_status = gr.Markdown()
                        save_button.click(
                            fn=save_local_settings,
                            inputs=[models_dir, logs_dir, log_level, log_retention, allowed_networks, admin_session],
                            outputs=settings_status,
                        )
                    unlock.click(
                        fn=exchange_owner_code,
                        inputs=owner_code,
                        outputs=[admin_session, admin_status],
                    ).then(
                        fn=management_controls,
                        inputs=admin_session,
                        outputs=[install_model, install_button, cancel_install, save_button],
                    )

            with gr.Column(scale=5, elem_classes="studio-panel", elem_id="listen-panel"):
                gr.Markdown("### Playback")
                gr.Markdown("Audio starts playing as it is generated.")
                with gr.Accordion("Live preview", open=True):
                    live_audio = gr.Audio(
                        label="Audio as it generates",
                        type="filepath",
                        autoplay=True,
                        streaming=True,
                        format="wav",
                    )
                audio = gr.Audio(
                    label="Generated speech",
                    type="filepath",
                    format="wav",
                    elem_id="out-audio",
                )
                download = gr.File(label="Download complete WAV")
                status = gr.Markdown(f"Local API: `{API_URL}`")

        generation_event = generate.click(
            fn=generate_speech,
            inputs=[
                text,
                max_chars,
                min_chars,
                split_on_comma,
                retry_silent,
                frames_after_eos,
                eos_threshold,
                pause_multiplier,
                language,
                model_speed,
                post_speed,
                voice_name,
                voice,
            ],
            outputs=[live_audio, audio, status, download],
        )
        stop.click(fn=None, inputs=None, outputs=None, cancels=[generation_event])
        clear.click(
            fn=lambda: ("", None, None, f"Local API: `{API_URL}`", None),
            outputs=[text, live_audio, audio, status, download],
        )
        clear_voice.click(fn=lambda: None, outputs=voice)
        language.change(fn=_voice_choices, inputs=language, outputs=[voice_name, model_speed, text, voice])
        interface.load(fn=show_local_management, inputs=None, outputs=management)
        interface.load(
            fn=load_local_settings, inputs=None,
            outputs=[models_dir, logs_dir, log_level, log_retention, allowed_networks],
        )
        interface.load(
            fn=load_runtime_settings,
            inputs=None,
            outputs=[model, runtime, device, runtime_status],
        )
    return interface


def main() -> None:
    config = load_config()
    build_interface().queue().launch(
        server_name=config.webui_host,
        server_port=config.webui_port,
        theme=_THEME,
        css=_CSS,
    )


if __name__ == "__main__":
    main()
