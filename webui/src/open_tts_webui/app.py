"""Gradio interface for the local Persian speech API."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import gradio as gr
import httpx

API_URL = os.getenv("OPEN_TTS_API_URL", "http://127.0.0.1:8000").rstrip("/")

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
    voice_path: str | None,
) -> tuple[str | None, str]:
    if not text or not text.strip():
        return None, "Enter Persian text first."

    form = {
        "text": text,
        "max_chars": str(int(max_chars)),
        "min_chars": str(int(min_chars)),
        "split_on_comma": str(split_on_comma).lower(),
        "retry_silent_chunks": str(retry_silent_chunks).lower(),
        "frames_after_eos": str(int(frames_after_eos)),
        "eos_threshold": str(float(eos_threshold)),
    }
    files = None

    try:
        if voice_path:
            voice_file = Path(voice_path)
            with voice_file.open("rb") as stream:
                files = {"voice": (voice_file.name, stream, "audio/wav")}
                response = httpx.post(
                    f"{API_URL}/v1/speech",
                    data=form,
                    files=files,
                    timeout=httpx.Timeout(600, connect=10),
                )
        else:
            response = httpx.post(
                f"{API_URL}/v1/speech",
                data=form,
                timeout=httpx.Timeout(600, connect=10),
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json().get("detail", exc.response.text)
        except ValueError:
            detail = exc.response.text
        return None, f"Speech generation failed: {detail}"
    except httpx.RequestError as exc:
        return None, f"Cannot reach the API at {API_URL}: {exc}"

    with NamedTemporaryFile(prefix="open-tts-", suffix=".wav", delete=False) as output:
        output.write(response.content)
    return output.name, "Speech ready. You can play or download the WAV."


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Persian Voice Studio") as interface:
        gr.Markdown(
            """# Persian Voice Studio
Write in Persian and listen to natural speech. Generation runs locally; the
finished audio appears here when synthesis is complete.""",
            elem_id="studio-heading",
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=6, elem_classes="studio-panel"):
                gr.Markdown("### Your script")
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

                with gr.Accordion("Voice", open=False):
                    voice = gr.Audio(
                        label="Optional WAV reference (up to 5 seconds)",
                        sources=["upload"],
                        type="filepath",
                        format="wav",
                    )
                    clear_voice = gr.Button("Use default voice", size="sm")

            with gr.Column(scale=5, elem_classes="studio-panel", elem_id="listen-panel"):
                gr.Markdown("### Playback")
                gr.Markdown("Generated audio appears here when the request finishes.")
                audio = gr.Audio(
                    label="Generated speech",
                    type="filepath",
                    autoplay=True,
                    elem_id="out-audio",
                )
                status = gr.Markdown(f"Local API: `{API_URL}`")

        generate.click(
            fn=generate_speech,
            inputs=[
                text,
                max_chars,
                min_chars,
                split_on_comma,
                retry_silent,
                frames_after_eos,
                eos_threshold,
                voice,
            ],
            outputs=[audio, status],
        )
        clear.click(fn=lambda: ("", None, f"Local API: `{API_URL}`"), outputs=[text, audio, status])
        clear_voice.click(fn=lambda: None, outputs=voice)

    return interface


def main() -> None:
    build_interface().queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("OPEN_TTS_WEBUI_PORT", "7862")),
        theme=_THEME,
        css=_CSS,
    )


if __name__ == "__main__":
    main()
