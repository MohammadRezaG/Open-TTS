"""UI-side checks for streaming PCM playback and the saved WAV."""

import base64
from contextlib import contextmanager
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import wave

from open_tts_webui import app


class FakeResponse:
    is_error = False

    def iter_lines(self):
        yield json.dumps({"event": "format", "sample_rate": 24000, "encoding": "pcm_s16le", "channels": 1})
        yield json.dumps({"event": "audio", "pcm": base64.b64encode(b"\x01\x00" * 240).decode()})
        yield json.dumps({"event": "done"})


class StreamingUiTests(unittest.TestCase):
    def test_streamed_audio_and_download_use_the_same_samples(self):
        @contextmanager
        def fake_stream(*_args, **_kwargs):
            yield FakeResponse()

        with patch.object(app.httpx, "stream", fake_stream):
            updates = list(
                app.generate_speech(
                    "سلام.", 100, 30, True, True, 2, -4, 1.0,
                    "fa", 1.0, 1.0, "Default", None
                )
            )

        self.assertEqual(len(updates), 2)
        played_audio, _final_audio, status, _download = updates[0]
        self.assertIn("Generating", status)
        with wave.open(io.BytesIO(played_audio), "rb") as playback:
            self.assertEqual(playback.getnframes(), 240)
        self.assertIsNone(updates[1][0])
        self.assertIn("Speech ready", updates[1][2])
        output_path = Path(updates[1][3])
        try:
            with wave.open(str(output_path), "rb") as saved:
                self.assertEqual(saved.readframes(saved.getnframes()), b"\x01\x00" * 240)
        finally:
            output_path.unlink(missing_ok=True)
