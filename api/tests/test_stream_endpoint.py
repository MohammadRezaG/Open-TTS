"""Lightweight API streaming checks using a fake speech engine."""

import asyncio
import base64
import json
from threading import Event
import time
from types import SimpleNamespace
import unittest

from fastapi import Request

from open_tts_api.main import app, synthesize, synthesize_stream
from open_tts_engine.audio import AudioChunk
from open_tts_common import AppConfig


class FakeEngine:
    def __init__(self):
        self.stop_event: Event | None = None

    def stream_synthesize(self, *, text, options, voice_path, stop_event):
        self.stop_event = stop_event
        self.options = options
        yield AudioChunk(24000, b"\x01\x00" * 12000)
        yield AudioChunk(24000, b"", pause_boundary=True)
        yield AudioChunk(24000, b"\x02\x00" * 120)

    def synthesize(self, *, text, options, voice_path, stop_event=None):
        self.options = options
        return b"fake-wav"


def request_for_test() -> Request:
    return Request(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/speech/stream",
            "raw_path": b"/v1/speech/stream",
            "query_string": b"",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("test", 80),
            "app": app,
        }
    )


class StreamingEndpointTests(unittest.TestCase):
    def test_api_passes_language_and_both_speed_controls(self):
        engine = FakeEngine()
        app.state.speech_engine = engine
        app.state.app_config = AppConfig()
        from unittest.mock import patch

        with patch("open_tts_api.main.shutil.which", return_value="ffmpeg"):
            response = synthesize(
                request_for_test(), text="hello 2", language="en", model_speed=1.25,
                post_speed=0.9, voice_name="Bella",
            )
        self.assertEqual(response.body, b"fake-wav")
        self.assertEqual(engine.options.language, "en")
        self.assertEqual(engine.options.model_speed, 1.25)
        self.assertEqual(engine.options.post_speed, 0.9)
        self.assertEqual(engine.options.voice_name, "Bella")

    def test_stream_records_preserve_format_audio_and_completion(self):
        engine = FakeEngine()
        app.state.speech_engine = engine
        response = synthesize_stream(request_for_test(), text="سلام.")

        async def collect():
            return [line async for line in response.body_iterator]

        records = [json.loads(line) for line in asyncio.run(collect())]
        self.assertEqual(
            [record["event"] for record in records], ["format", "audio", "audio", "done"]
        )
        self.assertEqual(records[0]["sample_rate"], 24000)
        self.assertEqual(base64.b64decode(records[1]["pcm"]), b"\x01\x00" * 12000)
        self.assertEqual(base64.b64decode(records[2]["pcm"]), b"\x02\x00" * 120)

    def test_closing_client_stream_sets_engine_stop_event(self):
        class LongEngine(FakeEngine):
            def stream_synthesize(self, *, text, options, voice_path, stop_event):
                self.stop_event = stop_event
                while not stop_event.is_set():
                    yield AudioChunk(24000, b"\x01\x00" * 12000)
                    time.sleep(0.01)

        engine = LongEngine()
        app.state.speech_engine = engine
        response = synthesize_stream(request_for_test(), text="سلام.")

        async def close_after_first_record():
            iterator = response.body_iterator
            await anext(iterator)
            await iterator.aclose()

        asyncio.run(close_after_first_record())
        deadline = time.monotonic() + 1
        while engine.stop_event is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(engine.stop_event)
        self.assertTrue(engine.stop_event.wait(1))
