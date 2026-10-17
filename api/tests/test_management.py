from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from fastapi import HTTPException, Request

from open_tts_api import main
from open_tts_common import AppConfig, OwnerSessions
from open_tts_engine import EngineSettings, LanguageRoutedEngine


def make_request(host: str, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    return Request({
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/", "raw_path": b"/",
        "query_string": b"", "headers": headers or [], "client": (host, 1234),
        "server": ("127.0.0.1", 8000), "app": main.app,
    })


class ManagementTests(unittest.TestCase):
    def setUp(self):
        main.app.state.app_config = AppConfig(allowed_networks=("192.0.2.0/24",))
        main.app.state.owner_sessions = OwnerSessions(timeout_minutes=60)

    def _owner_token(self) -> str:
        output = StringIO()
        with redirect_stdout(output):
            main.request_owner_code(make_request("127.0.0.1"))
        code = output.getvalue().split(": ", 1)[1].strip()
        return main.exchange_owner_code(main.OwnerCode(code=code), make_request("127.0.0.1"))["session_key"]

    def test_owner_code_exchange_and_admin_call(self):
        token = self._owner_token()
        request = make_request("::1", [(b"x-open-tts-session", token.encode())])
        main._require_owner(request)
        with self.assertRaises(HTTPException) as expired_or_used:
            main.exchange_owner_code(main.OwnerCode(code="wrong"), make_request("127.0.0.1"))
        self.assertEqual(expired_or_used.exception.status_code, 401)

    def test_management_rejects_non_loopback_even_with_session(self):
        token = self._owner_token()
        request = make_request("192.0.2.5", [(b"x-open-tts-session", token.encode())])
        with self.assertRaises(HTTPException) as denied:
            main._require_owner(request)
        self.assertEqual(denied.exception.status_code, 403)

    def test_speech_cidr_accepts_configured_network_and_rejects_other(self):
        main._require_speech_network(make_request("192.0.2.15"))
        with self.assertRaises(HTTPException) as denied:
            main._require_speech_network(make_request("198.51.100.15"))
        self.assertEqual(denied.exception.status_code, 403)

    def test_onnx_gpu_options_do_not_depend_on_pytorch_cuda(self):
        from unittest.mock import patch

        settings = EngineSettings(model_id="kitten_mini_08", runtime="onnxruntime")
        with patch.object(main, "_onnx_cuda_available", return_value=True), patch(
            "torch.cuda.get_device_name", side_effect=AssertionError("PyTorch CUDA queried")
        ):
            options = main._runtime_options(settings, AppConfig())
        self.assertIn({"value": "cuda", "label": "🎮 GPU — CUDA"}, options["devices"])

    def test_downloaded_model_without_python_backend_reports_not_ready(self):
        main.app.state.speech_engine = None
        with patch.object(
            main,
            "installed_models",
            return_value={"pocket_farsi_v2": {"installed": False}, "kitten_mini_08": {"installed": True}},
        ), patch.object(main, "_backend_available", return_value=False):
            status = main.health(make_request("127.0.0.1"))
            options = main._runtime_options(
                EngineSettings(model_id="kitten_mini_08", runtime="onnxruntime"),
                AppConfig(active_model="kitten_mini_08", runtime="onnxruntime"),
            )
        self.assertEqual(status["status"], "backend_required")
        kitten = next(item for item in options["models"] if item["value"] == "kitten_mini_08")
        self.assertTrue(kitten["installed"])
        self.assertFalse(kitten["ready"])

    def test_selecting_model_without_backend_returns_actionable_error(self):
        main.app.state.app_config = AppConfig()
        main.app.state.engine_settings = EngineSettings()
        main.app.state.engine_update_lock = __import__("threading").Lock()
        request = make_request("127.0.0.1")
        with patch.object(
            main,
            "installed_models",
            return_value={"pocket_farsi_v2": {"installed": False}, "kitten_mini_08": {"installed": True}},
        ), patch.object(main, "_backend_available", return_value=False):
            with self.assertRaises(HTTPException) as unavailable:
                main.set_runtime_options(
                    main.RuntimeSelection(model="kitten_mini_08", runtime="onnxruntime", device="cpu"),
                    request,
                )
        self.assertEqual(unavailable.exception.status_code, 422)
        self.assertIn("install.ps1", unavailable.exception.detail)

    def test_runtime_selection_is_loopback_only_and_needs_no_owner_session(self):
        class FakeEngine:
            pass

        main.app.state.app_config = AppConfig()
        main.app.state.engine_settings = EngineSettings()
        main.app.state.speech_engine = LanguageRoutedEngine({})
        main.app.state.loaded_speech_engines = {}
        main.app.state.engine_update_lock = __import__("threading").Lock()
        request = make_request("127.0.0.1")
        with patch.object(main, "installed_models", return_value={
            "pocket_farsi_v2": {"installed": True}, "kitten_mini_08": {"installed": False},
        }), patch.object(main, "_backend_available", return_value=True), patch.object(
            main, "_load_engine", return_value=FakeEngine()
        ), patch.object(main, "save_config"):
            result = main.set_runtime_options(
                main.RuntimeSelection(model="pocket_farsi_v2", runtime="pytorch", device="cpu"),
                request,
            )
        self.assertEqual(result["selection"]["model"], "pocket_farsi_v2")
        with self.assertRaises(HTTPException) as denied:
            main.set_runtime_options(
                main.RuntimeSelection(model="pocket_farsi_v2", runtime="pytorch", device="cpu"),
                make_request("192.0.2.5"),
            )
        self.assertEqual(denied.exception.status_code, 403)

    def test_unauthorized_settings_update_is_rejected(self):
        with self.assertRaises(HTTPException) as denied:
            main.update_settings(main.SettingsUpdate(log_level="DEBUG"), make_request("127.0.0.1"))
        self.assertEqual(denied.exception.status_code, 401)

    def test_failed_runtime_switch_keeps_current_engine(self):
        class ExistingEngine:
            pass

        old_engine = ExistingEngine()
        router = LanguageRoutedEngine({"fa": old_engine})
        main.app.state.speech_engine = router
        main.app.state.loaded_speech_engines = {"fa": old_engine}
        main.app.state.engine_settings = EngineSettings()
        main.app.state.engine_update_lock = __import__("threading").Lock()
        token = self._owner_token()
        request = make_request("127.0.0.1", [(b"x-open-tts-session", token.encode())])
        with patch.object(main, "_load_engine", side_effect=RuntimeError("mock load failure")):
            with self.assertRaises(HTTPException):
                main.set_runtime_options(
                    main.RuntimeSelection(model="pocket_farsi_v2", runtime="pytorch", device="cpu"),
                    request,
                )
        self.assertIs(main.app.state.speech_engine, router)
        self.assertIs(router._engines["fa"], old_engine)
        self.assertEqual(main.app.state.engine_settings.device, "auto")


if __name__ == "__main__":
    unittest.main()
