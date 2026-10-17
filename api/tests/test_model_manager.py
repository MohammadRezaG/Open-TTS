import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from open_tts_api.models import install_model, installed_models
from open_tts_common import AppConfig


class ModelManagerTests(unittest.TestCase):
    def test_install_records_revisions_and_all_assets_locally(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / ".tmp") as temporary:
            config = AppConfig(models_dir=Path(temporary))

            def download(*, repo_id, revision, local_dir):
                folder = Path(local_dir)
                folder.mkdir(parents=True, exist_ok=True)
                if repo_id == "mehdi-hf/pocket-tts-farsi-v2":
                    (folder / "model.yaml").write_text(
                        "weights_path: placeholder\nflow_lm:\n  lookup_table:\n    tokenizer_path: placeholder\n",
                        encoding="utf-8",
                    )
                    for name in ("model.safetensors", "tokenizer_ph.model"):
                        (folder / name).write_bytes(b"asset")
                else:
                    for name in ("config.json", "model.safetensors"):
                        (folder / name).write_bytes(b"asset")

            with patch("open_tts_api.models.model_info", return_value=SimpleNamespace(sha="abc123")), patch(
                "open_tts_api.models.snapshot_download", side_effect=download
            ):
                installed = install_model("pocket_farsi_v2", config)

            self.assertTrue(installed["installed"])
            self.assertEqual(installed["version"], "abc123")
            manifest = json.loads((Path(temporary) / "pocket_farsi_v2" / "manifest.json").read_text())
            self.assertEqual(manifest["g2p_revision"], "abc123")
            local_config = (Path(temporary) / "pocket_farsi_v2" / "model.yaml").read_text()
            self.assertIn(str(Path(temporary) / "pocket_farsi_v2" / "model.safetensors").replace("\\", "/"), local_config)

    def test_cancelled_install_cleans_staging_and_preserves_previous_install(self):
        from threading import Event

        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / ".tmp") as temporary:
            config = AppConfig(models_dir=Path(temporary))
            target = Path(temporary) / "kitten_mini_08"
            target.mkdir()
            (target / "keep.txt").write_text("previous")
            cancelled = Event()
            cancelled.set()
            with self.assertRaises(InterruptedError):
                install_model("kitten_mini_08", config, cancelled)
            self.assertEqual((target / "keep.txt").read_text(), "previous")
            self.assertEqual(list(Path(temporary).glob(".kitten_mini_08-*")), [])

    def test_missing_assets_do_not_count_as_installed(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / ".tmp") as temporary:
            root = Path(temporary) / "kitten_mini_08"
            root.mkdir()
            (root / "manifest.json").write_text('{"revision":"r1"}')
            self.assertFalse(installed_models(AppConfig(models_dir=Path(temporary)))["kitten_mini_08"]["installed"])

    def test_download_failure_keeps_previous_install_and_removes_temporary_files(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[2] / ".tmp") as temporary:
            config = AppConfig(models_dir=Path(temporary))
            target = Path(temporary) / "kitten_mini_08"
            target.mkdir()
            (target / "keep.txt").write_text("previous")
            with patch("open_tts_api.models.model_info", return_value=SimpleNamespace(sha="r2")), patch(
                "open_tts_api.models.snapshot_download", side_effect=RuntimeError("network failed")
            ):
                with self.assertRaisesRegex(RuntimeError, "network failed"):
                    install_model("kitten_mini_08", config)
            self.assertEqual((target / "keep.txt").read_text(), "previous")
            self.assertEqual(list(Path(temporary).glob(".kitten_mini_08-*")), [])


if __name__ == "__main__":
    unittest.main()
