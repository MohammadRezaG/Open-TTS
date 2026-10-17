from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from open_tts_common import AppConfig, load_config, save_config


class ConfigTests(unittest.TestCase):
    def test_config_round_trips_paths_networks_and_runtime_selection(self):
        workspace_temp = Path(__file__).resolve().parents[2] / ".tmp"
        workspace_temp.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=workspace_temp) as temporary:
            config_path = Path(temporary) / "config.toml"
            expected = AppConfig(
                allowed_networks=("192.0.2.0/24",),
                models_dir=Path(temporary) / "models",
                logs_dir=Path(temporary) / "logs",
                log_level="DEBUG",
                active_model="kitten_mini_08",
                runtime="onnxruntime",
                device="cpu",
            )

            save_config(expected, config_path)
            actual = load_config(config_path)

        self.assertEqual(actual, expected)

    def test_model_storage_environment_override_wins_over_toml(self):
        workspace_temp = Path(__file__).resolve().parents[2] / ".tmp"
        workspace_temp.mkdir(exist_ok=True)
        with TemporaryDirectory(dir=workspace_temp) as temporary:
            config_path = Path(temporary) / "config.toml"
            config = AppConfig(models_dir=Path(temporary) / "configured-models")
            save_config(config, config_path)
            with patch.dict(
                "os.environ",
                {
                    "OPEN_TTS_CONFIG": str(config_path),
                    "OPEN_TTS_MODELS_DIR": str(Path(temporary) / "environment-models"),
                },
            ):
                actual = load_config()

        self.assertEqual(actual.models_dir, Path(temporary) / "environment-models")


if __name__ == "__main__":
    unittest.main()
