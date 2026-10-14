"""Per-user TOML configuration with environment overrides for deployment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import tempfile
import tomllib


def default_data_dir() -> Path:
    if os.name == "nt":
        return Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "OpenTTS"
    return Path(os.getenv("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "open-tts"


def config_path() -> Path:
    override = os.getenv("OPEN_TTS_CONFIG")
    return Path(override).expanduser() if override else default_data_dir() / "config.toml"


@dataclass(frozen=True, slots=True)
class AppConfig:
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    webui_host: str = "127.0.0.1"
    webui_port: int = 7863
    allowed_networks: tuple[str, ...] = ()
    models_dir: Path = default_data_dir() / "models"
    logs_dir: Path = default_data_dir() / "logs"
    log_level: str = "INFO"
    log_retention: int = 5
    admin_session_minutes: int = 60
    active_model: str = "pocket_farsi_v2"
    runtime: str = "pytorch"
    device: str = "auto"

    def __post_init__(self) -> None:
        if not 1 <= self.api_port <= 65535 or not 1 <= self.webui_port <= 65535:
            raise ValueError("API and web UI ports must be between 1 and 65535.")
        if self.log_level.upper() not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError("Log level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL.")
        if not 1 <= self.log_retention <= 30:
            raise ValueError("Log retention must be between 1 and 30 files.")
        if not 1 <= self.admin_session_minutes <= 1440:
            raise ValueError("Admin session timeout must be between 1 and 1440 minutes.")


def load_config(path: Path | None = None) -> AppConfig:
    path = path or config_path()
    if not path.exists():
        return AppConfig()
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    network = data.get("network", {})
    storage = data.get("storage", {})
    logging = data.get("logging", {})
    security = data.get("security", {})
    model = data.get("model", {})
    defaults = AppConfig()
    values = {
        "api_host": network.get("api_host", defaults.api_host),
        "api_port": network.get("api_port", defaults.api_port),
        "webui_host": network.get("webui_host", defaults.webui_host),
        "webui_port": network.get("webui_port", defaults.webui_port),
        "allowed_networks": tuple(network.get("allowed_networks", defaults.allowed_networks)),
        "models_dir": Path(storage.get("models_dir", defaults.models_dir)).expanduser(),
        "logs_dir": Path(storage.get("logs_dir", defaults.logs_dir)).expanduser(),
        "log_level": logging.get("level", defaults.log_level),
        "log_retention": logging.get("retention", defaults.log_retention),
        "admin_session_minutes": security.get("admin_session_minutes", defaults.admin_session_minutes),
        "active_model": model.get("active", defaults.active_model),
        "runtime": model.get("runtime", defaults.runtime),
        "device": model.get("device", defaults.device),
    }
    env_fields = {
        "api_host": "OPEN_TTS_API_HOST",
        "api_port": "OPEN_TTS_API_PORT",
        "webui_host": "OPEN_TTS_WEBUI_HOST",
        "webui_port": "OPEN_TTS_WEBUI_PORT",
        "models_dir": "OPEN_TTS_MODELS_DIR",
        "logs_dir": "OPEN_TTS_LOGS_DIR",
    }
    for field, variable in env_fields.items():
        if variable in os.environ:
            values[field] = os.environ[variable]
    for field in ("api_port", "webui_port"):
        values[field] = int(values[field])
    for field in ("models_dir", "logs_dir"):
        values[field] = Path(values[field])
    return replace(defaults, **values)


def save_config(config: AppConfig, path: Path | None = None) -> Path:
    """Atomically write user configuration. Secrets never belong in this file."""
    path = path or config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[network]",
        f'api_host = "{_toml_string(config.api_host)}"',
        f"api_port = {config.api_port}",
        f'webui_host = "{_toml_string(config.webui_host)}"',
        f"webui_port = {config.webui_port}",
        "allowed_networks = [" + ", ".join(f'"{_toml_string(item)}"' for item in config.allowed_networks) + "]",
        "",
        "[storage]",
        f'models_dir = "{_toml_string(str(config.models_dir))}"',
        f'logs_dir = "{_toml_string(str(config.logs_dir))}"',
        "",
        "[logging]",
        f'level = "{config.log_level.upper()}"',
        f"retention = {config.log_retention}",
        "",
        "[security]",
        f"admin_session_minutes = {config.admin_session_minutes}",
        "",
        "[model]",
        f'active = "{_toml_string(config.active_model)}"',
        f'runtime = "{_toml_string(config.runtime)}"',
        f'device = "{_toml_string(config.device)}"',
        "",
    ]
    content = "\n".join(lines)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(content)
        temporary = Path(tmp.name)
    temporary.replace(path)
    return path


def _toml_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")
