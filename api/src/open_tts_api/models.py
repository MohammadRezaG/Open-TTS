"""Model catalog and local Hugging Face asset installation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from threading import Event
import json
from pathlib import Path
import tempfile

from huggingface_hub import model_info, snapshot_download

from open_tts_common import AppConfig


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    label: str
    language: str
    repository: str
    required_files: tuple[str, ...]
    runtime: str
    devices: tuple[str, ...]
    voices: tuple[str, ...]
    g2p_repository: str | None = None


MODEL_CATALOG = (
    ModelSpec(
        "pocket_farsi_v2", "Pocket TTS Persian v2", "fa",
        "mehdi-hf/pocket-tts-farsi-v2",
        ("model.yaml", "model.safetensors", "tokenizer_ph.model"), "pytorch", ("cpu", "cuda"), ("Default",),
        "mehdi-hf/Homo-GE2PE-Persian-HF",
    ),
    ModelSpec(
        "kitten_mini_08", "Kitten TTS Mini 0.8", "en",
        "KittenML/kitten-tts-mini-0.8",
        ("config.json", "kitten_tts_mini_v0_8.onnx", "voices.npz"),
        "onnxruntime", ("cpu", "cuda"),
        ("Bella", "Jasper", "Luna", "Bruno", "Rosie", "Hugo", "Kiki", "Leo"),
    ),
)


def installed_models(config: AppConfig) -> dict[str, dict[str, object]]:
    result = {}
    for spec in MODEL_CATALOG:
        directory = config.models_dir / spec.model_id
        manifest_path = directory / "manifest.json"
        manifest = _read_manifest(manifest_path)
        required = [directory / name for name in spec.required_files]
        if spec.g2p_repository:
            required += [directory / "g2p" / "config.json", directory / "g2p" / "model.safetensors"]
        complete = (
            manifest.get("model_id") == spec.model_id
            and manifest.get("repository") == spec.repository
            and bool(manifest.get("revision"))
            and all(path.is_file() for path in required)
        )
        result[spec.model_id] = {
            "installed": complete,
            "version": manifest.get("revision") if manifest else None,
            "directory": str(directory),
            "missing_assets": [str(path.relative_to(directory)) for path in required if not path.is_file()],
        }
    return result


def catalog(config: AppConfig) -> list[dict[str, object]]:
    installed = installed_models(config)
    return [
        {
            **asdict(spec),
            "installed": installed[spec.model_id]["installed"],
            "version": installed[spec.model_id]["version"],
            "missing_assets": installed[spec.model_id]["missing_assets"],
        }
        for spec in MODEL_CATALOG
    ]


def install_model(
    model_id: str, config: AppConfig, cancel_event: Event | None = None
) -> dict[str, object]:
    spec = next((item for item in MODEL_CATALOG if item.model_id == model_id), None)
    if spec is None:
        raise ValueError(f"Unsupported model: {model_id}.")
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Model installation was cancelled.")

    target = config.models_dir / spec.model_id
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{spec.model_id}-", dir=target.parent))
    backup = target.with_name(f".{target.name}-previous")
    try:
        revision = model_info(spec.repository, revision="main").sha
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Model installation was cancelled.")
        snapshot_download(
            repo_id=spec.repository,
            revision=revision,
            local_dir=staging,
        )
        g2p_revision = None
        if spec.g2p_repository:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Model installation was cancelled.")
            g2p_revision = model_info(spec.g2p_repository, revision="main").sha
            snapshot_download(
                repo_id=spec.g2p_repository,
                revision=g2p_revision,
                local_dir=staging / "g2p",
            )
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Model installation was cancelled.")
        missing = [name for name in spec.required_files if not (staging / name).is_file()]
        if missing:
            raise ValueError(f"Model download is incomplete; missing: {', '.join(missing)}.")
        if spec.g2p_repository:
            yaml_path = staging / "model.yaml"
            yaml_text = yaml_path.read_text(encoding="utf-8")
            yaml_text = _set_yaml_path(yaml_text, "weights_path", target / "model.safetensors")
            yaml_text = _set_yaml_path(yaml_text, "tokenizer_path", target / "tokenizer_ph.model")
            yaml_path.write_text(yaml_text, encoding="utf-8")
        manifest = {
            "model_id": spec.model_id,
            "repository": spec.repository,
            "revision": revision,
            "g2p_repository": spec.g2p_repository,
            "g2p_revision": g2p_revision,
            "required_files": list(spec.required_files),
            "assets": [
                {
                    "path": str(path.relative_to(staging)).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                }
                for path in staging.rglob("*")
                if path.is_file() and ".cache" not in path.parts
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        if backup.exists():
            import shutil

            shutil.rmtree(backup)
        had_target = target.exists()
        if had_target:
            target.replace(backup)
        try:
            staging.replace(target)
        except Exception:
            if had_target and backup.exists():
                backup.replace(target)
            raise
        if backup.exists():
            import shutil

            shutil.rmtree(backup)
        return installed_models(config)[spec.model_id]
    except Exception:
        import shutil

        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_manifest(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _set_yaml_path(contents: str, key: str, path: Path) -> str:
    import re

    value = str(path.resolve()).replace("\\", "/")
    updated, count = re.subn(
        rf"(?m)^(\s*{re.escape(key)}:\s*).*$", rf'\g<1>"{value}"', contents
    )
    if count != 1:
        raise ValueError(f"Model config must contain exactly one {key} entry.")
    return updated
