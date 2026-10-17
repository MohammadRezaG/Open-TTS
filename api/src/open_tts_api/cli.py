"""Run the local API with Uvicorn."""

import uvicorn
from open_tts_common import load_config


def main() -> None:
    config = load_config()
    uvicorn.run(
        "open_tts_api.main:app",
        host=config.api_host,
        port=config.api_port,
    )
