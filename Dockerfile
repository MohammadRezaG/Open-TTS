FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 HF_HOME=/models/huggingface
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    "pocket-tts @ git+https://github.com/mallahyari/pocket-tts@main" \
    gradio scipy transformers \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY ref-project/webui-v2 /app/webui-v2
WORKDIR /app/webui-v2

EXPOSE 7862
CMD ["python", "farsi_webui.py"]
