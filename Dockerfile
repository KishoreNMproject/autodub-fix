FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    sox \
    libsox-dev \
    libsox-fmt-all \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY requirements-vits.txt .
RUN pip install -r requirements-vits.txt

RUN mkdir -p /app/models/hf_cache /app/models/tts_cache /app/models/xtts-v2

ENV MODELS_ROOT=/app/models \
    HF_HOME=/app/models/hf_cache \
    TTS_HOME=/app/models/tts_cache \
    XTTS_MODEL_DIR=/app/models/xtts-v2 \
    XTTS_MODEL_REPO=coqui/XTTS-v2 \
    XTTS_AUTO_DOWNLOAD=1 \
    XTTS_DOWNLOAD_LOCK_TIMEOUT=300 \
    XTTS_FILE_DOWNLOAD_TIMEOUT=900 \
    XTTS_FILE_DOWNLOAD_RETRIES=2 \
    XTTS_MAX_DOWNLOAD_SECONDS=1800 \
    HF_HUB_ETAG_TIMEOUT=30 \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    VITS_ALLOW_MODEL_NAME_FALLBACK=0 \
    VITS_INTERNAL_FINETUNE_ENABLED=1 \
    VITS_INTERNAL_REFERENCE_SECONDS=60 \
    TRANSLATION_RETRIES=2

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
