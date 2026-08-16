FROM python:3.11-slim

WORKDIR /code
ENV PYTHONPATH=/code
ENV PORT=7860

COPY backend/requirements.txt /code/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/backend/requirements.txt

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates wget \
    && rm -rf /var/lib/apt/lists/*

# Build from the repository root. The artifact repository must expose these
# paths: checkpoints/best.pt, vocabulary.json, config/experiment_config.json,
# and datasets/scaling_summary.json.
ARG ARTIFACT_BASE_URL
ARG BEST_SHA256=16e3fbd6b950e7fca124f970e3b1579c5f8f125bc9535f8d6fa0217cb4cb0700
ARG VOCABULARY_SHA256=adca960cd5bbd89500e8977a485f2e5ae5e07f80560fd648044f7bf309a1f267
ARG CONFIG_SHA256=75a32ece3ae612962aedc87021e9f5d2dfb6a665adb7828e5b96fcf09e8a7929
ARG SCALING_SUMMARY_SHA256=6ba78eabf9833bc6308619b4fb6366f761534b4d68da85d7ea778eab50b0c09a

RUN test -n "$ARTIFACT_BASE_URL" \
    && mkdir -p /code/local_artifacts/gru_large_2500/checkpoints \
        /code/local_artifacts/gru_large_2500/config \
        /code/local_artifacts/gru_large_2500/datasets \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_large_2500/checkpoints/best.pt \
        "$ARTIFACT_BASE_URL/checkpoints/best.pt" \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_large_2500/vocabulary.json \
        "$ARTIFACT_BASE_URL/vocabulary.json" \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_large_2500/config/experiment_config.json \
        "$ARTIFACT_BASE_URL/config/experiment_config.json" \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_large_2500/datasets/scaling_summary.json \
        "$ARTIFACT_BASE_URL/datasets/scaling_summary.json" \
    && echo "$BEST_SHA256  /code/local_artifacts/gru_large_2500/checkpoints/best.pt" | sha256sum -c - \
    && echo "$VOCABULARY_SHA256  /code/local_artifacts/gru_large_2500/vocabulary.json" | sha256sum -c - \
    && echo "$CONFIG_SHA256  /code/local_artifacts/gru_large_2500/config/experiment_config.json" | sha256sum -c - \
    && echo "$SCALING_SUMMARY_SHA256  /code/local_artifacts/gru_large_2500/datasets/scaling_summary.json" | sha256sum -c -

COPY backend /code/backend
COPY src /code/src
COPY data/seed /code/data/seed

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import json, urllib.request; response = urllib.request.urlopen('http://127.0.0.1:7860/', timeout=5); payload = json.load(response); assert payload['model_loaded'] is True"

CMD ["gunicorn", "--chdir", "/code", "backend.app:app", "--bind", "0.0.0.0:7860", "--timeout", "120"]
