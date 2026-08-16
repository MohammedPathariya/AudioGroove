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
# paths: checkpoints/deploy.pt, vocabulary.json, config/experiment_config.json,
# and datasets/scaling_summary.json.
ARG ARTIFACT_BASE_URL
ARG DEPLOY_CHECKPOINT_SHA256=17e7ec8a4e6fc7e9f02d02ce93fd36b91aab9820b5b59fde8349afa58e227bb7
ARG VOCABULARY_SHA256=adca960cd5bbd89500e8977a485f2e5ae5e07f80560fd648044f7bf309a1f267
ARG CONFIG_SHA256=75a32ece3ae612962aedc87021e9f5d2dfb6a665adb7828e5b96fcf09e8a7929
ARG SCALING_SUMMARY_SHA256=6ba78eabf9833bc6308619b4fb6366f761534b4d68da85d7ea778eab50b0c09a

RUN test -n "$ARTIFACT_BASE_URL" \
    && mkdir -p /code/local_artifacts/gru_large_2500/checkpoints \
        /code/local_artifacts/gru_large_2500/config \
        /code/local_artifacts/gru_large_2500/datasets \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_large_2500/checkpoints/deploy.pt \
        "$ARTIFACT_BASE_URL/checkpoints/deploy.pt" \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_large_2500/vocabulary.json \
        "$ARTIFACT_BASE_URL/vocabulary.json" \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_large_2500/config/experiment_config.json \
        "$ARTIFACT_BASE_URL/config/experiment_config.json" \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_large_2500/datasets/scaling_summary.json \
        "$ARTIFACT_BASE_URL/datasets/scaling_summary.json" \
    && echo "$DEPLOY_CHECKPOINT_SHA256  /code/local_artifacts/gru_large_2500/checkpoints/deploy.pt" | sha256sum -c - \
    && echo "$VOCABULARY_SHA256  /code/local_artifacts/gru_large_2500/vocabulary.json" | sha256sum -c - \
    && echo "$CONFIG_SHA256  /code/local_artifacts/gru_large_2500/config/experiment_config.json" | sha256sum -c - \
    && echo "$SCALING_SUMMARY_SHA256  /code/local_artifacts/gru_large_2500/datasets/scaling_summary.json" | sha256sum -c -

COPY backend /code/backend
COPY src /code/src
COPY data/seed /code/data/seed

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import json, os, urllib.request; port = os.environ.get('PORT', '7860'); response = urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=5); payload = json.load(response); assert payload['model_loaded'] is True"

CMD ["sh", "-c", "exec gunicorn --chdir /code backend.app:app --bind 0.0.0.0:${PORT:-7860} --timeout 120"]
