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
# and deployment_manifest.json.
ARG ARTIFACT_BASE_URL
ARG DEPLOY_CHECKPOINT_SHA256=4ad344ba119e9d4a2603b191ae6a175d1de184bdab19ef2518970fcd5a367e01
ARG VOCABULARY_SHA256=6a67a6b3d51d4c23a5af820175605b1c8612d178ba9f229e32ca7f3a90bec010
ARG CONFIG_SHA256=d002e7663c8d72756b507281f9d7fe515c23bc8bb3e92f3157b9180b20432587
ARG DEPLOY_MANIFEST_SHA256=73a4ed654fdfdc97c15c92bebf445d3c34c7ae8f3e90a2e96942aab23d81d519

RUN test -n "$ARTIFACT_BASE_URL" \
    && mkdir -p /code/local_artifacts/gru_small_250/checkpoints \
        /code/local_artifacts/gru_small_250/config \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_small_250/checkpoints/deploy.pt \
        "$ARTIFACT_BASE_URL/checkpoints/deploy.pt" \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_small_250/vocabulary.json \
        "$ARTIFACT_BASE_URL/vocabulary.json" \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_small_250/config/experiment_config.json \
        "$ARTIFACT_BASE_URL/config/experiment_config.json" \
    && wget --https-only --retry-connrefused --tries=3 \
        -O /code/local_artifacts/gru_small_250/deployment_manifest.json \
        "$ARTIFACT_BASE_URL/deployment_manifest.json" \
    && echo "$DEPLOY_CHECKPOINT_SHA256  /code/local_artifacts/gru_small_250/checkpoints/deploy.pt" | sha256sum -c - \
    && echo "$VOCABULARY_SHA256  /code/local_artifacts/gru_small_250/vocabulary.json" | sha256sum -c - \
    && echo "$CONFIG_SHA256  /code/local_artifacts/gru_small_250/config/experiment_config.json" | sha256sum -c - \
    && echo "$DEPLOY_MANIFEST_SHA256  /code/local_artifacts/gru_small_250/deployment_manifest.json" | sha256sum -c -

COPY backend /code/backend
COPY src /code/src
COPY data/seed /code/data/seed

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import json, os, urllib.request; port = os.environ.get('PORT', '7860'); response = urllib.request.urlopen(f'http://127.0.0.1:{port}/', timeout=5); payload = json.load(response); assert payload['model_loaded'] is True"

CMD ["sh", "-c", "exec gunicorn --chdir /code backend.app:app --bind 0.0.0.0:${PORT:-7860} --timeout 120"]
