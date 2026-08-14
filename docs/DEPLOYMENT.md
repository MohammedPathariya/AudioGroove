# AudioGroove Deployment

## Deployment status

The repository contains deployment-oriented files and references to a Vercel frontend and Hugging Face backend, but a current end-to-end hosted generation run has not been verified in this working session. Treat the URLs in the README as targets until the smoke test below passes.

The recovered 2,500-song GRU-large vocabulary and checkpoint are available
locally under the ignored `local_artifacts/gru_large_2500/` package. The
existing Flask backend still targets the older enhanced-LSTM artifact contract
and is not yet wired to this recovered compact GRU model.

## Local development

### Backend prerequisites

- Python 3.11 is the intended runtime.
- Install root dependencies from `requirements.txt`.
- Install backend dependencies from `backend/requirements.txt` if running the API separately.
- Ensure the processed vocabulary and compatible checkpoint exist before importing `backend/app.py`.
- For the recovered model, use `python -m src.generation.run_local_gru_2500`
  before changing the Flask integration.

### Device selection

Local training should prefer Apple MPS:

```python
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
```

The application must log the selected device. Do not report CPU and MPS results as equivalent without recording the distinction.

### Local API expectations

The Flask API exposes:

- `GET /`: health response with model-loaded state
- `POST /generate`: optional `seed_midi` upload, returning a MIDI file

Before starting the API, verify:

1. The vocabulary exists.
2. The checkpoint matches the vocabulary size and model configuration.
3. The seed directory contains valid MIDI files.
4. The selected port is available.

## Artifact requirements

Each deployable model artifact must be accompanied by:

- model configuration
- vocabulary file or vocabulary hash
- dataset version
- training commit
- training device
- validation metrics
- generation evaluation report
- artifact checksum
- data modality and representation version
- MLflow run ID and tracking backend
- Dask preprocessing configuration and worker or partition limits

Do not deploy a checkpoint if its vocabulary, representation, model configuration, or audio preprocessing configuration is unknown.

## Audio deployment considerations

If the project moves from MIDI output to MP3 or another audio output:

- the API must define accepted input formats and maximum duration
- decoding must enforce sample rate, channel count, and file-size limits
- processing must be isolated from untrusted media where practical
- output encoding must be deterministic enough for evaluation
- health checks must distinguish model loading from audio backend readiness
- latency and memory limits must be measured separately from MIDI generation

Raw audio generation is likely to require a different serving architecture from the current lightweight Flask and MIDI-file response path.

## Optional cloud training

Cloud GPU training is allowed only after the 250-song pilot is reproducible locally on the smoke or development dataset. The first cloud-scale run must use the pilot-approved Dask and MLflow configuration and the cloud run must use:

- a pinned repository commit
- a pinned dataset or manifest
- a fixed random seed
- a saved configuration
- a resumable checkpoint
- an exported evaluation report
- an exported MLflow run or tracking archive with its run ID
- the Dask preprocessing configuration and bounded partition limits

Cloud compute is for the larger LMDClean experiment after the pilot gate passes, not for debugging broken preprocessing or training code. The pilot comparison itself must remain reproducible on the bounded development setup where practical.

## Container requirements

The backend image must:

- pin the Python version
- install pinned dependencies
- download model artifacts from immutable revisions where possible
- fail clearly if an artifact download fails
- expose the configured service port
- run a health check after startup

The Dockerfile currently downloads the model and vocabulary with `wget`. This should eventually be changed to verify checksums and fail if the downloaded file is an LFS pointer, an HTML error page, or an incomplete artifact.

## Deployment smoke test

Record the following for every deployment:

```text
frontend_url:
backend_url:
frontend_commit:
backend_commit:
model_revision:
health_status:
model_loaded:
generation_status:
returned_output_valid:
generation_latency_seconds:
cold_start_latency_seconds:
known_limitations:
```

The smoke test is successful only when:

1. The frontend loads.
2. The backend health endpoint responds.
3. The backend reports the model as loaded.
4. A request without a seed succeeds.
5. A request with a valid seed succeeds when seeded input is supported.
6. The returned output passes the selected modality's parser or decoder.
7. The returned output contains valid musical content for the selected modality.
