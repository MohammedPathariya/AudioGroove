# AudioGroove Deployment

## Production status

AudioGroove is deployed as a static Vercel frontend and a Docker-based Render
API. The deployed backend serves the recovered 250-song GRU-small artifact.
Both hosted generation paths and the production browser-origin CORS contract
were verified on 2026-08-28.

| Component | Production URL | Deployment source |
| --- | --- | --- |
| Frontend | `https://audiogroove-eosin.vercel.app` | Vercel, `frontend/` root directory, commit `47caa71` |
| Backend | `https://audiogroove-api.onrender.com` | Render Docker service, commit `47caa71` |
| Model artifact | `https://huggingface.co/pathmohd123/audiogroove-gru-small-250` | Hugging Face commit `aabd26b9344551f0a54d7977680e3846d18608b7` |

The frontend sends `seed_midi` requests to `POST /generate`. Render permits
`https://audiogroove-eosin.vercel.app` through the `FRONTEND_URL` environment
variable. The API returns an `audio/midi` response for download.

## Production architecture

```text
Browser
  -> Vercel static frontend
  -> Render POST /generate
  -> Flask and Gunicorn
  -> GRU-small inference model
  -> MIDI response

Render Docker build
  -> immutable Hugging Face artifact revision
  -> SHA-256 verification
  -> application startup
```

## Model artifact contract

The deployed model is `gru_small_250`:

- compact GRU-small trained on 250 songs
- 18,849-token train-only vocabulary
- 6,236,001 parameters
- inference-only `deploy.pt`
- deployment manifest with checkpoint, vocabulary, configuration, dataset,
  parameter count, report-hash, and HPC-provenance validation

The public artifact revision contains only:

```text
checkpoints/deploy.pt
vocabulary.json
config/experiment_config.json
deployment_manifest.json
```

`deploy.pt` contains model weights, dataset revision, and vocabulary hash. It
does not include optimizer or scheduler state. The application Dockerfile pins
all four artifact SHA-256 values and refuses an artifact mismatch.

## Runtime configuration

### Render

| Setting | Value |
| --- | --- |
| Dockerfile | `backend/Dockerfile` |
| Artifact base URL | `https://huggingface.co/pathmohd123/audiogroove-gru-small-250/resolve/aabd26b9344551f0a54d7977680e3846d18608b7` |
| Frontend origin | `https://audiogroove-eosin.vercel.app` |
| Python | 3.11 |
| Torch | `2.6.0+cpu` from the PyTorch CPU wheel index |
| Gunicorn workers | 1, selected by Render for the free CPU tier |

The CPU-only Torch pin is required. An unpinned Linux Torch installation pulled
CUDA libraries, produced a 3.06 GB local image, and reached the 512 MiB limit.

### Vercel

| Setting | Value |
| --- | --- |
| Root directory | `frontend` |
| Framework preset | Other |
| Build command | None |
| API endpoint | `https://audiogroove-api.onrender.com/generate` |

## Verification record

### Local 512 MiB gate

The GRU-small container was run locally with memory and swap limited to 512 MiB.

- Health reported `gru_small_250`, GRU `small`, dataset `250`, and vocabulary
  `18,849`.
- Unseeded and uploaded-seed generation returned parseable MIDI files.
- Final live memory was 220.9 MiB; cgroup peak memory was 236.2 MiB.
- Cgroup allocation denials, OOM events, and OOM kills were zero.

### Hosted verification

| Check | Result |
| --- | --- |
| Render health | HTTP 200; model loaded with the expected GRU-small contract |
| Unseeded generation | HTTP 200, 898-byte MIDI, 67.86 seconds, parsed with note events |
| Uploaded-seed generation | HTTP 200, 879-byte MIDI, 65.97 seconds, parsed with note events |
| Vercel CORS response | `Access-Control-Allow-Origin: https://audiogroove-eosin.vercel.app` |
| Browser-origin generation | HTTP 200, `audio/midi`, 879 bytes, 67.19 seconds, parsed with note events |

## Known limitations

- Render Free spins down after inactivity, so a cold request can be delayed by
  50 seconds or more.
- Warm generation takes about 66 to 68 seconds on the free CPU tier.
- The hosted service survived the verified requests, but Render Metrics has not
  yet supplied an exact hosted memory peak.
- The frontend must keep a visible loading state and prevent duplicate requests
  while a generation request is active.
- The GRU-small deployment choice addresses serving constraints. It does not
  replace the scientific model-selection record for GRU-large.

## Local development

Use a compatible ignored artifact package to run a local model check:

```bash
python3 -m src.generation.run_local_model
```

Run tests from the repository root:

```bash
python3 -m pytest -q
```

For a local container gate, build `backend/Dockerfile.local-test` with the
`local_artifacts/gru_small_250` build context, then run the image with both
memory and swap limited to 512 MiB.
