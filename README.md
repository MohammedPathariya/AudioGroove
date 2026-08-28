# AudioGroove

AudioGroove is a symbolic MIDI continuation application. A user selects a
curated MIDI sketch or uploads a MIDI file, the backend generates a continuation
with a recovered GRU-small model, and the browser offers the returned MIDI for
download. The deployed system is a bounded product integration, not a claim of
musical quality, originality, or broad generalization.

## Live deployment

| Frontend | Backend |
| --- | --- |
| [AudioGroove on Vercel](https://audiogroove-eosin.vercel.app) | [AudioGroove API on Render](https://audiogroove-api.onrender.com) |
| Vanilla HTML, CSS, and JavaScript | Flask and Gunicorn in Docker |
| Selects a sample or accepts a MIDI upload | Generates and returns `audio/midi` |
| Checks backend health and shows a loading state | Loads GRU-small and validates its artifact contract |

## Request flow

```text
User
  -> Vercel static frontend
  -> POST /generate with an optional seed_midi file
  -> Render Flask API
  -> GRU-small inference model
  -> MIDI response
  -> Browser download

Render builds the API image from GitHub and downloads the immutable model
artifact package from Hugging Face Hub. CORS permits the production Vercel
origin.
```

## Verified production contract

- Frontend deployment: `https://audiogroove-eosin.vercel.app`
- Backend deployment: `https://audiogroove-api.onrender.com`
- Application commit: `47caa71` (`allow production frontend CORS`)
- Model artifact: `pathmohd123/audiogroove-gru-small-250` at commit
  `aabd26b9344551f0a54d7977680e3846d18608b7`
- Model: compact GRU-small, trained on 250 songs
- Vocabulary: 18,849 train-only tokens
- Parameters: 6,236,001
- Hosted health endpoint reports the expected model, profile, dataset size, and
  vocabulary size.
- Hosted unseeded and uploaded-seed generation both returned HTTP 200 MIDI
  files that parsed successfully and contained note events.
- The production API returns the required CORS header for
  `https://audiogroove-eosin.vercel.app`.

## Architecture

### Frontend

- Static files in `frontend/`, deployed from the `frontend` root directory on
  Vercel.
- Curated MIDI sketches and audio previews are served with the site.
- The browser sends the selected or uploaded MIDI file as `seed_midi` to the
  Render API.
- The frontend disables generation controls and displays progress while a
  request is running.

### Backend

- `backend/app.py` exposes `GET /` for health and `POST /generate` for MIDI
  generation.
- The Docker image uses Python 3.11 and `torch==2.6.0+cpu`.
- Render runs one Gunicorn worker and receives the Vercel origin through the
  `FRONTEND_URL` environment variable.
- The artifact loader verifies model, vocabulary, configuration, dataset
  revision, parameter count, and SHA-256 hashes before inference.

### Model artifact

The model binary is intentionally outside the application repository. The
Hugging Face artifact revision contains only the deployment contract:

```text
checkpoints/deploy.pt
vocabulary.json
config/experiment_config.json
deployment_manifest.json
```

`deploy.pt` is inference-only. It includes model state, dataset revision, and
vocabulary hash, but not optimizer or scheduler state.

## Verification evidence

The GRU-small candidate passed two deployment gates on 2026-08-28.

| Gate | Result |
| --- | --- |
| Local Docker memory limit | 512 MiB limit, 236.2 MiB cgroup peak, zero allocation denials and OOM events |
| Local generation | Unseeded and uploaded-seed MIDI parsed successfully |
| Render health | Loaded `gru_small_250`, GRU `small`, dataset `250`, vocabulary `18,849` |
| Render generation | Unseeded generation completed in 67.86 seconds; seeded generation completed in 65.97 seconds |
| Vercel to Render CORS | Render returned `Access-Control-Allow-Origin: https://audiogroove-eosin.vercel.app` |
| Browser-origin generation | HTTP 200, `audio/midi`, 879 bytes, 67.19 seconds, MIDI parsed with note events |

## Limitations

- Render Free can spin down after inactivity. The first request may take 50
  seconds or more before generation begins.
- Generation currently takes about 66 to 68 seconds on the free CPU tier.
- The local 512 MiB gate is strong deployment evidence, but an exact hosted
  memory peak has not been recorded from Render Metrics.
- The deployed GRU-small model is selected for free-tier serving, not because it
  surpassed the GRU-large research result.
- The project has no completed musical-quality, originality, or human-listening
  evaluation harness.

## Local development

From the repository root, run the local model check with a compatible ignored
artifact package:

```bash
python3 -m src.generation.run_local_model
```

Run the full test suite:

```bash
python3 -m pytest -q
```

Detailed operational evidence and deployment instructions are in
[`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md). Training and experiment history are
recorded in [`docs/STATUS.md`](docs/STATUS.md).

## License

This project is licensed under the MIT License.
