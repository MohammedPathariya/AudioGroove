# AudioGroove Status

Updated: 2026-08-03

## Current state

AudioGroove is an end-to-end prototype for seeded MIDI generation. It has source code for MIDI extraction, vocabulary construction, chunked dataset creation, LSTM training, autoregressive sampling, a Flask API, and a vanilla JavaScript frontend.

## Verified in the current checkout

- The repository is on `main` and was previously synchronized with `origin/main`.
- Ten seed MIDI files are present under `data/seed/`.
- The enhanced model definition contains a stacked bidirectional LSTM and multi-head self-attention.
- The generation path uses a 32-token context, temperature sampling, and top-k sampling.
- The frontend supports optional MIDI upload, generation, regeneration, and download.
- Python syntax compilation passed during the initial review.

## Not verified

- The large cleaned training dataset is not present locally.
- `data/processed/` is empty in this checkout.
- The tracked checkpoint files are Git LFS pointer files, not usable local weights.
- A local training run has not been completed.
- A current end-to-end hosted generation request has not been verified in this session.
- No defensible model-quality metrics have yet been produced.

## Known technical blockers

1. Dataset paths are inconsistent across data-preparation scripts.
2. Chunk split and merge scripts import path constants that are not defined in `src/utils/paths.py`.
3. The current training target alignment does not match one-step autoregressive generation.
4. The current representation loses timing, duration, velocity, and reliable polyphony.
5. The merge-to-one-tensor workflow is unsafe for an 8 GB M1 machine.
6. The current code selects CUDA or CPU and does not properly support MPS as the primary local device.
7. The project has no automated evaluation harness or human evaluation protocol.
8. Existing local source changes and untracked RNN files have not been reviewed, tested, or committed.

## Modality and model decision pending

The project is not required to remain MIDI-only or LSTM-only.

- MIDI is currently the easiest path for interpretable preprocessing, compact experiments, and low-cost generation.
- MP3 or raw audio generation is a larger problem, not a simpler replacement. It requires audio decoding, segmentation, loudness and sample-rate policy, much larger datasets, a different model family, and audio-quality evaluation.
- A practical audio route may use pretrained audio representations or a pretrained audio-generation model rather than training raw waveform generation from scratch.
- Candidate model families include compact Transformers, GRUs, temporal convolutional networks, symbolic music Transformers, and pretrained audio or neural-codec models.
- Google Colab is available for larger training after local smoke tests pass.

The modality and model choice must be recorded in `DECISIONS.md` before Day 3 implementation work begins.

## Definition of a successful first week

By the end of the week, the project should have:

- a reproducible small dataset and source-file split manifest
- a corrected streaming dataset loader
- MPS and CPU device selection
- a corrected next-token training objective
- a compact baseline that completes locally
- a selected-model comparison experiment, if memory and time permit
- automated predictive, validity, musical-statistics, originality, and latency reports
- documented local startup and deployment smoke-test procedures
- no unsupported performance claims

The week does not require training a raw-audio model from scratch. It requires making and documenting a defensible modality choice, then producing one reproducible baseline.

## Current priority

Fix the data and experiment foundations before increasing model size. The first meaningful result is a trustworthy baseline report, not a larger checkpoint.
