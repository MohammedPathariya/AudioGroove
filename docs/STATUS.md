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

## Day 1 foundation completed

- `data/raw/LMDClean` is the canonical raw-dataset path.
- Chunk outputs use `data/chunks/train`, `data/chunks/val`, and `data/chunks/test`.
- Repository Python modules use `src.*` package imports when run from the repository root.
- The first-week decision is symbolic MIDI generation with the existing compact recurrent model as a baseline. Audio and alternative model families remain follow-up comparisons.

## Not verified

- The large cleaned training dataset is not present locally.
- `data/processed/` is empty in this checkout.
- The tracked checkpoint files are Git LFS pointer files, not usable local weights.
- A local training run has not been completed.
- A current end-to-end hosted generation request has not been verified in this session.
- No defensible model-quality metrics have yet been produced.

## Known technical blockers

1. The current training target alignment does not match one-step autoregressive generation.
2. The current representation loses timing, duration, velocity, and reliable polyphony.
3. The merge-to-one-tensor workflow is unsafe for an 8 GB M1 machine.
4. The current code selects CUDA or CPU and does not properly support MPS as the primary local device.
5. The project has no automated evaluation harness or human evaluation protocol.
6. The large cleaned training dataset is not present locally, and the active environment is missing `natsort` for chunk-management imports.
7. Dask preprocessing and MLflow experiment tracking have been specified but not yet integrated into or verified by a pilot training run.

## Modality settled, representation and model selection pending

The project is not required to remain MIDI-only or LSTM-only.

- MIDI is currently the easiest path for interpretable preprocessing, compact experiments, and low-cost generation.
- MP3 or raw audio generation is a larger problem, not a simpler replacement. It requires audio decoding, segmentation, loudness and sample-rate policy, much larger datasets, a different model family, and audio-quality evaluation.
- A practical audio route may use pretrained audio representations or a pretrained audio-generation model rather than training raw waveform generation from scratch.
- Candidate model families include compact Transformers, GRUs, temporal convolutional networks, symbolic music Transformers, and pretrained audio or neural-codec models.
- Google Colab is available for larger training after the 250-song pilot passes the scale-up gate.

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

Use an approximately 250-song LMDClean subset as a controlled pilot before any larger-corpus training. The first meaningful result is a reproducible model comparison and benchmark report, not a larger checkpoint.

## Next development phase: 250-song pilot

The planned pilot is a bounded development benchmark, not the final generalization test. It will:

- Select approximately 250 songs deterministically from a pinned LMDClean version.
- Record exact source paths, file hashes, selection seed, exclusions, and parser failures.
- Split by source song before windowing, grouping by artist or album when reliable metadata exists.
- Freeze the pilot test split before model comparison.
- Train and compare a compact LSTM, GRU, and compact Transformer using the same representation, source split, seed policy, training budget, and evaluation sample count.
- Run bounded preprocessing through Dask with a recorded worker or partition configuration.
- Track every baseline and comparison run in local MLflow under `runs/mlruns/`, including dataset revision, Dask configuration, Git commit, model configuration, metrics, resources, checkpoints, and benchmark artifacts.
- Report predictive, MIDI-validity, musical-statistics, originality, latency, memory, and throughput metrics.

The larger LMDClean run is blocked until the pilot has a reproducible loader, verified Dask preprocessing, a stable bounded training run tracked in MLflow, verified MIDI serialization, a frozen evaluation report, and a documented resource profile.

## Day 2 data audit completed

Command:

```bash
python3 -m src.data_prep.day2_audit
```

The bounded audit used the local `data/seed/` corpus. The configured
`data/raw/LMDClean/` directory was absent and no download was attempted.
Results are recorded in `data/audit/day2/`:

- 10 source MIDI files found, 9 parsed, 0 unreadable, and 1 overlong file.
- `data/seed/Let_em_In.1.mid` was copied to
  `data/audit/day2/quarantine/49d3f754018557cd_Let_em_In.1.mid`; the source
  file was not deleted or moved.
- Maximum duration was 600 seconds. The quarantined file measured
  1018.636092 seconds.
- 9 eligible files were assigned by source file with seed `20260803`: train 7,
  validation 1, test 1.
- Smoke data contains 3 source files, 34,418 windows, and 135 bounded chunk
  files. Development data contains 8 source files, 101,462 windows, and 397
  bounded chunk files. Sequence length is 32 and the maximum chunk size is
  256 windows.
- `training_started` is `false`. These artifacts use mido note-on and note-off
  event tokens for preprocessing only; model training remains out of scope.
- Generated `.pt` chunks and the quarantined MIDI copy remain local and are
  ignored by Git. The manifests and `src/data_prep/day2_audit.py` regenerate
  them without publishing source media or machine-specific paths.

## LMDClean 250-song pilot selected

The cleaned corpus is available locally at `data/clean_midi/`. The bounded
pilot selector is `src/data_prep/build_lmdclean_pilot.py` and copies selected
source files into the ignored local folder `data/pilot_250/`, organized as
`train/`, `val/`, and `test/` by artist group. The source corpus is not moved
or modified.

Command:

```bash
python3 -m src.data_prep.build_lmdclean_pilot
```

Pilot evidence is recorded in `data/audit/lmdclean_pilot_250/`:

- Dataset revision: `cb79c82e90dc9087dc7f525d5ddf48648c0e7ba64d39fcdce6619acb94fbe62d`.
  This revision is the SHA-256 of sorted relative MIDI paths and byte sizes.
  Selected files also have content SHA-256 values in `selected_manifest.jsonl`.
- 17,256 MIDI files were inventoried, representing 10,277 unique artist and
  normalized song identities after numbered filename variants were grouped.
- Exactly 250 songs were selected with selection seed `20260803` and copied
  without source mutation.
- Frozen source-level split with split seed `20260804`: train 175, validation
  37, test 38. Artist groups were kept within one split. Album grouping was not
  used because no reliable album metadata or sidecar file was present.
- The pilot contains 2,503,234 event tokens and 2,495,234 next-token windows
  across 9,749 bounded chunks. Each chunk contains at most 256 windows and
  uses sequence length 32.
- Exclusions: 6,979 duplicate song identities, 3 overlong files, and 2
  unreadable files. No selected file is unreadable or overlong.
- Leakage checks passed: zero song-identity overlap, zero artist-group overlap,
  and zero SHA-256 overlap across splits.
- `training_started` and `larger_corpus_training_started` are both `false`.
