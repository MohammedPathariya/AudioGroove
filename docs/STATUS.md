# AudioGroove Status

Updated: 2026-08-05

## Current state

AudioGroove is an end-to-end prototype for seeded MIDI generation. It has source code for MIDI extraction, vocabulary construction, chunked dataset creation, LSTM training, autoregressive sampling, a Flask API, and a vanilla JavaScript frontend.

The local training state has been reset for HPC migration. Previous local run
directories, MLflow data, prepared training chunks, generated training logs,
and checkpoint artifacts were deleted. The source code and 250-song selection
audit remain. Training will proceed on Big Red 200, not Colab.

## Verified in the current checkout

- The repository is on `main` and was previously synchronized with `origin/main`.
- Ten seed MIDI files are present under `data/seed/`.
- The enhanced model definition contains a stacked bidirectional LSTM and multi-head self-attention.
- The generation path uses a 32-token context, temperature sampling, and top-k sampling.
- The frontend supports optional MIDI upload, generation, regeneration, and download.
- Python syntax compilation passed during the initial review.
- Big Red 200 access is verified through Slurm project `r00284`.
- A Slurm GPU smoke test completed on an NVIDIA A100-SXM4 40 GB node with
  PyTorch `2.2.0+cu118`, CUDA available, and a successful CUDA matrix
  multiplication.
- The HPC Python module is `python/gpu/3.11.5`; personal scratch is writable at
  `/N/scratch/mjpathar`.

## Day 1 foundation completed

- `data/raw/LMDClean` is the canonical raw-dataset path.
- Chunk outputs use `data/chunks/train`, `data/chunks/val`, and `data/chunks/test`.
- Repository Python modules use `src.*` package imports when run from the repository root.
- The first-week decision is symbolic MIDI generation with the existing compact recurrent model as a baseline. Audio and alternative model families remain follow-up comparisons.

## Not verified

- The large cleaned training dataset is not present locally.
- `data/processed/` is empty in this checkout.
- A local training run has not been completed.
- An HPC AudioGroove training run has not been completed.
- Parallel Slurm model execution and HPC MLflow tracking have not been verified.
- A current end-to-end hosted generation request has not been verified in this session.
- No defensible model-quality metrics have yet been produced.

## Known technical blockers

1. The HPC training and MLflow execution path still needs implementation and verification.
2. The current bounded representation must be checked against the intended timing,
   velocity, program, and tempo contract before comparison training.
3. The project has no complete automated evaluation harness or human evaluation protocol.
4. The current 250-song pilot has no reliable genre metadata and is not a
   genre-stratified generalization benchmark.
5. The large cleaned training dataset is not present locally, and the active environment is missing `natsort` for chunk-management imports.
6. Model registry, deployment promotion, and post-deployment monitoring are not implemented.

## Modality and product strategy

The immediate product and MLOps path is symbolic MIDI generation. MIDI is not
treated as the final audio experience; it is the controlled representation for
model comparison, reproducible evaluation, and deployment integration.

- Complete the MIDI-first LSTM, GRU, and causal Transformer benchmark on HPC.
- Render valid generated MIDI to audio previews for the product experience.
- Treat genre-specific benchmarks as separate evaluation tracks because the
  current pilot has no reliable genre metadata.
- Defer raw MP3/audio modeling until the project has licensed audio, an explicit
  representation, an audio evaluation protocol, and a justified data scale.

## Definition of a successful pilot phase

By the end of the pilot phase, the project should have:

- a frozen 250-song source manifest and split
- a reproducible bounded loader on HPC
- a completed LSTM, GRU, and causal Transformer comparison
- MLflow records for every configuration and seed
- automated predictive, validity, musical-statistics, originality, and latency reports
- a held-out test evaluation performed only after model selection
- a registered model artifact with compatible vocabulary and representation metadata
- documented API and frontend deployment smoke tests
- no unsupported performance claims

The pilot does not require raw-audio model training. It requires a defensible
symbolic benchmark and an MLOps chain from data manifest through deployable
model artifact.

## Current priority

Move the frozen 250-song pilot to Big Red 200, verify HPC preprocessing and
MLflow tracking, run the parallel model comparison, and produce the first
model-promotion decision. Do not start larger-corpus training or raw-audio
training until this gate passes.

The detailed execution plan is in [`docs/MLOPS_PLAN.md`](MLOPS_PLAN.md).

## Next development phase: 250-song pilot

The planned pilot is a bounded development benchmark, not the final generalization test. It will:

- Select approximately 250 songs deterministically from a pinned LMDClean version.
- Record exact source paths, file hashes, selection seed, exclusions, and parser failures.
- Split by source song before windowing, grouping by artist or album when reliable metadata exists.
- Freeze the pilot test split before model comparison.
- Train and compare a compact LSTM, GRU, and compact Transformer using the same representation, source split, seed policy, training budget, and evaluation sample count.
- Run bounded preprocessing through Dask with a recorded worker or partition configuration.
- Track every baseline and comparison run in HPC MLflow, using a concurrency-safe tracking configuration, including dataset revision, Dask configuration, Git commit, model configuration, metrics, resources, checkpoints, and benchmark artifacts.
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
- On 2026-08-05, the ignored isolated copy was regenerated from
  `selected_manifest.jsonl` and verified against all 250 source SHA-256 hashes.
  It contains 190 distinct artists and remains the reproducible development
  pilot. The corpus layout has no reliable genre metadata, so this is not
  presented as a genre-stratified sample.
- The previous exploratory training state is not retained locally. New HPC
  training must start without a checkpoint or prior MLflow run.
- The migration target supplied for the clean HPC run is dataset revision
  `bf670db4f3390249537a2181cbab4635a7f9123fd864e74904c066ebe843d9fc`, with
  vocabulary size 22,481 and split window counts of 2,757,737 train,
  464,678 validation, and 586,728 test. These values must be checked against
  the retained manifests before training; the currently retained audit summary
  records an older source revision and must not be silently treated as the
  migration input.
