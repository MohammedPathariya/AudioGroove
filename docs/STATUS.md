# AudioGroove Status

Updated: 2026-08-06

## Current state

AudioGroove is an end-to-end prototype for seeded MIDI generation. The current
pilot path has deterministic MIDI preprocessing, compact LSTM, GRU, and causal
Transformer families, a shared training and generation contract, MLflow
tracking, Slurm launch scripts, a Flask API, and a vanilla JavaScript frontend.

The local training state has been reset for HPC migration. Previous local run
directories, MLflow data, prepared training chunks, generated training logs,
and checkpoint artifacts were deleted. The source code and 250-song selection
audit remain. Training will proceed on Big Red 200, not Colab.

## Verified in the current checkout

- The repository is on `main` and was previously synchronized with `origin/main`.
- Ten seed MIDI files are present under `data/seed/`.
- The controlled comparison has unidirectional LSTM and GRU models plus an
  explicitly masked causal Transformer. All return one next-token distribution.
- The shared generation path uses a 32-token context, temperature sampling,
  top-k sampling, and a validation-split seed during model selection.
- The frontend supports optional MIDI upload, generation, regeneration, and download.
- Python syntax compilation passed during the initial review.
- Big Red 200 access is verified through Slurm project `r00284`.
- A Slurm GPU smoke test completed on an NVIDIA A100-SXM4 40 GB node with
  PyTorch `2.2.0+cu118`, CUDA available, and a successful CUDA matrix
  multiplication.
- The HPC Python module is `python/gpu/3.11.5`; personal scratch is writable at
  `/N/scratch/mjpathar`.
- The HPC copy contains 250/250 hash-verified source MIDI files. The first
  prepared revision `bf670db4...` produced 22,481 vocabulary tokens and
  2,757,737, 464,678, and 586,728 train, validation, and test windows, but fit
  the vocabulary on all three splits and is now classified as preliminary.
- LSTM GPU smoke job `7900523` completed with exit code `0:0`, CUDA training,
  checkpoint reload, MLflow run `ba27493426aa4baebd1f1082bdba50fe`, and valid
  generated MIDI. Its 16-example metrics are infrastructure evidence only.
- Preliminary full-budget jobs completed successfully for LSTM `7900534`, GRU
  `7900535`, and Transformer `7900536`. GRU had the best validation loss at
  6.9306, but these results cannot be the final benchmark because of the
  vocabulary-fit leakage. The 1.4 GB `preliminary-v1` archive contains 226
  checksum-verified files; all three MLflow runs reopen and generated MIDI
  files parse. Exact evidence is in
  [`docs/PRELIMINARY_V1_RESULTS.md`](PRELIMINARY_V1_RESULTS.md).
- Leakage-corrected HPC preprocessing job `7903822` completed with exit code
  `0:0` and produced revision `a68aee4e...` with an 18,849-token vocabulary fit
  on training songs only.
  Validation OOV is 7,229 of 465,862 tokens (1.55%); test OOV is 30,427 of
  587,944 tokens (5.18%).
- Corrected baseline jobs `7903830` LSTM, `7903831` GRU, and `7903832`
  Transformer have been submitted. Their completion and results are not yet
  verified.

## Day 1 foundation completed

- `data/raw/LMDClean` is the canonical raw-dataset path.
- Chunk outputs use `data/chunks/train`, `data/chunks/val`, and `data/chunks/test`.
- Repository Python modules use `src.*` package imports when run from the repository root.
- The first-week decision is symbolic MIDI generation with the existing compact recurrent model as a baseline. Audio and alternative model families remain follow-up comparisons.

## Not verified

- The large cleaned training dataset is not present locally.
- `data/processed/` is empty in this checkout.
- The leakage-corrected baseline runs have not yet completed.
- The nine-profile sweep and three-seed finalist comparison have not run.
- The isolated per-job MLflow stores have not been consolidated into a shared
  tracking server.
- A current end-to-end hosted generation request has not been verified in this session.
- No defensible model-quality metrics have yet been produced.

## Known technical blockers

1. Corrected baseline jobs must complete before the profile sweep.
3. The project has no complete musical-statistics, originality, or human
   evaluation harness yet.
4. The current 250-song pilot has no reliable genre metadata and is not a
   genre-stratified generalization benchmark.
5. The large cleaned training dataset is not present locally.
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

Monitor the three corrected baseline jobs and validate their reports,
checkpoints, MLflow runs, and generated MIDI. Do not run the profile sweep
until those reports pass validation. Do not evaluate the frozen test split,
start larger-corpus training, or begin raw-audio training before the
three-seed finalist selection is frozen.

The detailed execution plan is in [`docs/MLOPS_PLAN.md`](MLOPS_PLAN.md), and
the cluster runbook is in [`docs/HPC_TRAINING.md`](HPC_TRAINING.md).

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
- HPC preprocessing produced derived revision `bf670db4...`, vocabulary size
  22,481, and the expected window counts. Subsequent review found that its
  vocabulary was fit on train, validation, and test tokens. It is retained as
  preliminary evidence, not an accepted final data contract. The corrected
  train-only-vocabulary revision is `a68aee4e...`.
