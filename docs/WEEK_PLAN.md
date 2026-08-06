# AudioGroove Pilot MLOps Weekly Plan

Updated: 2026-08-06

This plan covers the next implementation phase: a reproducible, HPC-backed
comparison of a compact LSTM, GRU, and causal Transformer on the frozen
250-song symbolic MIDI pilot. Raw MP3/audio modeling is explicitly deferred
to a separate licensed-data feasibility phase.

Do not advance past a phase while its stop condition is false unless the
failure is recorded in `docs/STATUS.md` and the relevant MLflow or audit
artifact.

## Phase 1: HPC and repository foundation

### Objective

Make the Big Red 200 environment reproducible before transferring data or
starting training.

### Work

- Clone the approved repository commit on Big Red 200.
- Use Slurm project `r00284` and personal scratch at `/N/scratch/mjpathar`.
- Load `python/gpu/3.11.5`.
- Record Python, PyTorch, CUDA, driver, GPU, Slurm, and Git metadata.
- Create separate directories for repository, data, checkpoints, MLflow, logs,
  and reports.
- Run the GPU smoke test and a repository import test.

### Stop condition

One Slurm job completes with CUDA-enabled PyTorch, and the environment record
can be reproduced on another GPU node.

## Phase 2: Frozen data contract

### Objective

Ensure every model sees exactly the same dataset and representation.

### Work

- Transfer or rebuild the isolated 250-song pilot from the original
  `data/clean_midi` corpus.
- Verify the selected manifest and every source SHA-256 hash.
- Resolve the source revision mismatch between the retained `cb79c82...`
  audit and the previous Colab-reported `bf670db4...` prepared dataset.
- Freeze the source-level split: 175 train, 37 validation, 38 test.
- Verify zero song, artist, and content-hash leakage.
- Fit the vocabulary on training songs only, map unseen validation and test
  tokens to `<UNK>`, and rebuild bounded chunks on HPC with deterministic
  ordering.
- Record vocabulary size, window counts, chunk counts, sequence length, and
  representation configuration.

### Stop condition

The HPC manifest matches the approved data contract and is identical for all
future model runs. The test split is frozen before tuning begins.

## Phase 3: Preprocessing and loader verification

### Objective

Prove that the real pilot can be loaded on a GPU node without relying on
Colab or Drive artifacts.

### Work

- Run Dask preprocessing with a recorded worker and ordering configuration.
- Validate chunk shapes, token ranges, vocabulary compatibility, and target
  alignment.
- Run a short loader-only job.
- Run a short forward/backward pass for all three model families.
- Confirm checkpoints can be saved and reloaded.

### Stop condition

All three models complete the same bounded smoke run on HPC, and the artifacts
can be reopened from scratch using only repository-relative configuration and
HPC data paths.

### Current evidence

- The 250-song HPC source copy passed all 250 SHA-256 checks.
- Revision `bf670db4...` produced the approved split and window counts but fit
  vocabulary on all splits, so its completed runs are preliminary only.
- Corrected HPC preprocessing job `7903822` produced revision `a68aee4e...`,
  vocabulary 18,849, validation OOV 1.55%, and test OOV 5.18% with exit code
  `0:0`.
- LSTM job `7900523` completed on CUDA, wrote MLflow run
  `ba27493426aa4baebd1f1082bdba50fe`, reloaded a checkpoint, and generated a
  parseable MIDI artifact.
- Local forward/backward and optimizer checks pass for LSTM, GRU, and causal
  Transformer through the shared trainer.
- Preliminary LSTM, GRU, and Transformer production jobs all completed on HPC,
  were archived with verified checksums, and have recoverable MLflow runs and
  parseable generated MIDI. Corrected baseline jobs `7903830`, `7903831`, and
  `7903832` completed successfully with verified reports, checkpoints, clean
  Git provenance, and parseable generated MIDI.

## Phase 4: Controlled baseline comparison

### Objective

Compare the three model families under one fixed training contract.

### Models

- Compact unidirectional LSTM
- Compact unidirectional GRU
- Compact causal Transformer

### Initial configurations

| Model | Embedding or model size | Layers | Other |
|---|---:|---:|---|
| LSTM | 128 embedding, 256 hidden | 1 | unidirectional |
| GRU | 128 embedding, 256 hidden | 1 | unidirectional |
| Transformer | 256 `d_model` | 2 | 4 heads, FFN 512 |

The complete small/baseline/large profile matrix is stored in
`training/configs/pilot_experiments.json`. Baseline trainable parameter counts
are 7,652,129 LSTM, 7,553,313 GRU, and 10,732,449 Transformer parameters with
the corrected 18,849-token vocabulary. This
is a fixed representative-family comparison, not a parameter-matched claim.

Keep fixed across models:

- sequence length
- effective batch size
- optimizer policy
- maximum epochs or steps
- gradient clipping
- evaluation sample count
- training and generation seeds

### Execution

Submit three independent one-GPU Slurm jobs. Slurm controls physical placement;
parallel jobs may share a multi-GPU node. Each supplied job uses a unique
MLflow file store and records a unique run and Slurm ID. A tracking-server URI
can be supplied when a shared server is available.

### Stop condition

All three baseline runs finish or fail with a recorded explanation. MLflow
contains parameters, epoch metrics, checkpoints, resources, and generated
MIDI artifacts for every run.

## Phase 5: Small hyperparameter sweep

### Objective

Measure configuration sensitivity without creating an uncontrolled grid.

### Sweep

Run small, baseline, and larger configurations for each family:

- LSTM and GRU: embedding 128 or 192; hidden size 192, 256, or 384; one or
  two layers.
- Transformer: `d_model` 192 or 256; two or four layers; four or eight heads;
  FFN size 512 or 1024.
- Learning rates: start with `1e-3` for LSTM/GRU and `3e-4` for Transformer.
- Weight decay: fixed initially at `1e-4`.
- Training budget: fixed five-epoch comparison budget before any longer run.

Select configurations using validation metrics and resource behavior only.

### Stop condition

The sweep report identifies the best configuration per model family, including
uncertainty caused by early stopping, runtime, and GPU memory.

## Phase 6: Seed confirmation and held-out evaluation

### Objective

Separate model selection from final generalization measurement.

### Work

- Run at least three training seeds for each finalist.
- Select the final candidate using validation results and seed stability.
- Evaluate each final candidate once on the frozen 38-song test split.
- Generate MIDI with identical seed files, generation lengths, temperatures,
  and sampling seeds.
- Produce one final comparison table.

### Metrics

- Cross-entropy and perplexity
- Token accuracy and top-5 accuracy
- Training time and throughput
- Peak GPU memory
- Checkpoint size
- MIDI parse and validity rate
- Generation latency and failure rate
- Pitch, note-density, velocity, duration, and time-shift statistics
- Repetition and training-overlap measures

### Stop condition

The test results are produced once, after model and configuration selection,
and the report clearly labels validation versus held-out test evidence.

## Phase 7: Registry, deployment, and monitoring

### Objective

Turn the selected pilot model into a verifiable product artifact.

### Work

- Register the selected checkpoint with its vocabulary and representation
  metadata.
- Promote it only after compatibility and held-out evaluation checks pass.
- Connect the model to the generation API.
- Validate MIDI generation with and without a seed file.
- Render generated MIDI to an audio preview where appropriate.
- Connect the frontend and test download and regeneration flows.
- Record API latency, invalid-output rate, generation failures, and basic
  output statistics.

### Stop condition

The deployed service loads the registered artifact, generates valid output,
and exposes a reproducible model and data revision.

## Scale-up gate

Do not start larger-corpus LMDClean training until all of the following are
true:

- the frozen pilot data contract is verified
- all three model families have completed controlled runs
- MLflow tracking is complete and recoverable
- the evaluation report is reproducible
- test evaluation is separated from model selection
- generated MIDI serialization is verified
- resource requirements are documented
- a model promotion decision is recorded

## Deferred audio and genre tracks

Raw MP3/audio training is a separate phase. It requires licensed or owned
recordings, a larger data contract, an explicit audio representation, and
audio-specific evaluation. The current heterogeneous pilot also lacks reliable
genre metadata, so genre-specific studies should use separate manifests and
separate MLflow experiments rather than silently relabeling the current pilot.
