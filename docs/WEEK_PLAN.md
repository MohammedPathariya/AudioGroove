# AudioGroove Seven-Day Plan

This plan is designed for an Apple MacBook Air M1 with 8 GB unified memory. Each day has one primary phase, a concrete deliverable, and a stopping condition. Do not move to the next phase while the stopping condition is false unless the failure is documented.

## Day 1: Repository, modality, and experiment foundation

### Objective

Make the repository internally consistent, decide what the system should generate, and establish the experiment contract.

### Work

- Standardize all data and output paths.
- Define missing chunk-directory constants.
- Standardize Python import and module execution conventions.
- Remove accidental `.DS_Store` changes from the intended work.
- Define a configuration object or file for dataset, model, device, seed, and output paths.
- Add a dataset manifest schema.
- Record the current prototype limitations in `STATUS.md`.
- Define the primary task: symbolic continuation, symbolic composition, audio continuation, or audio generation.
- Compare MIDI, extracted audio features, and raw MP3/audio against data availability, rights, storage, compute, and evaluation requirements.
- Decide whether the first week will deliver a symbolic baseline or an audio baseline.
- List candidate model families that fit the chosen representation.

### Deliverable

The data-preparation and model modules can be imported without path-name errors, one configuration identifies every input and output location, and `DECISIONS.md` records the initial modality and model-family choice.

### Stop condition

Focused import checks and syntax checks pass. No training is attempted yet.

## Day 2: Data audit and bounded preprocessing

### Objective

Create a small, reproducible dataset that is safe for 8 GB unified memory.

### Work

- Select approximately 250 songs from the approved LMDClean version with a fixed seed and record exact paths and hashes.
- Decide which MIDI corpus is legally and technically usable.
- Run the scanner in quarantine mode.
- Generate a manifest with parse status, duration, token count, and source identity.
- Create deterministic source-file train, validation, and test splits.
- Build bounded chunk files without merging all windows.
- Create smoke and development dataset manifests.
- Keep the pilot test split frozen before model comparison.

### Deliverable

A reproducible smoke dataset and development dataset exist, with counts and split assignments recorded. The next data target is an approximately 250-song LMDClean pilot, while the current 10-song result remains the smoke test.

### Stop condition

The same input manifest and seed reproduce the same split and chunk counts.

## Day 3: Representation and dataset loader

### Objective

Use a representation that preserves enough information for meaningful generation and evaluation. For MIDI, this means event tokens. For audio, this means a defined sample rate, segment length, feature representation, and reconstruction path.

### Work

- Split the pilot by source song before windowing. Group by artist or album when reliable metadata is available.
- Use Dask for bounded MIDI parsing and chunk preparation, with deterministic source ordering and a recorded worker or partition configuration.
- Implement or finalize event tokens for note starts, note ends or durations, time shifts, velocity, and instrument information.
- Define vocabulary special tokens and unknown-token behavior.
- Implement a streaming or bounded-chunk `Dataset`.
- Add shape, range, and round-trip tests.
- Confirm that generated event sequences can be converted back to valid MIDI.
- If audio is selected, define the audio preprocessing and reconstruction or playback path before training.

### Deliverable

The loader can train from bounded chunks, and the selected representation can be converted into an output that can be evaluated.

### Stop condition

A small sample of real files passes parse, preprocessing, reconstruction or serialization, and output validation checks. The pilot test split is frozen before model comparison.

## Day 4: Correct compact baseline

### Objective

Train a compact baseline locally on the bounded 250-song pilot using the same objective used during generation.

### Work

- Implement next-token targets.
- Add MPS selection with CPU fallback.
- Start with the smallest suitable baseline for the chosen representation. For symbolic data, use a compact unidirectional LSTM, GRU, temporal convolutional model, or small Transformer. For audio, prefer a pretrained representation or a bounded spectrogram baseline rather than raw waveform generation from scratch.
- If the symbolic baseline is selected, start with:
  - embedding 128
  - hidden size 256
  - one or two layers
  - batch size 8 to 16
  - sequence length 32 or 64
- Use gradient accumulation instead of a large physical batch.
- Disable CUDA-only assumptions and start with float32 on MPS.
- Add checkpoint resume and early stopping.
- Track the run in local MLflow under `runs/mlruns/`. Log the dataset revision, source and window counts, selection and split seeds, Git commit, model configuration, device, training budget, metrics, checkpoint, and generation artifacts.
- Keep Dask preprocessing configuration and MLflow run metadata in the reproducibility record.
- If the local device is insufficient, move the training run to Google Colab only after the local smoke test passes.

### Deliverable

The compact baseline completes a bounded pilot run and produces a checkpoint, an MLflow run, and a metrics report.

### Stop condition

Training loss, validation loss, perplexity, token accuracy, and generation latency are recorded.

## Day 5: Evaluation harness and baseline report

### Objective

Measure whether the baseline produces technically valid and statistically plausible output on the frozen pilot test split.

### Work

- Implement predictive metrics:
  - cross-entropy
  - perplexity
  - token accuracy
  - top-5 accuracy
- Implement validity metrics:
  - MIDI parse success for symbolic output
  - audio decode success for audio output
  - non-empty output rate
  - valid event or waveform rate
  - generation failure rate
- Implement musical distribution metrics appropriate to the modality:
  - symbolic: note density, pitch range, pitch-class distribution, duration distribution, time-shift distribution, repetition ratio
  - audio: loudness, spectral statistics, segment duration, clipping rate, and a defined audio-quality measure
- Compare generated outputs with held-out real files.
- Measure nearest-training-example similarity and n-gram overlap.

### Deliverable

A versioned JSON or CSV baseline report with fixed random seeds and saved sample MIDI files.

### Stop condition

The report can be regenerated from one command and clearly states dataset revision, pilot-selection seed, split seed, model, device, Dask preprocessing configuration, MLflow run ID, source-file counts, and sample count. The test split is not changed during tuning.

## Day 6: Model comparison and resource profiling

### Objective

Compare the compact LSTM, GRU, and compact Transformer on the same bounded 250-song pilot. Attention is only one candidate comparison.

### Work

- Add the selected comparison model while keeping the same representation and split. Possible comparisons include LSTM versus GRU, LSTM versus compact Transformer, or a symbolic model versus a pretrained audio-feature baseline.
- Keep the experiment controlled:
  - same dataset
  - same seed
  - same frozen source-file split
  - same Dask preprocessing configuration
  - same number of epochs or training steps
  - same evaluation sample count
- Log every comparison run and artifact to MLflow. Do not select a model from untracked terminal output.
- Record:
  - peak memory
  - samples per second
  - epoch duration
  - generation latency
  - audio quality metrics if the selected output is audio
  - validation perplexity
  - musical and validity metrics
- Stop or reduce the model if the machine begins swapping or becomes unstable.

### Deliverable

A baseline-versus-selected-model comparison table with resource and quality metrics, plus a decision about whether the pilot supports scaling to the larger corpus.

### Stop condition

The project can state whether the comparison model improved the agreed metrics, degraded them, or remains inconclusive.

## Day 7: Integration, deployment readiness, and handoff

### Objective

Make the verified pilot result reproducible and define whether the larger LMDClean run is justified.

### Work

- Connect the selected model artifact, representation configuration, and vocabulary or preprocessing assets to the backend.
- Add readiness checks for artifact compatibility.
- Run local API generation with and without a seed file.
- Validate returned MIDI or audio files according to the selected modality.
- Run frontend smoke testing.
- Update `STATUS.md`, `DECISIONS.md`, and `DEPLOYMENT.md` with actual results.
- Decide whether a larger cloud run is justified.
- Do not start the larger run unless the pilot passes the scale-up gate: reproducible preprocessing, frozen test split, stable bounded training, verified MIDI output, model comparison report, and documented resource profile.
- Keep README claims limited to verified evidence.

### Deliverable

A local end-to-end demo, evaluation report, deployment smoke-test record, and next-phase recommendation.

### Stop condition

The selected pilot result can be reproduced from the documented commands without relying on undocumented local files, and the larger-corpus decision is recorded with evidence.

## After Day 7: Larger LMDClean training gate

The larger corpus is a separate phase. It may begin only after the pilot gate passes. The larger run must pin the approved Dask preprocessing configuration, repository revision, dataset version, split policy, random seeds, model configuration, MLflow tracking configuration, checkpoint location, and evaluation procedure. The pilot test split and the larger-run final test split must remain held out from model selection.

## Laptop resource guardrails

- Prefer MPS, with explicit CPU fallback.
- Start with batch size 8 or 16.
- Use zero or one data-loader worker.
- Avoid `pin_memory=True` unless measured to help.
- Do not merge all dataset chunks into one tensor.
- Keep sequence lengths at 32 or 64 initially.
- Save checkpoints frequently.
- Watch memory pressure and stop before macOS begins heavy swapping.
- Use cloud GPU only after the local pipeline is reproducible.
- Use Google Colab for larger runs when local MPS memory or runtime is insufficient.
- Do not train raw audio generation from scratch during this week unless the data, model, and compute budget are already available.

## Metrics to report every experiment

```text
dataset_version
source_file_count
window_count
split_seed
model_name
parameter_count
device
batch_size
gradient_accumulation_steps
sequence_length
training_steps
epoch_time_seconds
peak_memory_if_available
validation_loss
perplexity
token_accuracy
top5_accuracy
output_validity_rate
generation_failure_rate
mean_generation_latency_ms
note_density_distance
pitch_distribution_distance
duration_distribution_distance
repetition_ratio
training_overlap_rate
data_modality
audio_sample_rate_if_applicable
segment_length_if_applicable
model_family
colab_runtime_and_gpu_if_applicable
```
