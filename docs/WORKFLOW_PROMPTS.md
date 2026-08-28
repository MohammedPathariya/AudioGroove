# AudioGroove Workflow Prompts

These prompts are intended for focused daily work. Use one prompt at a time. Each prompt requires evidence before claiming completion.

## General coding prompt

```text
You are working in /Users/mohammedpathariya/Docs/IUB Docs/Projects/AudioGroove.

Before editing, inspect the current git status and the relevant files. Preserve unrelated user changes. The machine is an Apple MacBook Air M1 with 8 GB unified memory, so avoid loading unbounded datasets or using CUDA-only assumptions.

Restate the exact problem in one sentence. Make the smallest targeted change. Do not add speculative abstractions. Run the narrowest relevant tests, then run the broader available checks. Report changed files, commands, results, failures, and any unverified claim.
```

## Modality and model selection prompt

```text
Before changing the AudioGroove pipeline, compare three options for the stated task: symbolic MIDI, extracted audio features such as spectrograms, and raw MP3/audio generation. For each option, assess data availability and rights, storage, preprocessing complexity, model requirements, M1 feasibility, Google Colab feasibility, output quality metrics, and deployment complexity. Then compare LSTM, GRU, compact Transformer, temporal convolutional, and relevant pretrained audio or symbolic models. Recommend one first-week path and identify what remains unverified. Do not implement until the decision is recorded in docs/DECISIONS.md.
```

## Foundation prompt

```text
Implement only the AudioGroove repository foundation work from docs/WEEK_PLAN.md. Standardize data paths, define missing chunk-directory constants, make import conventions consistent, and record the modality and model-family decision without implementing the new model or representation yet. Preserve unrelated changes. Verify imports and syntax, and report every remaining blocker.
```

## Source audit prompt

```text
Implement only the source audit and bounded preprocessing work. Inspect available MIDI sources, quarantine unreadable or overlong files without destructive deletion, create a deterministic manifest, and build smoke/development chunk artifacts. Do not start model training. Record exact counts, paths, seeds, and failures.
```

## 250-song pilot prompt

```text
Implement the bounded LMDClean pilot stage for AudioGroove. Select approximately 250 songs deterministically from the available cleaned corpus and record the source-selection seed, dataset revision, exact paths, hashes, and any exclusions. Split by source song before windowing, group by artist or album when reliable metadata exists, and freeze the test split before model comparison. Reuse one preprocessing configuration and bounded chunk format. Do not start larger-corpus training. Report source counts, split counts, token and window counts, quarantine or parse failures, leakage checks, and resource limits.
```

## MIDI representation prompt

```text
Implement only the representation and bounded dataset loader for the modality selected in docs/DECISIONS.md. For symbolic data, preserve timing, duration or note-off information, velocity, and instrument information where the source supports it. For audio, define sample rate, segmenting, feature extraction, and reconstruction or playback validation. Add round-trip and shape tests. Do not start large training.
```

## Pilot baseline prompt

```text
Implement only the compact pilot baseline. Use next-token prediction so training matches autoregressive generation. Use Dask for bounded preprocessing with deterministic ordering and a recorded worker or partition configuration. Track the run in local MLflow under `runs/mlruns/`, logging the dataset revision, source and window counts, seeds, Git commit, model configuration, device, training budget, losses, perplexity, accuracy, resources, checkpoint, and generation artifacts. Add MPS selection with CPU fallback, small physical batches, gradient accumulation, checkpoint resume, and early stopping. Start with the 250-song pilot. Report device, memory behavior, training time, MLflow run ID, losses, perplexity, accuracy, and generation result.
```

## Pilot evaluation prompt

```text
Implement only the evaluation harness. Add predictive, output-validity, musical-distribution, originality, and latency metrics appropriate to the selected modality. Use fixed random seeds and a held-out source-file split. Save machine-readable reports and representative MIDI or audio outputs. Do not claim that perplexity alone measures musical quality.
```

## Model comparison prompt

```text
Implement only the controlled model comparison. Keep dataset, split, seed, Dask preprocessing configuration, training budget, and evaluation procedure fixed relative to the compact baseline. Compare the selected alternative model, which may be attention, GRU, compact Transformer, temporal convolution, or a pretrained audio-feature approach. Log each run and artifact in MLflow. Profile memory, speed, and output quality. Stop if the M1 begins swapping or the run becomes unstable. State whether the comparison model improved, degraded, or did not conclusively change the measured results.
```

## Deployment integration prompt

```text
Implement only the integration and deployment-readiness work. Connect the
verified local model artifact and representation assets to the backend, add
compatibility checks, run supported seeded and unseeded generation, validate
the returned MIDI or audio output, and run frontend smoke tests. For a
free-tier container target, use a CPU-only dependency build and record the
memory limit, live usage, cgroup peak, allocation-denial count, and OOM count.
Update docs/STATUS.md and docs/DEPLOYMENT.md with exact evidence. Do not
publish unsupported hosted or model-quality claims.
```

## Review prompt

```text
Review the current AudioGroove changes against docs/DECISIONS.md and docs/WEEK_PLAN.md. Look specifically for dataset leakage, unbounded memory use, MPS/CUDA mistakes, training/inference mismatch, invalid MIDI output, unsupported metrics, and deployment claims without evidence. Do not edit files. Report findings by severity with file and line references.
```
