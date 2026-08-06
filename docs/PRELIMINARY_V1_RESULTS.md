# Preliminary V1 Pilot Results

Updated: 2026-08-06

## Classification

These runs verify the HPC training, checkpoint, MLflow, reporting, and MIDI
generation infrastructure. They are not the final 250-song benchmark and must
not be used to select a production model.

Revision
`bf670db4f3390249537a2181cbab4635a7f9123fd864e74904c066ebe843d9fc`
built its 22,481-token vocabulary from training, validation, and test songs.
The models did not train on held-out windows, but the fitted vocabulary exposed
held-out token identities. The runs are therefore classified as
`preliminary-v1`.

The held-out test split was not evaluated.

## Validation results

| Model | Slurm job | MLflow run | Parameters | Epochs | Best epoch | Validation loss | Perplexity | Accuracy | Top-5 accuracy | Trainer time | Peak GPU memory | Checkpoint |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | 7900535 | `94fe8140...` | 8,951,633 | 3 | 1 | **6.9306** | **1,023.11** | **10.43%** | **23.51%** | 16m 03s | 222.32 MB | 102.45 MB |
| LSTM | 7900534 | `27668eb8...` | 9,050,449 | 3 | 1 | 7.0284 | 1,128.25 | 9.79% | 22.63% | **14m 41s** | 223.63 MB | 103.58 MB |
| Transformer | 7900536 | `3f2e7719...` | 12,595,665 | 5 | 3 | 7.0971 | 1,208.41 | 9.24% | 23.20% | 33m 26s | 305.11 MB | 144.18 MB |

GRU was the preliminary validation leader. That ranking is descriptive only.
It must be tested again on the corrected train-only-vocabulary revision before
any model-selection decision.

## Execution evidence

| Model | Slurm state | Exit code | Slurm elapsed | Node |
|---|---|---|---:|---|
| LSTM | Completed | `0:0` | 16m 32s | `nid0666` |
| GRU | Completed | `0:0` | 17m 55s | `nid0666` |
| Transformer | Completed | `0:0` | 34m 27s | `nid0666` |

- Training seed: `20260805`
- Repository commit: `40c462928865fede3948e358dd143d0eb9d81c1a`
- Source split: 175 train, 37 validation, 38 test songs
- Sequence length: 32
- Generated MIDI validation seed split: validation
- Archive: `/N/scratch/mjpathar/AudioGroove/archive/preliminary-v1`
- Archive size: 1.4 GB
- Archived files: 226
- Archive checksums: verified

## Artifact verification

Each isolated MLflow store reopened successfully and contained its expected
single experiment and run. The archived best and last checkpoints, JSON
reports, environment records, experiment configurations, logs, generated MIDI,
dataset manifest, vocabulary, and Slurm accounting records are preserved.

| Model | MIDI tracks | MIDI messages | Duration | Parse result |
|---|---:|---:|---:|---|
| GRU | 1 | 253 | 21.645s | Valid |
| LSTM | 1 | 309 | 4.286s | Valid |
| Transformer | 1 | 251 | 16.342s | Valid |

MIDI parse success establishes technical validity only. It does not establish
musical quality, originality, structure, or listener preference.

## What this run establishes

- Big Red 200 can train all three model families on the full pilot window set.
- The shared trainer writes recoverable checkpoints, metrics, reports, and
  generated MIDI.
- Isolated per-job MLflow file stores remain readable after archival.
- Early stopping and validation checkpoint selection execute successfully.
- The Transformer baseline costs more time, memory, and storage under this
  configuration.

## What this run does not establish

- A final winning model
- Held-out test performance
- Stability across random seeds
- Musical quality or originality
- Genre-specific generalization
- Performance on the corrected train-only vocabulary

## Corrective follow-up

Corrected revision
`a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31`
fits an 18,849-token vocabulary on the 175 training songs and maps unseen
validation and test tokens to `<UNK>`. CPU preprocessing job `7903822`
completed with exit code `0:0`. Corrected LSTM, GRU, and Transformer baseline
jobs `7903830`, `7903831`, and `7903832` have been submitted. Their reports,
not `preliminary-v1`, determine whether the profile sweep may begin.
