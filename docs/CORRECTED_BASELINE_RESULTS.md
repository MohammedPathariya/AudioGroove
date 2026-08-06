# Corrected Pilot Baseline Results

Updated: 2026-08-06

## Classification

These are the accepted single-seed baselines for corrected dataset revision
`a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31`.
The 18,849-token vocabulary was fit on the 175 training songs only. Unseen
validation and test tokens map to `<UNK>`.

The held-out test split was not evaluated. These results permit the controlled
profile sweep but do not select a final model.

## Validation results

| Model | Slurm job | MLflow run | Parameters | Epochs | Best epoch | Validation loss | Perplexity | Accuracy | Top-5 accuracy | Trainer time | Peak GPU memory | Checkpoint |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | 7903831 | `a78943ad...` | 7,553,313 | 3 | 1 | **6.9753** | **1,069.86** | 9.70% | **23.19%** | 12m 09s | **189.92 MB** | **86.45 MB** |
| LSTM | 7903830 | `8f57d31e...` | 7,652,129 | 3 | 1 | 7.0215 | 1,120.48 | **9.79%** | 22.90% | **11m 38s** | 191.57 MB | 87.58 MB |
| Transformer | 7903832 | `8b3931b0...` | 10,732,449 | 4 | 2 | 7.1028 | 1,215.37 | 8.28% | 21.65% | 23m 02s | 261.91 MB | 122.85 MB |

GRU leads the primary validation-loss metric, while LSTM has slightly higher
top-1 accuracy and lower runtime. The validation-loss difference between GRU
and LSTM is approximately 0.0462. One seed is not enough to treat that gap as a
stable model-family advantage.

## Execution evidence

| Model | Slurm state | Exit code | Slurm elapsed | Node |
|---|---|---|---:|---|
| LSTM | Completed | `0:0` | 13m 09s | `nid0661` |
| GRU | Completed | `0:0` | 13m 41s | `nid0661` |
| Transformer | Completed | `0:0` | 23m 18s | `nid0661` |

- Training seed: `20260805`
- Repository commit: `5b7ccd84209b418e0564892c12aeb52d8c1b9f07`
- Git dirty state: false
- Vocabulary hash:
  `6a67a6b3d51d4c23a5af820175605b1c8612d178ba9f229e32ca7f3a90bec010`
- Sequence length: 32
- Source split: 175 train, 37 validation, 38 test songs
- Validation OOV rate: 1.55%
- Test OOV rate: 5.18%

## Artifact verification

Every run has a JSON report, best checkpoint, last checkpoint, environment
record, experiment configuration, and generated MIDI file. The reports agree
on dataset revision, vocabulary policy, vocabulary hash, and clean Git commit.

| Model | MIDI messages | Duration | Parse result |
|---|---:|---:|---|
| GRU | 378 | 15.322s | Valid |
| LSTM | 265 | 9.000s | Valid |
| Transformer | 309 | 13.807s | Valid |

MIDI parse success is a technical validity result. Musical quality,
originality, long-range structure, and listener preference remain unmeasured.

## Decision

Proceed to the controlled size-profile sweep. The accepted baseline profile
runs already cover three of the nine family/profile combinations. Submit only
the six missing `small` and `large` profiles, then combine `baseline` and
`sweep` reports when selecting one profile per family.

Do not evaluate the test split. Do not promote GRU from this single-seed result.
