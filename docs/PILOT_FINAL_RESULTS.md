# 250-Song Pilot Final Results

Updated: 2026-08-06

## Final protocol

The 250-song pilot used corrected dataset revision
`a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31`. The
vocabulary was fit on the 175 training songs only. Unseen validation and test
tokens were mapped to `<UNK>`.

The protocol compared the three model families, swept four profiles, confirmed
the family finalists with three seeds, selected using validation loss, and
evaluated the frozen selection on the test split exactly once.

## Final selection

| Selected model | Profile | Seeds | Mean best validation loss | Mean validation accuracy |
|---|---|---:|---:|---:|
| GRU | large | 3 | **6.8435** | 10.57% |

The representative checkpoint was the GRU-large run with training seed
`20260808`, selected as the lowest-validation-loss run within the winning
configuration.

## One-time held-out test result

| Metric | Result |
|---|---:|
| Test loss | 7.7766293511 |
| Test perplexity | 2384.2248765 |
| Test accuracy | 9.0173% |
| Test top-5 accuracy | 20.6124% |
| Test examples | 586,728 |
| Evaluation time | 72.25 seconds |
| Throughput | 8,120 examples/second |

The test loss and perplexity are materially worse than validation. This is a
measured generalization gap on a small, diverse pilot, not a training-pipeline
failure. Generated-MIDI validity passed, but musical quality, originality, and
human-listening evaluation are not yet measured.

## Reproducibility evidence

- Final test Slurm job: `7906662`, exit code `0:0`.
- Final MLflow run: `34437eaf1c234baab5c95fe66110312a`.
- Final-test archive: `$AG_SCRATCH/archive/final-test-v1`.
- The final-test archive checksum manifest completed successfully.
- Checkpoints, MLflow stores, prepared chunks, logs, reports, manifests, and
  provenance records remain on HPC scratch archives. Binary artifacts are not
  committed to Git.

## Next phase

Run deterministic 500-song, 1,000-song, and 2,500-song experiments with the
selected GRU-large configuration, followed later by the 10K dataset. Each
scale must fit vocabulary on its training songs only, map validation/test OOV
tokens to `<UNK>`, receive a new dataset revision, and preserve source-level
splits. Use validation for decisions and reserve the final test evaluation for
the selected 10K configuration.
