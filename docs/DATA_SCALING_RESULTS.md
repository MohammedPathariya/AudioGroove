# GRU Data-Scaling Phase

Updated: 2026-08-14

## Protocol

The selected GRU-large configuration was trained with the same representation,
sequence length, five-epoch budget, optimizer policy, and training seed across
the 500-song, 1,000-song, and 2,500-song nested datasets. The original
250-song pilot is included in each larger manifest. Cross-size overlap is
intentional; duplicate content hashes and artist-group overlap within each
dataset were rejected.

All scaling test splits remain unevaluated. Model decisions must use validation
results only until the full eligible corpus candidate is selected.

## Dataset contracts

| Dataset | Train / validation / test songs | Vocabulary | Train windows | Validation windows | Test windows | Preparation revision |
|---:|---:|---:|---:|---:|---:|---|
| 500 | 349 / 75 / 76 | 23,863 | recorded in manifest | recorded in manifest | recorded in manifest | `966f71ae...` |
| 1,000 | 700 / 150 / 150 | 28,989 | recorded in manifest | recorded in manifest | recorded in manifest | `5c2ed9d5...` |
| 2,500 | 1,750 / 375 / 375 | 35,707 | recorded in manifest | recorded in manifest | recorded in manifest | `d3e2b88f...` |
| 9,956 eligible full scale | 6,953 / 1,512 / 1,491 | 48,169 | 103,823,676 | 21,116,873 | 23,662,010 | `5ec47bf2...` |

The full eligible scale contains 9,956 songs rather than exactly 10,000. The
source corpus contains 10,277 unique song identities, but the deterministic
eligibility rules exclude the remaining candidates. Full-scale preparation
recorded validation OOV of 14,843 tokens (0.0701%) and test OOV of 18,101
tokens (0.0763%). Its vocabulary was fit on the 6,953 training songs only.

## HPC evidence

- Manifest job `7921983` completed with exit code `0:0`.
- Full-scale preparation job `7923108` produced the 9,956-song manifest with
  an empty stderr log.
- GRU scaling jobs completed successfully:
  - 500 songs: job `7908059`, 31m 48s.
  - 1,000 songs: job `7908060`, 1h 06m 23s.
  - 2,500 songs: job `7908061`, 4h 38m 45s.
- The 500/1K/2.5K runs and prepared datasets are preserved under the
  `scaling-gru-v1` HPC archive. The 9,956-song prepared dataset is staged
  separately and must be archived with its final model results.

## Final scaling decision

The 2,500-song GRU-large run is the largest recoverable trained model. Its
best checkpoint and complete run evidence were recovered locally under
`local_artifacts/gru_large_2500/`. The 9,956-song dataset was prepared but
never trained because the HPC Slurm account association was removed. No
full-scale model-selection or test-evaluation claim is made.
