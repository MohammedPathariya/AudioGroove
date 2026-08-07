# Corrected Pilot Profile Sweep

Updated: 2026-08-06

## Classification

This comparison combines the three corrected single-seed baseline-v2 runs with
nine corrected profile-sweep runs on dataset revision
`a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31`.
The vocabulary was fit on the 175 training songs only, with unseen validation
and test tokens mapped to `<UNK>`.

The held-out test split was not evaluated. All selections use validation loss
only.

## Validation comparison

| Family | Profile | Parameters | Best epoch | Validation loss | Perplexity | Accuracy | Top-5 accuracy | Trainer time | Peak GPU memory |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GRU | large | 12,428,769 | 1 | **6.8221** | **917.88** | 10.43% | 23.61% | 11m 28s | 309.47 MB |
| Transformer | large | 12,837,281 | 3 | 6.9166 | 1,008.88 | **10.63%** | **25.20%** | 40m 14s | 310.10 MB |
| LSTM | large | 12,946,401 | 1 | 6.9213 | 1,013.62 | 10.18% | 23.75% | 12m 53s | 323.30 MB |
| GRU | larger | 17,253,537 | 1 | 6.9474 | 1,040.45 | 10.48% | 23.57% | 12m 29s | 418.67 MB |
| Transformer | larger | 17,024,929 | 3 | 6.9447 | 1,037.64 | 10.21% | 24.61% | 38m 42s | 420.51 MB |
| GRU | small | 6,236,001 | 1 | 6.9546 | 1,048.01 | 10.09% | 23.33% | 8m 44s | 159.77 MB |
| GRU | baseline | 7,553,313 | 1 | 6.9753 | 1,069.86 | 9.70% | 23.19% | 12m 09s | 189.92 MB |
| LSTM | larger | 18,173,089 | 1 | 6.9859 | 1,081.30 | 10.21% | 23.98% | 13m 18s | 437.27 MB |
| Transformer | small | 7,956,001 | 3 | 7.0785 | 1,186.14 | 8.72% | 22.83% | 24m 50s | 200.34 MB |
| LSTM | baseline | 7,652,129 | 1 | 7.0215 | 1,120.48 | 9.79% | 22.90% | 11m 38s | 191.57 MB |
| LSTM | small | 6,297,825 | 1 | 7.0494 | 1,152.17 | 9.70% | 22.85% | 14m 04s | 161.38 MB |
| Transformer | baseline | 10,732,449 | 2 | 7.1028 | 1,215.37 | 8.28% | 21.65% | 23m 02s | 261.91 MB |

The three family finalists selected by validation loss are:

| Family | Selected profile | Validation loss |
|---|---|---:|
| GRU | large | **6.8221** |
| Transformer | large | 6.9166 |
| LSTM | large | 6.9213 |

All three families selected the `large` profile. The `larger` profile did not
improve validation loss for any family, indicating that the largest tested
configuration was not useful under this five-epoch budget.

## Artifact and provenance status

- Nine sweep reports, checkpoints, generated MIDI files, and isolated MLflow
  stores were verified on HPC.
- All runs use corrected dataset revision `a68aee4e...` and vocabulary hash
  `6a67a6b3...`.
- All reports have `test: null` and successful MIDI generation.
- The sweep artifacts were archived under
  `$AG_SCRATCH/archive/profile-sweep-v1` with checksum verification.
- The three-seed finalist results are documented in
  [`FINALIST_RESULTS.md`](FINALIST_RESULTS.md).

## Next step

The validation selection is frozen. Submit exactly one held-out test
evaluation for the representative GRU-large checkpoint after the finalist
results have been committed and pulled on HPC.
