# Corrected Pilot Finalist Results

Updated: 2026-08-06

## Validation-only finalist selection

The three selected `large` configurations were each trained with seeds
`20260807`, `20260808`, and `20260809` on corrected dataset revision
`a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31`.
The held-out test split was evaluated exactly once after this selection was
frozen. The result is recorded in
[`PILOT_FINAL_RESULTS.md`](PILOT_FINAL_RESULTS.md).

| Family | Profile | Seeds | Mean best validation loss | Loss standard deviation | Mean validation accuracy | Mean runtime |
|---|---|---:|---:|---:|---:|---:|
| GRU | large | 3 | **6.8435** | **0.0091** | 10.57% | **13m 01s** |
| Transformer | large | 3 | 6.9044 | 0.0299 | **11.01%** | 40m 25s |
| LSTM | large | 3 | 6.9795 | 0.0367 | 10.63% | 15m 55s |

The final validation-selected candidate is GRU-large. Its mean validation loss
is lower than Transformer-large by 0.0609 and lower than LSTM-large by 0.1359.
Transformer has the highest mean token accuracy, but validation loss is the
primary selection metric and Transformer requires substantially more runtime.

The representative run is the GRU-large run with training seed `20260808`:

- Source checkpoint:
  `$AG_SCRATCH/runs/pilot_comparison_v2/gru/large/20260806-191055-7906111/checkpoints/best.pt`
- Dataset revision: `a68aee4e...`
- Test evaluated: false
- Selection rule: lowest validation loss within the winning configuration

## Decision and gate

The validation selection was frozen before test evaluation. The one-time test
evaluation completed as Slurm job `7906662` with exit code `0:0`. Do not use
that test result to change the selected family, profile, or seed. It is a final
measurement of generalization for this pilot.
