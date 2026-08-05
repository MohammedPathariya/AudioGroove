# HPC Pilot Training

Updated: 2026-08-05

This runbook covers the frozen 250-song comparison on Big Red 200. It assumes
the prepared dataset already exists at
`/N/scratch/mjpathar/AudioGroove/prepared/pilot_dataset` with revision
`bf670db4f3390249537a2181cbab4635a7f9123fd864e74904c066ebe843d9fc`.

## Experiment contract

The executable configuration is
`training/configs/pilot_experiments.json`. All runs use sequence length 32,
batch size 64, five full epochs, AdamW, weight decay `1e-4`, gradient clipping
at 1.0, plateau scheduling, early-stopping patience 2, CUDA AMP, training seed
`20260805`, and generation seed `20260806`.

| Family | Profile | Architecture | Learning rate | Parameters |
|---|---|---|---:|---:|
| LSTM | small | embed 128, hidden 192, 1 layer, dropout 0.1 | `1e-3` | 7,463,697 |
| LSTM | baseline | embed 128, hidden 256, 1 layer, dropout 0.1 | `1e-3` | 9,050,449 |
| LSTM | large | embed 192, hidden 384, 2 layers, dropout 0.2 | `1e-3` | 15,042,065 |
| GRU | small | embed 128, hidden 192, 1 layer, dropout 0.1 | `1e-3` | 7,401,873 |
| GRU | baseline | embed 128, hidden 256, 1 layer, dropout 0.1 | `1e-3` | 8,951,633 |
| GRU | large | embed 192, hidden 384, 2 layers, dropout 0.2 | `1e-3` | 14,524,433 |
| Transformer | small | model 192, 2 layers, 4 heads, FFN 512, dropout 0.1 | `3e-4` | 9,354,321 |
| Transformer | baseline | model 256, 2 layers, 4 heads, FFN 512, dropout 0.1 | `3e-4` | 12,595,665 |
| Transformer | large | model 256, 4 layers, 8 heads, FFN 1024, dropout 0.2 | `3e-4` | 14,700,497 |

The profiles are fixed architecture bundles, not a factorial sensitivity
study. The baseline comparison is also not parameter matched. Parameter count,
runtime, and memory must accompany quality metrics.

## Update and verify the HPC checkout

```bash
ssh mjpathar@bigred200.uits.iu.edu

export AG_HOME_ROOT=/N/u/mjpathar/BigRed200
export AG_REPO="$AG_HOME_ROOT/AudioGroove"
export AG_SCRATCH=/N/scratch/mjpathar/AudioGroove

cd "$AG_REPO"
git fetch origin
git switch main
git pull --ff-only origin main
git status --short
git rev-parse HEAD

module purge
module load python/gpu/3.11.5
source "$AG_SCRATCH/venv/bin/activate"

python -m pytest -q
python -m src.training.pilot_comparison --help
```

The Git status must be understood before submission. An untracked local smoke
script is harmless to training but will correctly cause MLflow to record
`git_dirty=true`.

## Baseline submission

Do not include `--evaluate-test` during model or profile selection.

Submit all three baseline jobs:

```bash
cd "$AG_REPO"
bash training/slurm/submit_pilot_baselines.sh
```

Or submit one model explicitly:

```bash
sbatch training/slurm/train_pilot_model.sh transformer baseline baseline
```

Each job requests one GPU from the `gpu` partition. Independent jobs can run
in parallel, but Slurm decides whether they occupy the same physical node or
different nodes.

Monitor jobs:

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
```

Logs are written under `/N/scratch/mjpathar/AudioGroove/logs`. Checkpoints,
reports, environment manifests, and generated MIDI are written under
`/N/scratch/mjpathar/AudioGroove/runs/pilot_comparison`.

## MLflow isolation

The supplied Slurm script gives every job its own file-backed MLflow store:

```text
/N/scratch/mjpathar/AudioGroove/mlruns/<job>-<family>-<profile>
```

This avoids concurrent writes to one file store. The trainer also accepts
`--tracking-uri` when a shared MLflow server is available. Isolated stores must
be preserved and compared after the baseline batch; they are not equivalent to
a verified shared tracking server.

## Sweep and finalist policy

Do not submit the nine-job sweep until all three baseline jobs have valid
reports and recoverable checkpoints. When approved:

```bash
bash training/slurm/submit_pilot_sweep.sh
```

Choose the best profile in each family using validation data only. Then rerun
each finalist with at least three training seeds using `--training-seed`.
Evaluate the frozen test split once, only after the finalists and selection
rule are fixed, by adding `--evaluate-test` to dedicated final-evaluation jobs.
