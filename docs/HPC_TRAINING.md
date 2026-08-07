# HPC Pilot Training

Updated: 2026-08-06

This runbook covers the leakage-corrected 250-song comparison on Big Red 200.
The prepared dataset must be stored at
`/N/scratch/mjpathar/AudioGroove/prepared/pilot_dataset_train_vocab` with
revision
`a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31`.
The earlier revision `bf670db4...` fit its vocabulary on all splits. Its three
completed runs are preliminary evidence only.

## Corrected data contract

The vocabulary is fit only on the 175 training songs. Tokens found in
validation or test but absent from that vocabulary map to `<UNK>`. A local
deterministic regeneration produced:

| Split | Songs | Windows | Tokens | OOV tokens | OOV rate |
|---|---:|---:|---:|---:|---:|
| Train | 175 | 2,757,737 | 2,763,337 | 0 | 0.00% |
| Validation | 37 | 464,678 | 465,862 | 7,229 | 1.55% |
| Test | 38 | 586,728 | 587,944 | 30,427 | 5.18% |

The corrected vocabulary contains 18,849 tokens. The test OOV rate is a
dataset limitation and must not be used to select a model or profile.

## Experiment contract

The executable configuration is
`training/configs/pilot_experiments.json`. All profiles use sequence length 32,
batch size 64, five full epochs, AdamW, weight decay `1e-4`, gradient clipping
at 1.0, plateau scheduling, early-stopping patience 2, CUDA AMP, training seed
`20260805`, and generation seed `20260806`.

| Family | Profile | Architecture | Learning rate | Parameters |
|---|---|---|---:|---:|
| LSTM | small | embed 128, hidden 192, 1 layer, dropout 0.1 | `1e-3` | 6,297,825 |
| LSTM | baseline | embed 128, hidden 256, 1 layer, dropout 0.1 | `1e-3` | 7,652,129 |
| LSTM | large | embed 192, hidden 384, 2 layers, dropout 0.2 | `1e-3` | 12,946,401 |
| LSTM | larger | embed 256, hidden 512, 2 layers, dropout 0.2 | `1e-3` | 18,173,089 |
| GRU | small | embed 128, hidden 192, 1 layer, dropout 0.1 | `1e-3` | 6,236,001 |
| GRU | baseline | embed 128, hidden 256, 1 layer, dropout 0.1 | `1e-3` | 7,553,313 |
| GRU | large | embed 192, hidden 384, 2 layers, dropout 0.2 | `1e-3` | 12,428,769 |
| GRU | larger | embed 256, hidden 512, 2 layers, dropout 0.2 | `1e-3` | 17,253,537 |
| Transformer | small | model 192, 2 layers, 4 heads, FFN 512, dropout 0.1 | `3e-4` | 7,956,001 |
| Transformer | baseline | model 256, 2 layers, 4 heads, FFN 512, dropout 0.1 | `3e-4` | 10,732,449 |
| Transformer | large | model 256, 4 layers, 8 heads, FFN 1024, dropout 0.2 | `3e-4` | 12,837,281 |
| Transformer | larger | model 320, 4 layers, 8 heads, FFN 1280, dropout 0.2 | `3e-4` | 17,024,929 |

These are architecture profiles, not a factorial sensitivity study. The
baseline models are not parameter matched. The larger models are roughly
parameter matched at 17.0 to 18.2 million parameters. Report quality beside
parameter count, runtime, and memory.

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
python -m src.evaluation.pilot_results --help
```

Do not submit from an unexplained dirty checkout. MLflow records the commit and
dirty state for every run.

## Preserve preliminary-v1

Copy the completed jobs into a clearly labelled archive. This does not delete
or relocate the originals.

```bash
export PRELIMINARY_ARCHIVE="$AG_SCRATCH/archive/preliminary-v1"
test ! -e "$PRELIMINARY_ARCHIVE"
mkdir -p "$PRELIMINARY_ARCHIVE"/{runs,mlruns,logs}

cp -a "$AG_SCRATCH/runs/pilot_comparison" "$PRELIMINARY_ARCHIVE/runs/"

cp -a \
  "$AG_SCRATCH/mlruns/7900534-lstm-baseline" \
  "$AG_SCRATCH/mlruns/7900535-gru-baseline" \
  "$AG_SCRATCH/mlruns/7900536-transformer-baseline" \
  "$PRELIMINARY_ARCHIVE/mlruns/"

cp -a "$AG_SCRATCH/logs/pilot_7900534.out" "$PRELIMINARY_ARCHIVE/logs/"
cp -a "$AG_SCRATCH/logs/pilot_7900534.err" "$PRELIMINARY_ARCHIVE/logs/"
cp -a "$AG_SCRATCH/logs/pilot_7900535.out" "$PRELIMINARY_ARCHIVE/logs/"
cp -a "$AG_SCRATCH/logs/pilot_7900535.err" "$PRELIMINARY_ARCHIVE/logs/"
cp -a "$AG_SCRATCH/logs/pilot_7900536.out" "$PRELIMINARY_ARCHIVE/logs/"
cp -a "$AG_SCRATCH/logs/pilot_7900536.err" "$PRELIMINARY_ARCHIVE/logs/"

cd "$PRELIMINARY_ARCHIVE"
find . -type f ! -name checksums.sha256 -print0 | sort -z | xargs -0 sha256sum > checksums.sha256
sha256sum -c checksums.sha256
```

Keep this archive out of Git. It contains large checkpoints and MLflow
artifacts.

## Regenerate the corrected dataset

Use the new output directory so the preliminary chunks remain recoverable.
Preprocessing runs as a CPU Slurm job rather than on a shared login node.

```bash
cd "$AG_REPO"
sbatch training/slurm/prepare_pilot_dataset.sh
```

Monitor the submitted job and inspect both logs after it leaves the queue:

```bash
squeue -j JOB_ID
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
cat "$AG_SCRATCH/logs/prepare_v2_JOB_ID.out"
cat "$AG_SCRATCH/logs/prepare_v2_JOB_ID.err"
```

The job verifies all 250 source hashes before writing, refuses to overwrite an
existing output directory, and validates the revision, vocabulary policy,
vocabulary size, and OOV counts before reporting success.

## Corrected baseline submission

```bash
cd "$AG_REPO"
bash training/slurm/submit_pilot_baselines.sh
```

Each job requests one GPU. Slurm decides whether independent jobs run on the
same physical node or different nodes.

```bash
squeue -u "$USER"
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,AllocTRES,MaxRSS
```

Logs are written under `$AG_SCRATCH/logs`. Run artifacts are written under
`$AG_SCRATCH/runs/pilot_comparison_v2`. Each job receives an isolated MLflow
store under `$AG_SCRATCH/mlruns/v2/<job>-<family>-<profile>`.

After all three jobs complete successfully, create the corrected comparison:

```bash
python -m src.evaluation.pilot_results \
  --run-root "$AG_SCRATCH/runs/pilot_comparison_v2" \
  --dataset-revision a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31 \
  --phases baseline \
  --output-dir "$AG_SCRATCH/reports/pilot_v2/baselines"
```

Verify that all three reports have the corrected revision, successful
generation, recoverable best checkpoints, and `test: null`.

## Preserve baseline-v2

Before updating the repository for the profile sweep, preserve the accepted
baseline artifacts under `$AG_SCRATCH/archive/baseline-v2`. The completed
archive contains the three run directories, three isolated MLflow stores,
Slurm logs, comparison reports, corrected dataset manifest, baseline
experiment configuration, and training commit.

The archive was created before the repository advanced from `5b7ccd8` to
`cd887a1`. It is 1.1 GB with 243 files, and every entry in its
`checksums.sha256` manifest passed verification. Keep this archive unchanged
and outside Git.

## Profile sweep

The three corrected baseline runs already cover the `baseline` profile for each
family. Do not repeat them. Submit only the nine missing `small`, `large`, and
`larger` profiles after the corrected baselines pass the checks above.

```bash
bash training/slurm/submit_pilot_sweep.sh
```

Select one profile per family using validation loss only:

```bash
python -m src.evaluation.pilot_results \
  --run-root "$AG_SCRATCH/runs/pilot_comparison_v2" \
  --dataset-revision a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31 \
  --phases baseline sweep \
  --output-dir "$AG_SCRATCH/reports/pilot_v2/sweep" \
  --select-family-finalists
```

The completed profile sweep is documented in
[`PILOT_SWEEP_RESULTS.md`](PILOT_SWEEP_RESULTS.md). It selected the `large`
profile for each family using validation loss. The nine sweep artifacts are
preserved under `$AG_SCRATCH/archive/profile-sweep-v1` with verified checksums.

## Three-seed finalists

Run each selected family/profile with the frozen seeds `20260807`, `20260808`,
and `20260809`:

```bash
bash training/slurm/submit_pilot_finalists.sh \
  "$AG_SCRATCH/reports/pilot_v2/sweep/family_finalists.json"
```

Aggregate the finalist results and freeze the winner. The primary selection
metric is mean best validation loss. Mean runtime is only a deterministic
tie-breaker. The representative seed is the lowest-validation-loss run inside
the winning configuration.

```bash
python -m src.evaluation.pilot_results \
  --run-root "$AG_SCRATCH/runs/pilot_comparison_v2" \
  --dataset-revision a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31 \
  --phases finalist \
  --output-dir "$AG_SCRATCH/reports/pilot_v2/finalists" \
  --select-winner \
  --required-seeds 3
```

Inspect and preserve `final_selection.json`. It authorizes one family,
profile, dataset revision, and seed for final evaluation.

The completed three-seed finalist comparison selected GRU-large with mean best
validation loss `6.8435`. The representative seed is `20260808`. The detailed
validation evidence is in [`FINALIST_RESULTS.md`](FINALIST_RESULTS.md).

## One-time held-out test evaluation

Submit exactly one final test job:

```bash
bash training/slurm/submit_pilot_final_test.sh \
  "$AG_SCRATCH/reports/pilot_v2/finalists/final_selection.json"
```

The job loads the exact representative checkpoint named in the frozen
selection manifest and evaluates the test split without retraining. The
evaluator atomically creates
`$AG_SCRATCH/runs/pilot_comparison_v2/final_test_evaluation.json`. Any second
test submission fails while that file exists.

If the job fails, inspect its Slurm logs and the gate file. Do not delete the
gate and retry until the failure is understood and the recovery decision is
documented. Test metrics must never affect profile, seed, epoch, or
hyperparameter selection.
