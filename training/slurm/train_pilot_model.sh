#!/bin/bash

#SBATCH -J ag-pilot
#SBATCH -A r00284
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o /N/scratch/mjpathar/AudioGroove/logs/pilot_%j.out
#SBATCH -e /N/scratch/mjpathar/AudioGroove/logs/pilot_%j.err

set -euo pipefail

MODEL_FAMILY="${1:-}"
MODEL_PROFILE="${2:-baseline}"
RUN_PHASE="${3:-baseline}"
TRAINING_SEED="${4:-}"
if [[ ! "$MODEL_FAMILY" =~ ^(lstm|gru|transformer)$ ]]; then
    echo "usage: sbatch $0 {lstm|gru|transformer} [small|baseline|large] [baseline|sweep|finalist] [training-seed]" >&2
    exit 2
fi
if [[ ! "$MODEL_PROFILE" =~ ^(small|baseline|large)$ ]]; then
    echo "usage: sbatch $0 {lstm|gru|transformer} [small|baseline|large] [baseline|sweep|finalist] [training-seed]" >&2
    exit 2
fi
if [[ ! "$RUN_PHASE" =~ ^(baseline|sweep|finalist)$ ]]; then
    echo "usage: sbatch $0 {lstm|gru|transformer} [small|baseline|large] [baseline|sweep|finalist] [training-seed]" >&2
    exit 2
fi

module purge
module load python/gpu/3.11.5

export AG_HOME_ROOT=/N/u/mjpathar/BigRed200
export AG_REPO="$AG_HOME_ROOT/AudioGroove"
export AG_SCRATCH=/N/scratch/mjpathar/AudioGroove
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

source "$AG_SCRATCH/venv/bin/activate"
cd "$AG_REPO"

ACTUAL_GIT_COMMIT="$(git rev-parse HEAD)"
if [[ -n "${AG_EXPECTED_GIT_COMMIT:-}" && "$ACTUAL_GIT_COMMIT" != "$AG_EXPECTED_GIT_COMMIT" ]]; then
    echo "repository changed after submission: expected $AG_EXPECTED_GIT_COMMIT, found $ACTUAL_GIT_COMMIT" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "refusing to train from a dirty repository" >&2
    exit 2
fi

RUN_ARGUMENTS=()
if [[ -n "$TRAINING_SEED" ]]; then
    RUN_ARGUMENTS+=(--training-seed "$TRAINING_SEED")
fi

python -u -m src.training.pilot_comparison \
    --model-family "$MODEL_FAMILY" \
    --profile "$MODEL_PROFILE" \
    --run-phase "$RUN_PHASE" \
    --config "$AG_REPO/training/configs/pilot_experiments.json" \
    --audit-dir "$AG_REPO/data/audit/lmdclean_pilot_250" \
    --dataset-dir "$AG_SCRATCH/prepared/pilot_dataset_train_vocab" \
    --tracking-dir "$AG_SCRATCH/mlruns/v2/${SLURM_JOB_ID}-${MODEL_FAMILY}-${MODEL_PROFILE}" \
    --run-root "$AG_SCRATCH/runs/pilot_comparison_v2" \
    "${RUN_ARGUMENTS[@]}"
