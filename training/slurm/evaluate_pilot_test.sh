#!/bin/bash

#SBATCH -J ag-final-test
#SBATCH -A r00284
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o /N/scratch/mjpathar/AudioGroove/logs/final_test_%j.out
#SBATCH -e /N/scratch/mjpathar/AudioGroove/logs/final_test_%j.err

set -euo pipefail

SELECTION_PATH="${1:-}"
if [[ -z "$SELECTION_PATH" || ! -f "$SELECTION_PATH" ]]; then
    echo "usage: sbatch $0 /path/to/final_selection.json" >&2
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

python -u -m src.evaluation.evaluate_pilot_test \
    --selection "$SELECTION_PATH" \
    --config "$AG_REPO/training/configs/pilot_experiments.json" \
    --dataset-dir "$AG_SCRATCH/prepared/pilot_dataset_train_vocab" \
    --tracking-dir "$AG_SCRATCH/mlruns/v2/final-test-${SLURM_JOB_ID}" \
    --run-root "$AG_SCRATCH/runs/pilot_comparison_v2"
