#!/bin/bash
#SBATCH -J ag-gru-scaling
#SBATCH -A r00284
#SBATCH -p gpu
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH -o /N/scratch/mjpathar/AudioGroove/logs/gru_scaling_%j.out
#SBATCH -e /N/scratch/mjpathar/AudioGroove/logs/gru_scaling_%j.err

set -euo pipefail

SIZE="${1:-}"
PROFILE="${2:-large}"
if [[ "$SIZE" != "500" && "$SIZE" != "1000" && "$SIZE" != "2500" && "$SIZE" != "10000" ]]; then
    echo "usage: sbatch $0 {500|1000|2500|10000} [large|larger]" >&2
    exit 2
fi
if [[ "$PROFILE" != "large" && "$PROFILE" != "larger" ]]; then
    echo "usage: sbatch $0 {500|1000|2500|10000} [large|larger]" >&2
    exit 2
fi

module purge
module load python/gpu/3.11.5

export AG_REPO=/N/u/mjpathar/BigRed200/AudioGroove
export AG_SCRATCH=/N/scratch/mjpathar/AudioGroove
source "$AG_SCRATCH/venv/bin/activate"
cd "$AG_REPO"

if [[ -n "$(git status --porcelain)" ]]; then
    echo "refusing to train from a dirty repository" >&2
    exit 2
fi

DATASET_DIR="$AG_SCRATCH/prepared/scaling_${SIZE}_train_vocab"
AUDIT_DIR="$AG_SCRATCH/prepared/scaling_${SIZE}_audit"
DATASET_REVISION="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["dataset_revision"])' "$DATASET_DIR/manifest.json")"

python -u -m src.training.pilot_comparison \
    --model-family gru \
    --profile "$PROFILE" \
    --run-phase scaling \
    --config "$AG_REPO/training/configs/pilot_experiments.json" \
    --audit-dir "$AUDIT_DIR" \
    --dataset-dir "$DATASET_DIR" \
    --dataset-revision "$DATASET_REVISION" \
    --experiment-name "AudioGroove-${SIZE}-Song-GRU-${PROFILE}-Scaling" \
    --tracking-dir "$AG_SCRATCH/mlruns/scaling-${SIZE}-gru-${PROFILE}" \
    --run-root "$AG_SCRATCH/runs/scaling_gru/${SIZE}/${PROFILE}" \
    --training-seed 20260810
