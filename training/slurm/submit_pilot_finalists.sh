#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"
FINALISTS_PATH="${1:-/N/scratch/mjpathar/AudioGroove/reports/pilot_v2/sweep/family_finalists.json}"
SEEDS=(20260807 20260808 20260809)

if [[ ! -f "$FINALISTS_PATH" ]]; then
    echo "family finalist manifest not found: $FINALISTS_PATH" >&2
    exit 2
fi
if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "refusing to submit finalists from a dirty repository" >&2
    exit 2
fi

python - "$FINALISTS_PATH" <<'PY' | while read -r model_family model_profile; do
import json
import sys

for finalist in json.load(open(sys.argv[1], encoding="utf-8")):
    print(finalist["model_family"], finalist["model_profile"])
PY
    for training_seed in "${SEEDS[@]}"; do
        sbatch \
            --job-name "ag-${model_family}-${model_profile}-f${training_seed: -2}" \
            --export="ALL,AG_EXPECTED_GIT_COMMIT=$REPO_COMMIT" \
            "$REPO_ROOT/training/slurm/train_pilot_model.sh" \
            "$model_family" \
            "$model_profile" \
            finalist \
            "$training_seed"
    done
done
