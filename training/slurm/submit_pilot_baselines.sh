#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "refusing to submit baselines from a dirty repository" >&2
    exit 2
fi

for model_family in lstm gru transformer; do
    sbatch \
        --job-name "ag-${model_family}-v2" \
        --export="ALL,AG_EXPECTED_GIT_COMMIT=$REPO_COMMIT" \
        "$REPO_ROOT/training/slurm/train_pilot_model.sh" \
        "$model_family" \
        baseline \
        baseline
done
