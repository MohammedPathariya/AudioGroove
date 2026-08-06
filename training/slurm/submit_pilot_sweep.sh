#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_COMMIT="$(git -C "$REPO_ROOT" rev-parse HEAD)"

if [[ -n "$(git -C "$REPO_ROOT" status --porcelain)" ]]; then
    echo "refusing to submit sweep from a dirty repository" >&2
    exit 2
fi

for model_family in lstm gru transformer; do
    for model_profile in small large larger; do
        sbatch \
            --job-name "ag-${model_family}-${model_profile}" \
            --export="ALL,AG_EXPECTED_GIT_COMMIT=$REPO_COMMIT" \
            "$REPO_ROOT/training/slurm/train_pilot_model.sh" \
            "$model_family" \
            "$model_profile" \
            sweep
    done
done
