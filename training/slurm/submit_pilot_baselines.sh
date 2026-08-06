#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for model_family in lstm gru transformer; do
    sbatch \
        --job-name "ag-${model_family}-v2" \
        "$REPO_ROOT/training/slurm/train_pilot_model.sh" \
        "$model_family" \
        baseline \
        baseline
done
