#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

for model_family in lstm gru transformer; do
    for model_profile in small baseline large; do
        sbatch \
            "$REPO_ROOT/training/slurm/train_pilot_model.sh" \
            "$model_family" \
            "$model_profile" \
            sweep
    done
done
