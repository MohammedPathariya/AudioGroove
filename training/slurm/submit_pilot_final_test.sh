#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SELECTION_PATH="${1:-/N/scratch/mjpathar/AudioGroove/reports/pilot_v2/finalists/final_selection.json}"

if [[ ! -f "$SELECTION_PATH" ]]; then
    echo "final selection manifest not found: $SELECTION_PATH" >&2
    exit 2
fi

sbatch \
    "$REPO_ROOT/training/slurm/evaluate_pilot_test.sh" \
    "$SELECTION_PATH"
