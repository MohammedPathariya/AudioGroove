#!/bin/bash

#SBATCH -J ag-prepare-v2
#SBATCH -A r00284
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o /N/scratch/mjpathar/AudioGroove/logs/prepare_v2_%j.out
#SBATCH -e /N/scratch/mjpathar/AudioGroove/logs/prepare_v2_%j.err

set -euo pipefail

module purge
module load python/gpu/3.11.5

export AG_HOME_ROOT=/N/u/mjpathar/BigRed200
export AG_REPO="$AG_HOME_ROOT/AudioGroove"
export AG_SCRATCH=/N/scratch/mjpathar/AudioGroove
export CORRECTED_DATASET="$AG_SCRATCH/prepared/pilot_dataset_train_vocab"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

source "$AG_SCRATCH/venv/bin/activate"
cd "$AG_REPO"

if [[ -e "$CORRECTED_DATASET" ]]; then
    echo "refusing to overwrite existing corrected dataset: $CORRECTED_DATASET" >&2
    exit 2
fi

python -u - <<'PY'
import hashlib
import json
from pathlib import Path

from src.data_prep.day4_preprocessing import prepare_pilot_dataset

repo = Path.cwd()
audit_dir = repo / "data/audit/lmdclean_pilot_250"
selected_manifest = audit_dir / "selected_manifest.jsonl"
records = [json.loads(line) for line in selected_manifest.open() if line.strip()]

for record in records:
    path = repo / record["source_path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != record["sha256"]:
        raise RuntimeError(f"source hash mismatch: {path}")

if len(records) != 250:
    raise RuntimeError(f"expected 250 source records, found {len(records)}")

output = Path("/N/scratch/mjpathar/AudioGroove/prepared/pilot_dataset_train_vocab")
manifest = prepare_pilot_dataset(
    audit_dir=audit_dir,
    output_dir=output,
    sequence_length=32,
    max_windows_per_chunk=256,
    dask_workers=2,
)

expected_revision = "a68aee4e1f3f4dc4407beae45c10eae5b08d27252233d10fe2ff793ef7010d31"
if manifest["dataset_revision"] != expected_revision:
    raise RuntimeError(f"unexpected dataset revision: {manifest['dataset_revision']}")
if manifest["vocabulary_policy"] != "train_only":
    raise RuntimeError("vocabulary was not fit on the training split only")
if manifest["unknown_token_policy"] != "map_to_unk":
    raise RuntimeError("unseen held-out tokens were not mapped to <UNK>")
if manifest["vocabulary_size"] != 18849:
    raise RuntimeError(f"unexpected vocabulary size: {manifest['vocabulary_size']}")

expected_oov = {"train": 0, "val": 7229, "test": 30427}
for split, expected_count in expected_oov.items():
    actual = manifest["splits"][split]["oov_token_count"]
    if actual != expected_count:
        raise RuntimeError(f"unexpected {split} OOV count: {actual}")

print("Corrected pilot dataset verified")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

find "$CORRECTED_DATASET" -type f | wc -l
du -sh "$CORRECTED_DATASET"
