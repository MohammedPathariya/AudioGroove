#!/bin/bash
#SBATCH -J ag-prepare-scaling
#SBATCH -A r00284
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -o /N/scratch/mjpathar/AudioGroove/logs/prepare_scaling_%j.out
#SBATCH -e /N/scratch/mjpathar/AudioGroove/logs/prepare_scaling_%j.err

set -euo pipefail

module purge
module load python/gpu/3.11.5

export AG_REPO=/N/u/mjpathar/BigRed200/AudioGroove
export AG_SCRATCH=/N/scratch/mjpathar/AudioGroove
source "$AG_SCRATCH/venv/bin/activate"
cd "$AG_REPO"

python -u - <<'PY'
import hashlib
import json
import shutil
from pathlib import Path

from src.data_prep.day4_preprocessing import prepare_pilot_dataset

repo = Path.cwd()
scaling_root = Path("/N/scratch/mjpathar/AudioGroove/audits/lmdclean_scaling_v1")
prepared_root = Path("/N/scratch/mjpathar/AudioGroove/prepared")

for size in (500, 1000, 2500, 10000):
    rows = [
        json.loads(line)
        for line in (scaling_root / f"manifest_{size}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != size:
        raise RuntimeError(f"{size}: expected {size} manifest rows, found {len(rows)}")

    for row in rows:
        source = repo / row["source_path"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise RuntimeError(f"{size}: source hash mismatch: {source}")

    selection_revision = hashlib.sha256(
        json.dumps(
            [
                {key: row[key] for key in ("relative_path", "sha256", "split")}
                for row in sorted(rows, key=lambda item: item["relative_path"])
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    audit_dir = prepared_root / f"scaling_{size}_audit"
    output_dir = prepared_root / f"scaling_{size}_train_vocab"
    if audit_dir.exists() or output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing scale {size} artifacts")
    audit_dir.mkdir(parents=True)
    (audit_dir / "selected_manifest.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    )
    (audit_dir / "pilot_summary.json").write_text(
        json.dumps(
            {
                "dataset_revision": selection_revision,
                "selection_seed": 20260803,
                "split_seed": 20260804,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    manifest = prepare_pilot_dataset(
        audit_dir=audit_dir,
        output_dir=output_dir,
        sequence_length=32,
        max_windows_per_chunk=256,
        dask_workers=2,
        dataset_name=f"lmdclean_scaling_{size}",
    )
    print(json.dumps({"size": size, "manifest": manifest}, indent=2, sort_keys=True), flush=True)

print("All three scaling datasets prepared")
PY
