"""Create a validated inference-only deployment package from a recovered checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = ROOT / "local_artifacts" / "gru_small_250"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_deployment_checkpoint(artifact_dir: Path) -> dict[str, Any]:
    config_path = artifact_dir / "config" / "experiment_config.json"
    vocabulary_path = artifact_dir / "vocabulary.json"
    report_path = artifact_dir / "reports" / "report.json"
    source_checkpoint_path = artifact_dir / "checkpoints" / "best.pt"
    deploy_checkpoint_path = artifact_dir / "checkpoints" / "deploy.pt"
    manifest_path = artifact_dir / "deployment_manifest.json"

    config = json.loads(config_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary_hash = sha256_file(vocabulary_path)
    checkpoint = torch.load(source_checkpoint_path, map_location="cpu")

    if checkpoint["dataset_revision"] != config["dataset_revision"]:
        raise ValueError("checkpoint dataset revision does not match the experiment configuration")
    if checkpoint["vocabulary_hash"] != vocabulary_hash:
        raise ValueError("checkpoint vocabulary hash does not match vocabulary.json")
    if report["model_family"] != config["model_family"]:
        raise ValueError("report model family does not match the experiment configuration")
    if report["model_profile"] != config["model_profile"]:
        raise ValueError("report model profile does not match the experiment configuration")
    if report["dataset"]["dataset_revision"] != config["dataset_revision"]:
        raise ValueError("report dataset revision does not match the experiment configuration")
    if report["dataset"]["vocabulary_size"] != len(vocabulary):
        raise ValueError("report vocabulary size does not match vocabulary.json")

    deploy_checkpoint = {
        "model": checkpoint["model"],
        "dataset_revision": checkpoint["dataset_revision"],
        "vocabulary_hash": checkpoint["vocabulary_hash"],
    }
    torch.save(deploy_checkpoint, deploy_checkpoint_path)

    manifest = {
        "artifact_schema_version": 1,
        "artifact_id": artifact_dir.name,
        "checkpoint": {
            "path": "checkpoints/deploy.pt",
            "sha256": sha256_file(deploy_checkpoint_path),
        },
        "config": {
            "path": "config/experiment_config.json",
            "sha256": sha256_file(config_path),
        },
        "dataset": {
            "name": report["dataset"]["dataset_name"],
            "revision": config["dataset_revision"],
            "song_count": report["dataset"]["source_file_count"],
        },
        "model": {
            "family": config["model_family"],
            "profile": config["model_profile"],
            "parameter_count": report["parameter_count"],
        },
        "provenance": {
            "hpc_run_id": report["run_id"],
            "slurm_job_id": report["environment"]["slurm_job_id"],
            "report_sha256": sha256_file(report_path),
        },
        "vocabulary": {
            "path": "vocabulary.json",
            "sha256": vocabulary_hash,
            "size": len(vocabulary),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    manifest = export_deployment_checkpoint(args.artifact_dir)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
