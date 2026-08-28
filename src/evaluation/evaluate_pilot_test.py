"""Evaluate one validation-selected pilot checkpoint on the held-out test split."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.data_prep.prepare_pilot_dataset import UNKNOWN_TOKEN_POLICY, VOCABULARY_POLICY
from src.models.compact_midi_models import build_compact_model, count_parameters
from src.training.chunk_stream import iter_chunk_batches
from src.training.pilot_comparison import (
    environment_metadata,
    load_experiment_config,
    load_model_checkpoint,
    run_epoch,
    select_device,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "training" / "configs" / "pilot_experiments.json"


def validate_selection(
    selection: dict[str, Any],
    source_report: dict[str, Any],
    manifest: dict[str, Any],
    vocabulary_hash: str,
) -> None:
    if selection.get("test_evaluated") is not False:
        raise ValueError("selection manifest must state test_evaluated=false")
    if source_report.get("test") is not None:
        raise ValueError("selected source report already contains test metrics")
    expected = {
        "dataset_revision": manifest["dataset_revision"],
        "model_family": source_report["model_family"],
        "model_profile": source_report["model_profile"],
        "training_seed": source_report["experiment"]["training"]["seed"],
    }
    mismatches = {
        key: (selection.get(key), value)
        for key, value in expected.items()
        if selection.get(key) != value
    }
    if mismatches:
        raise ValueError(f"selection manifest mismatch: {mismatches}")
    if source_report["dataset"]["dataset_revision"] != manifest["dataset_revision"]:
        raise ValueError("selected report dataset revision does not match prepared data")
    if source_report["vocabulary_hash"] != vocabulary_hash:
        raise ValueError("selected report vocabulary does not match prepared data")
    if selection.get("source_checkpoint") != source_report["checkpoints"]["best"]:
        raise ValueError("selection checkpoint does not match the selected source report")


def evaluate(
    *,
    selection_path: Path,
    config_path: Path,
    dataset_dir: Path,
    tracking_dir: Path,
    run_root: Path,
) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_report_path = Path(selection["source_report"])
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("vocabulary_policy") != VOCABULARY_POLICY:
        raise ValueError("prepared vocabulary must be fit on training data only")
    if manifest.get("unknown_token_policy") != UNKNOWN_TOKEN_POLICY:
        raise ValueError("prepared dataset must map unseen tokens to <UNK>")
    vocabulary_path = dataset_dir / "vocabulary.json"
    vocabulary = json.loads(vocabulary_path.read_text(encoding="utf-8"))
    vocabulary_hash = sha256_file(vocabulary_path)
    validate_selection(selection, source_report, manifest, vocabulary_hash)

    experiment = load_experiment_config(
        config_path,
        selection["model_family"],
        selection["model_profile"],
    )
    experiment = replace(
        experiment,
        training=replace(experiment.training, seed=int(selection["training_seed"])),
    )
    if asdict(experiment) != source_report["experiment"]:
        raise ValueError("current experiment configuration differs from the selected run")

    checkpoint_path = Path(selection["source_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device, device_name = select_device()
    use_amp = experiment.training.amp and device.type == "cuda"
    model = build_compact_model(
        experiment.model_family,
        len(vocabulary),
        experiment.model_parameters,
    ).to(device)
    load_model_checkpoint(
        checkpoint_path,
        model=model,
        experiment=experiment,
        dataset_revision=manifest["dataset_revision"],
        vocabulary_hash=vocabulary_hash,
        device=device,
    )

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is required for final pilot evaluation") from exc
    tracking_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment(f"{experiment.experiment_name}-Final-Test")

    run_root.mkdir(parents=True, exist_ok=True)
    gate_path = run_root / "final_test_evaluation.json"
    gate = {
        "status": "started",
        "dataset_revision": manifest["dataset_revision"],
        "model_family": experiment.model_family,
        "model_profile": experiment.model_profile,
        "training_seed": experiment.training.seed,
        "selection_manifest": str(selection_path),
        "source_checkpoint": str(checkpoint_path),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        with gate_path.open("x", encoding="utf-8") as handle:
            json.dump(gate, handle, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise RuntimeError(f"final test is already reserved or complete: {gate_path}") from exc

    job_suffix = os.environ.get("SLURM_JOB_ID", "local")
    run_dir = run_root / "final_test" / f"{time.strftime('%Y%m%d-%H%M%S')}-{job_suffix}"
    run_dir.mkdir(parents=True, exist_ok=False)
    environment = environment_metadata(device, device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    with mlflow.start_run(run_name=f"final-test-{experiment.model_family}-{experiment.model_profile}") as run:
        mlflow.log_params(
            {
                "dataset_revision": manifest["dataset_revision"],
                "vocabulary_hash": vocabulary_hash,
                "model_family": experiment.model_family,
                "model_profile": experiment.model_profile,
                "training_seed": experiment.training.seed,
                "parameter_count": count_parameters(model),
                "source_run_id": source_report["run_id"],
                "source_checkpoint": str(checkpoint_path),
                "test_oov_token_count": manifest["splits"]["test"]["oov_token_count"],
                "test_oov_token_rate": manifest["splits"]["test"]["oov_token_rate"],
            }
        )
        test_metrics = run_epoch(
            model,
            iter_chunk_batches(
                dataset_dir / "test",
                experiment.training.batch_size,
                shuffle=False,
                seed=experiment.training.seed,
                max_batches=None,
            ),
            nn.CrossEntropyLoss(),
            device,
            optimizer=None,
            scaler=scaler,
            gradient_accumulation_steps=1,
            gradient_clip_norm=experiment.training.gradient_clip_norm,
            amp=use_amp,
            phase="held-out test",
        )
        mlflow.log_metrics({f"test_{name}": value for name, value in test_metrics.items()})
        report = {
            "run_id": run.info.run_id,
            "dataset": manifest,
            "vocabulary_hash": vocabulary_hash,
            "selection": selection,
            "source_run_id": source_report["run_id"],
            "source_checkpoint": str(checkpoint_path),
            "experiment": asdict(experiment),
            "environment": environment,
            "test": test_metrics,
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if device.type == "cuda"
                else None
            ),
        }
        report_path = run_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        mlflow.log_artifact(str(selection_path), artifact_path="selection")
        mlflow.log_artifact(str(report_path), artifact_path="reports")

    gate_path.write_text(
        json.dumps(
            {
                **gate,
                "status": "completed",
                "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "report": str(report_path),
                "test_metrics": test_metrics,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--tracking-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate(
        selection_path=args.selection.resolve(),
        config_path=args.config.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        tracking_dir=args.tracking_dir.resolve(),
        run_root=args.run_root.resolve(),
    )


if __name__ == "__main__":
    main()
