"""Run the Day 4 compact MIDI baseline on the bounded 250-song pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import resource
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mido
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data_prep.day4_preprocessing import DEFAULT_OUTPUT_DIR, load_selected_records, prepare_pilot_dataset
from src.data_prep.midi_representation import (
    DEFAULT_MAX_TIME_SHIFT_TICKS,
    DEFAULT_VELOCITY_BINS,
    BoundedChunkDataset,
    SequentialChunkDataset,
    decode_tokens,
    encode_midi,
)
from src.models.compact_midi_lstm import CompactMidiLSTM


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = ROOT / "data" / "audit" / "lmdclean_pilot_250"
DEFAULT_TRACKING_DIR = ROOT / "runs" / "mlruns"
DEFAULT_RUN_DIR = ROOT / "runs" / "day4" / "compact_baseline"


@dataclass
class BaselineConfig:
    sequence_length: int = 32
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 0.001
    max_epochs: int = 5
    max_train_steps_per_epoch: int = 500
    max_validation_batches: int = 100
    early_stopping_patience: int = 2
    gradient_clip_norm: float = 1.0
    dask_workers: int = 2
    chunk_size: int = 256
    max_time_shift_ticks: int = DEFAULT_MAX_TIME_SHIFT_TICKS
    velocity_bins: int = DEFAULT_VELOCITY_BINS
    full_epoch: bool = False
    generation_tokens: int = 64
    seed: int = 20260805
    generation_seed: int = 20260806


def select_device() -> tuple[torch.device, str]:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS"
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    print("MPS unavailable; falling back to CPU")
    return torch.device("cpu"), "CPU"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        return True


def peak_memory_mb() -> float:
    # macOS reports bytes; Linux reports KiB.
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if os.uname().sysname == "Darwin" else value / 1024


def device_memory_mb(device: torch.device) -> float | None:
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
        return float(torch.mps.current_allocated_memory()) / (1024 * 1024)
    if device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024)
    return None


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def loader(path: Path, batch_size: int, shuffle: bool, sequential: bool = False) -> DataLoader:
    dataset = SequentialChunkDataset(path) if sequential else BoundedChunkDataset(path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False if sequential else shuffle,
        num_workers=0,
    )


def run_epoch(
    model: CompactMidiLSTM,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    accumulation_steps: int,
    max_steps: int | None,
    clip_norm: float,
    phase: str = "epoch",
) -> dict[str, float]:
    training = optimizer is not None
    phase_start = time.perf_counter()
    print(f"[{phase}] starting", flush=True)
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0
    batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)
    for batch_index, (inputs, targets) in enumerate(data_loader):
        if max_steps is not None and batch_index >= max_steps:
            break
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if training:
                (loss / accumulation_steps).backward()
                if (batch_index + 1) % accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu()) * inputs.shape[0]
        correct += int((logits.argmax(dim=-1) == targets).sum().item())
        seen += inputs.shape[0]
        batches += 1
        if batch_index == 0 or (batch_index + 1) % 1000 == 0:
            elapsed = time.perf_counter() - phase_start
            print(
                f"[{phase}] batch={batch_index + 1} examples={seen} "
                f"loss={total_loss / max(seen, 1):.5f} elapsed={elapsed:.1f}s",
                flush=True,
            )
    if training and batches and batches % accumulation_steps:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    average_loss = total_loss / max(seen, 1)
    metrics = {
        "loss": average_loss,
        "perplexity": math.exp(min(average_loss, 20.0)),
        "accuracy": correct / max(seen, 1),
        "examples": float(seen),
        "batches": float(batches),
    }
    print(
        f"[{phase}] complete batches={batches} examples={seen} "
        f"loss={metrics['loss']:.5f} perplexity={metrics['perplexity']:.3f} "
        f"accuracy={metrics['accuracy']:.5f} "
        f"elapsed={time.perf_counter() - phase_start:.1f}s",
        flush=True,
    )
    return metrics


def save_checkpoint(
    path: Path,
    model: CompactMidiLSTM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    bad_epochs: int,
    config: BaselineConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "bad_epochs": bad_epochs,
            "config": asdict(config),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: CompactMidiLSTM,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # PyTorch 1.13 exposes weights_only but cannot read this full trusted
        # optimizer checkpoint format with that restricted unpickler.
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def generate_artifact(
    model: CompactMidiLSTM,
    seed_path: Path,
    vocabulary: dict[str, int],
    output_path: Path,
    device: torch.device,
    sequence_length: int,
    token_count: int,
    ticks_per_beat: int,
    generation_seed: int,
    max_time_shift_ticks: int,
    velocity_bins: int,
) -> dict[str, Any]:
    inverse = {value: key for key, value in vocabulary.items()}
    seed = encode_midi(
        seed_path,
        max_time_shift_ticks=max_time_shift_ticks,
        velocity_bins=velocity_bins,
    )
    ids = [vocabulary[token] for token in seed.tokens if token in vocabulary]
    if len(ids) < sequence_length:
        raise ValueError(f"generation seed has {len(ids)} tokens, needs {sequence_length}")
    generated = ids[:sequence_length]
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(generation_seed)
    with torch.no_grad():
        for _ in range(token_count):
            inputs = torch.tensor([generated[-sequence_length:]], dtype=torch.long, device=device)
            probabilities = torch.softmax(model(inputs)[0], dim=-1).cpu()
            next_id = int(torch.multinomial(probabilities, 1, generator=generator).item())
            generated.append(next_id)
    generated_tokens = [inverse[index] for index in generated]
    decode_tokens(
        generated_tokens,
        output_path,
        ticks_per_beat=ticks_per_beat,
        velocity_bins=velocity_bins,
    )
    parsed = mido.MidiFile(str(output_path))
    return {
        "success": True,
        "path": display_path(output_path),
        "token_count": len(generated_tokens),
        "duration_seconds": float(parsed.length),
        "seed_path": display_path(seed_path),
    }


def train(
    config: BaselineConfig,
    resume: Path | None = None,
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    dataset_output_dir: Path = DEFAULT_OUTPUT_DIR,
    tracking_dir: Path = DEFAULT_TRACKING_DIR,
    run_root: Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    set_seed(config.seed)
    device, device_name = select_device()
    dataset_manifest = prepare_pilot_dataset(
        audit_dir=audit_dir,
        output_dir=dataset_output_dir,
        sequence_length=config.sequence_length,
        max_windows_per_chunk=config.chunk_size,
        dask_workers=config.dask_workers,
        max_time_shift_ticks=config.max_time_shift_ticks,
        velocity_bins=config.velocity_bins,
    )
    dataset_dir = dataset_output_dir
    vocab = json.loads((dataset_dir / "vocabulary.json").read_text(encoding="utf-8"))
    vocab = {token: int(index) for token, index in vocab.items()}
    model = CompactMidiLSTM(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_loader = loader(
        dataset_dir / "train", config.batch_size,
        shuffle=not config.full_epoch, sequential=config.full_epoch,
    )
    val_loader = loader(
        dataset_dir / "val", config.batch_size,
        shuffle=False, sequential=config.full_epoch,
    )
    run_dir = run_root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = tracking_dir.resolve().as_uri()
    # MLflow 3.x blocks the local file store by default. This project uses
    # that store intentionally so runs remain portable on local disks or Drive.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is required for Day 4 training") from exc
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("AudioGroove-Day4-Compact-Baseline")
    start_time = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    bad_epochs = 0
    start_epoch = 0
    global_step = 0
    with mlflow.start_run(run_name="compact-midi-lstm") as run:
        mlflow.log_params(
            {
                "dataset_revision": dataset_manifest["dataset_revision"],
                "source_file_count": dataset_manifest["source_file_count"],
                "train_source_count": dataset_manifest["splits"]["train"]["source_file_count"],
                "val_source_count": dataset_manifest["splits"]["val"]["source_file_count"],
                "test_source_count": dataset_manifest["splits"]["test"]["source_file_count"],
                "train_window_count": dataset_manifest["splits"]["train"]["window_count"],
                "val_window_count": dataset_manifest["splits"]["val"]["window_count"],
                "test_window_count": dataset_manifest["splits"]["test"]["window_count"],
                "selection_seed": 20260803,
                "split_seed": 20260804,
                "training_seed": config.seed,
                "git_commit": git_commit(),
                "git_dirty": git_dirty(),
                "device": device_name,
                "model": "CompactMidiLSTM",
                "model_config": json.dumps(model.config, sort_keys=True),
                "training_budget": json.dumps(asdict(config), sort_keys=True),
                "dask_config": json.dumps(dataset_manifest["dask"], sort_keys=True),
                "generation_seed": config.generation_seed,
                "velocity_bins": config.velocity_bins,
            }
        )
        if resume:
            restored = load_checkpoint(resume, model, optimizer, device)
            start_epoch = int(restored["epoch"])
            global_step = int(restored["global_step"])
            best_val_loss = float(restored["best_val_loss"])
            bad_epochs = int(restored["bad_epochs"])
            mlflow.log_param("resumed_from", str(resume))
        for epoch in range(start_epoch, config.max_epochs):
            train_metrics = run_epoch(
                model, train_loader, criterion, optimizer, device,
                config.gradient_accumulation_steps,
                None if config.full_epoch else config.max_train_steps_per_epoch,
                config.gradient_clip_norm,
                phase=f"epoch {epoch + 1}/{config.max_epochs} train",
            )
            global_step += int(train_metrics["batches"])
            with torch.no_grad():
                val_metrics = run_epoch(
                    model, val_loader, criterion, None, device, 1,
                    None if config.full_epoch else config.max_validation_batches,
                    config.gradient_clip_norm,
                    phase=f"epoch {epoch + 1}/{config.max_epochs} val",
                )
            row = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
            history.append(row)
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_perplexity": train_metrics["perplexity"],
                    "train_accuracy": train_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                    "val_perplexity": val_metrics["perplexity"],
                    "val_accuracy": val_metrics["accuracy"],
                },
                step=epoch + 1,
            )
            save_checkpoint(
                checkpoint_dir / "last.pt", model, optimizer, epoch + 1, global_step,
                best_val_loss, bad_epochs, config,
            )
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                bad_epochs = 0
                save_checkpoint(
                    checkpoint_dir / "best.pt", model, optimizer, epoch + 1, global_step,
                    best_val_loss, bad_epochs, config,
                )
            else:
                bad_epochs += 1
                if bad_epochs >= config.early_stopping_patience:
                    break
        best_path = checkpoint_dir / "best.pt"
        restored = load_checkpoint(best_path, model, optimizer, device)
        seed_record = next(record for record in load_selected_records(audit_dir) if record["split"] == "test")
        generation = generate_artifact(
            model,
            ROOT / seed_record["source_path"],
            vocab,
            run_dir / "generated.mid",
            device,
            config.sequence_length,
            config.generation_tokens,
            encode_midi(ROOT / seed_record["source_path"]).ticks_per_beat,
            config.generation_seed,
            config.max_time_shift_ticks,
            config.velocity_bins,
        )
        elapsed = time.perf_counter() - start_time
        resources = {
            "peak_rss_mb": peak_memory_mb(),
            "device_allocated_mb": device_memory_mb(device),
            "elapsed_seconds": elapsed,
            "device": device_name,
            "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        }
        mlflow.log_metrics({key: value for key, value in resources.items() if isinstance(value, (int, float)) and value is not None})
        mlflow.log_metric("global_step", global_step)
        mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(run_dir / "generated.mid"), artifact_path="generation")
        report = {
            "run_id": run.info.run_id,
            "dataset": dataset_manifest,
            "device": resources,
            "model": model.config,
            "training": {"config": asdict(config), "history": history, "best_val_loss": best_val_loss, "epochs_completed": len(history), "global_step": global_step},
            "checkpoint": display_path(best_path),
            "generation": generation,
        }
        (run_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(run_dir / "report.json"), artifact_path="reports")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-epochs", type=int, default=5)
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=500)
    parser.add_argument("--max-validation-batches", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--dask-workers", type=int, default=2)
    parser.add_argument("--max-time-shift-ticks", type=int, default=DEFAULT_MAX_TIME_SHIFT_TICKS)
    parser.add_argument("--velocity-bins", type=int, default=DEFAULT_VELOCITY_BINS)
    parser.add_argument("--full-epoch", action="store_true")
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--dataset-output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIR)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = BaselineConfig(
        max_epochs=args.max_epochs,
        max_train_steps_per_epoch=args.max_train_steps_per_epoch,
        max_validation_batches=args.max_validation_batches,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dask_workers=args.dask_workers,
        max_time_shift_ticks=args.max_time_shift_ticks,
        velocity_bins=args.velocity_bins,
        full_epoch=args.full_epoch,
    )
    train(
        config,
        resume=args.resume,
        audit_dir=args.audit_dir.resolve(),
        dataset_output_dir=args.dataset_output_dir.resolve(),
        tracking_dir=args.tracking_dir.resolve(),
        run_root=args.run_root.resolve(),
    )


if __name__ == "__main__":
    main()
