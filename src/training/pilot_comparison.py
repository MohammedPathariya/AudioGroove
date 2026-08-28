"""Train one compact model family under the frozen 250-song pilot contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import resource
import shutil
import subprocess
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterator

import mido
import torch
from torch import nn

from src.data_prep.prepare_pilot_dataset import (
    DEFAULT_AUDIT_DIR,
    DEFAULT_OUTPUT_DIR,
    UNKNOWN_TOKEN_POLICY,
    VOCABULARY_POLICY,
    load_selected_records,
)
from src.data_prep.midi_representation import decode_tokens, encode_midi
from src.models.compact_midi_models import MODEL_FAMILIES, build_compact_model, count_parameters
from src.training.chunk_stream import iter_chunk_batches


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "training" / "configs" / "pilot_experiments.json"
DEFAULT_TRACKING_DIR = ROOT / "runs" / "mlruns"
DEFAULT_RUN_ROOT = ROOT / "runs" / "pilot_comparison"


@dataclass(frozen=True)
class TrainingConfig:
    sequence_length: int
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    max_epochs: int
    max_train_batches: int | None
    max_validation_batches: int | None
    early_stopping_patience: int
    gradient_clip_norm: float
    scheduler_factor: float
    scheduler_patience: int
    amp: bool
    seed: int


@dataclass(frozen=True)
class GenerationConfig:
    token_count: int
    temperature: float
    top_k: int | None
    seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    dataset_revision: str
    experiment_name: str
    model_family: str
    model_profile: str
    model_parameters: dict[str, Any]
    training: TrainingConfig
    generation: GenerationConfig


def load_experiment_config(
    path: Path,
    model_family: str,
    model_profile: str = "baseline",
) -> ExperimentConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    family = model_family.lower()
    if family not in MODEL_FAMILIES:
        raise ValueError(f"unsupported model family {model_family!r}")
    try:
        model_payload = payload["models"][family][model_profile]
        training_payload = dict(payload["training"])
        training_payload["learning_rate"] = model_payload["learning_rate"]
        config = ExperimentConfig(
            dataset_revision=str(payload["dataset_revision"]),
            experiment_name=str(payload["experiment_name"]),
            model_family=family,
            model_profile=model_profile,
            model_parameters=dict(model_payload["architecture"]),
            training=TrainingConfig(**training_payload),
            generation=GenerationConfig(**payload["generation"]),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid pilot configuration in {path}: {exc}") from exc
    validate_experiment_config(config)
    return config


def validate_experiment_config(config: ExperimentConfig) -> None:
    training = config.training
    generation = config.generation
    positive_integers = {
        "sequence_length": training.sequence_length,
        "batch_size": training.batch_size,
        "gradient_accumulation_steps": training.gradient_accumulation_steps,
        "max_epochs": training.max_epochs,
        "early_stopping_patience": training.early_stopping_patience,
        "scheduler_patience": training.scheduler_patience,
        "generation token_count": generation.token_count,
    }
    for name, value in positive_integers.items():
        if value < 1:
            raise ValueError(f"{name} must be positive")
    for name, value in {
        "max_train_batches": training.max_train_batches,
        "max_validation_batches": training.max_validation_batches,
    }.items():
        if value is not None and value < 1:
            raise ValueError(f"{name} must be positive or null")
    if training.learning_rate <= 0 or training.weight_decay < 0:
        raise ValueError("learning_rate must be positive and weight_decay non-negative")
    if training.gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive")
    if not 0 < training.scheduler_factor < 1:
        raise ValueError("scheduler_factor must be between zero and one")
    if generation.temperature <= 0:
        raise ValueError("generation temperature must be positive")
    if generation.top_k is not None and generation.top_k < 1:
        raise ValueError("generation top_k must be positive or null")
    transformer_limit = config.model_parameters.get("max_sequence_length")
    if transformer_limit is not None and training.sequence_length > int(transformer_limit):
        raise ValueError("training sequence length exceeds the Transformer positional limit")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> tuple[torch.device, str]:
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS"
    return torch.device("cpu"), "CPU"


def git_value(*arguments: str) -> str:
    try:
        return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def git_dirty() -> bool:
    return bool(git_value("status", "--porcelain"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if platform.system() == "Darwin" else value / 1024


def environment_metadata(device: torch.device, device_name: str) -> dict[str, Any]:
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": device_name,
        "gpu_name": gpu_name,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_node": os.environ.get("SLURMD_NODENAME") or os.environ.get("HOSTNAME"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_dirty": git_dirty(),
    }


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    clip_norm: float,
    gradient_multiplier: float = 1.0,
) -> None:
    scaler.unscale_(optimizer)
    if gradient_multiplier != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(gradient_multiplier)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def run_epoch(
    model: nn.Module,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float,
    amp: bool,
    phase: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if training:
        optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0
    examples = 0
    batch_count = 0
    pending_gradients = 0
    print(f"[{phase}] starting", flush=True)

    for inputs, targets in batches:
        inputs = inputs.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp and device.type == "cuda"
            else nullcontext()
        )
        with torch.set_grad_enabled(training), autocast:
            logits = model(inputs)
            loss = criterion(logits, targets)
        if training:
            scaler.scale(loss / gradient_accumulation_steps).backward()
            pending_gradients += 1
            if pending_gradients == gradient_accumulation_steps:
                _optimizer_step(model, optimizer, scaler, gradient_clip_norm)
                pending_gradients = 0

        batch_examples = int(inputs.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_examples
        predictions = logits.detach().topk(min(5, logits.shape[-1]), dim=-1).indices
        top1_correct += int((predictions[:, 0] == targets).sum().item())
        top5_correct += int((predictions == targets[:, None]).any(dim=1).sum().item())
        examples += batch_examples
        batch_count += 1
        if batch_count == 1 or batch_count % 1000 == 0:
            print(
                f"[{phase}] batch={batch_count} examples={examples} "
                f"loss={total_loss / examples:.5f} elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

    if training and pending_gradients:
        _optimizer_step(
            model,
            optimizer,
            scaler,
            gradient_clip_norm,
            gradient_accumulation_steps / pending_gradients,
        )
    if not batch_count:
        raise RuntimeError(f"{phase} produced no batches")
    elapsed = time.perf_counter() - started
    average_loss = total_loss / examples
    metrics = {
        "loss": average_loss,
        "perplexity": math.exp(min(average_loss, 20.0)),
        "accuracy": top1_correct / examples,
        "top5_accuracy": top5_correct / examples,
        "examples": float(examples),
        "batches": float(batch_count),
        "elapsed_seconds": elapsed,
        "examples_per_second": examples / elapsed,
    }
    print(
        f"[{phase}] complete batches={batch_count} examples={examples} "
        f"loss={metrics['loss']:.5f} perplexity={metrics['perplexity']:.3f} "
        f"accuracy={metrics['accuracy']:.5f} top5={metrics['top5_accuracy']:.5f} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return metrics


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.cuda.amp.GradScaler,
    experiment: ExperimentConfig,
    epoch: int,
    global_step: int,
    best_val_loss: float,
    bad_epochs: int,
    dataset_revision: str,
    vocabulary_hash: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "experiment": asdict(experiment),
            "epoch": epoch,
            "global_step": global_step,
            "best_val_loss": best_val_loss,
            "bad_epochs": bad_epochs,
            "dataset_revision": dataset_revision,
            "vocabulary_hash": vocabulary_hash,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    scaler: torch.cuda.amp.GradScaler,
    experiment: ExperimentConfig,
    dataset_revision: str,
    vocabulary_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = load_model_checkpoint(
        path,
        model=model,
        experiment=experiment,
        dataset_revision=dataset_revision,
        vocabulary_hash=vocabulary_hash,
        device=device,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def load_model_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    experiment: ExperimentConfig,
    dataset_revision: str,
    vocabulary_hash: str,
    device: torch.device,
) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(path, map_location=device)
    if checkpoint.get("dataset_revision") != dataset_revision:
        raise ValueError("checkpoint dataset revision does not match the prepared pilot")
    if checkpoint.get("vocabulary_hash") != vocabulary_hash:
        raise ValueError("checkpoint vocabulary does not match the prepared pilot")
    saved = checkpoint.get("experiment", {})
    if saved.get("model_family") != experiment.model_family:
        raise ValueError("checkpoint model family does not match the requested model")
    if saved.get("model_parameters") != experiment.model_parameters:
        raise ValueError("checkpoint architecture does not match the requested model")
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def generate_midi(
    model: nn.Module,
    seed_path: Path,
    vocabulary: dict[str, int],
    output_path: Path,
    device: torch.device,
    sequence_length: int,
    config: GenerationConfig,
) -> dict[str, Any]:
    encoded = encode_midi(seed_path)
    seed_ids = [vocabulary.get(token, vocabulary["<UNK>"]) for token in encoded.tokens]
    if len(seed_ids) < sequence_length:
        raise ValueError(f"generation seed has {len(seed_ids)} tokens, needs {sequence_length}")
    generated = seed_ids[:sequence_length]
    inverse = {index: token for token, index in vocabulary.items()}
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        for _ in range(config.token_count):
            context = torch.tensor(
                [generated[-sequence_length:]], dtype=torch.long, device=device
            )
            logits = model(context)[0].float().cpu() / config.temperature
            if config.top_k is not None:
                top_k = min(config.top_k, logits.shape[0])
                values, indices = torch.topk(logits, top_k)
                sampled = torch.multinomial(
                    torch.softmax(values, dim=-1), 1, generator=generator
                )
                next_id = int(indices[sampled].item())
            else:
                next_id = int(
                    torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).item()
                )
            generated.append(next_id)
    generation_seconds = time.perf_counter() - started
    decode_tokens(
        [inverse[index] for index in generated],
        output_path,
        ticks_per_beat=encoded.ticks_per_beat,
    )
    parsed = mido.MidiFile(str(output_path))
    return {
        "success": True,
        "path": str(output_path),
        "seed_path": str(seed_path),
        "seed_split": "val",
        "generated_token_count": config.token_count,
        "total_token_count": len(generated),
        "duration_seconds": float(parsed.length),
        "generation_seconds": generation_seconds,
        "tokens_per_second": config.token_count / generation_seconds,
    }


def train(
    experiment: ExperimentConfig,
    *,
    audit_dir: Path,
    dataset_dir: Path,
    tracking_dir: Path,
    run_root: Path,
    tracking_uri: str | None = None,
    run_phase: str = "baseline",
    resume: Path | None = None,
) -> dict[str, Any]:
    set_seed(experiment.training.seed)
    manifest_path = dataset_dir / "manifest.json"
    vocabulary_path = dataset_dir / "vocabulary.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("dataset_revision") != experiment.dataset_revision:
        raise ValueError(
            f"prepared dataset revision {manifest.get('dataset_revision')} does not match "
            f"configured revision {experiment.dataset_revision}"
        )
    if manifest.get("sequence_length") != experiment.training.sequence_length:
        raise ValueError("prepared sequence length does not match the experiment configuration")
    if manifest.get("vocabulary_policy") != VOCABULARY_POLICY:
        raise ValueError("prepared vocabulary must be fit on the training split only")
    if manifest.get("unknown_token_policy") != UNKNOWN_TOKEN_POLICY:
        raise ValueError("prepared dataset must map unseen tokens to <UNK>")
    vocabulary = {
        token: int(index)
        for token, index in json.loads(vocabulary_path.read_text(encoding="utf-8")).items()
    }
    vocabulary_hash = sha256_file(vocabulary_path)
    device, device_name = select_device()
    use_amp = experiment.training.amp and device.type == "cuda"
    model = build_compact_model(
        experiment.model_family,
        len(vocabulary),
        experiment.model_parameters,
    ).to(device)
    parameter_count = count_parameters(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=experiment.training.learning_rate,
        weight_decay=experiment.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=experiment.training.scheduler_factor,
        patience=experiment.training.scheduler_patience,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    criterion = nn.CrossEntropyLoss()

    job_suffix = os.environ.get("SLURM_JOB_ID", "local")
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = (
        run_root
        / experiment.model_family
        / experiment.model_profile
        / f"{timestamp}-{job_suffix}"
    )
    checkpoint_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=False)
    if tracking_uri is None:
        tracking_dir.mkdir(parents=True, exist_ok=True)
    environment = environment_metadata(device, device_name)
    config_artifact = run_dir / "experiment_config.json"
    environment_artifact = run_dir / "environment.json"
    config_artifact.write_text(
        json.dumps(asdict(experiment), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment_artifact.write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    bad_epochs = 0
    best_path = checkpoint_dir / "best.pt"
    last_path = checkpoint_dir / "last.pt"
    if resume is not None:
        restored = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            experiment=experiment,
            dataset_revision=manifest["dataset_revision"],
            vocabulary_hash=vocabulary_hash,
            device=device,
        )
        start_epoch = int(restored["epoch"])
        global_step = int(restored["global_step"])
        best_val_loss = float(restored["best_val_loss"])
        bad_epochs = int(restored["bad_epochs"])
        previous_best = resume.parent / "best.pt"
        if previous_best.is_file():
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous_best, best_path)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow is required for pilot comparison training") from exc
    mlflow.set_tracking_uri(tracking_uri or tracking_dir.resolve().as_uri())
    mlflow.set_experiment(experiment.experiment_name)
    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()

    with mlflow.start_run(
        run_name=f"pilot-{experiment.model_family}-{experiment.model_profile}"
    ) as mlflow_run:
        mlflow.set_tags(
            {
                "model_family": experiment.model_family,
                "model_profile": experiment.model_profile,
                "phase": run_phase,
                "slurm_job_id": environment.get("slurm_job_id") or "local",
                "git_dirty": str(environment["git_dirty"]).lower(),
            }
        )
        mlflow.log_params(
            {
                "model_family": experiment.model_family,
                "model_profile": experiment.model_profile,
                "model_parameters": json.dumps(experiment.model_parameters, sort_keys=True),
                "parameter_count": parameter_count,
                "dataset_revision": manifest["dataset_revision"],
                "source_dataset_revision": manifest["source_dataset_revision"],
                "vocabulary_size": len(vocabulary),
                "vocabulary_hash": vocabulary_hash,
                "vocabulary_policy": manifest["vocabulary_policy"],
                "unknown_token_policy": manifest["unknown_token_policy"],
                "val_oov_token_count": manifest["splits"]["val"]["oov_token_count"],
                "val_oov_token_rate": manifest["splits"]["val"]["oov_token_rate"],
                "test_oov_token_count": manifest["splits"]["test"]["oov_token_count"],
                "test_oov_token_rate": manifest["splits"]["test"]["oov_token_rate"],
                "sequence_length": experiment.training.sequence_length,
                "batch_size": experiment.training.batch_size,
                "effective_batch_size": (
                    experiment.training.batch_size
                    * experiment.training.gradient_accumulation_steps
                ),
                "learning_rate": experiment.training.learning_rate,
                "weight_decay": experiment.training.weight_decay,
                "max_epochs": experiment.training.max_epochs,
                "max_train_batches": experiment.training.max_train_batches or "full",
                "max_validation_batches": experiment.training.max_validation_batches or "full",
                "training_seed": experiment.training.seed,
                "generation_seed": experiment.generation.seed,
                "amp_requested": experiment.training.amp,
                "amp_enabled": use_amp,
                "git_commit": environment["git_commit"],
                "device": device_name,
                "gpu_name": environment.get("gpu_name") or "none",
            }
        )
        mlflow.log_artifact(str(config_artifact), artifact_path="configuration")
        mlflow.log_artifact(str(environment_artifact), artifact_path="configuration")

        for epoch in range(start_epoch, experiment.training.max_epochs):
            train_batches = iter_chunk_batches(
                dataset_dir / "train",
                experiment.training.batch_size,
                shuffle=True,
                seed=experiment.training.seed + epoch,
                max_batches=experiment.training.max_train_batches,
            )
            train_metrics = run_epoch(
                model,
                train_batches,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                gradient_accumulation_steps=experiment.training.gradient_accumulation_steps,
                gradient_clip_norm=experiment.training.gradient_clip_norm,
                amp=use_amp,
                phase=f"epoch {epoch + 1}/{experiment.training.max_epochs} train",
            )
            global_step += int(train_metrics["batches"])
            val_batches = iter_chunk_batches(
                dataset_dir / "val",
                experiment.training.batch_size,
                shuffle=False,
                seed=experiment.training.seed,
                max_batches=experiment.training.max_validation_batches,
            )
            val_metrics = run_epoch(
                model,
                val_batches,
                criterion,
                device,
                optimizer=None,
                scaler=scaler,
                gradient_accumulation_steps=1,
                gradient_clip_norm=experiment.training.gradient_clip_norm,
                amp=use_amp,
                phase=f"epoch {epoch + 1}/{experiment.training.max_epochs} val",
            )
            scheduler.step(val_metrics["loss"])
            improved = val_metrics["loss"] < best_val_loss
            if improved:
                best_val_loss = val_metrics["loss"]
                bad_epochs = 0
            else:
                bad_epochs += 1
            row = {
                "epoch": epoch + 1,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train": train_metrics,
                "val": val_metrics,
            }
            history.append(row)
            metric_payload = {
                **{f"train_{name}": value for name, value in train_metrics.items()},
                **{f"val_{name}": value for name, value in val_metrics.items()},
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            mlflow.log_metrics(metric_payload, step=epoch + 1)
            checkpoint_arguments = {
                "model": model,
                "optimizer": optimizer,
                "scheduler": scheduler,
                "scaler": scaler,
                "experiment": experiment,
                "epoch": epoch + 1,
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "bad_epochs": bad_epochs,
                "dataset_revision": manifest["dataset_revision"],
                "vocabulary_hash": vocabulary_hash,
            }
            save_checkpoint(last_path, **checkpoint_arguments)
            if improved:
                save_checkpoint(best_path, **checkpoint_arguments)
            if bad_epochs >= experiment.training.early_stopping_patience:
                print("early stopping triggered", flush=True)
                break

        if not best_path.is_file():
            raise RuntimeError("training completed without a best checkpoint")
        load_checkpoint(
            best_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            experiment=experiment,
            dataset_revision=manifest["dataset_revision"],
            vocabulary_hash=vocabulary_hash,
            device=device,
        )
        validation_seed = next(
            record for record in load_selected_records(audit_dir) if record["split"] == "val"
        )
        generation = generate_midi(
            model,
            ROOT / validation_seed["source_path"],
            vocabulary,
            run_dir / "generated.mid",
            device,
            experiment.training.sequence_length,
            experiment.generation,
        )
        elapsed = time.perf_counter() - training_started
        resources = {
            "elapsed_seconds": elapsed,
            "peak_rss_mb": peak_rss_mb(),
            "peak_gpu_memory_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if device.type == "cuda"
                else None
            ),
            "checkpoint_size_mb": best_path.stat().st_size / (1024 * 1024),
        }
        mlflow.log_metrics(
            {name: value for name, value in resources.items() if value is not None}
        )
        mlflow.log_metric("global_step", global_step)
        report = {
            "run_id": mlflow_run.info.run_id,
            "run_phase": run_phase,
            "model_family": experiment.model_family,
            "model_profile": experiment.model_profile,
            "model": model.config,
            "parameter_count": parameter_count,
            "dataset": manifest,
            "vocabulary_hash": vocabulary_hash,
            "experiment": asdict(experiment),
            "environment": environment,
            "training": {
                "epochs_completed": len(history),
                "global_step": global_step,
                "best_val_loss": best_val_loss,
                "history": history,
            },
            "test": None,
            "generation": generation,
            "resources": resources,
            "checkpoints": {"best": str(best_path), "last": str(last_path)},
        }
        report_path = run_dir / "report.json"
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        mlflow.log_artifact(str(best_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(last_path), artifact_path="checkpoints")
        mlflow.log_artifact(str(run_dir / "generated.mid"), artifact_path="generation")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, required=True)
    parser.add_argument(
        "--profile",
        choices=("small", "baseline", "large", "larger"),
        default="baseline",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tracking-dir", type=Path, default=DEFAULT_TRACKING_DIR)
    parser.add_argument("--tracking-uri")
    parser.add_argument(
        "--run-phase",
        choices=("baseline", "sweep", "finalist", "scaling"),
        default="baseline",
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation-steps", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--training-seed", type=int)
    parser.add_argument("--generation-seed", type=int)
    parser.add_argument("--generation-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--experiment-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = load_experiment_config(
        args.config.resolve(), args.model_family, args.profile
    )
    overrides: dict[str, Any] = {}
    for argument in (
        "max_epochs",
        "max_train_batches",
        "max_validation_batches",
        "batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
    ):
        value = getattr(args, argument)
        if value is not None:
            overrides[argument] = value
    if args.disable_amp:
        overrides["amp"] = False
    if args.training_seed is not None:
        overrides["seed"] = args.training_seed
    if overrides:
        experiment = replace(experiment, training=replace(experiment.training, **overrides))
    if args.dataset_revision is not None:
        experiment = replace(experiment, dataset_revision=args.dataset_revision)
    if args.experiment_name is not None:
        experiment = replace(experiment, experiment_name=args.experiment_name)
    generation_overrides = {}
    for argument, field in (
        ("generation_seed", "seed"),
        ("generation_tokens", "token_count"),
        ("temperature", "temperature"),
        ("top_k", "top_k"),
    ):
        value = getattr(args, argument)
        if value is not None:
            generation_overrides[field] = value
    if generation_overrides:
        experiment = replace(
            experiment,
            generation=replace(experiment.generation, **generation_overrides),
        )
    validate_experiment_config(experiment)
    train(
        experiment,
        audit_dir=args.audit_dir.resolve(),
        dataset_dir=args.dataset_dir.resolve(),
        tracking_dir=args.tracking_dir.resolve(),
        run_root=args.run_root.resolve(),
        tracking_uri=args.tracking_uri,
        run_phase=args.run_phase,
        resume=args.resume.resolve() if args.resume else None,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ExperimentConfig",
    "GenerationConfig",
    "TrainingConfig",
    "generate_midi",
    "environment_metadata",
    "load_experiment_config",
    "load_model_checkpoint",
    "run_epoch",
    "select_device",
    "sha256_file",
    "train",
    "validate_experiment_config",
]
