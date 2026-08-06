"""Deterministic, Dask-backed bounded preprocessing for the Day 4 pilot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from src.data_prep.midi_representation import (
    DEFAULT_MAX_TIME_SHIFT_TICKS,
    DEFAULT_VELOCITY_BINS,
    SPECIAL_TOKENS,
    MidiEventSequence,
    build_vocabulary,
    encode_midi,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = ROOT / "data" / "audit" / "lmdclean_pilot_250"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "day4" / "pilot_dataset"
VOCABULARY_POLICY = "train_only"
UNKNOWN_TOKEN_POLICY = "map_to_unk"


def _encode(path: str, max_time_shift_ticks: int, velocity_bins: int) -> MidiEventSequence:
    return encode_midi(
        path,
        max_time_shift_ticks=max_time_shift_ticks,
        velocity_bins=velocity_bins,
    )


def load_selected_records(audit_dir: Path = DEFAULT_AUDIT_DIR) -> list[dict[str, Any]]:
    manifest = audit_dir / "selected_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(records, key=lambda record: (record["split"], record["relative_path"]))


def dask_encode_records(
    records: list[dict[str, Any]],
    workers: int = 2,
    max_time_shift_ticks: int = DEFAULT_MAX_TIME_SHIFT_TICKS,
    velocity_bins: int = DEFAULT_VELOCITY_BINS,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Encode records in manifest order and return the recorded Dask policy."""
    try:
        import dask
        from dask import delayed
    except ImportError as exc:
        raise RuntimeError("Dask is required for Day 4 preprocessing") from exc

    if workers < 1:
        raise ValueError("workers must be positive")
    ordered = sorted(records, key=lambda record: (record["split"], record["relative_path"]))
    tasks = [
        delayed(_encode)(str(ROOT / record["source_path"]), max_time_shift_ticks, velocity_bins)
        for record in ordered
    ]
    with dask.config.set(scheduler="threads", num_workers=workers):
        sequences = dask.compute(*tasks)
    enriched = [dict(record, sequence=sequence) for record, sequence in zip(ordered, sequences)]
    config = {
        "scheduler": "threads",
        "workers": workers,
        "partition_count": len(tasks),
        "max_time_shift_ticks": max_time_shift_ticks,
        "velocity_bins": velocity_bins,
        "ordering": "split then relative_path; dask results consumed in task order",
    }
    return enriched, config


def _write_split_chunks(
    records: list[dict[str, Any]],
    vocabulary: dict[str, int],
    output_dir: Path,
    split: str,
    sequence_length: int,
    max_windows_per_chunk: int,
) -> dict[str, int | float]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    x_buffer: list[list[int]] = []
    y_buffer: list[int] = []
    chunk_index = 0
    window_count = 0
    token_count = 0
    oov_token_count = 0
    unique_oov_tokens: set[str] = set()
    unknown_id = vocabulary["<UNK>"]

    def flush() -> None:
        nonlocal chunk_index
        if not x_buffer:
            return
        torch.save(
            {
                "x": torch.tensor(x_buffer, dtype=torch.long),
                "y": torch.tensor(y_buffer, dtype=torch.long),
                "sequence_length": sequence_length,
                "split": split,
                "vocabulary_size": len(vocabulary),
            },
            split_dir / f"chunk_{chunk_index:04d}.pt",
        )
        x_buffer.clear()
        y_buffer.clear()
        chunk_index += 1

    for record in records:
        tokens = record["sequence"].tokens
        missing = [token for token in tokens if token not in vocabulary]
        token_count += len(tokens)
        oov_token_count += len(missing)
        unique_oov_tokens.update(missing)
        ids = [vocabulary.get(token, unknown_id) for token in tokens]
        for start in range(max(0, len(ids) - sequence_length)):
            x_buffer.append(ids[start : start + sequence_length])
            y_buffer.append(ids[start + sequence_length])
            window_count += 1
            if len(x_buffer) >= max_windows_per_chunk:
                flush()
    flush()
    return {
        "source_file_count": len(records),
        "window_count": window_count,
        "chunk_count": chunk_index,
        "token_count": token_count,
        "oov_token_count": oov_token_count,
        "oov_token_rate": oov_token_count / token_count if token_count else 0.0,
        "unique_oov_token_count": len(unique_oov_tokens),
    }


def prepare_pilot_dataset(
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sequence_length: int = 32,
    max_windows_per_chunk: int = 256,
    dask_workers: int = 2,
    max_time_shift_ticks: int = DEFAULT_MAX_TIME_SHIFT_TICKS,
    velocity_bins: int = DEFAULT_VELOCITY_BINS,
) -> dict[str, Any]:
    """Prepare all 250 songs into split-local bounded chunks."""
    if sequence_length < 1 or max_windows_per_chunk < 1:
        raise ValueError("sequence_length and max_windows_per_chunk must be positive")
    existing_manifest = output_dir / "manifest.json"
    if existing_manifest.is_file():
        cached = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if (
            cached.get("sequence_length") == sequence_length
            and cached.get("max_windows_per_chunk") == max_windows_per_chunk
            and cached.get("dask", {}).get("workers") == dask_workers
            and cached.get("max_time_shift_ticks") == max_time_shift_ticks
            and cached.get("velocity_bins") == velocity_bins
            and cached.get("vocabulary_policy") == VOCABULARY_POLICY
            and cached.get("unknown_token_policy") == UNKNOWN_TOKEN_POLICY
            and "vocabulary_breakdown" in cached
            and all((output_dir / split).is_dir() for split in ("train", "val", "test"))
            and all(
                "oov_token_rate" in cached.get("splits", {}).get(split, {})
                and len(list((output_dir / split).glob("chunk_*.pt")))
                == cached["splits"][split]["chunk_count"]
                for split in ("train", "val", "test")
            )
        ):
            return cached
    records = load_selected_records(audit_dir)
    enriched, dask_config = dask_encode_records(
        records,
        workers=dask_workers,
        max_time_shift_ticks=max_time_shift_ticks,
        velocity_bins=velocity_bins,
    )
    train_records = [record for record in enriched if record["split"] == "train"]
    vocabulary, _ = build_vocabulary(record["sequence"] for record in train_records)
    vocabulary_breakdown = Counter(token.split(":", 1)[0] for token in vocabulary)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    for split in ("train", "val", "test"):
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for stale_chunk in split_dir.glob("chunk_*.pt"):
            stale_chunk.unlink()
        split_records = [record for record in enriched if record["split"] == split]
        split_counts[split] = _write_split_chunks(
            split_records, vocabulary, output_dir, split, sequence_length, max_windows_per_chunk
        )

    summary = json.loads((audit_dir / "pilot_summary.json").read_text(encoding="utf-8"))
    derived_revision = hashlib.sha256(
        json.dumps(
            {
                "source_dataset_revision": summary["dataset_revision"],
                "representation": "bounded_time_shift_velocity_v2",
                "max_time_shift_ticks": max_time_shift_ticks,
                "velocity_bins": velocity_bins,
                "sequence_length": sequence_length,
                "max_windows_per_chunk": max_windows_per_chunk,
                "vocabulary_policy": VOCABULARY_POLICY,
                "unknown_token_policy": UNKNOWN_TOKEN_POLICY,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest = {
        "dataset_name": "lmdclean_pilot_250",
        "dataset_revision": derived_revision,
        "source_dataset_revision": summary["dataset_revision"],
        "representation": "bounded_time_shift_velocity_v2",
        "max_time_shift_ticks": max_time_shift_ticks,
        "velocity_bins": velocity_bins,
        "selection_seed": summary["selection_seed"],
        "split_seed": summary["split_seed"],
        "source_file_count": len(enriched),
        "sequence_length": sequence_length,
        "max_windows_per_chunk": max_windows_per_chunk,
        "vocabulary_policy": VOCABULARY_POLICY,
        "vocabulary_source_split": "train",
        "unknown_token_policy": UNKNOWN_TOKEN_POLICY,
        "vocabulary_size": len(vocabulary),
        "vocabulary_breakdown": dict(sorted(vocabulary_breakdown.items())),
        "special_tokens": list(SPECIAL_TOKENS),
        "dask": dask_config,
        "splits": split_counts,
    }
    (output_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


__all__ = [
    "DEFAULT_OUTPUT_DIR",
    "UNKNOWN_TOKEN_POLICY",
    "VOCABULARY_POLICY",
    "prepare_pilot_dataset",
]
