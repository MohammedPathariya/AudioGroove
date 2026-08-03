"""Deterministic, Dask-backed bounded preprocessing for the Day 4 pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from src.data_prep.midi_representation import SPECIAL_TOKENS, MidiEventSequence, build_vocabulary, encode_midi


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_DIR = ROOT / "data" / "audit" / "lmdclean_pilot_250"
DEFAULT_OUTPUT_DIR = ROOT / "runs" / "day4" / "pilot_dataset"


def _encode(path: str) -> MidiEventSequence:
    return encode_midi(path)


def load_selected_records(audit_dir: Path = DEFAULT_AUDIT_DIR) -> list[dict[str, Any]]:
    manifest = audit_dir / "selected_manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(records, key=lambda record: (record["split"], record["relative_path"]))


def dask_encode_records(records: list[dict[str, Any]], workers: int = 2) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Encode records in manifest order and return the recorded Dask policy."""
    try:
        import dask
        from dask import delayed
    except ImportError as exc:
        raise RuntimeError("Dask is required for Day 4 preprocessing") from exc

    if workers < 1:
        raise ValueError("workers must be positive")
    ordered = sorted(records, key=lambda record: (record["split"], record["relative_path"]))
    tasks = [delayed(_encode)(str(ROOT / record["source_path"])) for record in ordered]
    with dask.config.set(scheduler="threads", num_workers=workers):
        sequences = dask.compute(*tasks)
    enriched = [dict(record, sequence=sequence) for record, sequence in zip(ordered, sequences)]
    config = {
        "scheduler": "threads",
        "workers": workers,
        "partition_count": len(tasks),
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
) -> dict[str, int]:
    split_dir = output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    x_buffer: list[list[int]] = []
    y_buffer: list[int] = []
    chunk_index = 0
    window_count = 0

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
        ids = [vocabulary[token] for token in record["sequence"].tokens]
        for start in range(max(0, len(ids) - sequence_length)):
            x_buffer.append(ids[start : start + sequence_length])
            y_buffer.append(ids[start + sequence_length])
            window_count += 1
            if len(x_buffer) >= max_windows_per_chunk:
                flush()
    flush()
    return {"source_file_count": len(records), "window_count": window_count, "chunk_count": chunk_index}


def prepare_pilot_dataset(
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    sequence_length: int = 32,
    max_windows_per_chunk: int = 256,
    dask_workers: int = 2,
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
            and all((output_dir / split).is_dir() for split in ("train", "val", "test"))
        ):
            return cached
    records = load_selected_records(audit_dir)
    enriched, dask_config = dask_encode_records(records, workers=dask_workers)
    vocabulary, _ = build_vocabulary(record["sequence"] for record in enriched)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        split_records = [record for record in enriched if record["split"] == split]
        _write_split_chunks(
            split_records, vocabulary, output_dir, split, sequence_length, max_windows_per_chunk
        )

    summary = json.loads((audit_dir / "pilot_summary.json").read_text(encoding="utf-8"))
    split_counts = {}
    for split in ("train", "val", "test"):
        split_counts[split] = _count_split(output_dir / split, sequence_length)
        split_counts[split]["source_file_count"] = sum(
            record["split"] == split for record in enriched
        )
    manifest = {
        "dataset_name": "lmdclean_pilot_250",
        "dataset_revision": summary["dataset_revision"],
        "selection_seed": summary["selection_seed"],
        "split_seed": summary["split_seed"],
        "source_file_count": len(enriched),
        "sequence_length": sequence_length,
        "max_windows_per_chunk": max_windows_per_chunk,
        "vocabulary_size": len(vocabulary),
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


def _count_split(split_dir: Path, sequence_length: int) -> dict[str, int]:
    paths = sorted(split_dir.glob("chunk_*.pt"))
    windows = 0
    for path in paths:
        try:
            chunk = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            chunk = torch.load(path, map_location="cpu")
        if chunk["x"].ndim != 2 or chunk["x"].shape[1] != sequence_length:
            raise ValueError(f"Unexpected chunk shape in {path}")
        windows += int(chunk["x"].shape[0])
    return {"chunk_count": len(paths), "window_count": windows}


__all__ = ["DEFAULT_OUTPUT_DIR", "prepare_pilot_dataset"]
