"""Day 2 MIDI audit and bounded preprocessing.

This command is intentionally limited to source inspection, source-level
splitting, and bounded next-token chunk creation. It does not import or call
any training code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mido
import torch


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIRS = (ROOT / "data" / "raw" / "LMDClean", ROOT / "data" / "seed")
DEFAULT_OUTPUT_DIR = ROOT / "data" / "audit" / "day2"
DEFAULT_MAX_DURATION_SECONDS = 10 * 60
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_MAX_WINDOWS_PER_CHUNK = 256
DEFAULT_SEED = 20260803


@dataclass
class AuditRecord:
    source_path: str
    source_root: str
    source_identity: str
    sha256: str
    size_bytes: int
    parse_status: str
    quarantine_reason: str | None
    quarantine_path: str | None
    duration_seconds: float | None
    duration_quarter_length: float | None
    token_count: int | None
    split: str | None
    eligible_for_chunks: bool
    failure: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def midi_files(source_dirs: tuple[Path, ...]) -> list[tuple[Path, Path]]:
    found: dict[str, tuple[Path, Path]] = {}
    for root in source_dirs:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in {".mid", ".midi"}:
                resolved = path.resolve()
                found[str(resolved)] = (resolved, root.resolve())
    return sorted(found.values(), key=lambda item: str(item[0]))


def inspect_midi(path: Path) -> tuple[float, float, list[str]]:
    """Parse one file with mido and produce a bounded event-token sequence."""
    midi = mido.MidiFile(str(path))
    max_ticks = 0
    tokens: list[str] = []
    for track in midi.tracks:
        absolute_ticks = 0
        for message in track:
            absolute_ticks += message.time
            max_ticks = max(max_ticks, absolute_ticks)
            if message.type == "note_on" and message.velocity > 0:
                tokens.append(f"note_on:{message.note}")
            elif message.type == "note_off" or (
                message.type == "note_on" and message.velocity == 0
            ):
                tokens.append(f"note_off:{message.note}")
    quarter_length = max_ticks / midi.ticks_per_beat
    return float(midi.length), float(quarter_length), tokens


def split_records(records: list[AuditRecord], seed: int) -> None:
    eligible = [record for record in records if record.eligible_for_chunks]
    ordered = sorted(eligible, key=lambda record: record.source_identity)
    random.Random(seed).shuffle(ordered)
    total = len(ordered)
    train_count = max(1, int(total * 0.8)) if total else 0
    val_count = 1 if total >= 3 else (1 if total == 2 else 0)
    if train_count + val_count > total:
        train_count = max(0, total - val_count)
    for index, record in enumerate(ordered):
        record.split = "train" if index < train_count else "val" if index < train_count + val_count else "test"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_chunks(
    records: list[AuditRecord],
    output_dir: Path,
    sequence_length: int,
    max_windows_per_chunk: int,
    dataset_name: str,
) -> dict[str, Any]:
    chunk_dir = output_dir / "chunks" / dataset_name
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_rows: list[dict[str, Any]] = []
    total_windows = 0
    chunk_index = 0
    x_buffer: list[list[int]] = []
    y_buffer: list[int] = []

    token_values: dict[str, int] = {}

    def token_id(token: str) -> int:
        if token not in token_values:
            token_values[token] = len(token_values)
        return token_values[token]

    def flush() -> None:
        nonlocal chunk_index, x_buffer, y_buffer
        if not x_buffer:
            return
        chunk_path = chunk_dir / f"chunk_{chunk_index:04d}.pt"
        torch.save(
            {
                "x": torch.tensor(x_buffer, dtype=torch.long),
                "y": torch.tensor(y_buffer, dtype=torch.long),
                "sequence_length": sequence_length,
                "dataset_name": dataset_name,
            },
            chunk_path,
        )
        chunk_rows.append(
            {
                "chunk_path": str(chunk_path.relative_to(ROOT)),
                "window_count": len(x_buffer),
                "sequence_length": sequence_length,
                "token_count": len(token_values),
            }
        )
        chunk_index += 1
        x_buffer = []
        y_buffer = []

    selected = [record for record in records if record.eligible_for_chunks and record.split]
    if dataset_name == "smoke":
        selected = selected[:3]
    elif dataset_name == "development":
        selected = selected[:8]

    selected_paths: list[str] = []
    for record in selected:
        duration, _, tokens = inspect_midi(Path(record.source_path))
        del duration
        ids = [token_id(token) for token in tokens]
        for start in range(max(0, len(ids) - sequence_length)):
            x_buffer.append(ids[start : start + sequence_length])
            y_buffer.append(ids[start + sequence_length])
            total_windows += 1
            if len(x_buffer) >= max_windows_per_chunk:
                flush()
        selected_paths.append(record.source_path)
    flush()

    manifest = {
        "dataset_name": dataset_name,
        "sequence_length": sequence_length,
        "max_windows_per_chunk": max_windows_per_chunk,
        "source_file_count": len(selected),
        "source_paths": selected_paths,
        "chunk_count": len(chunk_rows),
        "window_count": total_windows,
        "vocabulary_size": len(token_values),
        "chunks": chunk_rows,
    }
    (output_dir / f"{dataset_name}_chunks.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / f"{dataset_name}_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def audit(
    source_dirs: tuple[Path, ...],
    output_dir: Path,
    seed: int,
    max_duration_seconds: float,
    sequence_length: int,
    max_windows_per_chunk: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    quarantine_dir = output_dir / "quarantine"
    quarantine_dir.mkdir(exist_ok=True)
    records: list[AuditRecord] = []
    for path, root in midi_files(source_dirs):
        digest = sha256_file(path)
        identity = f"{path.relative_to(root)}::{digest}"
        try:
            duration_seconds, quarter_length, tokens = inspect_midi(path)
            reason = "overlong" if duration_seconds > max_duration_seconds else None
            eligible = reason is None and len(tokens) > sequence_length
            failure = None if reason is None else f"duration_seconds={duration_seconds:.6f} exceeds max={max_duration_seconds:.6f}"
            status = "quarantined" if reason else "parsed"
        except Exception as exc:
            duration_seconds = quarter_length = None
            tokens = []
            reason = "unreadable"
            eligible = False
            failure = f"{type(exc).__name__}: {exc}"
            status = "quarantined"
        quarantine_path = None
        if reason:
            quarantine_path_obj = quarantine_dir / f"{digest[:16]}_{path.name}"
            shutil.copy2(path, quarantine_path_obj)
            quarantine_path = str(quarantine_path_obj.relative_to(ROOT))
        records.append(
            AuditRecord(
                source_path=str(path),
                source_root=str(root),
                source_identity=identity,
                sha256=digest,
                size_bytes=path.stat().st_size,
                parse_status=status,
                quarantine_reason=reason,
                quarantine_path=quarantine_path,
                duration_seconds=duration_seconds,
                duration_quarter_length=quarter_length,
                token_count=None if reason == "unreadable" else len(tokens),
                split=None,
                eligible_for_chunks=eligible,
                failure=failure,
            )
        )

    split_records(records, seed)
    record_dicts = [asdict(record) for record in records]
    write_jsonl(output_dir / "source_manifest.jsonl", record_dicts)
    split_manifest = {
        "seed": seed,
        "assignment": {
            record.source_path: record.split
            for record in records
            if record.split is not None
        },
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    smoke = write_chunks(records, output_dir, sequence_length, max_windows_per_chunk, "smoke")
    development = write_chunks(records, output_dir, sequence_length, max_windows_per_chunk, "development")
    summary = {
        "source_dirs": [str(path) for path in source_dirs],
        "missing_source_dirs": [str(path) for path in source_dirs if not path.is_dir()],
        "max_duration_seconds": max_duration_seconds,
        "sequence_length": sequence_length,
        "max_windows_per_chunk": max_windows_per_chunk,
        "seed": seed,
        "tokenization": "mido note_on/note_off event tokens; token_count counts note events",
        "source_file_count": len(records),
        "parsed_count": sum(record.parse_status == "parsed" for record in records),
        "quarantined_count": sum(record.quarantine_reason is not None for record in records),
        "unreadable_count": sum(record.quarantine_reason == "unreadable" for record in records),
        "overlong_count": sum(record.quarantine_reason == "overlong" for record in records),
        "eligible_count": sum(record.eligible_for_chunks for record in records),
        "split_counts": {
            split: sum(record.split == split for record in records)
            for split in ("train", "val", "test")
        },
        "smoke": smoke,
        "development": development,
        "training_started": False,
        "training_scope": "not run by this command",
    }
    (output_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", action="append", type=Path, dest="source_dirs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-duration-seconds", type=float, default=DEFAULT_MAX_DURATION_SECONDS)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-windows-per-chunk", type=int, default=DEFAULT_MAX_WINDOWS_PER_CHUNK)
    args = parser.parse_args()
    source_dirs = tuple(path.resolve() for path in (args.source_dirs or DEFAULT_SOURCE_DIRS))
    summary = audit(
        source_dirs,
        args.output_dir.resolve(),
        args.seed,
        args.max_duration_seconds,
        args.sequence_length,
        args.max_windows_per_chunk,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
