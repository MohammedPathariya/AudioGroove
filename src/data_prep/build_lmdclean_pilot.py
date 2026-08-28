"""Select and preprocess a deterministic, source-level LMDClean pilot.

This command copies approximately 250 eligible songs into a separate local
pilot directory, freezes source-level splits, and writes bounded chunks. It
does not train a model or modify the source corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from src.data_prep.audit_midi_sources import inspect_midi, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = ROOT / "data" / "clean_midi"
DEFAULT_PILOT_DIR = ROOT / "data" / "pilot_250"
DEFAULT_AUDIT_DIR = ROOT / "data" / "audit" / "lmdclean_pilot_250"
DEFAULT_SELECTION_SEED = 20260803
DEFAULT_SPLIT_SEED = 20260804
DEFAULT_TARGET_SONGS = 250
DEFAULT_MAX_DURATION_SECONDS = 600.0
DEFAULT_SEQUENCE_LENGTH = 32
DEFAULT_MAX_WINDOWS_PER_CHUNK = 256


@dataclass
class CorpusFile:
    source_path: str
    relative_path: str
    sha256: str | None
    size_bytes: int
    artist: str
    album: str | None
    title: str
    song_identity: str
    group_identity: str


@dataclass
class SelectedSong:
    source_path: str
    relative_path: str
    pilot_path: str
    sha256: str
    size_bytes: int
    artist: str
    album: str | None
    title: str
    song_identity: str
    group_identity: str
    duration_seconds: float
    duration_quarter_length: float
    token_count: int
    split: str


def normalized(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def song_title(path: Path) -> str:
    return re.sub(r"\.[0-9]+$", "", path.stem).strip()


def relative_source_path(path: Path, source_dir: Path) -> str:
    return path.relative_to(source_dir).as_posix()


def corpus_files(source_dir: Path) -> list[CorpusFile]:
    files = sorted(
        (
            path.resolve()
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
        ),
        key=lambda path: path.relative_to(source_dir).as_posix(),
    )
    records: list[CorpusFile] = []
    for path in files:
        relative = relative_source_path(path, source_dir)
        parts = Path(relative).parts
        artist = parts[0] if len(parts) > 1 else "__unknown_artist__"
        title = song_title(path)
        artist_key = normalized(artist) or "unknown artist"
        title_key = normalized(title) or normalized(path.stem)
        records.append(
            CorpusFile(
                source_path=str(path),
                relative_path=relative,
                sha256=None,
                size_bytes=path.stat().st_size,
                artist=artist,
                album=None,
                title=title,
                song_identity=f"{artist_key}::{title_key}",
                group_identity=f"artist::{artist_key}",
            )
        )
    return records


def inventory_revision(files: list[CorpusFile]) -> str:
    digest = hashlib.sha256()
    for record in files:
        digest.update(
            f"{record.relative_path}\t{record.size_bytes}\n".encode("utf-8")
        )
    return digest.hexdigest()


def choose_representatives(
    files: list[CorpusFile], exclusions: list[dict[str, Any]]
) -> list[CorpusFile]:
    grouped: dict[str, list[CorpusFile]] = defaultdict(list)
    for record in files:
        grouped[record.song_identity].append(record)
    representatives: list[CorpusFile] = []
    for identity in sorted(grouped):
        variants = sorted(grouped[identity], key=lambda item: (item.size_bytes, item.relative_path))
        representative = variants[0]
        representatives.append(representative)
        for duplicate in variants[1:]:
            exclusions.append(
                {
                    "relative_path": duplicate.relative_path,
                    "sha256": sha256_file(Path(duplicate.source_path)),
                    "reason": "duplicate_song_identity",
                    "song_identity": duplicate.song_identity,
                    "representative_path": representative.relative_path,
                }
            )
    return representatives


def select_songs(
    representatives: list[CorpusFile],
    target: int,
    selection_seed: int,
    max_duration_seconds: float,
    sequence_length: int,
    exclusions: list[dict[str, Any]],
) -> list[tuple[CorpusFile, float, float, int]]:
    ordered = sorted(representatives, key=lambda item: (item.song_identity, item.sha256))
    random.Random(selection_seed).shuffle(ordered)
    selected: list[tuple[CorpusFile, float, float, int]] = []
    for record in ordered:
        if len(selected) >= target:
            break
        try:
            duration_seconds, quarter_length, tokens = inspect_midi(Path(record.source_path))
        except Exception as exc:
            exclusions.append(
                {
                    "relative_path": record.relative_path,
                    "sha256": sha256_file(Path(record.source_path)),
                    "reason": "unreadable",
                    "failure": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if duration_seconds > max_duration_seconds:
            exclusions.append(
                {
                    "relative_path": record.relative_path,
                    "sha256": sha256_file(Path(record.source_path)),
                    "reason": "overlong",
                    "duration_seconds": duration_seconds,
                    "max_duration_seconds": max_duration_seconds,
                }
            )
            continue
        if len(tokens) <= sequence_length:
            exclusions.append(
                {
                    "relative_path": record.relative_path,
                    "sha256": sha256_file(Path(record.source_path)),
                    "reason": "too_short_for_sequence_length",
                    "token_count": len(tokens),
                    "sequence_length": sequence_length,
                }
            )
            continue
        record.sha256 = sha256_file(Path(record.source_path))
        selected.append((record, duration_seconds, quarter_length, len(tokens)))
    return selected


def assign_group_splits(
    selected: list[tuple[CorpusFile, float, float, int]], split_seed: int
) -> dict[str, str]:
    groups: dict[str, list[CorpusFile]] = defaultdict(list)
    for record, _, _, _ in selected:
        groups[record.group_identity].append(record)
    targets = {"train": 175, "val": 37, "test": 38}
    rng = random.Random(split_seed)
    group_names = sorted(groups)
    rng.shuffle(group_names)
    shuffled_order = {name: index for index, name in enumerate(group_names)}
    group_names.sort(key=lambda name: (-len(groups[name]), shuffled_order[name]))
    counts = {split: 0 for split in targets}
    assignments: dict[str, str] = {}
    for group_name in group_names:
        size = len(groups[group_name])
        split = min(
            targets,
            key=lambda candidate: (
                counts[candidate] / targets[candidate],
                abs((counts[candidate] + size) - targets[candidate]),
                candidate,
            ),
        )
        counts[split] += size
        assignments[group_name] = split
    return assignments


def copy_pilot_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def build_chunks(
    selected: list[SelectedSong],
    audit_dir: Path,
    sequence_length: int,
    max_windows_per_chunk: int,
) -> dict[str, Any]:
    token_set: set[str] = set()
    token_sequences: dict[str, list[str]] = {}
    for record in selected:
        _, _, tokens = inspect_midi(Path(record.source_path))
        token_sequences[record.source_path] = tokens
        token_set.update(tokens)
    vocabulary = {token: index for index, token in enumerate(sorted(token_set))}
    chunk_root = audit_dir / "chunks"
    chunks: list[dict[str, Any]] = []
    split_summaries: dict[str, Any] = {}
    for split in ("train", "val", "test"):
        split_records = [record for record in selected if record.split == split]
        x_buffer: list[list[int]] = []
        y_buffer: list[int] = []
        chunk_index = 0
        window_count = 0
        split_dir = chunk_root / split
        split_dir.mkdir(parents=True, exist_ok=True)

        def flush() -> None:
            nonlocal chunk_index, x_buffer, y_buffer
            if not x_buffer:
                return
            path = split_dir / f"chunk_{chunk_index:04d}.pt"
            torch.save(
                {
                    "x": torch.tensor(x_buffer, dtype=torch.long),
                    "y": torch.tensor(y_buffer, dtype=torch.long),
                    "sequence_length": sequence_length,
                    "dataset_name": "lmdclean_pilot_250",
                    "split": split,
                },
                path,
            )
            chunks.append(
                {
                    "chunk_path": str(path.relative_to(ROOT)),
                    "split": split,
                    "window_count": len(x_buffer),
                    "sequence_length": sequence_length,
                    "vocabulary_size": len(vocabulary),
                }
            )
            chunk_index += 1
            x_buffer = []
            y_buffer = []

        for record in split_records:
            ids = [vocabulary[token] for token in token_sequences[record.source_path]]
            for start in range(len(ids) - sequence_length):
                x_buffer.append(ids[start : start + sequence_length])
                y_buffer.append(ids[start + sequence_length])
                window_count += 1
                if len(x_buffer) >= max_windows_per_chunk:
                    flush()
        flush()
        split_summaries[split] = {
            "source_file_count": len(split_records),
            "token_count": sum(record.token_count for record in split_records),
            "window_count": window_count,
            "chunk_count": sum(chunk["split"] == split for chunk in chunks),
        }
    manifest = {
        "dataset_name": "lmdclean_pilot_250",
        "sequence_length": sequence_length,
        "max_windows_per_chunk": max_windows_per_chunk,
        "vocabulary_size": len(vocabulary),
        "split_summaries": split_summaries,
        "chunks": chunks,
    }
    (audit_dir / "chunk_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (audit_dir / "vocabulary.json").write_text(
        json.dumps(vocabulary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def build_pilot(
    source_dir: Path,
    pilot_dir: Path,
    audit_dir: Path,
    target: int,
    selection_seed: int,
    split_seed: int,
    max_duration_seconds: float,
    sequence_length: int,
    max_windows_per_chunk: int,
) -> dict[str, Any]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    pilot_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    exclusions: list[dict[str, Any]] = []
    files = corpus_files(source_dir)
    revision = inventory_revision(files)
    representatives = choose_representatives(files, exclusions)
    selected_raw = select_songs(
        representatives,
        target,
        selection_seed,
        max_duration_seconds,
        sequence_length,
        exclusions,
    )
    if len(selected_raw) < target:
        raise RuntimeError(f"only {len(selected_raw)} eligible songs found; target is {target}")
    assignments = assign_group_splits(selected_raw, split_seed)
    selected: list[SelectedSong] = []
    for record, duration_seconds, quarter_length, token_count in selected_raw:
        split = assignments[record.group_identity]
        pilot_relative = Path(split) / record.artist / Path(record.relative_path).name
        pilot_path = pilot_dir / pilot_relative
        copy_pilot_file(Path(record.source_path), pilot_path)
        selected.append(
            SelectedSong(
                source_path=f"data/clean_midi/{record.relative_path}",
                relative_path=record.relative_path,
                pilot_path=str(pilot_relative.as_posix()),
                sha256=record.sha256 or sha256_file(Path(record.source_path)),
                size_bytes=record.size_bytes,
                artist=record.artist,
                album=record.album,
                title=record.title,
                song_identity=record.song_identity,
                group_identity=record.group_identity,
                duration_seconds=duration_seconds,
                duration_quarter_length=quarter_length,
                token_count=token_count,
                split=split,
            )
        )
    selected.sort(key=lambda record: record.relative_path)
    write_jsonl(audit_dir / "selected_manifest.jsonl", [asdict(record) for record in selected])
    write_jsonl(audit_dir / "exclusions.jsonl", exclusions)
    split_manifest = {
        "selection_seed": selection_seed,
        "split_seed": split_seed,
        "test_split_frozen": True,
        "assignment": {record.song_identity: record.split for record in selected},
        "group_assignment": {record.group_identity: record.split for record in selected},
    }
    (audit_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    chunk_manifest = build_chunks(selected, audit_dir, sequence_length, max_windows_per_chunk)
    split_counts = {
        split: sum(record.split == split for record in selected)
        for split in ("train", "val", "test")
    }
    song_split_sets = {
        split: {record.song_identity for record in selected if record.split == split}
        for split in ("train", "val", "test")
    }
    group_split_sets = {
        split: {record.group_identity for record in selected if record.split == split}
        for split in ("train", "val", "test")
    }
    sha_split_sets = {
        split: {record.sha256 for record in selected if record.split == split}
        for split in ("train", "val", "test")
    }
    summary = {
        "dataset_name": "lmdclean_pilot_250",
        "dataset_revision": revision,
        "dataset_revision_method": "sha256 of sorted relative MIDI paths and byte sizes; selected files also have content SHA-256",
        "source_dir": "data/clean_midi",
        "pilot_dir": "data/pilot_250",
        "audit_dir": "data/audit/lmdclean_pilot_250",
        "source_file_count": len(files),
        "unique_song_identity_count": len(representatives),
        "target_song_count": target,
        "selected_song_count": len(selected),
        "selection_seed": selection_seed,
        "split_seed": split_seed,
        "test_split_frozen": True,
        "metadata": {
            "artist_grouping_used": True,
            "album_grouping_used": False,
            "album_metadata_reason": "No reliable album metadata was present in the corpus layout or sidecar files.",
        },
        "configuration": {
            "max_duration_seconds": max_duration_seconds,
            "sequence_length": sequence_length,
            "max_windows_per_chunk": max_windows_per_chunk,
            "tokenization": "mido note_on/note_off event tokens",
        },
        "split_counts": split_counts,
        "token_count": sum(record.token_count for record in selected),
        "window_count": sum(item["window_count"] for item in chunk_manifest["split_summaries"].values()),
        "chunk_manifest": chunk_manifest["split_summaries"],
        "exclusion_count": len(exclusions),
        "exclusion_counts_by_reason": {
            reason: sum(item.get("reason") == reason for item in exclusions)
            for reason in sorted({item.get("reason") for item in exclusions})
        },
        "leakage_checks": {
            "song_identity_overlap_count": len(song_split_sets["train"] & song_split_sets["val"] | song_split_sets["train"] & song_split_sets["test"] | song_split_sets["val"] & song_split_sets["test"]),
            "artist_group_overlap_count": len(group_split_sets["train"] & group_split_sets["val"] | group_split_sets["train"] & group_split_sets["test"] | group_split_sets["val"] & group_split_sets["test"]),
            "sha256_overlap_count": len(sha_split_sets["train"] & sha_split_sets["val"] | sha_split_sets["train"] & sha_split_sets["test"] | sha_split_sets["val"] & sha_split_sets["test"]),
        },
        "resource_limits": {
            "chunking": "bounded, one chunk buffer at a time",
            "max_windows_per_chunk": max_windows_per_chunk,
            "training_started": False,
            "larger_corpus_training_started": False,
            "model_training": "not run by this command",
        },
    }
    (audit_dir / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--pilot-dir", type=Path, default=DEFAULT_PILOT_DIR)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    parser.add_argument("--target-songs", type=int, default=DEFAULT_TARGET_SONGS)
    parser.add_argument("--selection-seed", type=int, default=DEFAULT_SELECTION_SEED)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SPLIT_SEED)
    parser.add_argument("--max-duration-seconds", type=float, default=DEFAULT_MAX_DURATION_SECONDS)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--max-windows-per-chunk", type=int, default=DEFAULT_MAX_WINDOWS_PER_CHUNK)
    args = parser.parse_args()
    summary = build_pilot(
        args.source_dir.resolve(),
        args.pilot_dir.resolve(),
        args.audit_dir.resolve(),
        args.target_songs,
        args.selection_seed,
        args.split_seed,
        args.max_duration_seconds,
        args.sequence_length,
        args.max_windows_per_chunk,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
