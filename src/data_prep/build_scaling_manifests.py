"""Build deterministic nested 500, 1K, and 2.5K LMDClean manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import signal
import unicodedata
from collections import defaultdict
from pathlib import Path

import mido


PILOT_SIZE = 250
SCALES = (500, 1000, 2500)
SELECTION_SEED = 20260803
SPLIT_SEED = 20260804
MAX_DURATION_SECONDS = 600.0
SEQUENCE_LENGTH = 32
PARSER_TIMEOUT_SECONDS = 10.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold()
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def source_records(source_root: Path) -> list[dict]:
    records = []
    for path in sorted(
        (p for p in source_root.rglob("*") if p.is_file() and p.suffix.lower() in {".mid", ".midi"}),
        key=lambda p: p.relative_to(source_root).as_posix(),
    ):
        relative = path.relative_to(source_root).as_posix()
        parts = Path(relative).parts
        artist = parts[0] if len(parts) > 1 else "__unknown_artist__"
        title = re.sub(r"\.[0-9]+$", "", path.stem).strip()
        records.append(
            {
                "path": path,
                "relative_path": relative,
                "artist": artist,
                "title": title,
                "song_identity": f"{normalize(artist)}::{normalize(title)}",
                "group_identity": f"artist::{normalize(artist)}",
            }
        )
    return records


def representatives(records: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for record in records:
        grouped[record["song_identity"]].append(record)
    return [
        sorted(variants, key=lambda item: (item["path"].stat().st_size, item["relative_path"]))[0]
        for _, variants in sorted(grouped.items())
    ]


def timeout_handler(_signum, _frame):
    raise TimeoutError(f"parser exceeded {PARSER_TIMEOUT_SECONDS:g}s")


def inspect_for_selection(path: Path) -> tuple[float, int]:
    previous = signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, PARSER_TIMEOUT_SECONDS)
    try:
        midi = mido.MidiFile(str(path))
        max_ticks = 0
        tempos = [(0, 500_000)]
        token_count = 0
        for track in midi.tracks:
            ticks = 0
            for message in track:
                ticks += message.time
                max_ticks = max(max_ticks, ticks)
                if message.type == "set_tempo":
                    tempos.append((ticks, message.tempo))
                if message.type == "note_on" and message.velocity > 0:
                    token_count += 1
                elif message.type == "note_off" or (
                    message.type == "note_on" and message.velocity == 0
                ):
                    token_count += 1
        tempos.sort()
        seconds = 0.0
        previous_tick = 0
        tempo = 500_000
        for tick, next_tempo in tempos:
            tick = min(max(tick, previous_tick), max_ticks)
            seconds += (tick - previous_tick) * tempo / 1_000_000 / midi.ticks_per_beat
            previous_tick = tick
            tempo = next_tempo
            if previous_tick == max_ticks:
                break
        seconds += (max_ticks - previous_tick) * tempo / 1_000_000 / midi.ticks_per_beat
        return seconds, token_count
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def load_pilot(audit_root: Path) -> tuple[list[dict], set[str], set[str], dict[str, str]]:
    rows = [json.loads(line) for line in (audit_root / "selected_manifest.jsonl").read_text().splitlines()]
    split = json.loads((audit_root / "split_manifest.json").read_text())
    return (
        rows,
        {row["relative_path"] for row in rows},
        {row["sha256"] for row in rows},
        split["group_assignment"],
    )


def assign_splits(records: list[dict], pilot_groups: dict[str, str], seed: int) -> dict[str, str]:
    targets = {"train": round(len(records) * 0.70), "val": round(len(records) * 0.15)}
    targets["test"] = len(records) - targets["train"] - targets["val"]
    groups = defaultdict(list)
    for record in records:
        groups[record["group_identity"]].append(record)
    assignments = {}
    counts = {split: 0 for split in targets}
    for group, split in pilot_groups.items():
        if group in groups:
            assignments[group] = split
            counts[split] += len(groups[group])
    new_groups = [group for group in groups if group not in assignments]
    random.Random(seed).shuffle(new_groups)
    for group in new_groups:
        split = min(
            targets,
            key=lambda candidate: (
                counts[candidate] / targets[candidate],
                abs(counts[candidate] + len(groups[group]) - targets[candidate]),
                candidate,
            ),
        )
        assignments[group] = split
        counts[split] += len(groups[group])
    return assignments


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def build(source_root: Path, pilot_root: Path, output_root: Path) -> dict:
    pilot_rows, pilot_paths, pilot_hashes, pilot_groups = load_pilot(pilot_root)
    records = source_records(source_root)
    reps = representatives(records)
    ordered = sorted(reps, key=lambda item: item["song_identity"])
    random.Random(SELECTION_SEED).shuffle(ordered)

    selected = [
        {
            "relative_path": row["relative_path"],
            "artist": row["artist"],
            "title": row["title"],
            "song_identity": row["song_identity"],
            "group_identity": row["group_identity"],
            "sha256": row["sha256"],
            "size_bytes": row["size_bytes"],
            "duration_seconds": row["duration_seconds"],
            "token_count": row["token_count"],
        }
        for row in pilot_rows
    ]
    seen_hashes = set(pilot_hashes)
    exclusions = []
    for index, record in enumerate(ordered, 1):
        if len(selected) == max(SCALES):
            break
        if record["relative_path"] in pilot_paths:
            continue
        if len(selected) >= PILOT_SIZE and index % 250 == 0:
            print(f"selected {len(selected)}/{max(SCALES)}; scanned {index}")
        try:
            duration, token_count = inspect_for_selection(record["path"])
        except Exception as exc:
            exclusions.append({"relative_path": record["relative_path"], "reason": type(exc).__name__, "error": str(exc)})
            continue
        if duration > MAX_DURATION_SECONDS or token_count <= SEQUENCE_LENGTH:
            exclusions.append({"relative_path": record["relative_path"], "reason": "ineligible", "duration_seconds": duration, "token_count": token_count})
            continue
        digest = sha256_file(record["path"])
        if digest in seen_hashes:
            exclusions.append({"relative_path": record["relative_path"], "reason": "duplicate_content_sha256", "sha256": digest})
            continue
        seen_hashes.add(digest)
        selected.append({**record, "sha256": digest, "size_bytes": record["path"].stat().st_size, "duration_seconds": duration, "token_count": token_count})

    output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for size in SCALES:
        rows = [dict(record) for record in selected[:size]]
        assignments = assign_splits(rows, pilot_groups, SPLIT_SEED)
        for row in rows:
            row.pop("path")
            row["source_path"] = f"data/clean_midi/{row['relative_path']}"
            row["split"] = assignments[row["group_identity"]]
        if len({row["sha256"] for row in rows}) != size:
            raise RuntimeError(f"{size}: duplicate content hash")
        groups = defaultdict(set)
        for row in rows:
            groups[row["split"]].add(row["group_identity"])
        if groups["train"] & groups["val"] or groups["train"] & groups["test"] or groups["val"] & groups["test"]:
            raise RuntimeError(f"{size}: artist group leakage")
        write_jsonl(output_root / f"manifest_{size}.jsonl", sorted(rows, key=lambda row: row["relative_path"]))
        summaries[str(size)] = {
            "song_count": size,
            "pilot_overlap_count": len({row["relative_path"] for row in rows} & pilot_paths),
            "new_song_count": size - PILOT_SIZE,
            "split_counts": {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")},
            "pilot_hash_overlap_count": len({row["sha256"] for row in rows} & pilot_hashes),
        }
    summary = {
        "dataset_name": "lmdclean_nested_scaling",
        "source_file_count": len(records),
        "unique_song_identity_count": len(reps),
        "selection_seed": SELECTION_SEED,
        "split_seed": SPLIT_SEED,
        "pilot_policy": "250-song pilot is the frozen prefix; cross-size overlap is intentional",
        "scales": summaries,
        "exclusion_count": len(exclusions),
    }
    (output_root / "scaling_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_jsonl(output_root / "exclusions.jsonl", exclusions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source_root.resolve(), args.pilot_root.resolve(), args.output_root.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
