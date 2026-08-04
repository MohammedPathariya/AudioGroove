"""Loss-aware symbolic MIDI representation and bounded chunk loader.

The representation uses strings so vocabularies remain serializable and easy to
inspect. MIDI delta times are kept in ticks rather than converted to seconds;
this makes serialization deterministic for a given ``ticks_per_beat`` and
preserves simultaneous events (zero-time shifts).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import mido
import torch
from torch.utils.data import Dataset


SPECIAL_TOKENS = ("<BOS>", "<EOS>", "<UNK>")
DEFAULT_MAX_TIME_SHIFT_TICKS = 96
DEFAULT_VELOCITY_BINS = 16


@dataclass(frozen=True)
class MidiEventSequence:
    tokens: tuple[str, ...]
    ticks_per_beat: int


def _velocity_bucket(velocity: int, bins: int) -> int:
    return min(bins - 1, max(0, ((velocity - 1) * bins) // 128))


def _bucket_velocity(bucket: int, bins: int) -> int:
    if bucket < 0 or bucket >= bins:
        raise ValueError(f"velocity bucket {bucket} outside 0..{bins - 1}")
    return min(127, bucket * 128 // bins + 64 // bins)


def _event_token(message: mido.Message, velocity_bins: int) -> str | None:
    if message.type == "note_on" and message.velocity > 0:
        bucket = _velocity_bucket(message.velocity, velocity_bins)
        return f"NOTE_ON:{message.channel}:{message.note}:{bucket}"
    if message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
        bucket = _velocity_bucket(message.velocity, velocity_bins)
        return f"NOTE_OFF:{message.channel}:{message.note}:{bucket}"
    if message.type == "program_change":
        return f"PROGRAM:{message.channel}:{message.program}"
    if message.type == "set_tempo":
        return f"TEMPO:{message.tempo}"
    return None


def encode_midi(
    path: str | Path,
    max_time_shift_ticks: int = DEFAULT_MAX_TIME_SHIFT_TICKS,
    velocity_bins: int = DEFAULT_VELOCITY_BINS,
) -> MidiEventSequence:
    """Encode supported messages with bounded, exact cumulative shifts."""
    if max_time_shift_ticks < 1:
        raise ValueError("max_time_shift_ticks must be positive")
    if velocity_bins < 1 or velocity_bins > 128:
        raise ValueError("velocity_bins must be between 1 and 128")
    midi = mido.MidiFile(str(path))
    events: list[tuple[int, int, str]] = []
    order = 0
    for track in midi.tracks:
        absolute_tick = 0
        for message in track:
            absolute_tick += int(message.time)
            token = _event_token(message, velocity_bins)
            if token is not None:
                events.append((absolute_tick, order, token))
                order += 1

    # Track order is retained for ties, while the token stream uses one merged
    # timeline. A TIME_SHIFT before each event is therefore unambiguous.
    events.sort(key=lambda item: (item[0], item[1]))
    tokens: list[str] = ["<BOS>"]
    previous_tick = 0
    for absolute_tick, _, token in events:
        delta = absolute_tick - previous_tick
        while delta > max_time_shift_ticks:
            tokens.append(f"TIME_SHIFT:{max_time_shift_ticks}")
            delta -= max_time_shift_ticks
        if delta:
            tokens.append(f"TIME_SHIFT:{delta}")
        tokens.append(token)
        previous_tick = absolute_tick
    tokens.append("<EOS>")
    return MidiEventSequence(tuple(tokens), midi.ticks_per_beat)


def _parse_token(token: str) -> tuple[str, tuple[int, ...]] | tuple[str, int] | None:
    parts = token.split(":")
    if parts[0] == "TIME_SHIFT" and len(parts) == 2:
        return "TIME_SHIFT", int(parts[1])
    if parts[0] in {"NOTE_ON", "NOTE_OFF", "PROGRAM"}:
        values = tuple(int(value) for value in parts[1:])
        expected = {"NOTE_ON": 3, "NOTE_OFF": 3, "PROGRAM": 2}[parts[0]]
        if len(values) == expected:
            return parts[0], values
    if parts[0] == "TEMPO" and len(parts) == 2:
        return "TEMPO", int(parts[1])
    return None


def decode_tokens(
    tokens: Sequence[str],
    output_path: str | Path,
    ticks_per_beat: int = 480,
    velocity_bins: int = DEFAULT_VELOCITY_BINS,
) -> Path:
    """Serialize supported tokens to a valid single-track MIDI file."""
    track = mido.MidiTrack()
    pending_ticks = 0
    for token in tokens:
        if token in {"<BOS>", "<EOS>", "<UNK>"}:
            continue
        parsed = _parse_token(token)
        if parsed is None:
            raise ValueError(f"Unsupported or malformed MIDI token: {token!r}")
        kind, values = parsed
        if kind == "TIME_SHIFT":
            pending_ticks += values  # type: ignore[operator]
            continue
        if kind == "NOTE_ON":
            channel, note, velocity_bucket = values  # type: ignore[misc]
            velocity = _bucket_velocity(velocity_bucket, velocity_bins)
            message = mido.Message("note_on", channel=channel, note=note, velocity=velocity)
        elif kind == "NOTE_OFF":
            channel, note, velocity_bucket = values  # type: ignore[misc]
            velocity = _bucket_velocity(velocity_bucket, velocity_bins)
            message = mido.Message("note_off", channel=channel, note=note, velocity=velocity)
        elif kind == "PROGRAM":
            channel, program = values  # type: ignore[misc]
            message = mido.Message("program_change", channel=channel, program=program)
        else:
            message = mido.MetaMessage("set_tempo", tempo=values)  # type: ignore[arg-type]
        message.time = pending_ticks
        track.append(message)
        pending_ticks = 0

    if pending_ticks:
        track.append(mido.MetaMessage("end_of_track", time=pending_ticks))
    midi = mido.MidiFile(ticks_per_beat=ticks_per_beat, tracks=[track])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(destination))
    return destination


def build_vocabulary(sequences: Iterable[MidiEventSequence]) -> tuple[dict[str, int], dict[int, str]]:
    """Build a deterministic vocabulary from already-encoded sequences."""
    tokens = set(SPECIAL_TOKENS)
    for sequence in sequences:
        tokens.update(sequence.tokens)
    ordered = list(SPECIAL_TOKENS) + sorted(tokens.difference(SPECIAL_TOKENS))
    token_to_id = {token: index for index, token in enumerate(ordered)}
    return token_to_id, {index: token for token, index in token_to_id.items()}


def make_next_token_chunks(
    paths: Iterable[str | Path],
    output_dir: str | Path,
    sequence_length: int = 32,
    max_windows_per_chunk: int = 256,
) -> dict[str, int]:
    """Create bounded ``.pt`` chunks without retaining the whole corpus."""
    if sequence_length < 1 or max_windows_per_chunk < 1:
        raise ValueError("sequence_length and max_windows_per_chunk must be positive")
    encoded = [encode_midi(path) for path in sorted((Path(path) for path in paths), key=str)]
    vocabulary, _ = build_vocabulary(encoded)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    x_buffer: list[list[int]] = []
    y_buffer: list[int] = []
    chunk_count = 0
    window_count = 0

    def flush() -> None:
        nonlocal chunk_count
        if not x_buffer:
            return
        torch.save(
            {"x": torch.tensor(x_buffer, dtype=torch.long), "y": torch.tensor(y_buffer, dtype=torch.long),
             "sequence_length": sequence_length, "vocabulary": vocabulary},
            destination / f"chunk_{chunk_count:04d}.pt",
        )
        x_buffer.clear()
        y_buffer.clear()
        chunk_count += 1

    for sequence in encoded:
        ids = [vocabulary[token] for token in sequence.tokens]
        for start in range(max(0, len(ids) - sequence_length)):
            x_buffer.append(ids[start : start + sequence_length])
            y_buffer.append(ids[start + sequence_length])
            window_count += 1
            if len(x_buffer) == max_windows_per_chunk:
                flush()
    flush()
    manifest = {"source_file_count": len(encoded), "chunk_count": chunk_count,
                "window_count": window_count, "sequence_length": sequence_length,
                "vocabulary_size": len(vocabulary)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


class BoundedChunkDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazy dataset that keeps only one bounded chunk resident at a time."""

    def __init__(self, chunk_dir: str | Path):
        self.chunk_paths = sorted(Path(chunk_dir).glob("chunk_*.pt"))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No chunk_*.pt files found in {chunk_dir}")
        self._lengths = [self._read_chunk(path)["x"].shape[0] for path in self.chunk_paths]
        self._active_index: int | None = None
        self._active_chunk: dict[str, torch.Tensor] | None = None

    @staticmethod
    def _read_chunk(path: Path) -> dict[str, torch.Tensor]:
        try:
            chunk = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:  # torch before weights_only was introduced
            chunk = torch.load(path, map_location="cpu")
        if not isinstance(chunk, dict) or not {"x", "y"}.issubset(chunk):
            raise ValueError(f"Invalid bounded chunk: {path}")
        if chunk["x"].ndim != 2 or chunk["y"].ndim != 1 or chunk["x"].shape[0] != chunk["y"].shape[0]:
            raise ValueError(f"Invalid chunk shapes: {path}")
        return chunk

    def __len__(self) -> int:
        return sum(self._lengths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        offset = index
        chunk_index = 0
        while offset >= self._lengths[chunk_index]:
            offset -= self._lengths[chunk_index]
            chunk_index += 1
        if self._active_index != chunk_index:
            self._active_chunk = self._read_chunk(self.chunk_paths[chunk_index])
            self._active_index = chunk_index
        assert self._active_chunk is not None
        return self._active_chunk["x"][offset], self._active_chunk["y"][offset]


class SequentialChunkDataset(torch.utils.data.IterableDataset[tuple[torch.Tensor, torch.Tensor]]):
    """Stream complete chunks in deterministic order for full-epoch training."""

    def __init__(self, chunk_dir: str | Path):
        self.chunk_paths = sorted(Path(chunk_dir).glob("chunk_*.pt"))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No chunk_*.pt files found in {chunk_dir}")

    def __iter__(self):
        for path in self.chunk_paths:
            chunk = BoundedChunkDataset._read_chunk(path)
            for index in range(chunk["x"].shape[0]):
                yield chunk["x"][index], chunk["y"][index]


__all__ = [
    "BoundedChunkDataset",
    "DEFAULT_MAX_TIME_SHIFT_TICKS",
    "DEFAULT_VELOCITY_BINS",
    "MidiEventSequence",
    "SequentialChunkDataset",
    "build_vocabulary",
    "decode_tokens",
    "encode_midi",
    "make_next_token_chunks",
]
