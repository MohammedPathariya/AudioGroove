from pathlib import Path

import mido
import torch

from src.data_prep.midi_representation import (
    BoundedChunkDataset,
    DEFAULT_VELOCITY_BINS,
    SequentialChunkDataset,
    decode_tokens,
    encode_midi,
    make_next_token_chunks,
)


def make_fixture(path: Path) -> None:
    track = mido.MidiTrack(
        [
            mido.MetaMessage("set_tempo", tempo=500000, time=0),
            mido.Message("program_change", channel=2, program=41, time=0),
            mido.Message("note_on", channel=2, note=60, velocity=91, time=12),
            mido.Message("note_on", channel=2, note=64, velocity=55, time=0),
            mido.Message("note_off", channel=2, note=60, time=36),
            mido.Message("note_off", channel=2, note=64, time=0),
        ]
    )
    mido.MidiFile(ticks_per_beat=96, tracks=[track]).save(path)


def test_round_trip_preserves_supported_events_and_timing(tmp_path: Path) -> None:
    original = tmp_path / "original.mid"
    restored = tmp_path / "restored.mid"
    make_fixture(original)
    encoded = encode_midi(original)
    decode_tokens(encoded.tokens, restored, encoded.ticks_per_beat)

    assert encode_midi(restored) == encoded


def test_velocity_buckets_are_bounded_and_round_trip_stably(tmp_path: Path) -> None:
    source = tmp_path / "velocity.mid"
    track = mido.MidiTrack(
        [
            mido.Message("note_on", note=60, velocity=1, time=0),
            mido.Message("note_off", note=60, velocity=127, time=12),
        ]
    )
    mido.MidiFile(ticks_per_beat=96, tracks=[track]).save(source)

    encoded = encode_midi(source)
    velocity_tokens = [
        token for token in encoded.tokens if token.startswith(("NOTE_ON:", "NOTE_OFF:"))
    ]

    assert [int(token.rsplit(":", 1)[1]) for token in velocity_tokens] == [0, 15]
    assert all(0 <= int(token.rsplit(":", 1)[1]) < DEFAULT_VELOCITY_BINS for token in velocity_tokens)

    restored = tmp_path / "velocity-restored.mid"
    decode_tokens(encoded.tokens, restored, encoded.ticks_per_beat)
    assert encode_midi(restored) == encoded


def test_bounded_loader_has_stable_shapes(tmp_path: Path) -> None:
    source = tmp_path / "fixture.mid"
    make_fixture(source)
    manifest = make_next_token_chunks([source], tmp_path / "chunks", sequence_length=4, max_windows_per_chunk=2)
    dataset = BoundedChunkDataset(tmp_path / "chunks")

    assert manifest["chunk_count"] == 3
    assert len(dataset) == manifest["window_count"]
    x, y = dataset[0]
    assert x.shape == (4,)
    assert y.shape == torch.Size([])
    assert x.dtype == torch.long
    assert y.dtype == torch.long


def test_sequential_chunk_stream_visits_each_window_once(tmp_path: Path) -> None:
    source = tmp_path / "fixture.mid"
    make_fixture(source)
    manifest = make_next_token_chunks([source], tmp_path / "chunks", sequence_length=4, max_windows_per_chunk=2)
    stream = SequentialChunkDataset(tmp_path / "chunks")

    assert sum(1 for _ in stream) == manifest["window_count"]
