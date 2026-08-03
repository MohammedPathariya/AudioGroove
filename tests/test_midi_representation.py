from pathlib import Path

import mido
import torch

from src.data_prep.midi_representation import (
    BoundedChunkDataset,
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
