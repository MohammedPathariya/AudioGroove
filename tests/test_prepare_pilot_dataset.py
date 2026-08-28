import json
from pathlib import Path

import torch

from src.data_prep.prepare_pilot_dataset import prepare_pilot_dataset
from src.data_prep.midi_representation import MidiEventSequence


def test_pilot_vocabulary_is_fit_on_train_only(tmp_path: Path, monkeypatch) -> None:
    audit_dir = tmp_path / "audit"
    output_dir = tmp_path / "prepared"
    audit_dir.mkdir()
    (audit_dir / "pilot_summary.json").write_text(
        json.dumps(
            {
                "dataset_revision": "source-revision",
                "selection_seed": 11,
                "split_seed": 12,
            }
        ),
        encoding="utf-8",
    )
    records = [
        {"split": "train", "relative_path": "train.mid", "source_path": "train.mid"},
        {"split": "val", "relative_path": "val.mid", "source_path": "val.mid"},
        {"split": "test", "relative_path": "test.mid", "source_path": "test.mid"},
    ]
    sequences = {
        "train": MidiEventSequence(("<BOS>", "TIME_SHIFT:1", "NOTE_ON:0:60:4", "<EOS>"), 96),
        "val": MidiEventSequence(("<BOS>", "VAL_ONLY", "NOTE_ON:0:60:4", "<EOS>"), 96),
        "test": MidiEventSequence(("<BOS>", "TEST_ONLY", "NOTE_ON:0:60:4", "<EOS>"), 96),
    }

    monkeypatch.setattr(
        "src.data_prep.prepare_pilot_dataset.load_selected_records",
        lambda _: records,
    )

    def fake_encode(selected, **_):
        enriched = [dict(record, sequence=sequences[record["split"]]) for record in selected]
        return enriched, {"workers": 1}

    monkeypatch.setattr(
        "src.data_prep.prepare_pilot_dataset.dask_encode_records",
        fake_encode,
    )

    manifest = prepare_pilot_dataset(
        audit_dir=audit_dir,
        output_dir=output_dir,
        sequence_length=2,
        max_windows_per_chunk=8,
        dask_workers=1,
    )
    vocabulary = json.loads((output_dir / "vocabulary.json").read_text(encoding="utf-8"))

    assert manifest["vocabulary_policy"] == "train_only"
    assert manifest["unknown_token_policy"] == "map_to_unk"
    assert "VAL_ONLY" not in vocabulary
    assert "TEST_ONLY" not in vocabulary
    assert manifest["splits"]["train"]["oov_token_count"] == 0
    assert manifest["splits"]["val"]["oov_token_count"] == 1
    assert manifest["splits"]["val"]["oov_token_rate"] == 0.25
    assert manifest["splits"]["test"]["oov_token_count"] == 1

    val_chunk = torch.load(output_dir / "val" / "chunk_0000.pt", weights_only=True)
    assert vocabulary["<UNK>"] in val_chunk["x"]
