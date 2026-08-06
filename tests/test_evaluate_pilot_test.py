import copy

import pytest

from src.evaluation.evaluate_pilot_test import validate_selection


def valid_inputs():
    manifest = {"dataset_revision": "revision"}
    report = {
        "model_family": "gru",
        "model_profile": "baseline",
        "experiment": {"training": {"seed": 17}},
        "dataset": {"dataset_revision": "revision"},
        "vocabulary_hash": "vocab-hash",
        "checkpoints": {"best": "/tmp/best.pt"},
        "test": None,
    }
    selection = {
        "dataset_revision": "revision",
        "model_family": "gru",
        "model_profile": "baseline",
        "training_seed": 17,
        "source_checkpoint": "/tmp/best.pt",
        "test_evaluated": False,
    }
    return selection, report, manifest


def test_final_test_selection_must_match_source_report() -> None:
    selection, report, manifest = valid_inputs()

    validate_selection(selection, report, manifest, "vocab-hash")

    changed = copy.deepcopy(selection)
    changed["training_seed"] = 18
    with pytest.raises(ValueError, match="selection manifest mismatch"):
        validate_selection(changed, report, manifest, "vocab-hash")


def test_final_test_rejects_prior_test_metrics() -> None:
    selection, report, manifest = valid_inputs()
    report["test"] = {"loss": 1.0}

    with pytest.raises(ValueError, match="already contains test metrics"):
        validate_selection(selection, report, manifest, "vocab-hash")
