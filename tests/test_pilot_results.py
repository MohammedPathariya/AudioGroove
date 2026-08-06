import json
from pathlib import Path

import pytest

from src.evaluation.pilot_results import select_family_finalists, select_final_winner, summarize_report


def make_report(path: Path, family: str, profile: str, seed: int, loss: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "run_id": f"{family}-{profile}-{seed}",
                "run_phase": "finalist",
                "model_family": family,
                "model_profile": profile,
                "parameter_count": 100,
                "dataset": {"dataset_revision": "revision"},
                "experiment": {"training": {"seed": seed}},
                "training": {
                    "epochs_completed": 1,
                    "history": [
                        {
                            "epoch": 1,
                            "val": {
                                "loss": loss,
                                "perplexity": 10.0,
                                "accuracy": 0.2,
                                "top5_accuracy": 0.5,
                            },
                        }
                    ],
                },
                "test": None,
                "generation": {"success": True},
                "resources": {
                    "elapsed_seconds": 5.0,
                    "peak_gpu_memory_mb": 10.0,
                    "checkpoint_size_mb": 2.0,
                },
                "checkpoints": {"best": str(path.parent / "best.pt")},
            }
        ),
        encoding="utf-8",
    )


def test_selection_uses_validation_only_and_aggregates_seeds(tmp_path: Path) -> None:
    rows = []
    losses_by_family = {
        "gru": [2.0, 2.2, 1.8],
        "lstm": [2.5, 2.4, 2.6],
        "transformer": [2.3, 2.4, 2.2],
    }
    for family, losses in losses_by_family.items():
        for seed, loss in enumerate(losses, start=1):
            path = tmp_path / family / str(seed) / "report.json"
            make_report(path, family, "baseline", seed, loss)
            rows.append(summarize_report(path))

    aggregates, winner = select_final_winner(rows, "revision", required_seeds=3)

    assert len(aggregates) == 3
    assert winner["model_family"] == "gru"
    assert winner["training_seed"] == 3
    assert winner["test_evaluated"] is False


def test_family_selection_requires_all_profiles(tmp_path: Path) -> None:
    rows = []
    for family in ("lstm", "gru", "transformer"):
        for index, profile in enumerate(("small", "baseline", "large"), start=1):
            path = tmp_path / family / profile / "report.json"
            make_report(path, family, profile, 1, 2.0 + index / 10)
            rows.append(summarize_report(path))

    finalists = select_family_finalists(rows)

    assert len(finalists) == 3
    assert {row["model_profile"] for row in finalists} == {"small"}


def test_selection_rejects_reports_with_test_metrics(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    make_report(path, "gru", "baseline", 1, 2.0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["test"] = {"loss": 1.0}
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="contains test metrics"):
        summarize_report(path)
