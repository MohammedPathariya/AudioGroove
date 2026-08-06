"""Summarize pilot reports and freeze validation-only model selections."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from src.models.compact_midi_models import MODEL_FAMILIES


EXPECTED_PROFILES = {"small", "baseline", "large", "larger"}

def summarize_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    history = report["training"]["history"]
    if not history:
        raise ValueError(f"training history is empty: {path}")
    if report.get("test") is not None:
        raise ValueError(f"selection input contains test metrics: {path}")
    best = min(history, key=lambda row: row["val"]["loss"])
    return {
        "report": str(path.resolve()),
        "run_id": report["run_id"],
        "run_phase": report["run_phase"],
        "dataset_revision": report["dataset"]["dataset_revision"],
        "model_family": report["model_family"],
        "model_profile": report["model_profile"],
        "training_seed": report["experiment"]["training"]["seed"],
        "parameter_count": report["parameter_count"],
        "epochs_completed": report["training"]["epochs_completed"],
        "best_epoch": best["epoch"],
        "best_val_loss": best["val"]["loss"],
        "best_val_perplexity": best["val"]["perplexity"],
        "best_val_accuracy": best["val"]["accuracy"],
        "best_val_top5_accuracy": best["val"]["top5_accuracy"],
        "elapsed_seconds": report["resources"]["elapsed_seconds"],
        "peak_gpu_memory_mb": report["resources"]["peak_gpu_memory_mb"],
        "checkpoint_size_mb": report["resources"]["checkpoint_size_mb"],
        "generation_success": report["generation"]["success"],
        "best_checkpoint": report["checkpoints"]["best"],
    }


def load_summaries(
    run_root: Path,
    dataset_revision: str,
    phases: Iterable[str],
) -> list[dict[str, Any]]:
    allowed_phases = set(phases)
    rows = []
    for path in sorted(run_root.glob("*/*/*/report.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if (
            metadata["dataset"]["dataset_revision"] != dataset_revision
            or metadata["run_phase"] not in allowed_phases
        ):
            continue
        row = summarize_report(path)
        rows.append(row)
    if not rows:
        raise ValueError("no matching reports found")
    return sorted(rows, key=lambda row: (row["best_val_loss"], row["model_family"], row["model_profile"]))


def select_family_finalists(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = {row["model_family"] for row in rows}
    if families != set(MODEL_FAMILIES):
        raise ValueError(f"expected reports for {MODEL_FAMILIES}, found {sorted(families)}")
    finalists = []
    for family in sorted(families):
        candidates = [row for row in rows if row["model_family"] == family]
        profiles = [row["model_profile"] for row in candidates]
        if len(profiles) != len(set(profiles)):
            raise ValueError(f"duplicate sweep profile for {family}")
        if set(profiles) != EXPECTED_PROFILES:
            raise ValueError(f"incomplete sweep profiles for {family}: {sorted(profiles)}")
        winner = min(candidates, key=lambda row: (row["best_val_loss"], row["model_profile"]))
        finalists.append(
            {
                "model_family": family,
                "model_profile": winner["model_profile"],
                "selection_metric": "best_validation_loss",
                "best_val_loss": winner["best_val_loss"],
                "source_report": winner["report"],
            }
        )
    return finalists


def select_final_winner(
    rows: list[dict[str, Any]],
    dataset_revision: str,
    required_seeds: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["model_family"], row["model_profile"])].append(row)

    families = {family for family, _ in groups}
    if families != set(MODEL_FAMILIES) or len(groups) != len(MODEL_FAMILIES):
        raise ValueError("finalist reports must contain one profile for each model family")

    aggregates = []
    for (family, profile), candidates in sorted(groups.items()):
        seeds = {row["training_seed"] for row in candidates}
        if len(seeds) != len(candidates):
            raise ValueError(f"duplicate training seeds for {family}/{profile}")
        if len(seeds) != required_seeds:
            raise ValueError(
                f"{family}/{profile} has {len(seeds)} finalist seeds; requires {required_seeds}"
            )
        losses = [row["best_val_loss"] for row in candidates]
        aggregates.append(
            {
                "model_family": family,
                "model_profile": profile,
                "seed_count": len(seeds),
                "training_seeds": sorted(seeds),
                "mean_best_val_loss": statistics.fmean(losses),
                "stdev_best_val_loss": statistics.stdev(losses) if len(losses) > 1 else 0.0,
                "mean_best_val_accuracy": statistics.fmean(
                    row["best_val_accuracy"] for row in candidates
                ),
                "mean_elapsed_seconds": statistics.fmean(
                    row["elapsed_seconds"] for row in candidates
                ),
            }
        )

    winning_group = min(
        aggregates,
        key=lambda row: (
            row["mean_best_val_loss"],
            row["mean_elapsed_seconds"],
            row["model_family"],
            row["model_profile"],
        ),
    )
    winning_runs = groups[(winning_group["model_family"], winning_group["model_profile"])]
    representative = min(
        winning_runs,
        key=lambda row: (row["best_val_loss"], row["training_seed"]),
    )
    selection = {
        "dataset_revision": dataset_revision,
        "model_family": winning_group["model_family"],
        "model_profile": winning_group["model_profile"],
        "training_seed": representative["training_seed"],
        "selection_metric": "mean_best_validation_loss_across_seeds",
        "mean_best_val_loss": winning_group["mean_best_val_loss"],
        "required_seeds": required_seeds,
        "representative_rule": "lowest_validation_loss_within_winning_configuration",
        "source_report": representative["report"],
        "source_checkpoint": representative["best_checkpoint"],
        "test_evaluated": False,
    }
    return aggregates, selection


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument(
        "--phases",
        nargs="+",
        choices=("baseline", "sweep", "finalist"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--select-family-finalists", action="store_true")
    parser.add_argument("--select-winner", action="store_true")
    parser.add_argument("--required-seeds", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.required_seeds < 1:
        raise ValueError("required seeds must be positive")
    rows = load_summaries(args.run_root, args.dataset_revision, args.phases)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "comparison.csv", rows)

    if args.select_family_finalists:
        finalists = select_family_finalists(rows)
        (args.output_dir / "family_finalists.json").write_text(
            json.dumps(finalists, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.select_winner:
        aggregates, selection = select_final_winner(
            rows,
            args.dataset_revision,
            args.required_seeds,
        )
        (args.output_dir / "finalist_aggregates.json").write_text(
            json.dumps(aggregates, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.output_dir / "final_selection.json").write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
