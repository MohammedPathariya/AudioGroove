from pathlib import Path

import torch
from torch import nn

from src.models.compact_midi_models import build_compact_model, count_parameters
from src.training.chunk_stream import iter_chunk_batches
from src.training.pilot_comparison import load_experiment_config, run_epoch


def write_chunk(path: Path, first_value: int) -> None:
    inputs = torch.arange(first_value, first_value + 24, dtype=torch.long).reshape(6, 4)
    targets = torch.arange(first_value, first_value + 6, dtype=torch.long)
    torch.save({"x": inputs, "y": targets}, path)


def test_chunk_stream_is_deterministic_and_bounded(tmp_path: Path) -> None:
    write_chunk(tmp_path / "chunk_0000.pt", 0)
    write_chunk(tmp_path / "chunk_0001.pt", 100)

    first = list(
        iter_chunk_batches(tmp_path, 2, shuffle=True, seed=17, max_batches=3)
    )
    second = list(
        iter_chunk_batches(tmp_path, 2, shuffle=True, seed=17, max_batches=3)
    )

    assert len(first) == 3
    assert all(inputs.shape == (2, 4) and targets.shape == (2,) for inputs, targets in first)
    assert all(
        torch.equal(first_inputs, second_inputs) and torch.equal(first_targets, second_targets)
        for (first_inputs, first_targets), (second_inputs, second_targets) in zip(first, second)
    )


def test_baseline_configuration_defines_all_model_families() -> None:
    config_path = Path("training/configs/pilot_experiments.json")

    lstm = load_experiment_config(config_path, "lstm")
    gru = load_experiment_config(config_path, "gru")
    transformer = load_experiment_config(config_path, "transformer")

    assert lstm.training.sequence_length == gru.training.sequence_length == 32
    assert lstm.training.batch_size == transformer.training.batch_size == 64
    assert lstm.training.learning_rate == gru.training.learning_rate == 0.001
    assert transformer.training.learning_rate == 0.0003
    assert transformer.model_parameters["num_heads"] == 4
    assert transformer.model_parameters["max_sequence_length"] == 32


def test_sweep_profiles_change_architecture_without_changing_budget() -> None:
    config_path = Path("training/configs/pilot_experiments.json")
    profile_names = ("small", "baseline", "large", "larger")

    for family in ("lstm", "gru", "transformer"):
        experiments = [
            load_experiment_config(config_path, family, profile)
            for profile in profile_names
        ]
        parameter_counts = [
            count_parameters(
                build_compact_model(family, 18_849, experiment.model_parameters)
            )
            for experiment in experiments
        ]

        assert parameter_counts == sorted(parameter_counts)
        assert len(set(parameter_counts)) == len(profile_names)
        assert all(
            experiment.training == experiments[0].training
            for experiment in experiments[1:]
        )


def test_epoch_metrics_count_each_example_once() -> None:
    class FixedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(inputs.shape[0], 7)
            logits[:, 0] = 1.0
            return logits + self.anchor

    batches = iter(
        [
            (torch.zeros(3, 4, dtype=torch.long), torch.zeros(3, dtype=torch.long)),
            (torch.zeros(2, 4, dtype=torch.long), torch.zeros(2, dtype=torch.long)),
        ]
    )

    metrics = run_epoch(
        FixedModel(),
        batches,
        nn.CrossEntropyLoss(),
        torch.device("cpu"),
        optimizer=None,
        scaler=torch.cuda.amp.GradScaler(enabled=False),
        gradient_accumulation_steps=1,
        gradient_clip_norm=1.0,
        amp=False,
        phase="metric test",
    )

    assert metrics["examples"] == 5
    assert metrics["batches"] == 2
    assert metrics["accuracy"] == 1.0
