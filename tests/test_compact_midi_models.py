from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.compact_midi_models import (
    CompactCausalTransformer,
    build_compact_model,
    count_parameters,
)


@pytest.mark.parametrize("family", ["lstm", "gru", "transformer"])
def test_compact_models_return_one_distribution_and_backpropagate(family: str) -> None:
    parameters = {
        "lstm": {"embed_dim": 8, "hidden_dim": 12, "num_layers": 1, "dropout": 0.0},
        "gru": {"embed_dim": 8, "hidden_dim": 12, "num_layers": 1, "dropout": 0.0},
        "transformer": {
            "d_model": 12,
            "num_layers": 1,
            "num_heads": 3,
            "ffn_dim": 24,
            "dropout": 0.0,
            "max_sequence_length": 8,
        },
    }[family]
    model = build_compact_model(family, vocab_size=17, parameters=parameters)
    inputs = torch.randint(0, 17, (4, 8), dtype=torch.long)
    targets = torch.randint(0, 17, (4,), dtype=torch.long)

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)
    loss.backward()

    assert logits.shape == (4, 17)
    assert count_parameters(model) > 0
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_transformer_receives_strict_causal_mask() -> None:
    class MaskRecorder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mask: torch.Tensor | None = None

        def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
            self.mask = mask
            return inputs

    model = CompactCausalTransformer(
        vocab_size=11,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        dropout=0.0,
        max_sequence_length=4,
    )
    recorder = MaskRecorder()
    model.encoder = recorder

    model(torch.randint(0, 11, (2, 4), dtype=torch.long))

    assert recorder.mask is not None
    assert torch.equal(
        recorder.mask,
        torch.tensor(
            [
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
                [False, False, False, False],
            ]
        ),
    )


def test_transformer_rejects_context_beyond_position_limit() -> None:
    model = CompactCausalTransformer(
        vocab_size=11,
        d_model=8,
        num_layers=1,
        num_heads=2,
        ffn_dim=16,
        max_sequence_length=4,
    )

    with pytest.raises(ValueError, match="exceeds configured maximum"):
        model(torch.randint(0, 11, (2, 5), dtype=torch.long))
