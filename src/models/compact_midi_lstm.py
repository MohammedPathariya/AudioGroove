"""Compact unidirectional LSTM for one-step symbolic MIDI prediction."""

from __future__ import annotations

import torch
from torch import nn


class CompactMidiLSTM(nn.Module):
    """Predict the token after a fixed context using only past context."""

    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, num_layers: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.config = {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "bidirectional": False,
            "objective": "next-token; logits from final context position",
        }

    def forward(self, tokens: torch.LongTensor) -> torch.Tensor:
        outputs, _ = self.lstm(self.embedding(tokens))
        return self.output(outputs[:, -1, :])


__all__ = ["CompactMidiLSTM"]
