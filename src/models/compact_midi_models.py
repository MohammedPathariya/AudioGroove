"""Compact autoregressive model families for the frozen MIDI pilot."""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import nn


MODEL_FAMILIES = ("lstm", "gru", "transformer")

BASELINE_MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "lstm": {
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.1,
    },
    "gru": {
        "embed_dim": 128,
        "hidden_dim": 256,
        "num_layers": 1,
        "dropout": 0.1,
    },
    "transformer": {
        "d_model": 256,
        "num_layers": 2,
        "num_heads": 4,
        "ffn_dim": 512,
        "dropout": 0.1,
        "max_sequence_length": 32,
    },
}


def _validate_tokens(tokens: torch.Tensor) -> None:
    if tokens.ndim != 2:
        raise ValueError(f"expected tokens with shape [batch, sequence], got {tuple(tokens.shape)}")
    if tokens.dtype != torch.long:
        raise TypeError(f"expected torch.long tokens, got {tokens.dtype}")


class CompactMidiLSTM(nn.Module):
    """Unidirectional LSTM that predicts one token after the context."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or embed_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("vocabulary, embedding, hidden, and layer sizes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.config = {
            "family": "lstm",
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": False,
            "objective": "next-token; logits from final context position",
        }

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _validate_tokens(tokens)
        outputs, _ = self.lstm(self.input_dropout(self.embedding(tokens)))
        return self.output(self.output_dropout(outputs[:, -1, :]))


class CompactMidiGRU(nn.Module):
    """Unidirectional GRU with the same next-token interface as the LSTM."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if vocab_size < 2 or embed_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("vocabulary, embedding, hidden, and layer sizes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output_dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.config = {
            "family": "gru",
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": False,
            "objective": "next-token; logits from final context position",
        }

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _validate_tokens(tokens)
        outputs, _ = self.gru(self.input_dropout(self.embedding(tokens)))
        return self.output(self.output_dropout(outputs[:, -1, :]))


class CompactCausalTransformer(nn.Module):
    """Decoder-style Transformer encoder with an explicit causal mask."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_sequence_length: int = 32,
    ) -> None:
        super().__init__()
        if min(vocab_size, d_model, num_layers, num_heads, ffn_dim, max_sequence_length) < 1:
            raise ValueError("Transformer dimensions must be positive")
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_sequence_length, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.config = {
            "family": "transformer",
            "vocab_size": vocab_size,
            "d_model": d_model,
            "num_layers": num_layers,
            "num_heads": num_heads,
            "ffn_dim": ffn_dim,
            "dropout": dropout,
            "max_sequence_length": max_sequence_length,
            "causal": True,
            "objective": "next-token; logits from final causally masked context position",
        }

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _validate_tokens(tokens)
        sequence_length = tokens.shape[1]
        if sequence_length > self.max_sequence_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds configured maximum "
                f"{self.max_sequence_length}"
            )
        positions = torch.arange(sequence_length, device=tokens.device)
        embeddings = self.token_embedding(tokens) * math.sqrt(self.config["d_model"])
        embeddings = self.embedding_dropout(embeddings + self.position_embedding(positions)[None, :, :])
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=tokens.device),
            diagonal=1,
        )
        encoded = self.encoder(embeddings, mask=causal_mask)
        return self.output(self.final_norm(encoded[:, -1, :]))


def build_compact_model(
    family: str,
    vocab_size: int,
    parameters: Mapping[str, Any] | None = None,
) -> nn.Module:
    """Build one supported family from an explicit, serializable configuration."""
    normalized = family.lower()
    if normalized not in MODEL_FAMILIES:
        raise ValueError(f"unsupported model family {family!r}; choose from {MODEL_FAMILIES}")
    config = dict(BASELINE_MODEL_CONFIGS[normalized])
    if parameters:
        config.update(parameters)
    model_class = {
        "lstm": CompactMidiLSTM,
        "gru": CompactMidiGRU,
        "transformer": CompactCausalTransformer,
    }[normalized]
    return model_class(vocab_size=vocab_size, **config)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


__all__ = [
    "BASELINE_MODEL_CONFIGS",
    "MODEL_FAMILIES",
    "CompactCausalTransformer",
    "CompactMidiGRU",
    "CompactMidiLSTM",
    "build_compact_model",
    "count_parameters",
]
