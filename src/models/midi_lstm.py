# models/midi_lstm.py

import torch
import torch.nn as nn

class MidiLSTMEnhanced(nn.Module):
    """
    Enhanced LSTM + self-attention model for MIDI note sequences.
    Input:  (batch, TIMESTEPS)
    Output: logits (batch, TIMESTEPS, vocab_size)
    We'll train on the last FUTURE_STEPS time-steps.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3,
        bidirectional: bool = True,
        attn_heads: int = 8
    ):
        super().__init__()
        self.embed_dim  = embed_dim
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1

        # 1) Token embedding + dropout
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.emb_dropout = nn.Dropout(0.2)

        # 2) Stacked, (optionally) bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        # 3) Multi-head self-attention over LSTM outputs
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * self.num_directions,
            num_heads=attn_heads,
            dropout=0.1,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim * self.num_directions)

        # 4) Final classifier
        self.fc = nn.Linear(hidden_dim * self.num_directions, vocab_size)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (batch, TIMESTEPS)
        embeds = self.embedding(x)         # (batch, TIMESTEPS, embed_dim)
        embeds = self.emb_dropout(embeds)

        outputs, _ = self.lstm(embeds)
        # outputs: (batch, TIMESTEPS, hidden_dim * num_directions)

        # Self-attention
        attn_out, _ = self.attn(outputs, outputs, outputs, need_weights=False)
        outputs = self.layer_norm(outputs + attn_out)

        logits = self.fc(outputs)          # (batch, TIMESTEPS, vocab_size)
        return logits
