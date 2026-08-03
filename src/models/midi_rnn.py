# src/models/midi_rnn.py

import torch
import torch.nn as nn

class MidiRNN(nn.Module):
    """
    A simple baseline RNN/LSTM model without bidirectionality or self-attention.
    This serves as a benchmark to prove the effectiveness of more complex models.
    """
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.3
    ):
        super().__init__()
        # 1) Token embedding
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # 2) A standard, unidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False # The key difference
        )
        
        # 3) Final classifier
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (batch, TIMESTEPS)
        embeds = self.embedding(x)  # (batch, TIMESTEPS, embed_dim)
        outputs, _ = self.lstm(embeds) # (batch, TIMESTEPS, hidden_dim)
        logits = self.fc(outputs)   # (batch, TIMESTEPS, vocab_size)
        return logits