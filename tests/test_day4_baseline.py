import torch

from src.models.compact_midi_lstm import CompactMidiLSTM


def test_compact_model_returns_one_next_token_distribution() -> None:
    model = CompactMidiLSTM(vocab_size=17, embed_dim=8, hidden_dim=12)
    inputs = torch.randint(0, 17, (4, 32), dtype=torch.long)
    targets = torch.randint(0, 17, (4,), dtype=torch.long)

    logits = model(inputs)
    loss = torch.nn.functional.cross_entropy(logits, targets)

    assert logits.shape == (4, 17)
    assert loss.ndim == 0
