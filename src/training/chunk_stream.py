"""Deterministic, bounded batch streaming for prepared pilot chunks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import torch

from src.data_prep.midi_representation import BoundedChunkDataset


def iter_chunk_batches(
    chunk_dir: str | Path,
    batch_size: int,
    *,
    shuffle: bool,
    seed: int,
    max_batches: int | None = None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Yield batches while keeping one prepared chunk resident at a time."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_batches is not None and max_batches < 1:
        raise ValueError("max_batches must be positive or None")
    paths = sorted(Path(chunk_dir).glob("chunk_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No chunk_*.pt files found in {chunk_dir}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if shuffle:
        order = torch.randperm(len(paths), generator=generator).tolist()
        paths = [paths[index] for index in order]

    emitted = 0
    for path in paths:
        chunk = BoundedChunkDataset._read_chunk(path)
        inputs = chunk["x"]
        targets = chunk["y"]
        indices = (
            torch.randperm(inputs.shape[0], generator=generator)
            if shuffle
            else torch.arange(inputs.shape[0])
        )
        for start in range(0, inputs.shape[0], batch_size):
            selection = indices[start : start + batch_size]
            yield inputs[selection], targets[selection]
            emitted += 1
            if max_batches is not None and emitted >= max_batches:
                return


__all__ = ["iter_chunk_batches"]
