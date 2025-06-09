# data_prep/split_chunks.py

import os
import torch
from natsort import natsorted
from utils.paths import CHUNKS_DIR, TRAIN_CHUNKS_DIR, VAL_CHUNKS_DIR, TEST_CHUNKS_DIR

if __name__ == "__main__":
    chunk_files = [
        os.path.join(CHUNKS_DIR, fn)
        for fn in natsorted(os.listdir(CHUNKS_DIR))
        if fn.startswith("chunk_") and fn.endswith(".pt")
    ]
    if not chunk_files:
        print("No chunk_*.pt files found in", CHUNKS_DIR)
        exit(1)

    os.makedirs(TRAIN_CHUNKS_DIR, exist_ok=True)
    os.makedirs(VAL_CHUNKS_DIR, exist_ok=True)
    os.makedirs(TEST_CHUNKS_DIR, exist_ok=True)

    for chunk_path in chunk_files:
        chunk_name = os.path.basename(chunk_path)
        X_chunk, Y_chunk = torch.load(chunk_path)
        N = X_chunk.size(0)
        perm = torch.randperm(N)
        X_chunk = X_chunk[perm]
        Y_chunk = Y_chunk[perm]

        train_end = int(0.7 * N)
        val_end   = int(0.9 * N)

        torch.save((X_chunk[:train_end], Y_chunk[:train_end]), os.path.join(TRAIN_CHUNKS_DIR, chunk_name))
        torch.save((X_chunk[train_end:val_end], Y_chunk[train_end:val_end]), os.path.join(VAL_CHUNKS_DIR, chunk_name))
        torch.save((X_chunk[val_end:], Y_chunk[val_end:]), os.path.join(TEST_CHUNKS_DIR, chunk_name))

        print(f"Split '{chunk_name}': "
              f"{train_end}→train, {val_end - train_end}→val, {N - val_end}→test")

    print("\n✅ All chunks split into train_chunks/, val_chunks/, test_chunks/.")
