# data_prep/merge_train_chunks.py

import os
import torch
from natsort import natsorted
from utils.paths import TRAIN_CHUNKS_DIR, TRAIN_PT

if __name__ == "__main__":
    chunk_files = [
        os.path.join(TRAIN_CHUNKS_DIR, fn)
        for fn in natsorted(os.listdir(TRAIN_CHUNKS_DIR))
        if fn.endswith(".pt")
    ]
    if not chunk_files:
        print(f"No *.pt files found in {TRAIN_CHUNKS_DIR}")
        exit(1)

    X_list, Y_list = [], []
    print(f"Loading {len(chunk_files)} train‐chunk files…")
    for path in chunk_files:
        Xc, Yc = torch.load(path)
        X_list.append(Xc)
        Y_list.append(Yc)

    X_all = torch.cat(X_list, dim=0)
    Y_all = torch.cat(Y_list, dim=0)
    torch.save((X_all, Y_all), TRAIN_PT)
    print(f"Saved merged train.pt with {X_all.size(0)} windows at '{TRAIN_PT}'")
