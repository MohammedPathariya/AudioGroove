# data_prep/build_dataset_chunks.py

import json
import os
import numpy as np
import torch
from tqdm import tqdm
from data_prep.extract_midi import extract_notes_from_midi
from utils.paths import VOCAB_JSONL, CHUNKS_DIR

def load_vocab_from_jsonl(jsonl_path=VOCAB_JSONL):
    note_to_int, int_to_note, midi_paths = {}, {}, []
    with open(jsonl_path, "r") as f:
        for line in f:
            record = json.loads(line)
            midi_paths.append(record["midi_path"])
            note_to_int = record["note_to_int"]
            int_to_note = {int(k): v for k, v in record["int_to_note"].items()}
    return note_to_int, int_to_note, midi_paths

def midi_to_int_sequence(midi_path, note_to_int, instrument_filter=None):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tokens = extract_notes_from_midi(midi_path, instrument_filter)
    return [note_to_int[t] for t in tokens if t in note_to_int]

def build_sequences_from_intseq(int_seq, timesteps=32, future_steps=8):
    seq_pairs = []
    total_length = len(int_seq)
    max_start = total_length - (timesteps + future_steps) + 1
    for i in range(max_start):
        x_seq = int_seq[i : i + timesteps]
        y_seq = int_seq[i + timesteps : i + timesteps + future_steps]
        seq_pairs.append((x_seq, y_seq))
    return seq_pairs

if __name__ == "__main__":
    timesteps = 32
    future_steps = 8
    SAVE_EVERY_N_WINDOWS = 200_000

    note_to_int, int_to_note, all_midis = load_vocab_from_jsonl()

    X_buffer, Y_buffer = [], []
    window_count, chunk_idx = 0, 0

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    for midi_path in tqdm(all_midis, desc="Building & chunking", ncols=80):
        int_seq = midi_to_int_sequence(midi_path, note_to_int, instrument_filter=None)
        if len(int_seq) < (timesteps + future_steps):
            continue

        pairs = build_sequences_from_intseq(int_seq, timesteps, future_steps)
        for x_seq, y_seq in pairs:
            X_buffer.append(x_seq)
            Y_buffer.append(y_seq)
            window_count += 1

            if window_count >= SAVE_EVERY_N_WINDOWS:
                X_arr = np.array(X_buffer, dtype=np.int64)
                Y_arr = np.array(Y_buffer, dtype=np.int64)
                out_path = os.path.join(CHUNKS_DIR, f"chunk_{chunk_idx}.pt")
                torch.save((torch.from_numpy(X_arr), torch.from_numpy(Y_arr)), out_path)

                chunk_idx += 1
                X_buffer.clear()
                Y_buffer.clear()
                window_count = 0

    if window_count > 0:
        X_arr = np.array(X_buffer, dtype=np.int64)
        Y_arr = np.array(Y_buffer, dtype=np.int64)
        out_path = os.path.join(CHUNKS_DIR, f"chunk_{chunk_idx}.pt")
        torch.save((torch.from_numpy(X_arr), torch.from_numpy(Y_arr)), out_path)

    print(f"\n🔖 Created {chunk_idx + 1} chunk file(s) in '{CHUNKS_DIR}'.")
