# data_prep/build_vocab.py

import os
import json
import time
from collections import Counter
from music21 import converter, instrument, note, chord
from tqdm import tqdm
from utils.paths import PROCESSED_DIR, VOCAB_JSONL

def extract_notes_from_midi_simple(midi_path, instrument_filter=None):
    try:
        midi = converter.parse(midi_path)
    except Exception:
        return []

    notes = []
    try:
        parts = instrument.partitionByInstrument(midi)
    except Exception:
        parts = None

    if parts:
        for part in parts.parts:
            instr = part.getInstrument()
            instr_name = getattr(instr, "instrumentName", "") or getattr(instr, "partName", "")
            if (instrument_filter is None) or (instrument_filter in instr_name):
                for elem in part.recurse():
                    if isinstance(elem, note.Note):
                        notes.append(str(elem.pitch))
                    elif isinstance(elem, chord.Chord):
                        notes.append(".".join(str(n) for n in elem.normalOrder))
    else:
        for elem in midi.flat:
            if isinstance(elem, note.Note):
                notes.append(str(elem.pitch))
            elif isinstance(elem, chord.Chord):
                notes.append(".".join(str(n) for n in elem.normalOrder))
    return notes

def build_vocab_from_full_folder(root_dir, instrument_filter=None, threshold=50):
    all_midis = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith(('.mid', '.midi')):
                all_midis.append(os.path.join(dirpath, fname))

    total_files = len(all_midis)
    if total_files == 0:
        print(f"No MIDI files found under: {root_dir}")
        return Counter(), {}, {}

    counter = Counter()
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    if os.path.exists(VOCAB_JSONL):
        os.remove(VOCAB_JSONL)

    start_time = time.time()
    with tqdm(total=total_files, desc="Building full vocab", ncols=80) as pbar:
        for idx, midi_path in enumerate(all_midis, 1):
            tokens = extract_notes_from_midi_simple(midi_path, instrument_filter)
            counter.update(tokens)

            frequent_tokens = [tok for tok, count in counter.items() if count >= threshold]
            frequent_sorted = sorted(frequent_tokens)
            note_to_int = {tok: i for i, tok in enumerate(frequent_sorted)}
            int_to_note = {i: tok for tok, i in note_to_int.items()}

            record = {
                "file_index": idx,
                "midi_path": midi_path,
                "tokens_from_this_file": len(tokens),
                "total_unique_tokens": len(counter),
                "vocab_size_at_threshold": len(note_to_int),
                "note_to_int": note_to_int,
                "int_to_note": int_to_note
            }
            with open(VOCAB_JSONL, "a") as f:
                f.write(json.dumps(record) + "\n")

            pbar.update(1)

    elapsed = time.time() - start_time
    print(f"\n✅ Done! Processed {total_files} files in {elapsed:.1f}s.")
    print(f"Final vocabulary size (count ≥ {threshold}): {len(note_to_int)}")
    print(f"Results saved to: {VOCAB_JSONL}")

    return counter, note_to_int, int_to_note

if __name__ == "__main__":
    from utils.paths import DATA_DIR
    full_folder = os.path.join(DATA_DIR, "LMDClean")
    build_vocab_from_full_folder(
        root_dir=full_folder,
        instrument_filter=None,
        threshold=50
    )
