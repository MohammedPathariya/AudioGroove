# data_prep/extract_midi.py

import os
from music21 import converter, instrument, note, chord
from utils.paths import DATA_DIR

def find_all_mid_files(root_dir):
    midi_paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith((".mid", ".midi")):
                midi_paths.append(os.path.join(dirpath, fname))
    return midi_paths

def extract_notes_from_midi(midi_path, instrument_filter=None):
    print(f"Parsing: {midi_path}")
    try:
        midi = converter.parse(midi_path)
    except Exception as e:
        print(f"  ⚠️  Failed to parse {midi_path}: {e}")
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

if __name__ == "__main__":
    root = os.path.join(DATA_DIR, "LMDClean")  # Dynamic path now
    all_midis = find_all_mid_files(root)
    print(f"Found {len(all_midis)} MIDI files.\n")
    for path in all_midis[:5]:
        tokens = extract_notes_from_midi(path, instrument_filter=None)
        print(f"  → Extracted {len(tokens)} tokens from {path}")
