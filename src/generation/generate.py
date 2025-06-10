import os
import sys
import json
import re
import torch
from tqdm import tqdm
from mido import MidiFile, MidiTrack, Message
from music21 import pitch

# ─── 1. Add project's 'src' directory to sys.path ───
src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, src_root)

# ─── 2. Import custom modules AFTER updating the path ───
from models.midi_lstm import MidiLSTMEnhanced
from data_prep.extract_midi import extract_notes_from_midi
from utils.paths import CHECKPOINT_DIR, VOCAB_JSONL, SEED_MIDI, OUTPUT_MIDI

# ─── Settings ───
TIMESTEPS    = 32
GENERATE_LEN = 200
TEMPERATURE  = 1.0
TOP_K        = 50
NOTE_DURATION = 480  # MIDI ticks

# ─── Resolve paths from utils.paths ───
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_epoch_03.pt")

# ─── Sanity check files ───
print("Looking for files...")
for name, path in [("checkpoint", CHECKPOINT_PATH), ("vocab", VOCAB_JSONL), ("seed MIDI", SEED_MIDI)]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Cannot find {name} at: {path}")
print("✅ All necessary files found.")


# ─── Load vocab (MODIFIED SECTION) ───
# This new logic reads the entire file and extracts the JSON from the LAST line.
print(f"Reading full vocabulary history from: {VOCAB_JSONL}")
note_to_int, int_to_note = {}, {}
with open(VOCAB_JSONL, "r") as f:
    last_line = None
    for line in f:
        # This will efficiently loop through and keep only the last non-empty line
        if line.strip():
            last_line = line

    if last_line is None:
        raise ValueError(f"Vocabulary file is empty or contains no valid lines: {VOCAB_JSONL}")

    # Parse the JSON from the last line
    rec = json.loads(last_line)
    note_to_int = rec["note_to_int"]
    int_to_note = {int(k): v for k, v in rec["int_to_note"].items()}

VOCAB_SIZE = len(note_to_int)
print(f"Vocabulary size from file (last line): {VOCAB_SIZE}")


# ─── Load model ───
print(f"Loading model configured for {VOCAB_SIZE} tokens...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MidiLSTMEnhanced(
    vocab_size=VOCAB_SIZE,
    embed_dim=256, hidden_dim=512,
    num_layers=3, dropout=0.3,
    bidirectional=True, attn_heads=8
).to(device)

# This line will now succeed because VOCAB_SIZE will be read from the final record.
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded onto {device}.")


# ... (The rest of the script is identical) ...


# ─── Convert seed MIDI to token IDs ───
def midi_to_token_ids(path):
    tokens = extract_notes_from_midi(path, instrument_filter=None)
    return [note_to_int[t] for t in tokens if t in note_to_int]

seed_ids = midi_to_token_ids(SEED_MIDI)
if len(seed_ids) < TIMESTEPS:
    raise RuntimeError(f"Seed MIDI only yielded {len(seed_ids)} tokens (<{TIMESTEPS})")
window = seed_ids[:TIMESTEPS]

# ─── Sampling helper ───
def sample_next(logits, temperature=1.0, top_k=None):
    logits = logits / (temperature or 1e-9)
    if top_k:
        values, indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, -float("Inf"))
        mask[indices] = logits[indices]
        logits = mask
    probs = torch.softmax(logits, dim=0)
    return torch.multinomial(probs, num_samples=1).item()

# ─── Generate ───
generated = window.copy()
for _ in tqdm(range(GENERATE_LEN), desc="Generating", ncols=80):
    inp = torch.LongTensor([generated[-TIMESTEPS:]]).to(device)
    with torch.no_grad():
        logits = model(inp)[0, -1, :]
    nxt = sample_next(logits, TEMPERATURE, TOP_K)
    generated.append(nxt)

# ─── Write to MIDI ───
PITCH_RE = re.compile(r"^[A-G](?:#|b|-)?\d+$")
mid = MidiFile()
track = MidiTrack()
mid.tracks.append(track)
track.append(Message('program_change', program=0, time=0))

for tok_id in generated:
    note_str = int_to_note.get(tok_id)
    if not note_str or "." in note_str or not PITCH_RE.match(note_str):
        continue
    try:
        midi_num = pitch.Pitch(note_str).midi
        track.append(Message('note_on', note=midi_num, velocity=64, time=0))
        track.append(Message('note_off', note=midi_num, velocity=64, time=NOTE_DURATION))
    except Exception as e:
        continue

mid.save(OUTPUT_MIDI)
print(f"✅ Wrote generated MIDI to: {OUTPUT_MIDI}")