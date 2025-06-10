import os
import sys
import json
import re
import torch
import random
import io
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from mido import MidiFile, MidiTrack, Message
from music21 import pitch

# ─── Flask App Initialization ───
app = Flask(__name__)
CORS(app) # Allows the frontend to make requests

# ─── 1. Standalone Path Definitions ───
# Since this script is in /backend/, we go up one level ('..') to find the project root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data")
SEED_FILES_DIR = os.path.join(DATA_DIR, "seed")
VOCAB_JSONL = os.path.join(DATA_DIR, "processed", "vocab_full_history.jsonl")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "training", "checkpoints", "lstm_enhanced")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_epoch_03.pt")


# ─── 2. Add the 'src' directory to Python's path ───
# This is necessary to find your custom modules like `models` and `data_prep`.
src_dir = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, src_dir)


# ─── 3. Import custom modules AFTER updating the path ───
from models.midi_lstm import MidiLSTMEnhanced
from data_prep.extract_midi import extract_notes_from_midi


# ─── Settings ───
TIMESTEPS    = 32
GENERATE_LEN = 200
TEMPERATURE  = 1.0
TOP_K        = 50
NOTE_DURATION = 480

# ─── Sanity Checks ───
for path, name in [(CHECKPOINT_PATH, "checkpoint"), (VOCAB_JSONL, "vocabulary"), (SEED_FILES_DIR, "seed directory")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Server HALTED: Cannot find {name} at: {path}")


# ─── Load Vocab & Model (on server startup) ───
print("🚀 Server starting...")

# Use the successful "last-line" loading logic
print(f"Reading full vocabulary history from: {VOCAB_JSONL}")
note_to_int, int_to_note = {}, {}
with open(VOCAB_JSONL, "r") as f:
    last_line = None
    for line in f:
        if line.strip(): # Find last non-empty line
            last_line = line
    if last_line is None:
        raise ValueError(f"Vocabulary file is empty: {VOCAB_JSONL}")

    rec = json.loads(last_line)
    note_to_int = rec["note_to_int"]
    int_to_note = {int(k): v for k, v in rec["int_to_note"].items()}

VOCAB_SIZE = len(note_to_int)
print(f"Vocabulary size from file (last line): {VOCAB_SIZE}")

# Initialize and load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MidiLSTMEnhanced(
    vocab_size=VOCAB_SIZE, embed_dim=256, hidden_dim=512,
    num_layers=3, dropout=0.3, bidirectional=True, attn_heads=8
).to(device)

print(f"Loading checkpoint with model configured for {VOCAB_SIZE} tokens...")
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
print(f"✅ Model loaded onto {device} and ready for requests.")


# ─── Helper Functions ───
def midi_to_token_ids(path_or_file):
    tokens = extract_notes_from_midi(path_or_file, instrument_filter=None)
    return [note_to_int[t] for t in tokens if t in note_to_int]

def sample_next(logits, temperature=1.0, top_k=None):
    logits = logits / (temperature + 1e-9)
    if top_k and top_k > 0:
        v, i = torch.topk(logits, top_k)
        mask = torch.full_like(logits, -float("Inf"))
        mask.scatter_(0, i, v)
        logits = mask
    probs = torch.softmax(logits, dim=0)
    return torch.multinomial(probs, num_samples=1).item()


# ─── API Endpoint for Music Generation ───
@app.route("/generate", methods=["POST"])
def generate_music_endpoint():
    # Handle user-provided seed file or fallback to a random one
    seed_file = None
    if 'seed_midi' in request.files and request.files['seed_midi'].filename != '':
        seed_file = request.files['seed_midi']
        print(f"🎤 Received user seed: {seed_file.filename}")
    else:
        print("🎧 No user seed provided. Using random seed from server.")
        seed_options = [f for f in os.listdir(SEED_FILES_DIR) if f.lower().endswith(('.mid', '.midi'))]
        random_seed_name = random.choice(seed_options)
        seed_file = os.path.join(SEED_FILES_DIR, random_seed_name)
        print(f"🎧 Selected random seed: {random_seed_name}")

    # Convert seed to tokens
    try:
        seed_ids = midi_to_token_ids(seed_file)
        if len(seed_ids) < TIMESTEPS:
            return jsonify({"error": f"Seed MIDI is too short. It has {len(seed_ids)} tokens, but {TIMESTEPS} are required."}), 400
        window = seed_ids[:TIMESTEPS]
    except Exception as e:
        return jsonify({"error": f"Could not process the provided seed MIDI file: {e}"}), 400

    # Generate music
    generated = window.copy()
    for _ in range(GENERATE_LEN):
        inp = torch.LongTensor([generated[-TIMESTEPS:]]).to(device)
        with torch.no_grad():
            logits = model(inp)[0, -1, :]
        nxt = sample_next(logits, TEMPERATURE, TOP_K)
        generated.append(nxt)

    # Convert generated tokens back to a MIDI file in memory
    PITCH_RE = re.compile(r"^[A-G](?:#|b|-)?\d+$")
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    track.append(Message('program_change', program=0, time=0))
    for tok_id in generated:
        note_str = int_to_note.get(tok_id)
        if note_str and PITCH_RE.match(note_str):
            try:
                midi_num = pitch.Pitch(note_str).midi
                track.append(Message('note_on', note=midi_num, velocity=64, time=0))
                track.append(Message('note_off', note=midi_num, velocity=64, time=NOTE_DURATION))
            except Exception:
                continue
    
    # Save to a byte stream and send back to the user
    midi_io = io.BytesIO()
    mid.save(file=midi_io)
    midi_io.seek(0)

    print("✅ Generation complete. Sending MIDI file to user.")
    return send_file(
        midi_io,
        mimetype='audio/midi',
        as_attachment=True,
        download_name='generated_music.mid'
    )

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=True)