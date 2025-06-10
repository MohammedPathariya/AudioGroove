import os
import sys
import json
import re
import torch
import random
import io
import traceback
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from mido import MidiFile, MidiTrack, Message
from music21 import pitch

# ─── Flask App Initialization ───
app = Flask(__name__)
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:8000")
CORS(app, origins=[FRONTEND_URL, "http://127.0.0.1:8000", "http://localhost:5173"])

# ─── 1. Corrected Path Definitions ───
# Since this script is in /backend/, we go up one level ('..') to get the project's root directory.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Define all other paths relative to this correct PROJECT_ROOT
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
TRAINING_DIR = os.path.join(PROJECT_ROOT, "training")

SEED_FILES_DIR = os.path.join(DATA_DIR, "seed")
VOCAB_JSONL = os.path.join(DATA_DIR, "processed", "vocab_full_history.jsonl")
CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "checkpoints", "lstm_enhanced")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_epoch_03.pt")

# ─── 2. Add the project's 'src' directory to Python's path ───
sys.path.insert(0, SRC_DIR)

# ─── 3. Import custom modules AFTER updating the path ───
from models.midi_lstm import MidiLSTMEnhanced
from data_prep.extract_midi import extract_notes_from_midi

# --- (The rest of the file is identical to the previous "production-ready" version) ---

# ─── Global Settings & Sanity Checks ───
TIMESTEPS    = 32
GENERATE_LEN = 100
TEMPERATURE  = 1.0
TOP_K        = 50
NOTE_DURATION = 480

for path, name in [(CHECKPOINT_PATH, "checkpoint"), (VOCAB_JSONL, "vocabulary"), (SEED_FILES_DIR, "seed directory")]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Server HALTED on startup: Cannot find {name} at: {path}")

# ─── Load Model and Vocab (on server startup for efficiency) ───
print("🚀 Server starting: Loading model and vocabulary...")
model = None # Initialize model as None
try:
    with open(VOCAB_JSONL, "r") as f:
        last_line = None
        for line in f:
            if line.strip():
                last_line = line
    if last_line is None:
        raise ValueError("Vocabulary file is empty.")

    rec = json.loads(last_line)
    note_to_int = rec["note_to_int"]
    int_to_note = {int(k): v for k, v in rec["int_to_note"].items()}
    VOCAB_SIZE = len(note_to_int)
    print(f"✅ Vocabulary loaded. Size: {VOCAB_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MidiLSTMEnhanced(
        vocab_size=VOCAB_SIZE, embed_dim=256, hidden_dim=512,
        num_layers=3, dropout=0.3, bidirectional=True, attn_heads=8
    ).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded onto {device} and ready for requests.")
except Exception as e:
    print(f"🔥🔥🔥 FATAL ERROR DURING STARTUP: {e}")
    traceback.print_exc()

@app.route("/")
def health_check():
    return jsonify(
        status="ok",
        message="AudioGroove backend is running.",
        model_loaded= (model is not None)
    ), 200

@app.route("/generate", methods=["POST"])
def generate_music_endpoint():
    if model is None:
        return jsonify({"error": "Model is not loaded. Server may have failed to start."}), 503

    seed_file = None
    if 'seed_midi' in request.files and request.files['seed_midi'].filename != '':
        seed_file = request.files['seed_midi']
        print(f"🎤 Received user seed: {seed_file.filename}")
    else:
        print("🎧 No user seed provided. Using random seed from server.")
        seed_options = [f for f in os.listdir(SEED_FILES_DIR) if f.lower().endswith(('.mid', '.midi'))]
        if not seed_options:
            return jsonify({"error": "The server's seed directory is empty."}), 500
        random_seed_name = random.choice(seed_options)
        seed_file = os.path.join(SEED_FILES_DIR, random_seed_name)
        print(f"🎧 Selected random seed: {random_seed_name}")

    try:
        seed_ids = midi_to_token_ids(seed_file)
        if len(seed_ids) < TIMESTEPS:
            return jsonify({"error": f"Seed MIDI is too short. It has {len(seed_ids)} tokens, but {TIMESTEPS} are required."}), 400
        window = seed_ids[:TIMESTEPS]
    except Exception as e:
        print(f"[ERROR] Could not process seed file. Type: {type(e).__name__}, Details: {e}")
        return jsonify({"error": "Could not process the provided seed MIDI file."}), 400

    try:
        generated = window.copy()
        for _ in range(GENERATE_LEN):
            inp = torch.LongTensor([generated[-TIMESTEPS:]]).to(device)
            with torch.no_grad():
                logits = model(inp)[0, -1, :]
            nxt = sample_next(logits, TEMPERATURE, TOP_K)
            generated.append(nxt)
    except Exception as e:
        print(f"[ERROR] Model generation failed. Type: {type(e).__name__}, Details: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred during music generation."}), 500

    midi_io = io.BytesIO()
    try:
        PITCH_RE = re.compile(r"^[A-G](?:#|b|-)?\d+$")
        mid = MidiFile()
        track = MidiTrack()
        mid.tracks.append(track)
        track.append(Message('program_change', program=0, time=0))
        for tok_id in generated:
            note_str = int_to_note.get(tok_id)
            if note_str and PITCH_RE.match(note_str):
                midi_num = pitch.Pitch(note_str).midi
                track.append(Message('note_on', note=midi_num, velocity=64, time=0))
                track.append(Message('note_off', note=midi_num, velocity=64, time=NOTE_DURATION))
        mid.save(file=midi_io)
        midi_io.seek(0)
    except Exception as e:
        print(f"[ERROR] Failed to write final MIDI file. Type: {type(e).__name__}, Details: {e}")
        return jsonify({"error": "Failed to construct the final MIDI file after generation."}), 500

    print("✅ Generation complete. Sending MIDI file to user.")
    return send_file(midi_io, mimetype='audio/midi', as_attachment=True, download_name='generated_music.mid')

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