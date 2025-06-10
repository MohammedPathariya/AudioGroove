import os

# ----------------------------------------------------------------------------
# Core Project Structure
# ----------------------------------------------------------------------------
# Root directory of the entire project.
# We are in AudioGroove/src/utils/, so we go up two levels to find the root.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# ----------------------------------------------------------------------------
# Data-Related Paths
# ----------------------------------------------------------------------------
# Main directory for all datasets and processed files, located at the project root.
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
LMDCLEAN_DIR = os.path.join(RAW_DIR, "LMDClean")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Directory containing all seed MIDI files
SEED_FILES_DIR = os.path.join(DATA_DIR, "seed")

# Vocabulary file
VOCAB_JSONL = os.path.join(PROCESSED_DIR, "vocab_full_history.jsonl")

# ----------------------------------------------------------------------------
# Model Training & Logging Paths
# ----------------------------------------------------------------------------
# These directories are at the project root, not inside /src.
TRAINING_DIR = os.path.join(ROOT_DIR, "training")
CHECKPOINT_DIR = os.path.join(TRAINING_DIR, "checkpoints", "lstm_enhanced")
LOG_DIR = os.path.join(TRAINING_DIR, "logs", "lstm_enhanced")

# ----------------------------------------------------------------------------
# Generation & Output Paths (for standalone scripts like src/generation/generate.py)
# ----------------------------------------------------------------------------
# Default seed MIDI file for the standalone script
DEFAULT_SEED_MIDI = os.path.join(SEED_FILES_DIR, "Boom_Boom_Boom.mid")

# Default output path for the standalone script
DEFAULT_OUTPUT_MIDI = os.path.join(ROOT_DIR, "generated_music.mid")

# --- For backwards compatibility with your original scripts ---
# These variables ensure that older scripts that might use them still work.
SEED_MIDI = DEFAULT_SEED_MIDI
OUTPUT_MIDI = DEFAULT_OUTPUT_MIDI