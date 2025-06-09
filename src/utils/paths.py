import os

# Root path of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

# Data directories
DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
SEED_DIR = os.path.join(DATA_DIR, "seed")
LMDCLEAN_DIR = os.path.join(RAW_DIR, "LMDClean")
CHUNKS_DIR = os.path.join(DATA_DIR, "chunks")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

# Chunked merged files (not split into subfolders)
TRAIN_PT = os.path.join(PROCESSED_DIR, "train.pt")
VAL_PT = os.path.join(PROCESSED_DIR, "val.pt")
TEST_PT = os.path.join(PROCESSED_DIR, "test.pt")

# Vocab file
VOCAB_JSONL = os.path.join(PROCESSED_DIR, "vocab_full_history.jsonl")

# Model and logging
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "training", "checkpoints", "lstm_enhanced")
LOG_DIR = os.path.join(ROOT_DIR, "training", "logs", "lstm_enhanced")

# Generation
SEED_MIDI = os.path.join(SEED_DIR, "Boom_Boom_Boom.mid")
OUTPUT_MIDI = os.path.join(ROOT_DIR, "generated_music1.mid")