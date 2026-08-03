# training/train_rnn.py

import os, json, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# Corrected line
from src.models.midi_rnn import MidiRNN

# Make sure to import ROOT_DIR
from src.utils.paths import TRAIN_PT, VAL_PT, VOCAB_JSONL, ROOT_DIR

# ───── Config ─────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
LR = 1e-3
NUM_EPOCHS = 50
TIMESTEPS = 32
FUTURE = 8
GRAD_CLIP = 5.0

# ... (load_vocab and MidiDataset code is fine) ...
def load_vocab(jsonl):
    nt2i, i2nt = {}, {}
    with open(jsonl) as f:
        # This logic correctly gets the final vocab from the jsonl file
        last_line = [line for line in f if line.strip()][-1]
    rec = json.loads(last_line)
    nt2i = rec["note_to_int"]
    i2nt = {int(k): v for k, v in rec["int_to_note"].items()}
    return nt2i, i2nt

note2int, int2note = load_vocab(VOCAB_JSONL)
VOCAB_SIZE = len(note2int)

class MidiDataset(Dataset):
    def __init__(self, path):
        self.X, self.Y = torch.load(path)
    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, i):
        return self.X[i], self.Y[i]

train_ds = MidiDataset(TRAIN_PT)
val_ds = MidiDataset(VAL_PT)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# ───── Model ─────
# Define a new checkpoint and log directory for the baseline
CHECKPOINT_DIR_RNN = os.path.join(ROOT_DIR, "training", "checkpoints", "rnn_baseline")
LOG_DIR_RNN = os.path.join(ROOT_DIR, "training", "logs", "rnn_baseline")
os.makedirs(CHECKPOINT_DIR_RNN, exist_ok=True)
os.makedirs(LOG_DIR_RNN, exist_ok=True) # Also create the log directory

# Instantiate the correct model
model = MidiRNN(vocab_size=VOCAB_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# <--- FIX 1, 2, 3: DEFINE MISSING VARIABLES --->
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
scaler = GradScaler()
writer = SummaryWriter(LOG_DIR_RNN) # Use the new log directory

# ... (train_epoch and validate functions are fine) ...
def train_epoch(e):
    model.train()
    total_loss, seen = 0.0, 0
    bar = tqdm(train_loader, desc=f"Train Epoch {e+1}", ncols=80)
    for Xb, Yb in bar:
        Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(Xb)
            loss = sum(criterion(logits[:, TIMESTEPS - FUTURE + i, :], Yb[:, i]) for i in range(FUTURE)) / FUTURE
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * Xb.size(0)
        seen += Xb.size(0)
        bar.set_postfix(loss=f"{total_loss / seen:.5f}")
    avg = total_loss / len(train_ds)
    writer.add_scalar("Loss/Train", avg, e + 1)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], e + 1)
    return avg

def validate(e):
    model.eval()
    total_loss, seen = 0.0, 0
    with torch.no_grad():
        bar = tqdm(val_loader, desc="Validate", ncols=80)
        for Xb, Yb in bar:
            Xb, Yb = Xb.to(DEVICE), Yb.to(DEVICE)
            logits = model(Xb)
            loss = sum(criterion(logits[:, TIMESTEPS - FUTURE + i, :], Yb[:, i]) for i in range(FUTURE)) / FUTURE
            total_loss += loss.item() * Xb.size(0)
            seen += Xb.size(0)
            bar.set_postfix(val_loss=f"{total_loss / seen:.5f}")
    avg = total_loss / len(val_ds)
    writer.add_scalar("Loss/Val", avg, e + 1)
    return avg


# ───── Main Loop ─────
if __name__ == "__main__":
    best = float("inf")
    for epoch in range(NUM_EPOCHS):
        tr_loss = train_epoch(epoch)
        val_loss = validate(epoch)
        scheduler.step(val_loss)

        print(f"Epoch {epoch + 1}: train={tr_loss:.5f}, val={val_loss:.5f}")

        if val_loss < best:
            best = val_loss
            # <--- FIX 4: SAVE TO THE CORRECT DIRECTORY --->
            path = os.path.join(CHECKPOINT_DIR_RNN, f"best_epoch_{epoch + 1:02d}.pt")
            torch.save(model.state_dict(), path)
            print("💾 Saved", path)
        else:
            print("— no improvement")

    writer.close()
