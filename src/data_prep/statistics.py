# data_prep/statistics.py
import os
import shutil
import csv
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import timedelta
from typing import List, Dict, Optional

from mido import MidiFile
from tqdm import tqdm
from utils.paths import DATA_DIR

# ─── Configuration ─────────────────────────────────────────────
ROOT_DIR = Path(DATA_DIR) / "SmallMIDFolder"
MAX_REASONABLE_DURATION = 3600  # seconds (1 hour)
QUARANTINE_DIR = ROOT_DIR.parent / "quarantine_midis"  # sibling to dataset

@dataclass
class MidiStat:
    path: str
    duration_sec: float

def scan_and_summarize(
    root_dir: Path,
    max_duration: float = MAX_REASONABLE_DURATION,
    delete: bool = False,
    quarantine: bool = True,
    csv_out: Optional[Path] = None,
    log_path: Optional[Path] = None,
) -> Dict:
    root_dir = Path(root_dir)
    all_midis = [*root_dir.rglob("*.mid"), *root_dir.rglob("*.midi")]
    total_files = len(all_midis)

    if total_files == 0:
        print(f"No MIDI files found under: {root_dir}")
        return {
            "root": str(root_dir),
            "total_files": 0,
            "deleted_count": 0,
            "valid_count": 0,
            "total_duration_sec": 0.0,
            "avg_duration_sec": 0.0,
            "longest": None,
            "shortest": None,
            "elapsed_sec": 0.0,
        }

    valid_stats: List[MidiStat] = []
    removed: List[Path] = []
    failures: List[str] = []

    if quarantine:
        QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    with tqdm(total=total_files, desc="Scanning MIDI files", ncols=90, unit="file") as pbar:
        for path in all_midis:
            try:
                mid = MidiFile(path)
                duration = float(mid.length)
                if duration <= max_duration:
                    valid_stats.append(MidiStat(str(path), duration))
                else:
                    # too long → quarantine/delete
                    removed.append(path)
                    if delete and not quarantine:
                        path.unlink(missing_ok=True)
                    else:
                        # move to quarantine with preserved relative structure
                        rel = path.relative_to(root_dir)
                        dest = QUARANTINE_DIR / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(path), str(dest))
            except Exception as e:
                # unreadable → quarantine/delete
                failures.append(f"{path} | {repr(e)}")
                removed.append(path)
                try:
                    if delete and not quarantine:
                        path.unlink(missing_ok=True)
                    else:
                        rel = path.relative_to(root_dir)
                        dest = QUARANTINE_DIR / rel
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if path.exists():
                            shutil.move(str(path), str(dest))
                except Exception as inner:
                    failures.append(f"(while handling failure) {path} | {repr(inner)}")
            finally:
                pbar.update(1)

    valid_count = len(valid_stats)
    total_duration_sec = sum(s.duration_sec for s in valid_stats)
    avg_duration_sec = (total_duration_sec / valid_count) if valid_count else 0.0

    longest = max(valid_stats, key=lambda s: s.duration_sec, default=None)
    shortest = min(valid_stats, key=lambda s: s.duration_sec, default=None)

    # Optional CSV export
    if csv_out and valid_stats:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["path", "duration_sec"])
            w.writeheader()
            for s in valid_stats:
                w.writerow(asdict(s))

    # Optional logging
    if log_path and failures:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("Unreadable/failed files:\n")
            for line in failures:
                f.write(line + "\n")

    elapsed_total = time.time() - start_time

    # Human-readable summary
    print("\n🎵 MIDI Dataset Summary")
    if not delete and quarantine:
        print("Note: Overlong/unreadable files were moved to quarantine (not deleted).")
    print(f"Scan root                   : {root_dir}")
    print(f"Original total MIDI files   : {total_files}")
    print(f"Removed or quarantined      : {len(removed)}")
    print(f"Remaining valid MIDI files  : {valid_count}")

    print(f"\nTotal playback time (valid) : {timedelta(seconds=total_duration_sec)} ({total_duration_sec:.2f} s)")
    print(f"Average file duration       : {timedelta(seconds=avg_duration_sec)} ({avg_duration_sec:.2f} s)")

    if longest:
        print(f"\nLongest valid file          : {longest.path} ({timedelta(seconds=longest.duration_sec)})")
    if shortest:
        print(f"Shortest valid file         : {shortest.path} ({timedelta(seconds=shortest.duration_sec)})")

    print(f"\n⏱️  Total time elapsed       : {elapsed_total:.2f} s\n")

    # Machine-friendly return
    return {
        "root": str(root_dir),
        "total_files": total_files,
        "deleted_count": len(removed),
        "valid_count": valid_count,
        "total_duration_sec": total_duration_sec,
        "avg_duration_sec": avg_duration_sec,
        "longest": asdict(longest) if longest else None,
        "shortest": asdict(shortest) if shortest else None,
        "elapsed_sec": elapsed_total,
        "csv_out": str(csv_out) if csv_out else None,
        "log_path": str(log_path) if log_path else None,
        "quarantine_dir": str(QUARANTINE_DIR if quarantine else ""),
    }

if __name__ == "__main__":
    print("=" * 60)
    print("🎵 MIDI SCANNER & SUMMARY")
    print("=" * 60)
    print(f"Target folder: {ROOT_DIR}")
    print(f"Maximum allowed duration: {timedelta(seconds=MAX_REASONABLE_DURATION)}")
    print()

    # Example invocation: quarantine instead of deleting; also write CSV & log.
    scan_and_summarize(
        ROOT_DIR,
        max_duration=MAX_REASONABLE_DURATION,
        delete=False,  # set True if you really want to delete
        quarantine=True,
        csv_out=ROOT_DIR.parent / "midi_valid_stats.csv",
        log_path=ROOT_DIR.parent / "midi_scan_failures.log",
    )
