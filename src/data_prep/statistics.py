# data_prep/statistics.py

import os
import time
from mido import MidiFile
from datetime import timedelta
from tqdm import tqdm
from utils.paths import DATA_DIR

# ─── Configuration ─────────────────────────────────────────────
root_dir = os.path.join(DATA_DIR, "SmallMIDFolder")
MAX_REASONABLE_DURATION = 3600  # 1 hour in seconds

def scan_and_summarize(root_dir, max_duration=3600):
    all_midis = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.mid'):
                all_midis.append(os.path.join(dirpath, fname))

    total_files = len(all_midis)
    if total_files == 0:
        print("No MIDI files found under:", root_dir)
        return

    valid_stats = []
    deleted_count = 0
    start_time = time.time()

    with tqdm(total=total_files, desc="Scanning MIDI files", ncols=80) as pbar:
        for path in all_midis:
            try:
                mid = MidiFile(path)
                duration = mid.length

                if duration <= max_duration:
                    valid_stats.append({'path': path, 'duration_sec': duration})
                else:
                    os.remove(path)
                    deleted_count += 1

            except Exception:
                try:
                    os.remove(path)
                    deleted_count += 1
                except Exception:
                    pass

            pbar.update(1)

    valid_count = len(valid_stats)
    total_duration_sec = sum(item['duration_sec'] for item in valid_stats)
    avg_duration_sec = total_duration_sec / valid_count if valid_count else 0

    total_duration_td = timedelta(seconds=total_duration_sec)
    avg_duration_td   = timedelta(seconds=avg_duration_sec)

    print("\n🎵 MIDI Dataset Summary (after deleting >1h files):")
    print(f"Original total MIDI files  : {total_files}")
    print(f"Deleted (duration >1h)     : {deleted_count}")
    print(f"Remaining valid MIDI files : {valid_count}")
    print(f"\nTotal playback time (valid)  : {total_duration_td} ({total_duration_sec:.2f} s)")
    print(f"Average file duration (valid): {avg_duration_td} ({avg_duration_sec:.2f} s)")

    if valid_stats:
        max_file = max(valid_stats, key=lambda x: x['duration_sec'], default=None)
        min_file = min(valid_stats, key=lambda x: x['duration_sec'], default=None)
        if max_file:
            print(f"\nLongest valid file           : {max_file['path']} ({timedelta(seconds=max_file['duration_sec'])})")
        if min_file:
            print(f"Shortest valid file          : {min_file['path']} ({timedelta(seconds=min_file['duration_sec'])})")

    elapsed_total = time.time() - start_time
    print(f"\n⏱️  Total time elapsed: {elapsed_total:.2f} seconds\n")

if __name__ == "__main__":
    print("=" * 60)
    print("🎵 MIDI SCANNER & SUMMARY")
    print("=" * 60)
    print(f"Target folder: {root_dir}")
    print(f"Maximum allowed duration: {timedelta(seconds=MAX_REASONABLE_DURATION)}")
    print()

    scan_and_summarize(root_dir, MAX_REASONABLE_DURATION)
