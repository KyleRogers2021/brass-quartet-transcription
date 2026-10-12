r"""
Unpack the downloaded CocoChorales brass archives.

Each .tar.bz2 file is a compressed bundle of many tracks. This script unpacks
every bundle into R:\datasets\cocochorales\extracted, keeping the same
layout as the official extraction script:

    extracted\main_dataset\train\brass_track048001\mix.wav ...
    extracted\f0\test\...

It unpacks several bundles at once (one per CPU core it uses) to save time.
If it gets interrupted, run it again: bundles already finished are skipped.

Run from the Miniforge Prompt:
    python R:\code\data_prep\extract_cocochorales.py
"""
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ZIPPED = Path(r"R:\datasets\cocochorales\zipped")
EXTRACTED = Path(r"R:\datasets\cocochorales\extracted")
DONE_DIR = EXTRACTED / "_done"   # one small marker file per finished bundle
WORKERS = 6                      # bundles unpacked at the same time

# Biggest data type first, so the long jobs start early.
PARTS = ["main_dataset", "f0", "note_expression", "metadata"]
SPLITS = ["train", "valid", "test"]


def unpack(archive, out_dir, marker):
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:bz2") as tar:
        tar.extractall(out_dir, filter="data")
    marker.touch()   # only written after the whole bundle unpacked cleanly
    return archive


def main():
    DONE_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for part in PARTS:
        for split in SPLITS:
            for archive in sorted((ZIPPED / part / split).glob("*.tar.bz2")):
                marker = DONE_DIR / f"{part}_{split}_{archive.name}.done"
                if not marker.exists():
                    jobs.append((archive, EXTRACTED / part / split, marker))

    if not jobs:
        print("Everything is already unpacked.")
        return

    print(f"Unpacking {len(jobs)} bundles, {WORKERS} at a time...")
    start = time.time()
    failures = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(unpack, *job): job[0] for job in jobs}
        for n, future in enumerate(as_completed(futures), 1):
            archive = futures[future]
            name = archive.relative_to(ZIPPED)
            minutes = (time.time() - start) / 60
            try:
                future.result()
                print(f"[{n}/{len(jobs)}] done  {name}   ({minutes:.1f} min elapsed)")
            except Exception as err:
                failures.append(name)
                print(f"[{n}/{len(jobs)}] FAILED {name}: {err}")

    print()
    if failures:
        print(f"{len(failures)} bundle(s) failed. Run the script again to retry them.")
    else:
        print("ALL DONE")


if __name__ == "__main__":   # required on Windows when using several processes
    main()
