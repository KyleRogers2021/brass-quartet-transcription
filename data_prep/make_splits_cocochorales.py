r"""
Write the train / validation / test track lists that every model shares.

Follows CocoChorales' official split (brass only), and also picks a fixed
random sample of validation tracks for the quick checks during training.
The lists hold track folder names only (e.g. brass_track048001), so they
work no matter where the data lives (Windows or Linux).

If the lists already exist, the script refuses to overwrite them unless
they match exactly. The split must stay fixed for the whole project.

Run from the Miniforge Prompt:
    python R:\code\data_prep\make_splits_cocochorales.py
"""
import random
import sys
from pathlib import Path

DATA = Path(r"R:\datasets\cocochorales\extracted\main_dataset")
OUT = Path(r"R:\splits\cocochorales_brass")
VALID_SAMPLE_SIZE = 200
SEED = 0
EXPECTED = {"train": 48000, "valid": 6000, "test": 6000}


def write_or_check(path, names):
    text = "\n".join(names) + "\n"
    if path.exists():
        if path.read_text() == text:
            print(f"  {path.name}: already exists and matches ({len(names)} tracks)")
            return
        sys.exit(f"STOPPED: {path} already exists and is DIFFERENT. "
                 "Not overwriting it. Tell Claude before changing the split.")
    path.write_text(text)
    print(f"  {path.name}: wrote {len(names)} tracks")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lists = {}
    for split, expected in EXPECTED.items():
        names = sorted(p.name for p in (DATA / split).iterdir() if p.is_dir())
        bad = [n for n in names if not n.startswith("brass_track")]
        if bad:
            sys.exit(f"STOPPED: non-brass folders in {split}, e.g. {bad[:3]}")
        if len(names) != expected:
            print(f"  WARNING: {split} has {len(names)} tracks, expected {expected}")
        lists[split] = names

    overlap = (set(lists["train"]) & set(lists["valid"])) | \
              (set(lists["train"]) & set(lists["test"])) | \
              (set(lists["valid"]) & set(lists["test"]))
    if overlap:
        sys.exit(f"STOPPED: {len(overlap)} tracks appear in more than one split")

    print("Writing lists to", OUT)
    for split, names in lists.items():
        write_or_check(OUT / f"{split}.txt", names)

    sample = sorted(random.Random(SEED).sample(lists["valid"], VALID_SAMPLE_SIZE))
    write_or_check(OUT / "valid_sample.txt", sample)

    print("\nNo track appears in more than one split.")
    print(f"Validation sample: {VALID_SAMPLE_SIZE} tracks, random seed {SEED}.")
    print("ALL DONE")


if __name__ == "__main__":
    main()
