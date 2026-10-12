r"""
One-time prep pass: turn each track's four MIDI stems into one note list.

For every track in R:\splits\cocochorales_brass (train, valid, test), reads stems_midi\*.mid and
writes R:\datasets\cocochorales\labels\<split>\<track>.tsv with one row per note:

    onset   offset   pitch   velocity   instrument
    (sec)   (sec)    (MIDI)  (1-127)    0=trumpet 1=horn 2=trombone 3=tuba

The original dataset is never changed. If the script is interrupted, run it
again: finished tracks are skipped. At the end it prints (and saves to
labels\summary.txt) exact pitch ranges and velocity statistics for the
whole dataset.

Run from the Miniforge Prompt (oaf environment):
    python R:\code\data_prep\prep_labels_cocochorales.py
"""
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mido

DATA = Path(r"R:\datasets\cocochorales\extracted\main_dataset")
SPLITS_DIR = Path(r"R:\splits\cocochorales_brass")
LABELS = Path(r"R:\datasets\cocochorales\labels")
INSTRUMENTS = ["trumpet", "horn", "trombone", "tuba"]   # stem files 1_ to 4_
WORKERS = 6
HEADER = "onset\toffset\tpitch\tvelocity\tinstrument\n"


def read_notes(midi_path):
    """List of (onset_sec, offset_sec, pitch, velocity) from one MIDI file."""
    notes, active, now = [], {}, 0.0
    for msg in mido.MidiFile(midi_path):          # times come out in seconds
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(msg.note, []).append((now, msg.velocity))
        elif msg.type in ("note_off", "note_on") and active.get(msg.note):
            onset, vel = active[msg.note].pop(0)
            notes.append((onset, now, msg.note, vel))
    return notes


def stats_from_rows(rows):
    """Summary numbers for one track, used for the dataset-wide report."""
    s = {"vel": Counter(), "lo": {}, "hi": {}, "count": Counter(), "zero_len": 0}
    for onset, offset, pitch, vel, inst in rows:
        s["vel"][vel] += 1
        s["count"][inst] += 1
        s["lo"][inst] = min(s["lo"].get(inst, 999), pitch)
        s["hi"][inst] = max(s["hi"].get(inst, -1), pitch)
        if offset <= onset:
            s["zero_len"] += 1
    return s


def process(job):
    split, track = job
    out = LABELS / split / f"{track}.tsv"
    if out.exists():                               # done on an earlier run
        rows = []
        for line in out.read_text().splitlines()[1:]:
            a, b, c, d, e = line.split("\t")
            rows.append((float(a), float(b), int(c), int(d), int(e)))
        return split, stats_from_rows(rows), None
    try:
        rows = []
        for i, name in enumerate(INSTRUMENTS):
            midi = DATA / split / track / "stems_midi" / f"{i + 1}_{name}.mid"
            rows += [(on, off, p, v, i) for on, off, p, v in read_notes(midi)]
        rows.sort()
        text = HEADER + "".join(f"{on:.6f}\t{off:.6f}\t{p}\t{v}\t{i}\n"
                                for on, off, p, v, i in rows)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(text)
        os.replace(tmp, out)                       # file appears only when complete
        return split, stats_from_rows(rows), None
    except Exception as err:
        return split, None, f"{split}/{track}: {err}"


def main():
    jobs = []
    for split in ["train", "valid", "test"]:
        (LABELS / split).mkdir(parents=True, exist_ok=True)
        tracks = (SPLITS_DIR / f"{split}.txt").read_text().split()
        jobs += [(split, t) for t in tracks]

    print(f"Preparing note lists for {len(jobs)} tracks, {WORKERS} at a time...")
    start = time.time()
    vel, count, failures = Counter(), Counter(), []
    lo, hi, zero_len = {}, {}, 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for n, (split, s, err) in enumerate(pool.map(process, jobs, chunksize=100), 1):
            if err:
                failures.append(err)
            else:
                vel.update(s["vel"]); count.update(s["count"]); zero_len += s["zero_len"]
                for i in s["lo"]:
                    lo[i] = min(lo.get(i, 999), s["lo"][i])
                    hi[i] = max(hi.get(i, -1), s["hi"][i])
            if n % 2000 == 0 or n == len(jobs):
                print(f"  {n}/{len(jobs)} tracks   ({(time.time() - start) / 60:.1f} min)")

    lines = ["", "Per instrument: lowest pitch, highest pitch, total notes"]
    for i, name in enumerate(INSTRUMENTS):
        lines.append(f"  {name:10s} {lo.get(i, '-'):>4} {hi.get(i, '-'):>5} {count[i]:>10}")
    lines.append(f"Notes with zero or negative length: {zero_len}")
    lines.append(f"Distinct velocity values: {len(vel)}")
    lines.append("Most common velocities (value: notes): " +
                 ", ".join(f"{v}: {c}" for v, c in vel.most_common(8)))
    lines.append(f"Velocity range: {min(vel) if vel else '-'} to {max(vel) if vel else '-'}")
    lines.append("")
    if failures:
        lines.append(f"{len(failures)} track(s) FAILED. Run again to retry. First few:")
        lines += ["  " + f for f in failures[:5]]
    else:
        lines.append("ALL DONE")
    report = "\n".join(lines)
    print(report)
    (LABELS / "summary.txt").write_text(report + "\n")


if __name__ == "__main__":     # required on Windows when using several processes
    main()
