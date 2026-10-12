r"""
Survey the unpacked CocoChorales brass data before we write the data loader.

This script only READS files; it doesn't change anything. It answers a few
questions the loader design depends on:

  1. What files does each track folder actually contain?
  2. Is the instrument order the same in every track (trumpet, horn,
     trombone, tuba)?
  3. On a random sample of training tracks:
       - audio sample rate and track lengths
       - the pitch range each instrument actually plays
       - whether each instrument ever plays two notes at once
       - whether the MIDI timing lines up with the audio length

Run from the Miniforge Prompt (with the oaf environment active):
    python R:\code\data_prep\survey_cocochorales.py
"""
import random
from collections import Counter, defaultdict
from pathlib import Path

import mido
import soundfile

ROOT = Path(r"R:\datasets\cocochorales\extracted\main_dataset")
SPLITS = ["train", "valid", "test"]
SAMPLE_SIZE = 300
SEED = 0


def read_notes(midi_path):
    """Return a list of (onset_sec, offset_sec, pitch) from a MIDI file."""
    notes, active, now = [], {}, 0.0
    for msg in mido.MidiFile(midi_path):      # iterating gives times in seconds
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(msg.note, []).append(now)
        elif msg.type in ("note_off", "note_on") and msg.note in active and active[msg.note]:
            notes.append((active[msg.note].pop(0), now, msg.note))
    return notes


def max_overlap(notes):
    """Largest number of notes sounding at the same moment."""
    events = sorted([(on, 1) for on, _, _ in notes] + [(off, -1) for _, off, _ in notes])
    current = best = 0
    for _, change in events:
        current += change
        best = max(best, current)
    return best


def main():
    # 1. One example track
    example = sorted((ROOT / "train").iterdir())[0]
    print("=" * 60)
    print(f"1. Files in one track folder ({example.name}):")
    for f in sorted(example.rglob("*")):
        if f.is_file():
            print(f"     {f.relative_to(example)}   ({f.stat().st_size / 1e6:.2f} MB)")

    # 2. Instrument order in every track
    print("=" * 60)
    print("2. Instrument order across all tracks (this takes a minute)...")
    tracks = {s: sorted(p for p in (ROOT / s).iterdir() if p.is_dir()) for s in SPLITS}
    orders = Counter()
    for split in SPLITS:
        for t in tracks[split]:
            stems = sorted(f.stem for f in (t / "stems_MIDI").glob("*.mid"))
            orders[tuple(stems)] += 1
    for order, count in orders.most_common():
        print(f"     {count:6d} tracks: {', '.join(order)}")

    # 3. Detailed look at a random sample
    print("=" * 60)
    print(f"3. Details from {SAMPLE_SIZE} random training tracks...")
    random.seed(SEED)
    sample = random.sample(tracks["train"], SAMPLE_SIZE)
    rates, lengths, gaps = Counter(), [], []
    pitch_lo, pitch_hi = defaultdict(lambda: 999), defaultdict(lambda: -1)
    overlap, note_counts = defaultdict(int), defaultdict(int)
    for t in sample:
        info = soundfile.info(t / "mix.wav")
        rates[info.samplerate] += 1
        lengths.append(info.duration)
        last_offset = 0.0
        for midi in sorted((t / "stems_MIDI").glob("*.mid")):
            inst = midi.stem.split("_", 1)[1]
            notes = read_notes(midi)
            if not notes:
                continue
            note_counts[inst] += len(notes)
            pitch_lo[inst] = min(pitch_lo[inst], min(p for _, _, p in notes))
            pitch_hi[inst] = max(pitch_hi[inst], max(p for _, _, p in notes))
            overlap[inst] = max(overlap[inst], max_overlap(notes))
            last_offset = max(last_offset, max(off for _, off, _ in notes))
        gaps.append(info.duration - last_offset)

    print(f"     Sample rates: {dict(rates)}")
    print(f"     Track length (sec): min {min(lengths):.1f}, "
          f"average {sum(lengths) / len(lengths):.1f}, max {max(lengths):.1f}")
    print("     Per instrument: lowest pitch, highest pitch, most notes at once, total notes")
    for inst in pitch_lo:
        print(f"       {inst:10s} {pitch_lo[inst]:4d} {pitch_hi[inst]:5d} "
              f"{overlap[inst]:8d} {note_counts[inst]:10d}")
    print(f"     Audio length minus last MIDI note end (sec): min {min(gaps):.2f}, "
          f"average {sum(gaps) / len(gaps):.2f}, max {max(gaps):.2f}")
    print("=" * 60)
    print("Survey finished.")


if __name__ == "__main__":
    main()
