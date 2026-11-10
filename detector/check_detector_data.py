r"""
Check the assumptions behind the detector design, using only the answer-key note
files (no audio, no GPU). Runs in a few minutes.

For CocoChorales (train, valid, test) and ChoraleBricks it reports:
  1. Pitch range per instrument, and whether every note still fits the planned
     image window (MIDI 20-99) after the largest pitch shift (default +/-9).
  2. Unisons: notes that overlap in time with another instrument's note at the
     same pitch, and how many of those would give nearly identical boxes
     (time overlap IoU >= 0.5 / 0.9), which is the training risk.
  3. One note at a time: notes of the same instrument that overlap in time
     (the duplicate-removal and window-merging plans assume this never happens).
  4. Note lengths in image pixels (16 ms per pixel).

Saves the report to R:\experiments\results\detector_data_check.txt.

Run from the Miniforge Prompt (oaf environment):
    python R:\code\detector\check_detector_data.py
    (add --max-tracks 2000 for a quicker look at a sample of each split)
"""
import argparse
import os
import random
from collections import defaultdict

import numpy as np

COCO_LABELS = r'R:\datasets\cocochorales\labels'
COCO_SPLITS = r'R:\splits\cocochorales_brass'
CB_LABELS = r'R:\datasets\choralebricks\labels\test'
CB_LIST = r'R:\splits\choralebricks_brass\all.txt'
OUT = r'R:\experiments\results\detector_data_check.txt'

NAMES = ['trumpet', 'horn', 'trombone', 'tuba']
WINDOW = (20, 99)          # planned image pitch range (MIDI, inclusive)
FRAME_S = 256 / 16000      # one image pixel in time: 16 ms
OVERLAP_TOL = 0.02         # ignore overlaps shorter than 20 ms (rounding at note joins)


def read_notes(path):
    """(onset, offset, pitch, instrument) rows of one answer-key file."""
    rows = np.loadtxt(path, delimiter='\t', skiprows=1, ndmin=2)
    return [(float(r[0]), float(r[1]), int(round(r[2])), int(r[4])) for r in rows]


def time_overlap(a, b):
    return min(a[1], b[1]) - max(a[0], b[0])


def time_iou(a, b):
    inter = max(0.0, time_overlap(a, b))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def check(files, shift):
    s = dict(tracks=0, notes=0, durations=[], pitches=defaultdict(list),
             unison_notes=0, iou50=0, iou90=0, unison_tracks=0,
             same_inst_overlaps=0, same_inst_tracks=0, missing=0)
    for path in files:
        if not os.path.exists(path):
            s['missing'] += 1
            continue
        notes = read_notes(path)
        s['tracks'] += 1
        s['notes'] += len(notes)
        for n in notes:
            s['durations'].append(n[1] - n[0])
            s['pitches'][n[3]].append(n[2])

        # unisons: same pitch, different instrument, overlapping in time
        by_pitch = defaultdict(list)
        for n in notes:
            by_pitch[n[2]].append(n)
        in_unison, track_had = set(), False
        for group in by_pitch.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a, b = group[i], group[j]
                    if a[3] != b[3] and time_overlap(a, b) > OVERLAP_TOL:
                        in_unison.update([a, b])
                        track_had = True
                        iou = time_iou(a, b)
                        s['iou50'] += iou >= 0.5
                        s['iou90'] += iou >= 0.9
        s['unison_notes'] += len(in_unison)
        s['unison_tracks'] += track_had

        # one note at a time per instrument
        by_inst = defaultdict(list)
        for n in notes:
            by_inst[n[3]].append(n)
        had = False
        for group in by_inst.values():
            group.sort()
            for a, b in zip(group, group[1:]):
                if time_overlap(a, b) > OVERLAP_TOL:
                    s['same_inst_overlaps'] += 1
                    had = True
        s['same_inst_tracks'] += had
    return s


def report(name, s, shift, log):
    log(f'\n=== {name}: {s["tracks"]} tracks, {s["notes"]} notes'
        + (f' ({s["missing"]} label files missing)' if s['missing'] else ''))
    if not s['notes']:
        return
    log('  Pitch range (MIDI, note name)        lowest     1%      99%   highest   '
        f'fits {WINDOW[0]}-{WINDOW[1]} after +/-{shift}?')
    for i, name_i in enumerate(NAMES):
        p = np.array(s['pitches'].get(i, []))
        if not len(p):
            log(f'    {name_i:<10} no notes')
            continue
        lo, hi = p.min(), p.max()
        fits = lo - shift >= WINDOW[0] and hi + shift <= WINDOW[1]
        log(f'    {name_i:<10} {lo:>3} {midi_name(lo):<5} {np.percentile(p, 1):>7.0f} '
            f'{np.percentile(p, 99):>7.0f}   {hi:>3} {midi_name(hi):<5}   '
            + ('yes' if fits else f'NO ({lo - shift} to {hi + shift})'))
    n = s['notes']
    log(f'  Unisons: {s["unison_notes"]} notes ({s["unison_notes"] / n:.2%}) share their pitch with '
        f'another instrument at the same time, in {s["unison_tracks"]} of {s["tracks"]} tracks')
    log(f'    pairs with nearly identical boxes: time IoU >= 0.5: {s["iou50"]}, >= 0.9: {s["iou90"]}')
    log(f'  One note at a time: {s["same_inst_overlaps"]} same-instrument overlaps '
        f'(> {OVERLAP_TOL * 1000:.0f} ms) in {s["same_inst_tracks"]} tracks')
    d = np.array(s['durations'])
    px = d / FRAME_S
    log(f'  Note length: median {np.median(d):.2f} s ({np.median(px):.0f} px), '
        f'shortest {d.min() * 1000:.0f} ms ({px.min():.1f} px); '
        f'under 50 ms: {(d < 0.05).mean():.2%}, under 100 ms: {(d < 0.1).mean():.2%}')


def midi_name(m):
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f'{names[m % 12]}{m // 12 - 1}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shift', type=int, default=9, help='largest planned pitch shift (default 9)')
    ap.add_argument('--max-tracks', type=int, help='check a random sample of each split instead of all')
    a = ap.parse_args()

    lines = []

    def log(text=''):
        print(text)
        lines.append(text)

    log(f'Detector data check (image window MIDI {WINDOW[0]}-{WINDOW[1]}, shift +/-{a.shift}, '
        f'{FRAME_S * 1000:.0f} ms per pixel)')
    for split in ['train', 'valid', 'test']:
        with open(os.path.join(COCO_SPLITS, f'{split}.txt')) as f:
            tracks = f.read().split()
        if a.max_tracks and len(tracks) > a.max_tracks:
            tracks = random.Random(0).sample(tracks, a.max_tracks)
        print(f'\nReading CocoChorales {split} ({len(tracks)} tracks) ...')
        files = [os.path.join(COCO_LABELS, split, f'{t}.tsv') for t in tracks]
        report(f'CocoChorales {split}', check(files, a.shift), a.shift, log)

    with open(CB_LIST) as f:
        items = f.read().split()
    files = [os.path.join(CB_LABELS, f'{t}.tsv') for t in items]
    report('ChoraleBricks (real recordings; never shifted, so checked without a shift)', check(files, 0), 0, log)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved to {OUT}')


if __name__ == '__main__':
    main()
