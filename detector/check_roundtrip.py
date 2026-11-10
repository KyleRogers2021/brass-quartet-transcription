r"""
Round-trip check of the detector's label geometry: real CocoChorales notes ->
overlapping 10.24 s windows -> YOLO box label files -> back to notes -> merged
across windows. Every note should come back exactly; anything else is a bug in
the conversion, found before any training data is generated.

Also reports what the boxes look like per clip (count, thinnest slivers at clip
edges, identical boxes from unisons), which sets a few YOLO training options.

Needs only the label files (no audio, no GPU). Run in the oaf environment:
    python R:\code\detector\check_roundtrip.py
    (--split valid --tracks 1000 by default)
"""
import argparse
import os
import random
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings as S                                                   # noqa: E402
from boxes import (notes_to_boxes, boxes_to_notes, window_starts, merge_notes,   # noqa: E402
                   write_yolo, read_yolo)

COCO_LABELS = r'R:\datasets\cocochorales\labels'
COCO_SPLITS = r'R:\splits\cocochorales_brass'
OUT = r'R:\experiments\results\detector_roundtrip_check.txt'
TOL = 0.001          # 1 ms


def read_notes(path):
    rows = np.loadtxt(path, delimiter='\t', skiprows=1, ndmin=2)
    return [(float(r[0]), float(r[1]), int(round(r[2])), int(r[4])) for r in rows]


def same(a, b):
    return a[2] == b[2] and a[3] == b[3] and abs(a[0] - b[0]) <= TOL and abs(a[1] - b[1]) <= TOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='valid')
    ap.add_argument('--tracks', type=int, default=1000)
    a = ap.parse_args()

    with open(os.path.join(COCO_SPLITS, f'{a.split}.txt')) as f:
        tracks = f.read().split()
    tracks = random.Random(0).sample(tracks, min(a.tracks, len(tracks)))

    total = exact = 0
    problems = []
    per_clip, slivers, dup_boxes, clips = [], 0, 0, 0
    px_w = S.CLIP_SECONDS / S.FRAMES
    with tempfile.TemporaryDirectory() as tmp:
        label_file = os.path.join(tmp, 'clip.txt')
        for track in tracks:
            notes = read_notes(os.path.join(COCO_LABELS, a.split, f'{track}.tsv'))
            duration = max(n[1] for n in notes) + 1.0
            recovered = []
            for t0 in window_starts(duration):
                boxes = notes_to_boxes(notes, t0)
                write_yolo(label_file, boxes)              # through the real file format
                back = read_yolo(label_file)
                recovered += boxes_to_notes(back, t0)
                clips += 1
                per_clip.append(len(boxes))
                slivers += sum(b[3] * S.CLIP_SECONDS < 2 * px_w for b in boxes)
                keys = [(round(b[1], 4), round(b[2], 4), round(b[3], 4)) for b in boxes]
                dup_boxes += len(keys) - len(set(keys))
            merged = merge_notes(recovered)
            total += len(notes)
            remaining = list(merged)
            for n in sorted(notes):
                hit = next((i for i, m in enumerate(remaining) if same(n, m)), None)
                if hit is None:
                    if len(problems) < 10:
                        problems.append((track, n))
                else:
                    exact += 1
                    remaining.pop(hit)
            if remaining and len(problems) < 10:
                problems.append((track, ('extra', remaining[0])))

    lines = [f'Round-trip check on {len(tracks)} CocoChorales {a.split} tracks ({clips} clips of '
             f'{S.CLIP_SECONDS:.2f} s, windows every {S.WINDOW_HOP_SECONDS:.2f} s)',
             f'  notes recovered exactly (pitch, instrument, onset and offset within 1 ms): '
             f'{exact} of {total} ({exact / max(total, 1):.4%})']
    if problems:
        lines.append('  first problems (track, note):')
        lines += [f'    {t}  {n}' for t, n in problems]
    pc = np.array(per_clip)
    lines += [f'  boxes per clip: median {np.median(pc):.0f}, max {pc.max()}',
              f'  slivers under 2 px wide (notes cut at a clip edge): {slivers} '
              f'({slivers / max(pc.sum(), 1):.2%} of boxes)',
              f'  identical boxes from exact unisons: {dup_boxes} ({dup_boxes / max(pc.sum(), 1):.2%} of boxes)',
              'ROUND TRIP PASSED' if exact == total else 'ROUND TRIP FAILED: see problems above']
    report = '\n'.join(lines)
    print(report)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(report + '\n')
    print(f'Saved to {OUT}')


if __name__ == '__main__':
    main()
