r"""
Two diagnostics for a model scored on ChoraleBricks, from the files
score_transcriptions.py already saved (nothing is rerun):

PART 1  Naming accuracy and register test  (same method as
        analyze_register_shortcut.py, so the numbers compare directly with
        Onsets and Frames and YourMT3+)
    - Pitch-only guesser: the most common instrument at each pitch in
      CocoChorales training (same 5,000-track sample, same seed).
    - Of the real notes the model found and labeled:
          label = true instrument     (naming accuracy)
          label = pitch-only guess    (register test)
      If the second is higher than the first, the model names by range, not sound.
    - Instrument x part table: how each instrument was labeled on each part it played.

PART 2  Near-miss check: what happened to the notes counted as MISSED?
    For each missed real note, look for a leftover prediction (one that matched
    nothing) in the same recording that starts within --window ms (default 150):
        timing     same pitch, but starts 50-150 ms off  (just outside the 50 ms rule)
        semitone   1 semitone off
        octave dn  12 semitones low
        octave up  12 semitones high
        12th up    19 semitones high (the 3rd harmonic)
        none       nothing nearby: truly missed
    Each leftover prediction is used at most once, closest first.
    Chance row: the same search with every real note moved 2 s later. Whatever
    it finds is coincidence (dense music, repeated notes), so read each category
    as "found minus chance".
    Also printed: for timing near-misses, the median offset (prediction minus
    true onset; positive = late), and whether the near-miss had the right label.

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\diagnose_model.py <model> [<model> ...]
A model is a results folder name (e.g. detector_medium_v1_choralebricks) or a
full path to a folder holding notes_reference.csv.
Options:  --window 150   near-miss window in ms

Writes R:\experiments\results\diagnostics\<model>.txt
"""
import argparse
import csv
import glob
import os
import random
import sys
from collections import Counter, defaultdict

import numpy as np

RESULTS = r'R:\experiments\results'
COCO_LABELS = r'R:\datasets\cocochorales\labels'
COCO_SPLITS = r'R:\splits\cocochorales_brass'
CB_ITEMS = r'R:\datasets\choralebricks\prepared\items.csv'
CB_LABELS = r'R:\datasets\choralebricks\labels\test'
CB_SOLO = r'R:\splits\choralebricks_brass\solo.txt'

NAMES = ['trumpet', 'horn', 'trombone', 'tuba']
PARTS = {1: 'S', 2: 'A', 3: 'T', 4: 'B'}
N_TRAIN = 5000            # same sample as analyze_register_shortcut.py
N_TEST = 1000
SEED = 0
ONSET_TOL = 0.05          # the scoring rule
CHANCE_SHIFT = 2.0        # seconds, for the chance row
CATEGORIES = [('timing', 0), ('semitone', 1), ('octave dn', -12), ('octave up', 12), ('12th up', 19)]


def as_name(value):
    value = str(value).strip()
    if value in NAMES:
        return value
    try:
        i = int(float(value))
        return NAMES[i] if 0 <= i < len(NAMES) else None
    except ValueError:
        return None


def find_scores(model):
    candidates = [model] if os.path.isdir(model) else []
    candidates += [os.path.join(RESULTS, model, sub) for sub in ('all', 'scores', '')]
    for c in candidates:
        if os.path.isfile(os.path.join(c, 'notes_reference.csv')):
            return c
    found = sorted(os.path.relpath(os.path.dirname(p), RESULTS) for p in
                   glob.glob(os.path.join(RESULTS, '**', 'notes_reference.csv'), recursive=True))
    sys.exit(f'No notes_reference.csv found for "{model}". Folders that have one:\n  ' + '\n  '.join(found))


# ---------------------------------------------------------------- part 1
def read_label_notes(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.split('\t')
            out.append((int(float(parts[2])), int(float(parts[4]))))
    return out


def sample_tracks(split, n):
    with open(os.path.join(COCO_SPLITS, f'{split}.txt')) as f:
        tracks = f.read().split()
    return random.Random(SEED).sample(tracks, min(n, len(tracks)))


def build_guesser():
    counts = defaultdict(Counter)
    for t in sample_tracks('train', N_TRAIN):
        for p, inst in read_label_notes(os.path.join(COCO_LABELS, 'train', f'{t}.tsv')):
            counts[p][inst] += 1
    table = {p: c.most_common(1)[0][0] for p, c in counts.items()}
    known = sorted(table)

    def guess(p):
        if p in table:
            return table[p]
        return table[min(known, key=lambda k: abs(k - p))]
    return guess


def register_part(rows, guess, log):
    labeled = [r for r in rows if as_name(r['given_to'])]
    if not labeled:
        log('   (no labeled notes)')
        return
    true_match = sum(as_name(r['given_to']) == as_name(r['instrument']) for r in labeled)
    pitch_match = sum(as_name(r['given_to']) == NAMES[guess(int(round(float(r['pitch']))))] for r in labeled)
    log(f'   real notes found and labeled: {len(labeled)}')
    log(f'   label = true instrument (naming accuracy): {true_match / len(labeled):.1%}')
    log(f'   label = pitch-only guess (register test):  {pitch_match / len(labeled):.1%}')

    part_of = {}
    if os.path.isfile(CB_ITEMS):
        with open(CB_ITEMS, newline='') as f:
            for row in csv.DictReader(f):
                for vi in row['voices'].split():
                    v, inst = vi.split(':')
                    part_of[(row['item'], as_name(inst) or inst)] = PARTS[int(v)]
    table = defaultdict(Counter)
    for r in rows:
        inst = as_name(r['instrument'])
        part = part_of.get((r['track'], inst), '?')
        table[(NAMES.index(inst), part)][as_name(r['given_to']) or 'missed'] += 1
    log('')
    log('   How each instrument was labeled, by the part it played (share of its notes):')
    log(f'   {"instrument (part)":<20}{"notes":>7}' + ''.join(f'{n:>10}' for n in NAMES + ['missed']))
    for key in sorted(table, key=lambda k: (k[0], 'SATB?'.index(k[1]) if k[1] in 'SATB?' else 9)):
        c = table[key]
        n = sum(c.values())
        log(f'   {NAMES[key[0]] + " (" + key[1] + ")":<20}{n:>7}'
            + ''.join(f'{c[x] / n:>10.0%}' for x in NAMES + ['missed']))


# ---------------------------------------------------------------- part 2
def near_misses(missed, extras, window, shift=0.0):
    """missed: list of (track, inst, pitch, onset). extras: {track: list of dict}.
    Returns per-note category, offset (s) and the label of the prediction used."""
    candidates = []                     # (|dt|, priority, note index, extra index, category, dt)
    for i, (track, inst, pitch, onset) in enumerate(missed):
        onset += shift
        for j, e in enumerate(extras.get(track, [])):
            dt = e['onset'] - onset
            if abs(dt) > window:
                continue
            for k, (cat, d) in enumerate(CATEGORIES):
                if (cat == 'semitone' and abs(e['pitch'] - pitch) == 1) or \
                   (cat != 'semitone' and e['pitch'] - pitch == d):
                    if cat == 'timing' and abs(dt) <= ONSET_TOL and shift == 0.0:
                        continue        # would have matched; can't happen, but be safe
                    candidates.append((k, abs(dt), i, j, cat, dt))
    candidates.sort()                   # category priority first, then closest
    used_note, used_extra = {}, set()
    for k, adt, i, j, cat, dt in candidates:
        key = (missed[i][0], j)
        if i in used_note or key in used_extra:
            continue
        used_note[i] = (cat, dt, extras[missed[i][0]][j]['instrument'])
        used_extra.add(key)
    return used_note


def near_miss_part(rows, extras_rows, window, log):
    missed = [(r['track'], as_name(r['instrument']), int(round(float(r['pitch']))), float(r['onset']))
              for r in rows if not as_name(r['given_to'])]
    extras = defaultdict(list)
    for r in extras_rows:
        extras[r['track']].append({'pitch': int(round(float(r['pitch']))), 'onset': float(r['onset']),
                                   'instrument': as_name(r['instrument'])})
    found = near_misses(missed, extras, window)
    chance = near_misses(missed, extras, window, shift=CHANCE_SHIFT)

    cats = [c for c, _ in CATEGORIES]
    log(f'   leftover predictions (matched nothing): {len(extras_rows)}')
    log(f'   window: {window * 1000:.0f} ms;  each row: count (chance count)')
    log(f'   {"instrument":<11}{"missed":>7}' + ''.join(f'{c:>14}' for c in cats) + f'{"none":>9}')
    totals = Counter()
    for inst in NAMES + ['ALL']:
        idx = [i for i, m in enumerate(missed) if inst == 'ALL' or m[1] == inst]
        if not idx:
            continue
        cf = Counter(found[i][0] for i in idx if i in found)
        cc = Counter(chance[i][0] for i in idx if i in chance)
        none = len(idx) - sum(cf.values())
        log(f'   {inst:<11}{len(idx):>7}' + ''.join(f'{f"{cf[c]} ({cc[c]})":>14}' for c in cats)
            + f'{none:>9}')
    explained = len(found)
    log(f'   leftover predictions explained as near-misses: {explained} of {len(extras_rows)} '
        f'({explained / max(len(extras_rows), 1):.0%}); chance: {len(chance)}')

    log('')
    log('   Timing near-misses: offset (prediction minus true onset) and label')
    log(f'   {"instrument":<11}{"count":>7}{"median ms":>11}{"late share":>12}{"right label":>13}')
    for inst in NAMES:
        t = [found[i] for i, m in enumerate(missed) if m[1] == inst and found.get(i, ('',))[0] == 'timing']
        if not t:
            log(f'   {inst:<11}{0:>7}')
            continue
        dts = np.array([x[1] for x in t]) * 1000
        right = sum(x[2] == inst for x in t)
        log(f'   {inst:<11}{len(t):>7}{np.median(dts):>11.0f}{(dts > 0).mean():>12.0%}{right / len(t):>13.0%}')


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='+')
    ap.add_argument('--window', type=float, default=150, help='near-miss window in ms')
    args = ap.parse_args()
    out_dir = os.path.join(RESULTS, 'diagnostics')
    os.makedirs(out_dir, exist_ok=True)

    print('Building the pitch-only guesser from CocoChorales training labels...')
    guess = build_guesser()
    with open(CB_SOLO) as f:
        solos = f.read().split()
    cb = [n for t in solos for n in read_label_notes(os.path.join(CB_LABELS, f'{t}.tsv'))]
    guess_acc = sum(guess(p) == i for p, i in cb) / max(len(cb), 1)

    for model in args.models:
        folder = find_scores(model)
        name = os.path.basename(os.path.normpath(model))
        lines = []

        def log(text=''):
            print(text)
            lines.append(text)

        with open(os.path.join(folder, 'notes_reference.csv'), newline='') as f:
            rows = list(csv.DictReader(f))
        with open(os.path.join(folder, 'notes_extra.csv'), newline='') as f:
            extras_rows = list(csv.DictReader(f))

        log(f'===== {name} =====')
        log('')
        log('PART 1  Naming accuracy and register test')
        log(f'   (pitch-only guesser on ChoraleBricks solos: {guess_acc:.1%}; random guessing: 25%)')
        register_part(rows, guess, log)
        log('')
        log('PART 2  Near-miss check on missed notes')
        near_miss_part(rows, extras_rows, args.window / 1000, log)
        log('')
        with open(os.path.join(out_dir, f'{name}.txt'), 'w') as f:
            f.write('\n'.join(lines) + '\n')
    print(f'Saved to {out_dir}')
    print('DIAGNOSTICS DONE')


if __name__ == '__main__':
    main()
