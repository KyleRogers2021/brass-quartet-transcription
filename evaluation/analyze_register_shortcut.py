r"""
Register-shortcut test: on real recordings, do a model's instrument labels
follow the instrument's sound, or just the pitch range (register)?

In CocoChorales every instrument always plays the same voice (trumpet S,
horn A, trombone T, tuba B), so pitch alone almost identifies the instrument.
A model trained there can score well by learning "high = trumpet, low = tuba".
ChoraleBricks breaks that link (e.g. trumpet also plays alto, trombone bass).

1. Pitch-only guesser: from CocoChorales TRAINING labels, the most common
   instrument at each pitch. Its accuracy on CocoChorales vs ChoraleBricks
   shows how much the shortcut is worth on each.
2. For each model with instrument labels: among real notes it found
   (assigned to one of the four instruments), how often its label equals
   (a) the true instrument, (b) the pitch-only guess.
3. Instrument x part table: how each instrument was labeled on each part.

Reads per-note records from score_choralebricks.py
(R:\experiments\results\<model>_choralebricks\all\notes_reference.csv).

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\analyze_register_shortcut.py
"""
import csv
import glob
import os
import random
from collections import Counter, defaultdict

COCO_LABELS = r'R:\datasets\cocochorales\labels'
COCO_SPLITS = r'R:\splits\cocochorales_brass'
CB_ITEMS = r'R:\datasets\choralebricks\prepared\items.csv'
CB_LABELS = r'R:\datasets\choralebricks\labels\test'
RESULTS = r'R:\experiments\results'
NAMES = ['trumpet', 'horn', 'trombone', 'tuba']
PARTS = {1: 'S', 2: 'A', 3: 'T', 4: 'B'}
N_TRAIN = 5000          # training tracks sampled for the pitch-only guesser
N_TEST = 1000           # CocoChorales test tracks sampled to measure it
SEED = 0


def read_label_notes(path):
    """(pitch, instrument) for each note of an answer-key file."""
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


def guesser_accuracy(guess, notes):
    return sum(guess(p) == inst for p, inst in notes) / max(len(notes), 1)


def main():
    lines = []

    def log(text=''):
        print(text)
        lines.append(text)

    guess = build_guesser()
    coco = [n for t in sample_tracks('test', N_TEST)
            for n in read_label_notes(os.path.join(COCO_LABELS, 'test', f'{t}.tsv'))]
    with open(os.path.join(r'R:\splits\choralebricks_brass', 'solo.txt')) as f:
        solos = f.read().split()
    cb = [n for t in solos for n in read_label_notes(os.path.join(CB_LABELS, f'{t}.tsv'))]
    log('1. Pitch-only guesser (instrument = most common one at that pitch in CocoChorales training)')
    log(f'   accuracy on CocoChorales test notes:      {guesser_accuracy(guess, coco):.1%}')
    log(f'   accuracy on ChoraleBricks (real) notes:   {guesser_accuracy(guess, cb):.1%}'
        '   (solo recordings, every note counted once)')

    # part (voice) of each instrument in each ChoraleBricks item
    part_of = {}
    with open(CB_ITEMS, newline='') as f:
        for row in csv.DictReader(f):
            for vi in row['voices'].split():
                v, inst = vi.split(':')
                part_of[(row['item'], inst)] = PARTS[int(v)]

    for path in sorted(glob.glob(os.path.join(RESULTS, '*_choralebricks', 'all', 'notes_reference.csv'))):
        model = os.path.basename(os.path.dirname(os.path.dirname(path))).replace('_choralebricks', '')
        with open(path, newline='') as f:
            rows = list(csv.DictReader(f))
        if not rows or rows[0]['given_to'] == 'found':
            continue                         # model gives no instrument labels
        labeled = [r for r in rows if r['given_to'] in NAMES]
        if not labeled:
            continue
        true_match = sum(r['given_to'] == r['instrument'] for r in labeled)
        pitch_match = sum(r['given_to'] == NAMES[guess(int(r['pitch']))] for r in labeled)
        log(f'\n2. {model}: real notes found and given one of the four instruments: {len(labeled)}')
        log(f'   label = true instrument:   {true_match / len(labeled):.1%}')
        log(f'   label = pitch-only guess:  {pitch_match / len(labeled):.1%}')

        log(f'\n3. {model}: how each instrument was labeled, by the part it played '
            '(share of its notes; all ChoraleBricks items)')
        log(f'   {"instrument (part)":<20}{"notes":>7}' + ''.join(f'{n:>10}' for n in NAMES + ['missed']))
        table = defaultdict(Counter)
        for r in rows:
            part = part_of.get((r['track'], r['instrument']), '?')
            table[(NAMES.index(r['instrument']), 'SATB?'.index(part), r['instrument'], part)][r['given_to']] += 1
        for key in sorted(table):
            _, _, inst, part = key
            c = table[key]
            total = sum(c.values())
            log(f'   {inst + " (" + part + ")":<20}{total:>7}' +
                ''.join(f'{c[n] / total:>10.0%}' for n in NAMES + ['missed']))

    out = os.path.join(RESULTS, 'register_shortcut.txt')
    with open(out, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nSaved to {out}')
    print('ANALYSIS DONE')


if __name__ == '__main__':
    main()
