r"""
Where is each instrument detected? Per-pitch counts for each instrument.

For every model given, writes four tables (trumpet, horn, trombone, tuba).
Each row is one pitch, and the columns are plain counts:

    train       CocoChorales training notes of this instrument at this pitch
                (0 = the model never saw this instrument play this pitch)
    real        real notes of this instrument at this pitch (answer key)
    correct     found, and given the right instrument
    wrong       real notes not counted correct  (= other inst + missed)
      other     found, but given a different instrument
      missed    not found at all
    wrong as    the most common wrong instrument, with its count
    false       notes the model called THIS instrument at this pitch that
                were not this instrument: real notes of another instrument
                it mislabeled as this one, plus notes that matched nothing
    recall      correct / real  (the only fraction, for quick reading)

A summary under each table gives, as pitch ranges:
    real notes, CocoChorales training, correctly detected, everything the
    model labeled as this instrument.

Reads only the files score_transcriptions.py already saved:
    notes_reference.csv, notes_extra.csv
so nothing needs to be rerun.

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\pitch_tables.py <model> [<model> ...]
A model is a results folder name (e.g. medium_v1_choralebricks) or a full
path to a folder holding notes_reference.csv.

Writes R:\experiments\results\pitch_tables\
    <model>_tables.txt          all four tables, as printed
    <model>_<instrument>.csv    one CSV per table
The CocoChorales training counts are cached in train_pitch_counts.csv there,
so only the first run reads the training labels (about a minute).
"""
import argparse
import csv
import glob
import os
import sys
from collections import Counter, defaultdict

RESULTS = r'R:\experiments\results'
TRAIN_LABELS = r'R:\datasets\cocochorales\labels\train'
TRAIN_LIST = r'R:\splits\cocochorales_brass\train.txt'

NAMES = ['trumpet', 'horn', 'trombone', 'tuba']
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


def note_name(midi):
    return f'{NOTE_NAMES[midi % 12]}{midi // 12 - 1}'


def as_name(value):
    """Instrument column may be a name or an index; '' / missed -> None."""
    value = str(value).strip()
    if value in NAMES:
        return value
    try:
        i = int(float(value))
        return NAMES[i] if 0 <= i < len(NAMES) else None
    except ValueError:
        return None


# ---------------------------------------------------------------- inputs
def find_scores(model):
    candidates = [model] if os.path.isabs(model) or os.path.isdir(model) else []
    for sub in ('all', 'scores', ''):
        candidates.append(os.path.join(RESULTS, model, sub))
    for c in candidates:
        if os.path.isfile(os.path.join(c, 'notes_reference.csv')):
            return c
    found = sorted(os.path.relpath(os.path.dirname(p), RESULTS) for p in
                   glob.glob(os.path.join(RESULTS, '**', 'notes_reference.csv'), recursive=True))
    sys.exit(f'No notes_reference.csv found for "{model}". Folders that have one:\n  '
             + '\n  '.join(found))


def train_counts(cache):
    """{instrument: Counter(pitch -> number of CocoChorales training notes)}"""
    counts = {n: Counter() for n in NAMES}
    if os.path.isfile(cache):
        with open(cache, newline='') as f:
            for r in csv.DictReader(f):
                counts[r['instrument']][int(r['pitch'])] = int(r['notes'])
        return counts
    if not os.path.isdir(TRAIN_LABELS):
        print(f'(CocoChorales training labels not found at {TRAIN_LABELS}; '
              'the train column will be empty)')
        return None
    with open(TRAIN_LIST) as f:
        tracks = [t.strip() for t in f if t.strip()]
    print(f'Reading CocoChorales training labels ({len(tracks)} tracks, once)...')
    for t in tracks:
        path = os.path.join(TRAIN_LABELS, t + '.tsv')
        if not os.path.isfile(path):
            continue
        with open(path) as f:
            next(f)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 5:
                    inst = as_name(parts[4])
                    if inst:
                        counts[inst][int(round(float(parts[2])))] += 1
    with open(cache, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['instrument', 'pitch', 'notes'])
        for n in NAMES:
            for p in sorted(counts[n]):
                w.writerow([n, p, counts[n][p]])
    return counts


def load_model(folder):
    """Per instrument: Counters by pitch for real/correct/other/missed/false, and
    which wrong instruments each pitch went to."""
    real, correct, other, missed, false = (defaultdict(Counter) for _ in range(5))
    wrong_to = defaultdict(lambda: defaultdict(Counter))
    with open(os.path.join(folder, 'notes_reference.csv'), newline='') as f:
        rows = list(csv.DictReader(f))
    if rows and as_name(rows[0]['given_to']) is None and rows[0]['given_to'] == 'found':
        sys.exit(f'{folder}: this model gives no instrument labels, so there is nothing to table.')
    for r in rows:
        inst, pitch = as_name(r['instrument']), int(round(float(r['pitch'])))
        given = as_name(r['given_to'])
        real[inst][pitch] += 1
        if given == inst:
            correct[inst][pitch] += 1
        elif given is None:
            missed[inst][pitch] += 1
        else:
            other[inst][pitch] += 1
            wrong_to[inst][pitch][given] += 1
            false[given][pitch] += 1               # mislabeled as `given` here
    with open(os.path.join(folder, 'notes_extra.csv'), newline='') as f:
        for r in csv.DictReader(f):
            inst = as_name(r['instrument'])
            if inst:
                false[inst][int(round(float(r['pitch'])))] += 1
    return real, correct, other, missed, false, wrong_to


# ---------------------------------------------------------------- output
def span(counter):
    ps = [p for p, c in counter.items() if c > 0]
    if not ps:
        return 'none'
    return f'{note_name(min(ps))}-{note_name(max(ps))} (MIDI {min(ps)}-{max(ps)})'


def tables(model, data, train, out_dir):
    real, correct, other, missed, false, wrong_to = data
    header = ['pitch', 'midi', 'train', 'real', 'correct', 'wrong', 'other', 'missed',
              'wrong as', 'false', 'recall']
    lines = [f'Model: {model}', '']
    for inst in NAMES:
        tr = train[inst] if train else Counter()
        pitches = sorted(set(real[inst]) | set(false[inst]))
        rows = []
        for p in pitches:
            w = other[inst][p] + missed[inst][p]
            top = wrong_to[inst][p].most_common(1)
            rows.append([note_name(p), p, tr[p] if train else '', real[inst][p],
                         correct[inst][p], w, other[inst][p], missed[inst][p],
                         f'{top[0][0]} ({top[0][1]})' if top else '',
                         false[inst][p],
                         f'{correct[inst][p] / real[inst][p]:.2f}' if real[inst][p] else ''])
        tot = [sum(real[inst].values()), sum(correct[inst].values()),
               sum(other[inst].values()) + sum(missed[inst].values()),
               sum(other[inst].values()), sum(missed[inst].values()),
               sum(false[inst].values())]
        overall_wrong = Counter()
        for p in wrong_to[inst]:
            overall_wrong.update(wrong_to[inst][p])
        top = overall_wrong.most_common(1)
        total_row = ['TOTAL', '', sum(tr.values()) if train else '', *tot[:5],
                     f'{top[0][0]} ({top[0][1]})' if top else '', tot[5],
                     f'{tot[1] / tot[0]:.2f}' if tot[0] else '']

        with open(os.path.join(out_dir, f'{model}_{inst}.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows + [total_row])

        widths = [6, 5, 7, 6, 8, 6, 6, 7, 18, 6, 7]
        fmt = lambda r: ''.join(f'{str(v):>{wd}}' if i != 8 else f'  {str(v):<{wd - 2}}'
                                for i, (v, wd) in enumerate(zip(r, widths)))
        lines.append(f'=== {inst.upper()} ===')
        lines.append(fmt(header))
        for r in rows:
            mark = '   <- not in training' if train and r[2] == 0 and r[3] else ''
            lines.append(fmt(r) + mark)
        lines.append(fmt(total_row))
        labeled = Counter(correct[inst]) + Counter(false[inst])
        lines.append(f'  real notes:               {span(real[inst])}')
        if train:
            lines.append(f'  CocoChorales training:    {span(tr)}')
        lines.append(f'  correctly detected:       {span(correct[inst])}')
        lines.append(f'  anything labeled {inst + ":":<9}{span(labeled)}')
        if train:
            outside = sum(real[inst][p] for p in real[inst] if tr[p] == 0)
            lines.append(f'  real notes at pitches never in training: {outside} of {tot[0]}')
        lines.append('')
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='+', help='results folder name(s) or path(s)')
    ap.add_argument('--out', default=os.path.join(RESULTS, 'pitch_tables'))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    train = train_counts(os.path.join(args.out, 'train_pitch_counts.csv'))
    for model in args.models:
        folder = find_scores(model)
        name = os.path.basename(os.path.normpath(model))
        lines = tables(name, load_model(folder), train, args.out)
        text = '\n'.join(lines)
        print(text)
        with open(os.path.join(args.out, f'{name}_tables.txt'), 'w') as f:
            f.write(text + '\n')
    print(f'Saved to {args.out}')


if __name__ == '__main__':
    main()
