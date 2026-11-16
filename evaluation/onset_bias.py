r"""
Onset bias check: does a model place note starts systematically early or late?

For every scored model in R:\experiments\results (every folder holding a
notes_reference.csv, CocoChorales and ChoraleBricks alike), it collects the
onset error, prediction minus true onset (negative = early), of:
    found     real notes the model matched (within the 50 ms rule; labeled
              right or wrong). Cut off at +/-50 ms, so it understates a bias.
    combined  found notes plus missed notes that have a leftover prediction at
              the same pitch within --window ms (default 150). This shows the
              bias the 50 ms rule hides.
and prints the median for each, overall and per instrument, plus the share of
combined notes that are early.

An unbiased model has medians near 0. If models trained on CocoChorales are
early on ChoraleBricks but not on CocoChorales, while YourMT3+ (not trained on
CocoChorales) is not, the two datasets mark onsets differently.

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\onset_bias.py
    python R:\code\evaluation\onset_bias.py <model> [<model> ...]   (only these)
Writes R:\experiments\results\onset_bias.txt and onset_bias.csv
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np

RESULTS = r'R:\experiments\results'
NAMES = ['trumpet', 'horn', 'trombone', 'tuba']


def read(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def errors(folder, window):
    """{instrument: (found errors ms, combined errors ms)}"""
    ref = read(os.path.join(folder, 'notes_reference.csv'))
    extra_path = os.path.join(folder, 'notes_extra.csv')
    extras = defaultdict(list)
    if os.path.isfile(extra_path):
        for r in read(extra_path):
            extras[(r['track'], int(round(float(r['pitch']))))].append(float(r['onset']))
    found, combined = defaultdict(list), defaultdict(list)
    missed = []
    for r in ref:
        inst = r['instrument']
        if r['onset_error_ms'] not in ('', None):
            e = float(r['onset_error_ms'])
            found[inst].append(e)
            combined[inst].append(e)
        else:
            missed.append(r)
    # each leftover prediction used once, closest pairs first
    cands = []
    for i, r in enumerate(missed):
        on = float(r['onset'])
        for j, t in enumerate(extras.get((r['track'], int(round(float(r['pitch'])))), [])):
            if abs(t - on) <= window:
                cands.append((abs(t - on), i, (r['track'], int(round(float(r['pitch'])))), j, (t - on) * 1000))
    cands.sort()
    used_i, used_j = set(), set()
    for _, i, key, j, e in cands:
        if i in used_i or (key, j) in used_j:
            continue
        used_i.add(i)
        used_j.add((key, j))
        combined[missed[i]['instrument']].append(e)
    return found, combined


def med(x):
    return f'{np.median(x):.0f}' if len(x) else '-'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('models', nargs='*')
    ap.add_argument('--window', type=float, default=150)
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(RESULTS, '**', 'notes_reference.csv'), recursive=True))
    folders = [os.path.dirname(p) for p in paths]
    if args.models:
        folders = [f for f in folders if any(m in os.path.relpath(f, RESULTS) for m in args.models)]

    lines, rows = [], []
    head = (f'{"model (results folder)":<58}{"found":>7}{"comb.":>7}{"early":>7}  '
            + ''.join(f'{n:>10}' for n in NAMES))
    lines += ['Onset error, prediction minus true onset, median ms (negative = early)',
              f'found = matched notes only (within 50 ms); comb. = plus same-pitch near-misses '
              f'within {args.window:.0f} ms; early = share of comb. notes that are early;',
              'per-instrument columns = comb. median', '', head]
    for folder in folders:
        found, comb = errors(folder, args.window / 1000)
        allf = [e for v in found.values() for e in v]
        allc = [e for v in comb.values() for e in v]
        if not allf:
            continue
        name = os.path.relpath(folder, RESULTS)
        early = np.mean(np.array(allc) < 0) if allc else float('nan')
        lines.append(f'{name[:57]:<58}{med(allf):>7}{med(allc):>7}{early:>7.0%}  '
                     + ''.join(f'{med(comb.get(n, [])):>10}' for n in NAMES))
        rows.append([name, len(allf), med(allf), len(allc), med(allc), round(early, 3)]
                    + [med(comb.get(n, [])) for n in NAMES])
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(RESULTS, 'onset_bias.txt'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(RESULTS, 'onset_bias.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['model', 'n_found', 'found_median_ms', 'n_combined', 'combined_median_ms',
                    'early_share'] + [f'{n}_median_ms' for n in NAMES])
        w.writerows(rows)
    print(f'\nSaved to {RESULTS}\\onset_bias.txt')
    print('ONSET BIAS DONE')


if __name__ == '__main__':
    main()
