r"""
Compare detector settings across seed repeats on ChoraleBricks.

For every run it reads the files score_transcriptions.py saved in
R:\experiments\results\detector_<run>_choralebricks and reports (all per-track
averages, note level, start within 50 ms):
    agnostic F1        notes found, instrument ignored
    aware F1           notes found with the right instrument (main measure)
    trumpet ... tuba   aware F1 of each instrument
    inst. avg          plain average of the four instrument scores
    naming             of found notes, share labeled with the true instrument
    register           of found notes, share labeled with the pitch-only guess
                       (same guesser as diagnose_model.py)
Then for each setting: mean, and min-max over its seeds. A setting is marked
  ++  every seed above every baseline seed     (clearly better)
  --  every seed below every baseline seed     (clearly worse)
  ~   ranges overlap                           (not distinguishable from noise)

Default settings (three seeds each):
  baseline  medium_v1, medium_v1_seed1, medium_v1_seed2           (+/-9, harmonics 1 2 3)
  C1 p3m12  medium_p3m12_v1, _seed1, _seed2                       (shift -12..+3)
  C2 h05    medium_h05_v1, _seed1, _seed2                         (harmonics 0.5 1 2)
Other groups:  --group "name=run1,run2,run3"  (repeatable; the first is the baseline)
100 ms scoring: --tol 100 reads the folders scored with --onset-tol 100
(detector_<run>_choralebricks_tol100) and writes compare_detector_runs_tol100.txt

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\compare_detector_runs.py
Writes R:\experiments\results\compare_detector_runs.txt and .csv
"""
import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from diagnose_model import build_guesser, as_name, NAMES      # noqa: E402

RESULTS = r'R:\experiments\results'
DEFAULT_GROUPS = [
    ('baseline', ['medium_v1', 'medium_v1_seed1', 'medium_v1_seed2']),
    ('C1 p3m12', ['medium_p3m12_v1', 'medium_p3m12_v1_seed1', 'medium_p3m12_v1_seed2']),
    ('C2 h05', ['medium_h05_v1', 'medium_h05_v1_seed1', 'medium_h05_v1_seed2']),
]
COLUMNS = ['agnostic', 'aware'] + NAMES + ['inst. avg', 'naming', 'register']


def run_scores(run, guess, suffix=''):
    folder = os.path.join(RESULTS, f'detector_{run}_choralebricks{suffix}')
    if not os.path.isfile(os.path.join(folder, 'results.csv')):
        return None
    out = {}
    with open(os.path.join(folder, 'results.csv'), newline='') as f:
        for r in csv.DictReader(f):
            if r['averaging'] == 'per_track' and r['measure'] == 'note':
                out[r['level']] = float(r['f1'])
    with open(os.path.join(folder, 'per_instrument.csv'), newline='') as f:
        for r in csv.DictReader(f):
            out[r['instrument']] = float(r['f1'])
    out['inst. avg'] = float(np.mean([out[n] for n in NAMES]))
    with open(os.path.join(folder, 'notes_reference.csv'), newline='') as f:
        labeled = [r for r in csv.DictReader(f) if as_name(r['given_to'])]
    n = max(len(labeled), 1)
    out['naming'] = sum(as_name(r['given_to']) == as_name(r['instrument']) for r in labeled) / n
    out['register'] = sum(as_name(r['given_to']) == NAMES[guess(int(round(float(r['pitch']))))]
                          for r in labeled) / n
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group', action='append', help='"name=run1,run2,run3"')
    ap.add_argument('--tol', type=int, default=50, help='note start tolerance the runs were scored with, ms')
    a = ap.parse_args()
    suffix = '' if a.tol == 50 else f'_tol{a.tol}'
    groups = DEFAULT_GROUPS if not a.group else \
        [(g.split('=', 1)[0], g.split('=', 1)[1].split(',')) for g in a.group]

    print('Building the pitch-only guesser...')
    guess = build_guesser()
    lines, rows, stats = [], [], {}
    head = f'{"run":<26}' + ''.join(f'{c:>10}' for c in COLUMNS)
    lines += [f'ChoraleBricks, per-track note F1 (onset within {a.tol} ms); naming/register as shares', '', head]
    for name, runs in groups:
        vals = []
        for run in runs:
            s = run_scores(run, guess, suffix)
            if s is None:
                lines.append(f'{run:<26}  (not scored yet: no results folder)')
                continue
            vals.append(s)
            lines.append(f'{run:<26}' + ''.join(f'{s[c]:>10.3f}' for c in COLUMNS))
            rows.append([name, run] + [round(s[c], 4) for c in COLUMNS])
        if vals:
            arr = {c: np.array([v[c] for v in vals]) for c in COLUMNS}
            stats[name] = arr
            lines.append(f'{"  " + name + " mean":<26}' + ''.join(f'{arr[c].mean():>10.3f}' for c in COLUMNS))
            lines.append(f'{"  " + name + " min-max":<26}'
                         + ''.join(f'{f"{arr[c].min():.2f}-{arr[c].max():.2f}":>10}' for c in COLUMNS))
        lines.append('')

    base = groups[0][0]
    if base in stats:
        lines.append(f'Against {base} (++ every seed above every {base} seed, -- every seed below, ~ overlap;')
        lines.append('for "register", lower is better, so its marks are reversed):')
        lines.append(f'{"":<26}' + ''.join(f'{c:>10}' for c in COLUMNS))
        for name, _ in groups[1:]:
            if name not in stats:
                continue
            marks = []
            for c in COLUMNS:
                g, b = stats[name][c], stats[base][c]
                diff = g.mean() - b.mean()
                better, worse = g.min() > b.max(), g.max() < b.min()
                if c == 'register':
                    better, worse = worse, better
                m = '++' if better else '--' if worse else '~'
                marks.append(f'{m} {diff:+.3f}')
            lines.append(f'{name:<26}' + ''.join(f'{m:>10}' for m in marks))
    text = '\n'.join(lines)
    print(text)
    with open(os.path.join(RESULTS, f'compare_detector_runs{suffix}.txt'), 'w') as f:
        f.write(text + '\n')
    with open(os.path.join(RESULTS, f'compare_detector_runs{suffix}.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['setting', 'run'] + COLUMNS)
        w.writerows(rows)
    print(f'\nSaved to {RESULTS}\\compare_detector_runs{suffix}.txt')
    print('COMPARISON DONE')


if __name__ == '__main__':
    main()
