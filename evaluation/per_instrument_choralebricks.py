r"""
Per-instrument results on ChoraleBricks for every Onsets and Frames run type,
averaged over the repeat runs, plus the other models for comparison.

Reads the per_instrument.csv files that compare_runs_choralebricks.py already
saved for each run (R:\experiments\results\compare_runs_choralebricks\<run>\),
so nothing is rescored and it takes seconds. Run the comparison first.
Other models (YourMT3+ etc.) come from R:\experiments\results\<model>_choralebricks\all\.

Prints, for each type (rows) and instrument (columns):
    1. aware F1, mean over the runs of that type (with the standard deviation)
    2. aware recall (share of each instrument's real notes found and named right)
    3. aware precision (share of notes given that instrument that were right)

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\per_instrument_choralebricks.py
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compare_runs_choralebricks import OUT, RESULTS, type_of, type_label, type_order, mean, sd  # noqa: E402

NAMES = ['trumpet', 'horn', 'trombone', 'tuba']
MEASURES = [('f1', 'Aware F1'), ('recall', 'Aware recall'), ('precision', 'Aware precision')]


def read_per_instrument(path):
    with open(path, newline='') as f:
        return {r['instrument']: {k: float(r[k]) for k in ('precision', 'recall', 'f1')}
                for r in csv.DictReader(f)}


def main():
    by_type = {}
    for path in sorted(glob.glob(os.path.join(OUT, 'oaf_cocochorales-*', 'per_instrument.csv'))):
        t = type_of(os.path.basename(os.path.dirname(path)))
        if t:
            by_type.setdefault(t, []).append(read_per_instrument(path))
    if not by_type:
        sys.exit(f'No per-run results in {OUT}. Run compare_runs_choralebricks.py first.')
    others = {}
    for path in sorted(glob.glob(os.path.join(RESULTS, '*_choralebricks', 'all', 'per_instrument.csv'))):
        name = os.path.basename(os.path.dirname(os.path.dirname(path)))[:-len('_choralebricks')]
        others[name] = read_per_instrument(path)

    lines = ['Per-instrument results on ChoraleBricks (all items, per-item averages).',
             'O&F types: mean over their runs (standard deviation in brackets for F1).']
    for key, title in MEASURES:
        lines += ['', title,
                  f'  {"":<22}{"runs":>5}' + ''.join(f'{n:>16}' for n in NAMES)]
        for t in sorted(by_type, key=type_order):
            runs = by_type[t]
            cells = ''
            for n in NAMES:
                vals = [r[n][key] for r in runs if n in r]
                text = f'{mean(vals):.3f}' + (f' ({sd(vals):.3f})' if key == 'f1' and len(vals) > 1 else '')
                cells += f'{text:>16}'
            lines.append(f'  {"O&F " + type_label(t):<22}{len(runs):>5}{cells}')
        for name, r in others.items():
            lines.append(f'  {name:<22}{1:>5}' + ''.join(
                f'{r[n][key]:>16.3f}' if n in r else f'{"":>16}' for n in NAMES))
    report = '\n'.join(lines)
    print('\n' + report)
    with open(os.path.join(OUT, 'per_instrument_by_type.txt'), 'w') as f:
        f.write(report + '\n')
    print(f'\nSaved to {os.path.join(OUT, "per_instrument_by_type.txt")}')


if __name__ == '__main__':
    main()
