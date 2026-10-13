r"""
Does longer training make Onsets and Frames worse on real recordings?

Scores every checkpoint of an O&F run that has ChoraleBricks predictions
(made with predict_oaf.py --dataset choralebricks --checkpoints ...) on all 158
items, with the shared scoring code (score_transcriptions.py), and prints one
line per checkpoint:
    note F1 (agnostic and aware), precision and recall (aware),
    instrument naming (share of found notes given the right instrument),
    and the share of real tuba notes found at all (any instrument label).

This is an analysis of how real-audio performance changes during training. It
is NOT a way to choose the final checkpoint: picking by ChoraleBricks score
would turn the test set into a validation set.

Full details per checkpoint go to
R:\experiments\results\<run>_checkpoints_choralebricks\step_<N>\.

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\score_checkpoints_choralebricks.py
    add --run <run name> for a specific run (default: newest oaf_cocochorales run)
"""
import argparse
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_transcriptions import run   # noqa: E402

PRED = r'R:\experiments\predictions'
RESULTS = r'R:\experiments\results'
REF = r'R:\datasets\choralebricks\labels\test'
ALL_LIST = r'R:\splits\choralebricks_brass\all.txt'
FINAL_STEP = 50000          # predictions of the final checkpoint live in 'choralebricks'
NAMES = ['trumpet', 'horn', 'trombone', 'tuba']


def get(table, measure, level):
    for avg, m, lv, p, r, f1 in table:
        if avg == 'per_track' and m == measure and lv == level:
            return p, r, f1
    return None


def confusion_numbers(path):
    """(instrument naming accuracy on found notes, share of tuba notes found)."""
    with open(path) as f:
        rows = {r['true_instrument']: r for r in csv.DictReader(f)}
    correct = found = 0
    for name in NAMES:
        r = rows[name]
        correct += int(r[name])
        found += sum(int(r[c]) for c in NAMES + ['other'] if c in r and r[c] != '')
    tuba = rows['tuba']
    tuba_total = sum(int(tuba[c]) for c in tuba if c != 'true_instrument' and tuba[c] != '')
    tuba_found = tuba_total - int(tuba['missed'])
    return (correct / found if found else 0.0), (tuba_found / tuba_total if tuba_total else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', help='run name (default: newest oaf_cocochorales run with predictions)')
    a = ap.parse_args()
    if a.run:
        run_name = os.path.basename(os.path.normpath(a.run))
    else:
        runs = sorted(glob.glob(os.path.join(PRED, 'oaf_cocochorales-*')), key=os.path.getmtime)
        if not runs:
            sys.exit(f'No oaf_cocochorales predictions found in {PRED}')
        run_name = os.path.basename(runs[-1])
    base = os.path.join(PRED, run_name)

    steps = {}
    for d in glob.glob(os.path.join(base, 'choralebricks*')):
        name = os.path.basename(d)
        if name == 'choralebricks':
            steps[FINAL_STEP] = d
        else:
            m = re.fullmatch(r'choralebricks_step(\d+)', name)
            if m:
                steps[int(m.group(1))] = d
    if not steps:
        sys.exit(f'No ChoraleBricks predictions found in {base}')

    with open(ALL_LIST) as f:
        n_items = len(f.read().split())
    out_base = os.path.join(RESULTS, f'{run_name}_checkpoints_choralebricks')
    rows = []
    for step in sorted(steps):
        d = steps[step]
        done = len(glob.glob(os.path.join(d, '*.tsv')))
        if done < n_items:
            print(f'  step {step}: only {done} of {n_items} items predicted, skipped')
            continue
        out = os.path.join(out_base, f'step_{step}')
        print(f'  scoring step {step} ...')
        table = run(d, REF, ALL_LIST, out, f'O&F step {step} / ChoraleBricks all', quiet=True)
        agn = get(table, 'note', 'agnostic')
        aw = get(table, 'note', 'aware')
        naming, tuba = confusion_numbers(os.path.join(out, 'confusion.csv'))
        rows.append([step, agn[2], aw[0], aw[1], aw[2], naming, tuba])

    lines = [f'O&F checkpoints on ChoraleBricks (all {n_items} items, per-item average note scores)',
             f'  {"step":>6}{"agn F1":>9}{"aware P":>9}{"aware R":>9}{"aware F1":>10}'
             f'{"naming":>9}{"tuba found":>12}']
    for step, f_agn, p, r, f_aw, naming, tuba in rows:
        lines.append(f'  {step:>6}{f_agn:>9.3f}{p:>9.3f}{r:>9.3f}{f_aw:>10.3f}{naming:>9.1%}{tuba:>12.1%}')
    lines += ['', '  naming = share of found notes given the right instrument (chance: 25% random, '
                  'about 35% always-trumpet)',
              '  tuba found = share of real tuba notes found at all (right pitch, start within 50 ms, '
              'any instrument label)']
    report = '\n'.join(lines)
    print('\n' + report)
    os.makedirs(out_base, exist_ok=True)
    with open(os.path.join(out_base, 'summary.txt'), 'w') as f:
        f.write(report + '\n')
    with open(os.path.join(out_base, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['step', 'agnostic_f1', 'aware_precision', 'aware_recall', 'aware_f1',
                    'naming_accuracy', 'tuba_found'])
        w.writerows(rows)
    print(f'\nSaved to {out_base}')
    print('SCORING DONE')


if __name__ == '__main__':
    main()
