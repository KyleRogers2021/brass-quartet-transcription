r"""
Compare all Onsets and Frames runs (baseline and augmented) on ChoraleBricks.

Finds every O&F run that has ChoraleBricks predictions in
R:\experiments\predictions, scores each with the shared scoring code
(score_transcriptions.py) on all 158 items, and prints one comparison table:
    note F1 (agnostic and aware), aware precision and recall,
    instrument naming (share of found notes given the right instrument),
    register test (share of labels that match the pitch-only guess),
    and the share of real tuba notes found at all.

Which predictions are used for each run:
    finished runs:   the final checkpoint (the 'choralebricks' folder)
    unfinished runs: their latest scored checkpoint ('choralebricks_step<N>'),
                     and the baseline at that same step is added for a fair comparison.

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\compare_runs_choralebricks.py
"""
import csv
import glob
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_transcriptions import run                       # noqa: E402
from analyze_register_shortcut import build_guesser        # noqa: E402

PRED = r'R:\experiments\predictions'
RESULTS = r'R:\experiments\results'
REF = r'R:\datasets\choralebricks\labels\test'
ALL_LIST = r'R:\splits\choralebricks_brass\all.txt'
OUT = os.path.join(RESULTS, 'compare_runs_choralebricks')
NAMES = ['trumpet', 'horn', 'trombone', 'tuba']
ORDER = ['baseline', 'pitch', 'stems', 'both']


def variant_of(run_name):
    m = re.fullmatch(r'oaf_cocochorales-(?:aug_(\w+?)-)?\d{6}-\d{6}', run_name)
    return None if not m else (m.group(1) or 'baseline')


def get(table, measure, level):
    for avg, m, lv, p, r, f1 in table:
        if avg == 'per_track' and m == measure and lv == level:
            return p, r, f1


def steps_of(run_dir):
    """{step: folder} of ChoraleBricks predictions; the final one is stored as step None."""
    out = {}
    for d in glob.glob(os.path.join(run_dir, 'choralebricks*')):
        name = os.path.basename(d)
        if name == 'choralebricks':
            out[None] = d
        else:
            m = re.fullmatch(r'choralebricks_step(\d+)', name)
            if m:
                out[int(m.group(1))] = d
    return out


def main():
    with open(ALL_LIST) as f:
        n_items = len(f.read().split())
    runs = {}
    for d in glob.glob(os.path.join(PRED, 'oaf_cocochorales-*')):
        v = variant_of(os.path.basename(d))
        if v and steps_of(d) and (v not in runs or os.path.getmtime(d) > os.path.getmtime(runs[v])):
            runs[v] = d                       # newest run of each variant
    if 'baseline' not in runs:
        sys.exit('No baseline O&F predictions found.')

    # which (label, folder) to score
    todo, extra_baseline_steps = [], set()
    for v in sorted(runs, key=ORDER.index):
        steps = steps_of(runs[v])
        if None in steps:
            todo.append((v, steps[None], 'final'))
        else:
            last = max(steps)
            todo.append((v, steps[last], f'step {last}, unfinished'))
            extra_baseline_steps.add(last)
    base_steps = steps_of(runs['baseline'])
    for s in sorted(extra_baseline_steps):
        if s in base_steps:
            todo.insert(1, ('baseline', base_steps[s], f'step {s}'))
        else:
            print(f'  (no baseline predictions at step {s}; run predict_oaf.py --checkpoints {s} to add them)')

    guess = build_guesser()
    rows = []
    for v, folder, when in todo:
        done = len(glob.glob(os.path.join(folder, '*.tsv')))
        label = f'{v} ({when})'
        if done < n_items:
            print(f'  {label}: only {done} of {n_items} items predicted, skipped')
            continue
        out = os.path.join(OUT, f'{v}_{when.split(",")[0].replace(" ", "")}')
        print(f'  scoring {label} ...')
        table = run(folder, REF, ALL_LIST, out, f'O&F {label} / ChoraleBricks all', quiet=True)
        agn, aw = get(table, 'note', 'agnostic'), get(table, 'note', 'aware')
        with open(os.path.join(out, 'notes_reference.csv'), newline='') as f:
            notes = list(csv.DictReader(f))
        labeled = [r for r in notes if r['given_to'] in NAMES]
        naming = sum(r['given_to'] == r['instrument'] for r in labeled) / max(len(labeled), 1)
        pitchy = sum(r['given_to'] == NAMES[guess(int(r['pitch']))] for r in labeled) / max(len(labeled), 1)
        tuba = [r for r in notes if r['instrument'] == 'tuba']
        tuba_found = sum(r['given_to'] != 'missed' for r in tuba) / max(len(tuba), 1)
        rows.append([label, agn[2], aw[0], aw[1], aw[2], naming, pitchy, tuba_found])

    lines = [f'Onsets and Frames runs on ChoraleBricks (all {n_items} items, per-item average note scores)',
             f'  {"run":<34}{"agn F1":>8}{"aware P":>9}{"aware R":>9}{"aware F1":>10}'
             f'{"naming":>9}{"=pitch":>9}{"tuba":>8}']
    for label, f_agn, p, r, f_aw, naming, pitchy, tuba in rows:
        lines.append(f'  {label:<34}{f_agn:>8.3f}{p:>9.3f}{r:>9.3f}{f_aw:>10.3f}'
                     f'{naming:>9.1%}{pitchy:>9.1%}{tuba:>8.1%}')
    lines += ['',
              '  naming = share of found notes given the right instrument (25% random, ~35% always-trumpet)',
              '  =pitch = share of labels that match the pitch-only guess (register test: if it is above naming,',
              '           the labels follow register more than sound; the guess itself is right ~22% of the time)',
              '  tuba   = share of real tuba notes found at all (any instrument label)']
    report = '\n'.join(lines)
    print('\n' + report)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
        f.write(report + '\n')
    with open(os.path.join(OUT, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 'agnostic_f1', 'aware_precision', 'aware_recall', 'aware_f1',
                    'naming_accuracy', 'label_matches_pitch_guess', 'tuba_found'])
        w.writerows(rows)
    print(f'\nSaved to {OUT}')
    print('SCORING DONE')


if __name__ == '__main__':
    main()
