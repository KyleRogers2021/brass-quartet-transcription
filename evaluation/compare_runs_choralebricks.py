r"""
Compare all Onsets and Frames runs (baseline and augmented) on ChoraleBricks,
averaging the repeat runs of each type.

Finds every O&F run with finished ChoraleBricks predictions (the final
checkpoint, folder 'choralebricks') in R:\experiments\predictions, scores each
with the shared scoring code (score_transcriptions.py) on all 158 items, and
groups the runs by type: baseline, pitch, stems, both (+/-5), and the
shift-size runs both +/-3, +/-7, +/-9 (folders ..._aug_both_s3-...).

Prints three tables:
    1. each type: the mean over its runs, the spread of aware F1 (standard
       deviation and min-max) and the number of runs
    2. the shift-size ablation: stem subsets with shift 0 (the 'stems' runs),
       3, 5, 7 and 9 semitones
    3. every single run, so any number can be traced back
Measures: note F1 (agnostic and aware), aware precision and recall, instrument
naming (share of found notes given the right instrument), register test (share
of labels that match the pitch-only guess), and the share of real tuba notes found.

A run is scored once; its scores are kept in R:\experiments\results\
compare_runs_choralebricks\<run name>\ and reused next time (delete that
folder to rescore it). Runs whose predictions are missing or incomplete are
listed and left out.

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
DEFAULT_SHIFT = 5
RUN_RE = re.compile(r'oaf_cocochorales-(?:aug_(pitch|stems|both)(?:_s(\d+))?-)?\d{6}-\d{6}')
MEASURES = ['agnostic_f1', 'aware_precision', 'aware_recall', 'aware_f1',
            'naming_accuracy', 'label_matches_pitch_guess', 'tuba_found']


def type_of(run_name):
    """('both', 3) for ..._aug_both_s3-...; ('baseline', 0) for the plain runs; None if not an O&F run."""
    m = RUN_RE.fullmatch(run_name)
    if not m:
        return None
    v = m.group(1) or 'baseline'
    shift = (int(m.group(2)) if m.group(2) else DEFAULT_SHIFT) if v in ('pitch', 'both') else 0
    return v, shift


def type_label(t):
    v, shift = t
    return f'{v} +/-{shift}' if v in ('pitch', 'both') else v


def type_order(t):
    return (['baseline', 'pitch', 'stems', 'both'].index(t[0]), t[1])


def get(table, measure, level):
    for avg, m, lv, p, r, f1 in table:
        if avg == 'per_track' and m == measure and lv == level:
            return p, r, f1


def score_run(name, folder, guess):
    """Scores of one run's final predictions, as a dict of MEASURES (cached on disk)."""
    out = os.path.join(OUT, name)
    cache = os.path.join(out, 'scores.csv')
    if os.path.exists(cache):
        with open(cache, newline='') as f:
            row = next(csv.DictReader(f))
        return {k: float(row[k]) for k in MEASURES}
    print(f'  scoring {name} ...')
    table = run(folder, REF, ALL_LIST, out, f'O&F {name} / ChoraleBricks all', quiet=True)
    agn, aw = get(table, 'note', 'agnostic'), get(table, 'note', 'aware')
    with open(os.path.join(out, 'notes_reference.csv'), newline='') as f:
        notes = list(csv.DictReader(f))
    labeled = [r for r in notes if r['given_to'] in NAMES]
    tuba = [r for r in notes if r['instrument'] == 'tuba']
    scores = dict(
        agnostic_f1=agn[2], aware_precision=aw[0], aware_recall=aw[1], aware_f1=aw[2],
        naming_accuracy=sum(r['given_to'] == r['instrument'] for r in labeled) / max(len(labeled), 1),
        label_matches_pitch_guess=sum(r['given_to'] == NAMES[guess(int(r['pitch']))]
                                      for r in labeled) / max(len(labeled), 1),
        tuba_found=sum(r['given_to'] != 'missed' for r in tuba) / max(len(tuba), 1))
    with open(cache, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=MEASURES)
        w.writeheader()
        w.writerow(scores)
    return scores


def mean(xs):
    return sum(xs) / len(xs)


def sd(xs):
    """Sample standard deviation (0 for a single run)."""
    if len(xs) < 2:
        return 0.0
    m = mean(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def row_text(label, s, extra=''):
    return (f'  {label:<44}{s["agnostic_f1"]:>8.3f}{s["aware_precision"]:>9.3f}{s["aware_recall"]:>9.3f}'
            f'{s["aware_f1"]:>10.3f}{s["naming_accuracy"]:>9.1%}{s["label_matches_pitch_guess"]:>9.1%}'
            f'{s["tuba_found"]:>8.1%}{extra}')


HEADER = (f'  {"":<44}{"agn F1":>8}{"aware P":>9}{"aware R":>9}{"aware F1":>10}'
          f'{"naming":>9}{"=pitch":>9}{"tuba":>8}')


def main():
    with open(ALL_LIST) as f:
        n_items = len(f.read().split())
    guess = build_guesser()

    by_type, skipped = {}, []
    for d in sorted(glob.glob(os.path.join(PRED, 'oaf_cocochorales-*'))):
        name = os.path.basename(d)
        t = type_of(name)
        if t is None:
            continue
        folder = os.path.join(d, 'choralebricks')
        done = len(glob.glob(os.path.join(folder, '*.tsv')))
        if done < n_items:
            skipped.append(f'{name} ({done} of {n_items} items predicted)')
            continue
        by_type.setdefault(t, []).append((name, score_run(name, folder, guess)))
    if not by_type:
        sys.exit('No finished O&F predictions on ChoraleBricks found.')
    types = sorted(by_type, key=type_order)

    lines = [f'Onsets and Frames on ChoraleBricks (all {n_items} items, per-item average note scores)',
             '',
             '1. Each type: mean over its runs; spread = aware F1 standard deviation and min-max',
             HEADER + f'{"aware F1 spread":>24}{"runs":>6}']
    summary = {}
    for t in types:
        runs = [s for _, s in by_type[t]]
        m = {k: mean([s[k] for s in runs]) for k in MEASURES}
        f1s = [s['aware_f1'] for s in runs]
        summary[t] = (m, f1s)
        spread = f'sd {sd(f1s):.3f} ({min(f1s):.3f}-{max(f1s):.3f})' if len(f1s) > 1 else '(one run)'
        lines.append(row_text(type_label(t), m, f'{spread:>24}{len(f1s):>6}'))

    ablation = [t for t in types if t == ('stems', 0) or t[0] == 'both']
    if len(ablation) > 1:
        lines += ['', '2. Shift-size ablation (all with stem subsets; shift 0 = the stems runs)',
                  HEADER + f'{"aware F1 spread":>24}{"runs":>6}']
        for t in sorted(ablation, key=lambda t: t[1]):
            m, f1s = summary[t]
            spread = f'sd {sd(f1s):.3f} ({min(f1s):.3f}-{max(f1s):.3f})' if len(f1s) > 1 else '(one run)'
            lines.append(row_text('shift 0  (stems)' if t[1] == 0 else f'shift +/-{t[1]}', m,
                                  f'{spread:>24}{len(f1s):>6}'))

    lines += ['', '3. Every run', HEADER]
    for t in types:
        for name, s in by_type[t]:
            lines.append(row_text(f'{type_label(t)}  {name[len("oaf_cocochorales-"):]}', s))
    lines += ['',
              '  naming = share of found notes given the right instrument (25% random, ~35% always-trumpet)',
              '  =pitch = share of labels that match the pitch-only guess (register test: if it is above naming,',
              '           the labels follow register more than sound; the guess itself is right ~22% of the time)',
              '  tuba   = share of real tuba notes found at all (any instrument label)']
    if skipped:
        lines += ['', 'Left out (predictions missing or incomplete):'] + [f'  {x}' for x in skipped]
    report = '\n'.join(lines)
    print('\n' + report)

    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'summary.txt'), 'w') as f:
        f.write(report + '\n')
    with open(os.path.join(OUT, 'summary_by_type.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['type', 'variant', 'max_shift', 'runs'] + [f'{k}_mean' for k in MEASURES]
                   + ['aware_f1_sd', 'aware_f1_min', 'aware_f1_max'])
        for t in types:
            m, f1s = summary[t]
            w.writerow([type_label(t), t[0], t[1], len(f1s)] + [m[k] for k in MEASURES]
                       + [sd(f1s), min(f1s), max(f1s)])
    with open(os.path.join(OUT, 'summary_by_run.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['type', 'run'] + MEASURES)
        for t in types:
            for name, s in by_type[t]:
                w.writerow([type_label(t), name] + [s[k] for k in MEASURES])
    print(f'\nSaved to {OUT}')
    print('SCORING DONE')


if __name__ == '__main__':
    main()
