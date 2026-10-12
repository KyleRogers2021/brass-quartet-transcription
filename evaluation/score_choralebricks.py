r"""
Score every model that has ChoraleBricks predictions, separately for solos,
duets, trios and quartets, with the shared scoring code (score_transcriptions.py,
same rules as CocoChorales), and print one compact comparison table.

Full details for each model and size (per-instrument scores, confusion tables,
per-note records) are saved in R:\experiments\results\<model>_choralebricks\<size>\.

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\score_choralebricks.py
"""
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score_transcriptions import run   # noqa: E402

PRED = r'R:\experiments\predictions'
RESULTS = r'R:\experiments\results'
REF = r'R:\datasets\choralebricks\labels\test'
LISTS = r'R:\splits\choralebricks_brass'
SIZES = ['solo', 'duet', 'trio', 'quartet', 'all']
OAF_RUNS = sorted(glob.glob(os.path.join(PRED, 'oaf_cocochorales-*')), key=os.path.getmtime)
MODELS = {
    'Onsets and Frames': OAF_RUNS[-1] if OAF_RUNS else None,
    'BasicPitch': os.path.join(PRED, 'basicpitch'),
    'YourMT3+': os.path.join(PRED, 'yourmt3'),
}


def get(table, measure, level):
    for avg, m, lv, p, r, f1 in table:
        if avg == 'per_track' and m == measure and lv == level:
            return f1
    return None


def main():
    rows = []
    for model, base in MODELS.items():
        pred = os.path.join(base, 'choralebricks') if base else ''
        if not pred or not os.path.isdir(pred):
            print(f'{model}: no ChoraleBricks predictions yet, skipped')
            continue
        for size in SIZES:
            lst = os.path.join(LISTS, f'{size}.txt')
            with open(lst) as f:
                tracks = f.read().split()
            if not tracks:
                continue
            missing = [t for t in tracks if not os.path.exists(os.path.join(pred, f'{t}.tsv'))]
            if missing:
                print(f'{model}: {len(missing)} {size} items not predicted yet, skipped')
                continue
            out = os.path.join(RESULTS, os.path.basename(os.path.normpath(base)) + '_choralebricks', size)
            table = run(pred, REF, lst, out, f'{model} / ChoraleBricks {size}', quiet=True)
            rows.append([model, size, len(tracks),
                         get(table, 'note', 'agnostic'), get(table, 'note', 'aware'),
                         get(table, 'note_offsets', 'agnostic'), get(table, 'frame', 'agnostic')])

    fmt = lambda v: f'{v:.3f}' if v is not None else '  n/a'
    lines = ['ChoraleBricks real recordings, per-item average F1',
             f'  {"model":<18}{"size":<9}{"items":>6}{"note":>9}{"note":>9}{"offsets":>9}{"frame":>9}',
             f'  {"":<18}{"":<9}{"":>6}{"agnostic":>9}{"aware":>9}{"agnostic":>9}{"agnostic":>9}']
    last = None
    for model, size, n, a, w, o, fr in rows:
        if last and model != last:
            lines.append('')
        lines.append(f'  {model if model != last else "":<18}{size:<9}{n:>6}'
                     f'{fmt(a):>9}{fmt(w):>9}{fmt(o):>9}{fmt(fr):>9}')
        last = model
    print('\n' + '\n'.join(lines))

    for model, base in MODELS.items():
        conf = os.path.join(RESULTS, os.path.basename(os.path.normpath(base or '')) + '_choralebricks',
                            'all', 'confusion.csv')
        if base and os.path.exists(conf):
            print(f'\n{model}: where each instrument\'s notes went (all ChoraleBricks items)')
            with open(conf) as f:
                for row in csv.reader(f):
                    print('  ' + ''.join(f'{c:>10}' if i else f'{c:<16}' for i, c in enumerate(row)))
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, 'choralebricks_summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'\nTable saved to {os.path.join(RESULTS, "choralebricks_summary.txt")}')
    print('SCORING DONE')


if __name__ == '__main__':
    main()
