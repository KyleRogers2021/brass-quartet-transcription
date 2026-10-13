r"""
Do real brass recordings use pitches CocoChorales never trained on, and are
low notes the ones being missed?

1. For each instrument, compares the pitch range of the real ChoraleBricks solo
   recordings with the range that instrument plays in CocoChorales (from
   R:\datasets\cocochorales\labels\summary.txt), and counts real notes outside it.
2. For tuba solos, shows how many real notes each model found, by pitch band.
   "Found" = some predicted note has the same pitch and starts within 50 ms,
   whatever instrument it was labeled. (A simple check, close to but not
   identical to mir_eval's one-to-one matching.)

Solos only, so another instrument can't "find" a tuba note by accident.

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\analyze_pitch_range_choralebricks.py
"""
import glob
import os
from collections import Counter, defaultdict

REF = r'R:\datasets\choralebricks\labels\test'
SOLO_LIST = r'R:\splits\choralebricks_brass\solo.txt'
PRED = r'R:\experiments\predictions'
OUT = r'R:\experiments\results\pitch_range_choralebricks.txt'
INSTRUMENTS = ['trumpet', 'horn', 'trombone', 'tuba']
# CocoChorales pitch range per instrument, all 60,000 tracks (labels\summary.txt)
COCO_RANGE = {0: (60, 84), 1: (56, 77), 2: (47, 72), 3: (35, 67)}
NAMES = 'C C# D D# E F F# G G# A A# B'.split()
TUBA_BANDS = [(0, 34, 'below Coco range'), (35, 40, ''), (41, 46, ''), (47, 52, ''),
              (53, 67, ''), (68, 127, 'above Coco range')]


def note_name(p):
    return f'{NAMES[p % 12]}{p // 12 - 1}'


def read(path):
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            a, b, c, d, e = line.rstrip('\n').split('\t')
            rows.append((float(a), int(c), int(e)))      # onset, pitch, instrument
    return rows


def found(ref_note, pred_by_pitch):
    onset, pitch, _ = ref_note
    return any(abs(o - onset) <= 0.05 for o in pred_by_pitch.get(pitch, ()))


def main():
    solos = open(SOLO_LIST).read().split()
    ref = {s: read(os.path.join(REF, f'{s}.tsv')) for s in solos}
    lines = []

    # ---- 1. pitch ranges
    pitches = defaultdict(list)
    for rows in ref.values():
        for _, p, i in rows:
            pitches[i].append(p)
    lines += ['1. Real (ChoraleBricks solo) pitch range vs CocoChorales range',
              f'  {"instrument":10s} {"notes":>6}  {"real range":>16}  {"Coco range":>16}  {"below":>6}  {"above":>6}']
    for i, name in enumerate(INSTRUMENTS):
        ps = pitches[i]
        if not ps:
            lines.append(f'  {name:10s} no solo notes'); continue
        lo, hi = COCO_RANGE[i]
        below = sum(p < lo for p in ps) / len(ps)
        above = sum(p > hi for p in ps) / len(ps)
        lines.append(f'  {name:10s} {len(ps):>6}  {min(ps):>3}-{max(ps):<3} ({note_name(min(ps))}-{note_name(max(ps))})'.ljust(37)
                     + f'{lo:>3}-{hi:<3} ({note_name(lo)}-{note_name(hi)})'.ljust(18)
                     + f'{below:>6.1%}  {above:>6.1%}')

    # ---- 2. tuba notes found, by pitch band
    models = {}
    oaf = sorted(glob.glob(os.path.join(PRED, 'oaf_cocochorales-*')), key=os.path.getmtime)
    if oaf:
        models['O&F'] = os.path.join(oaf[-1], 'choralebricks')
    models['BasicPitch'] = os.path.join(PRED, 'basicpitch', 'choralebricks')
    models['YourMT3+'] = os.path.join(PRED, 'yourmt3', 'choralebricks')
    tuba_items = [s for s in solos if any(i == 3 for _, _, i in ref[s])]

    lines += ['', f'2. Tuba solo notes found, by pitch band ({len(tuba_items)} tuba solos; '
              'right pitch, start within 50 ms, any instrument label)']
    header = f'  {"band":24s} {"notes":>6}' + ''.join(f'{m:>12}' for m in models)
    lines.append(header)
    totals = Counter(); hits = defaultdict(Counter)
    preds = {}
    for m, d in models.items():
        if not os.path.isdir(d):
            continue
        preds[m] = {}
        for s in tuba_items:
            byp = defaultdict(list)
            for o, p, _ in read(os.path.join(d, f'{s}.tsv')):
                byp[p].append(o)
            preds[m][s] = byp
    for s in tuba_items:
        for note in ref[s]:
            if note[2] != 3:
                continue
            band = next(b for b in TUBA_BANDS if b[0] <= note[1] <= b[1])
            totals[band] += 1
            for m in preds:
                hits[m][band] += found(note, preds[m][s])
    for lo, hi, label in TUBA_BANDS:
        b = (lo, hi, label)
        if not totals[b]:
            continue
        name = f'{lo}-{hi} ({note_name(lo)}-{note_name(min(hi, 127))})' if label == '' else label
        lines.append(f'  {name:24s} {totals[b]:>6}' +
                     ''.join(f'{hits[m][b] / totals[b]:>12.1%}' if m in preds else f'{"n/a":>12}' for m in models))
    n = sum(totals.values())
    lines.append(f'  {"all tuba notes":24s} {n:>6}' +
                 ''.join(f'{sum(hits[m].values()) / n:>12.1%}' if m in preds and n else f'{"n/a":>12}' for m in models))

    report = '\n'.join(lines)
    print(report)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        f.write(report + '\n')
    print(f'\nSaved to {OUT}')
    print('ANALYSIS DONE')


if __name__ == '__main__':
    main()
