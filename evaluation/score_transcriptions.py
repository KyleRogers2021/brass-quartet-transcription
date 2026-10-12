r"""
Shared scoring for every model (Onsets and Frames, BasicPitch, YourMT3+, YOLO).

Compares each model's predicted note lists with the answer key, using the
same rules for every model. It never runs a model; it only reads note lists.

Input format (answer key and predictions alike), one .tsv per track, header row:
    onset   offset   pitch   velocity   instrument
    (sec)   (sec)    (MIDI)  (ignored)  0=trumpet 1=horn 2=trombone 3=tuba,
                                        4=other (any non-brass-quartet instrument),
                                        -1=unknown
A model that doesn't identify instruments (BasicPitch) writes -1, and gets
instrument-agnostic scores only. Notes labeled 4 (other) count in agnostic
scoring but can never be correct in aware scoring.

Measures (mir_eval note matching, the field's standard library):
    note           pitch right, start within 50 ms
    note_offsets   also end within 20% of note length or 50 ms (whichever is larger)
    frame          which pitches are sounding, checked every 10 ms
Each at two levels:
    agnostic       instruments ignored: did it find the music?
    aware          notes only match within the same instrument
And two ways of averaging over tracks:
    per_track      score each track, then average (every track counts equally;
                   what most papers report)
    pooled         add up all notes first, then score (busy tracks count more)
Velocity is never scored.

Writes to the output folder:
    summary.txt             the printed results tables
    results.csv             every measure x level x averaging, P / R / F1
    per_instrument.csv      aware scores for each instrument
    confusion.csv           where each instrument's notes went (right, other
                            instrument, missed) + extra notes per instrument
    per_track.csv           every track's scores
    notes_reference.csv     every answer-key note: found? given to whom? how far off?
    notes_extra.csv         every predicted note that matched nothing

Run from the Miniforge Prompt (oaf environment):
    python R:\code\evaluation\score_transcriptions.py --selftest
    python R:\code\evaluation\score_transcriptions.py --pred <prediction folder> --name <run name>
Options (defaults are the CocoChorales brass test set):
    --ref   answer-key folder   (R:\datasets\cocochorales\labels\test)
    --list  track list          (R:\splits\cocochorales_brass\test.txt)
    --out   output folder       (R:\experiments\results\<name>\scores)
"""
import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from mir_eval.transcription import match_notes
from mir_eval.util import midi_to_hz

INSTRUMENTS = ['trumpet', 'horn', 'trombone', 'tuba']
NAMES = INSTRUMENTS + ['other']      # predictions may also use 4 = other
ONSET_TOL = 0.05
OFFSET_RATIO = 0.2
OFFSET_MIN = 0.05
FRAME_HOP = 0.01
WORKERS = 6

DEFAULT_REF = r'R:\datasets\cocochorales\labels\test'
DEFAULT_LIST = r'R:\splits\cocochorales_brass\test.txt'
DEFAULT_RESULTS = r'R:\experiments\results'
MEASURES = ['note', 'note_offsets', 'frame']


# ---------------------------------------------------------------- reading
def read_notes(path):
    """Array of rows (onset, offset, pitch, instrument); velocity dropped."""
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 5:
                on, off, pitch, _vel, inst = parts[:5]
                rows.append((float(on), float(off), float(pitch), float(inst)))
    return np.array(rows, dtype=float).reshape(-1, 4)


# ---------------------------------------------------------------- matching
def match(ref, est, offsets):
    """List of (ref_index, est_index) pairs, mir_eval bipartite matching."""
    if len(ref) == 0 or len(est) == 0:
        return []
    return match_notes(ref[:, :2], midi_to_hz(ref[:, 2]), est[:, :2], midi_to_hz(est[:, 2]),
                       onset_tolerance=ONSET_TOL,
                       offset_ratio=OFFSET_RATIO if offsets else None,
                       offset_min_tolerance=OFFSET_MIN)


def match_aware(ref, est, offsets):
    """Match within each instrument separately; indices refer to the full arrays."""
    pairs = []
    for i in range(len(INSTRUMENTS)):
        ri, ei = np.where(ref[:, 3] == i)[0], np.where(est[:, 3] == i)[0]
        pairs += [(ri[a], ei[b]) for a, b in match(ref[ri], est[ei], offsets)]
    return pairs


def frame_counts(ref, est, aware):
    """(true positives, n_ref, n_est) over 10 ms frames x pitch (x instrument)."""
    end = max(ref[:, 1].max() if len(ref) else 0, est[:, 1].max() if len(est) else 0)
    n = int(np.ceil(end / FRAME_HOP)) + 1
    width = 128 * (len(NAMES) if aware else 1)

    def roll(notes):
        r = np.zeros((n, width), bool)
        for on, off, p, inst in notes:
            col = int(round(p)) + (128 * int(inst) if aware else 0)
            a, b = int(round(on / FRAME_HOP)), int(round(off / FRAME_HOP))
            r[a:max(b, a + 1), col] = True
        return r
    r, e = roll(ref), roll(est)
    return int((r & e).sum()), int(r.sum()), int(e.sum())


def prf(tp, n_ref, n_est):
    p = tp / n_est if n_est else 0.0
    r = tp / n_ref if n_ref else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


# ---------------------------------------------------------------- one track
def score_track(args):
    track, ref_path, pred_path = args
    ref, est = read_notes(ref_path), read_notes(pred_path)
    if len(est) and ((est[:, 3] < -1) | (est[:, 3] > 4)).any():
        raise ValueError(f'{pred_path}: instrument must be -1 or 0-4')
    has_inst = len(est) == 0 or bool((est[:, 3] >= 0).all())
    out = {'track': track, 'n_ref': len(ref), 'n_est': len(est), 'has_inst': has_inst}

    # counts[level][measure] = (tp, n_ref, n_est)
    counts = {'agnostic': {}, 'aware': {}}
    for offsets, m in ((False, 'note'), (True, 'note_offsets')):
        counts['agnostic'][m] = (len(match(ref, est, offsets)), len(ref), len(est))
        if has_inst:
            counts['aware'][m] = (len(match_aware(ref, est, offsets)), len(ref), len(est))
    counts['agnostic']['frame'] = frame_counts(ref, est, aware=False)
    if has_inst:
        counts['aware']['frame'] = frame_counts(ref, est, aware=True)
    out['counts'] = counts

    # per-instrument aware note counts: (tp, n_ref, n_est)
    per_inst = {}
    notes, extras = [], []
    if has_inst:
        pairs = match_aware(ref, est, offsets=False)
        pairs_off = set(r for r, _ in match_aware(ref, est, offsets=True))
        found = {r: e for r, e in pairs}
        for i in range(len(INSTRUMENTS)):
            tp = sum(1 for r, _ in pairs if ref[r, 3] == i)
            per_inst[i] = (tp, int((ref[:, 3] == i).sum()), int((est[:, 3] == i).sum()))
        # confusion: among leftover notes, match ignoring instrument
        left_r = np.array([k for k in range(len(ref)) if k not in found], int)
        used_e = set(found.values())
        left_e = np.array([k for k in range(len(est)) if k not in used_e], int)
        conf = {}
        for a, b in match(ref[left_r], est[left_e], offsets=False):
            conf[left_r[a]] = left_e[b]
        conf_e = set(conf.values())
    else:
        pairs_ag = match(ref, est, offsets=False)
        found = {r: e for r, e in pairs_ag}
        pairs_off = set(r for r, _ in match(ref, est, offsets=True))
        conf, conf_e = {}, set()
        used_e = set(found.values())

    # per-note records, with context useful for finding patterns in misses
    order = np.lexsort((ref[:, 0], ref[:, 3]))
    prev = {}
    for k in order:
        on, off, p, inst = ref[k]
        last = prev.get(inst)
        repeated = last is not None and last[2] == p
        gap = (on - last[1]) * 1000 if last is not None else ''
        prev[inst] = ref[k]
        unison = bool(((ref[:, 2] == p) & (ref[:, 3] != inst) &
                       (ref[:, 0] <= on + ONSET_TOL) & (ref[:, 1] > on)).any())
        if k in found:
            given = INSTRUMENTS[int(inst)] if has_inst else 'found'
            err = round((est[found[k], 0] - on) * 1000, 1)
        elif k in conf:
            given = NAMES[int(est[conf[k], 3])]
            err = round((est[conf[k], 0] - on) * 1000, 1)
        else:
            given, err = 'missed', ''
        notes.append([track, INSTRUMENTS[int(inst)], int(p), round(on, 3), round(off - on, 3),
                      k in found, k in pairs_off, given, err, repeated,
                      round(gap, 1) if gap != '' else '', unison])
    for k in range(len(est)):
        if k not in used_e and k not in conf_e:
            on, off, p, inst = est[k]
            extras.append([track, NAMES[int(inst)] if inst >= 0 else 'unknown', int(p),
                           round(on, 3), round(off - on, 3)])
    out['per_inst'], out['notes'], out['extras'] = per_inst, notes, extras
    return out


# ---------------------------------------------------------------- whole run
def run(pred_dir, ref_dir, list_path, out_dir, name, quiet=False):
    with open(list_path) as f:
        tracks = f.read().split()
    missing = [t for t in tracks if not os.path.exists(os.path.join(pred_dir, f'{t}.tsv'))]
    if missing:
        sys.exit(f'{len(missing)} of {len(tracks)} tracks have no prediction file in {pred_dir} '
                 f'(first: {missing[0]}). Scoring stopped so a partial run is not mistaken '
                 f'for a full one.')
    os.makedirs(out_dir, exist_ok=True)
    jobs = [(t, os.path.join(ref_dir, f'{t}.tsv'), os.path.join(pred_dir, f'{t}.tsv'))
            for t in tracks]
    t0 = time.time()
    results = []
    with Pool(WORKERS) as pool:
        for k, r in enumerate(pool.imap(score_track, jobs, chunksize=20), 1):
            results.append(r)
            if not quiet and (k % 500 == 0 or k == len(jobs)):
                print(f'  scored {k}/{len(jobs)} tracks ({time.time() - t0:.0f} s)')

    has_inst = all(r['has_inst'] for r in results)
    levels = ['agnostic', 'aware'] if has_inst else ['agnostic']
    lines = [f'Scores for {name}: {len(results)} tracks', '']
    table = []
    for avg in ('per_track', 'pooled'):
        lines.append(f'{"Per-track average" if avg == "per_track" else "Pooled"}:')
        lines.append(f'  {"measure":<14}{"level":<10}{"precision":>10}{"recall":>9}{"F1":>8}')
        for m in MEASURES:
            for lv in levels:
                if avg == 'per_track':
                    vals = np.array([prf(*r['counts'][lv][m]) for r in results])
                    p, rc, f1 = vals.mean(axis=0)
                else:
                    tp, nr, ne = np.array([r['counts'][lv][m] for r in results]).sum(axis=0)
                    p, rc, f1 = prf(tp, nr, ne)
                table.append([avg, m, lv, round(p, 4), round(rc, 4), round(f1, 4)])
                lines.append(f'  {m:<14}{lv:<10}{p:>10.3f}{rc:>9.3f}{f1:>8.3f}')
        if not has_inst:
            lines.append('  (aware: n/a, this model does not identify instruments)')
        lines.append('')
    with open(os.path.join(out_dir, 'results.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['averaging', 'measure', 'level', 'precision', 'recall', 'f1'])
        w.writerows(table)

    with open(os.path.join(out_dir, 'per_track.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['track', 'n_ref', 'n_est'] + [f'{lv}_{m}_f1' for lv in levels for m in MEASURES])
        for r in results:
            w.writerow([r['track'], r['n_ref'], r['n_est']] +
                       [round(prf(*r['counts'][lv][m])[2], 4) for lv in levels for m in MEASURES])

    with open(os.path.join(out_dir, 'notes_reference.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['track', 'instrument', 'pitch', 'onset', 'duration', 'found',
                    'found_with_offset', 'given_to', 'onset_error_ms', 'repeated_pitch',
                    'gap_before_ms', 'unison'])
        for r in results:
            w.writerows(r['notes'])
    with open(os.path.join(out_dir, 'notes_extra.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['track', 'instrument', 'pitch', 'onset', 'duration'])
        for r in results:
            w.writerows(r['extras'])

    if has_inst:
        lines.append('Per instrument (aware note scores, per-track average | pooled F1):')
        lines.append(f'  {"instrument":<11}{"precision":>10}{"recall":>9}{"F1":>8}{"pooled F1":>11}')
        per_rows = []
        for i, nm in enumerate(INSTRUMENTS):
            vals = np.array([prf(*r['per_inst'][i]) for r in results if r['per_inst'][i][1]])
            p, rc, f1 = vals.mean(axis=0) if len(vals) else (0, 0, 0)
            pooled = prf(*np.array([r['per_inst'][i] for r in results]).sum(axis=0))[2]
            per_rows.append([nm, round(p, 4), round(rc, 4), round(f1, 4), round(pooled, 4)])
            lines.append(f'  {nm:<11}{p:>10.3f}{rc:>9.3f}{f1:>8.3f}{pooled:>11.3f}')
        with open(os.path.join(out_dir, 'per_instrument.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['instrument', 'precision', 'recall', 'f1', 'pooled_f1'])
            w.writerows(per_rows)

        cols = NAMES + ['missed']
        conf = {a: {b: 0 for b in cols} for a in INSTRUMENTS}
        extra = {a: 0 for a in NAMES}
        for r in results:
            for n in r['notes']:
                conf[n[1]][n[7]] += 1
            for e in r['extras']:
                if e[1] in extra:
                    extra[e[1]] += 1
        lines += ['', 'Where each instrument\'s notes went (rows = true instrument, '
                      'note level, start within 50 ms):',
                  '  ' + f'{"":<11}' + ''.join(f'{c:>10}' for c in cols)]
        for a in INSTRUMENTS:
            lines.append(f'  {a:<11}' + ''.join(f'{conf[a][c]:>10}' for c in cols))
        lines.append(f'  {"extra":<11}' + ''.join(f'{extra[c]:>10}' for c in NAMES) +
                     '   (predicted notes matching nothing)')
        with open(os.path.join(out_dir, 'confusion.csv'), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['true_instrument'] + cols)
            for a in INSTRUMENTS:
                w.writerow([a] + [conf[a][c] for c in cols])
            w.writerow(['extra'] + [extra[c] for c in NAMES] + [''])

    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    if not quiet:
        print('\n'.join(lines))
        print(f'\nSaved to {out_dir}')
    return table


# ---------------------------------------------------------------- self-test
def selftest(ref_dir, list_path, base):
    """Score the answer key against known changes of itself; results must match expectations."""
    import random
    import shutil
    rng = random.Random(0)
    with open(list_path) as f:
        tracks = f.read().split()[:200]
    tmp = os.path.join(base, '_scoring_selftest')
    shutil.rmtree(tmp, ignore_errors=True)
    lst = os.path.join(tmp, 'list.txt')
    os.makedirs(tmp)
    with open(lst, 'w') as f:
        f.write('\n'.join(tracks))
    header = 'onset\toffset\tpitch\tvelocity\tinstrument\n'

    def make(name, change):
        d = os.path.join(tmp, name)
        os.makedirs(d)
        for t in tracks:
            notes = read_notes(os.path.join(ref_dir, f'{t}.tsv'))
            with open(os.path.join(d, f'{t}.tsv'), 'w') as f:
                f.write(header)
                for on, off, p, inst in change(notes):
                    f.write(f'{on:.4f}\t{off:.4f}\t{int(p)}\t64\t{int(inst)}\n')
        return d

    def get(table, avg, m, lv):
        return next(r[5] for r in table if r[0] == avg and r[1] == m and r[2] == lv)

    def drop_half_horn(n):
        keep = [row for row in n if not (row[3] == 1 and rng.random() < 0.5)]
        return np.array(keep).reshape(-1, 4)

    def horn_as_other(n):
        n = n.copy()
        n[n[:, 3] == 1, 3] = 4
        return n

    def horn_as_trombone(n):
        n = n.copy()
        n[n[:, 3] == 1, 3] = 2
        return n

    cases = [
        ('identical copy', lambda n: n,
         [('note', 'agnostic', 1.0), ('note', 'aware', 1.0), ('frame', 'aware', 1.0)]),
        ('all starts 30 ms late (inside 50 ms)', lambda n: n + [0.03, 0.03, 0, 0],
         [('note', 'agnostic', 1.0), ('note', 'aware', 1.0)]),
        # agnostic can stay slightly above 0: a shifted note may land on ANOTHER
        # instrument's note of the same pitch, which agnostic scoring rightly counts
        ('all starts 80 ms late (outside 50 ms)', lambda n: n + [0.08, 0.08, 0, 0],
         [('note', 'agnostic', '<0.01'), ('note', 'aware', 0.0)]),
        ('horn labeled as trombone', horn_as_trombone,
         [('note', 'agnostic', 1.0), ('note', 'aware', '<1')]),
        ('horn labeled as other (4)', horn_as_other,
         [('note', 'agnostic', 1.0), ('note', 'aware', '<1')]),
        ('no instruments given (-1)', lambda n: n * [1, 1, 1, 0] - [0, 0, 0, 1],
         [('note', 'agnostic', 1.0)]),
        ('half the horn notes deleted', drop_half_horn,
         [('note', 'agnostic', '<1'), ('note', 'aware', '<1')]),
    ]
    ok = True
    print('Self-test: scoring the answer key against known changes of itself (200 tracks)')
    for name, change, expect in cases:
        d = make(name.split(' (')[0].replace(' ', '_'), change)
        table = run(d, ref_dir, lst, os.path.join(d, 'scores'), name, quiet=True)
        got = []
        for m, lv, want in expect:
            f1 = get(table, 'per_track', m, lv)
            if want == '<1':
                good = f1 < 0.999
            elif want == '<0.01':
                good = f1 < 0.01
            else:
                good = abs(f1 - want) < 1e-6
            ok &= good
            got.append(f'{lv} {m} F1 {f1:.3f}{"" if good else " (WRONG)"}')
        print(f'  {name:<40}' + '   '.join(got))
    shutil.rmtree(tmp, ignore_errors=True)
    print('SELF-TEST PASSED' if ok else 'SELF-TEST FAILED: send me this output')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Score predicted note lists against the answer key.')
    ap.add_argument('--pred', help='folder of predicted <track>.tsv files')
    ap.add_argument('--name', help='run name, used for the output folder')
    ap.add_argument('--ref', default=DEFAULT_REF)
    ap.add_argument('--list', default=DEFAULT_LIST)
    ap.add_argument('--out')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        selftest(a.ref, a.list, DEFAULT_RESULTS)
    elif a.pred and a.name:
        run(a.pred, a.ref, a.list, a.out or os.path.join(DEFAULT_RESULTS, a.name, 'scores'), a.name)
    else:
        ap.error('give --pred and --name, or --selftest')
