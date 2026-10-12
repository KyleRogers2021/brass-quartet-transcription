r"""
Overlap check: does the same music appear in both the training split and the
test (or validation) split of CocoChorales brass?

If it does, the model could partly be recognizing pieces it has already heard
instead of transcribing them, which would inflate its test scores.

How it works
  Two independent tests:

  1. Fragments. Each track becomes a sequence of "states": every time any
     note starts or ends, which instrument plays which pitch. Only the ORDER
     of starts and ends is used, never their times, so a tempo change can't
     alter it. The sequence is cut into overlapping fragments of N states,
     each stored relative to its own lowest note (so transposing doesn't
     matter either). For every test/validation track, the script measures
     what share of its fragments also occur anywhere in training.

  2. Whole-piece voice match. Each instrument's pitch sequence, in order,
     with no timing at all, relative to the track's lowest note. Two tracks
     match if all four lines are identical. This catches the same piece even
     if the timing between voices was performed differently.

  Short shared fragments are expected by chance (common cadences). A copied
  or disguised piece shows up as a share near 100%.

  Built-in checks: 20 training tracks are disguised two ways and must be
  caught: (a) transposed up 3 and 15% slower: both tests should find them;
  (b) the same plus random timing shifts of up to 20 ms per note: the
  whole-piece voice match must still find them.

  It also measures how the MIDI timing behaves (do voices start exactly
  together?), which matters for interpreting the results.

Writes to R:\experiments\results\overlap_check_cocochorales\:
  overlap_test.csv, overlap_valid.csv   one row per track
  summary.txt                           the same summary that is printed

Run from the Miniforge Prompt (oaf environment):
    python R:\code\data_prep\check_overlap_cocochorales.py
"""
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SPLITS_DIR = Path(r"R:\splits\cocochorales_brass")
LABELS = Path(r"R:\datasets\cocochorales\labels")
OUT = Path(r"R:\experiments\results\overlap_check_cocochorales")
N = 12              # states per fragment
DETAIL_FROM = 0.2   # find the closest training track when share >= 20%

log_lines = []


def log(text=''):
    print(text)
    log_lines.append(text)


def read_notes(path):
    """List of (onset, offset, pitch, instrument), sorted by onset."""
    notes = []
    with open(path) as f:
        next(f)                                    # header row
        for line in f:
            on, off, pitch, _vel, inst = line.strip().split('\t')
            notes.append((float(on), float(off), int(float(pitch)), int(float(inst))))
    notes.sort()
    return notes


def to_states(notes):
    """Sequence of states: tuple of (instrument, pitch) sounding between consecutive
    note starts/ends. Uses only the order of those times, not their values."""
    bounds = sorted({t for on, off, *_ in notes for t in (on, off)})
    states, active, i = [], [], 0
    for a, b in zip(bounds, bounds[1:]):
        m = (a + b) / 2
        while i < len(notes) and notes[i][0] <= m:
            active.append(notes[i])
            i += 1
        active = [n for n in active if n[1] > m]
        if active:
            st = tuple(sorted((n[3], n[2]) for n in active))
            if not states or states[-1] != st:
                states.append(st)
    return states


def voice_key(notes):
    """Each instrument's pitch sequence, relative to the track's lowest note."""
    if not notes:
        return None
    ref = min(n[2] for n in notes)
    return hash(tuple(tuple(n[2] - ref for n in notes if n[3] == inst) for inst in range(4)))


def relative(chords):
    ref = min(p for _, p in chords[0])
    return tuple(tuple((inst, p - ref) for inst, p in c) for c in chords)


def fingerprints(notes):
    """(array of distinct fragment hashes, whole-piece voice key)."""
    states = to_states(notes)
    frags = {hash(relative(states[i:i + N])) for i in range(len(states) - N + 1)}
    return np.fromiter(frags, dtype=np.int64, count=len(frags)), voice_key(notes)


def load_split(name):
    tracks = (SPLITS_DIR / f'{name}.txt').read_text().split()
    result = []
    t0 = time.time()
    for k, track in enumerate(tracks, 1):
        notes = read_notes(LABELS / name / f'{track}.tsv')
        result.append((track, *fingerprints(notes)))
        if k % 5000 == 0 or k == len(tracks):
            print(f'  {name}: {k}/{len(tracks)} tracks read ({time.time() - t0:.0f} s)')
    return result


class TrainIndex:
    def __init__(self, train):
        self.names = [t for t, _, _ in train]
        hashes = np.concatenate([h for _, h, _ in train])
        owners = np.concatenate([np.full(len(h), i, np.int32) for i, (_, h, _) in enumerate(train)])
        order = np.argsort(hashes, kind='stable')
        self.hashes, self.owners = hashes[order], owners[order]
        self.whole = {w: t for t, _, w in train if w is not None}

    def compare(self, frags, whole):
        """share of fragments found in train, closest train track and its share, whole-piece match."""
        if len(frags) == 0:
            return float('nan'), '', float('nan'), self.whole.get(whole, '')
        lo = np.searchsorted(self.hashes, frags, 'left')
        hi = np.searchsorted(self.hashes, frags, 'right')
        found = hi > lo
        share = found.mean()
        best, best_share = '', 0.0
        if share >= DETAIL_FROM:
            counts = Counter()
            for a, b in zip(lo[found], hi[found]):
                counts.update(set(self.owners[a:b].tolist()))
            tid, c = counts.most_common(1)[0]
            best, best_share = self.names[tid], c / len(frags)
        return share, best, best_share, self.whole.get(whole, '')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    log(f'Overlap check, CocoChorales brass  (fragments of {N} states)')
    log('Reading note lists...')
    splits = {name: load_split(name) for name in ('train', 'valid', 'test')}
    index = TrainIndex(splits['train'])
    log(f'Training index: {len(index.hashes):,} fragments from {len(index.names):,} tracks')

    # How does the MIDI timing behave? (first 2,000 training tracks)
    together = near = apart = overlaps = consecutive = 0
    for track in index.names[:2000]:
        notes = read_notes(LABELS / 'train' / f'{track}.tsv')
        onsets = sorted(n[0] for n in notes)
        for k, t in enumerate(onsets):
            gaps = [abs(t - onsets[j]) for j in (k - 1, k + 1) if 0 <= j < len(onsets)]
            g = min(gaps) if gaps else 1.0
            together += g < 0.001
            near += 0.001 <= g < 0.06
            apart += g >= 0.06
        for inst in range(4):
            line = [n for n in notes if n[3] == inst]
            consecutive += max(len(line) - 1, 0)
            overlaps += sum(a[1] > b[0] + 0.001 for a, b in zip(line, line[1:]))
    total = together + near + apart
    log('\nMIDI timing (2,000 training tracks):')
    log(f'  note starts exactly with another voice (<1 ms): {together / total:.1%}')
    log(f'  note starts slightly off another voice (1-60 ms): {near / total:.1%}')
    log(f'  note starts on their own (60 ms+ from any other): {apart / total:.1%}')
    log(f'  notes that overlap the next note in the same voice: {overlaps / max(consecutive, 1):.1%}')

    # Built-in checks: disguised copies of training tracks must be caught
    rng = random.Random(0)
    picks = rng.sample(index.names, 20)
    results = {'a': ([], 0), 'b': ([], 0)}
    for track in picks:
        base = read_notes(LABELS / 'train' / f'{track}.tsv')
        a = sorted((on * 1.15, off * 1.15, p + 3, inst) for on, off, p, inst in base)
        b = sorted((on * 1.15 + rng.uniform(-0.02, 0.02), off * 1.15 + rng.uniform(-0.02, 0.02),
                    p + 3, inst) for on, off, p, inst in base)
        for key, notes in (('a', a), ('b', b)):
            share, _, _, wmatch = index.compare(*fingerprints(notes))
            shares, wins = results[key]
            shares.append(share)
            results[key] = (shares, wins + bool(wmatch))
    sa, wa = results['a']
    sb, wb = results['b']
    sa_clean = [x for x in sa if not np.isnan(x)]
    sb_clean = [x for x in sb if not np.isnan(x)]
    log('\nBuilt-in checks (20 training tracks, disguised):')
    log(f'  (a) transposed +3, 15% slower:            lowest fragment share '
        f'{min(sa_clean):.1%}   whole-piece matches {wa}/20')
    log(f'  (b) same, plus timing shifts up to 20 ms:  lowest fragment share '
        f'{min(sb_clean):.1%}   whole-piece matches {wb}/20')
    check_ok = min(sa_clean) > 0.999 and wa == 20 and wb == 20
    log('  PASSED: the check catches disguised copies' if check_ok else
        '  FAILED: send me this output before trusting the results below')
    log('  (fragment share in (b) is for information only: timing shifts can reorder')
    log('   near-simultaneous notes, which is why the whole-piece voice match exists)')

    for name in ('test', 'valid'):
        rows = [(t, len(f), *index.compare(f, w)) for t, f, w in splits[name]]
        with open(OUT / f'overlap_{name}.csv', 'w', newline='') as fh:
            wr = csv.writer(fh)
            wr.writerow(['track', 'fragments', 'share_in_train', 'closest_train_track',
                         'closest_share', 'whole_piece_match'])
            wr.writerows(rows)
        s = np.array([r[2] for r in rows if not np.isnan(r[2])])
        log(f'\n{name.upper()} vs TRAIN  ({len(rows):,} tracks)')
        log(f'  whole-piece voice matches (same four lines, any key/timing): {sum(bool(r[5]) for r in rows)}')
        log(f'  share of fragments also in train:  median {np.median(s):.1%}   '
            f'95th percentile {np.percentile(s, 95):.1%}   max {s.max():.1%}')
        for cut in (0.25, 0.5, 0.9):
            log(f'  tracks with share above {cut:.0%}: {(s > cut).sum()}')
        top = sorted(rows, key=lambda r: -r[2] if not np.isnan(r[2]) else 0)[:5]
        log('  highest 5:')
        for t, nf, sh, best, bsh, _ in top:
            extra = f'   closest train track {best} ({bsh:.0%})' if best else ''
            log(f'    {t}  {sh:.1%} of {nf} fragments{extra}')

    (OUT / 'summary.txt').write_text('\n'.join(log_lines) + '\n')
    print(f'\nSaved to {OUT}')
    print('OVERLAP CHECK DONE')


if __name__ == '__main__':
    main()
