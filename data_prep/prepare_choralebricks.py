r"""
Prepare a real-recording test set from ChoraleBricks for the brass quartet
instruments (trumpet, French horn, trombone, tuba), in the same format as
CocoChorales so every model's prediction script and the shared scoring work
unchanged.

Graded by number of players (same design as planned):
    solo     every usable single recording
    duet     2 different instruments on 2 different voices, same chorale
    trio     3 ...
    quartet  4 ... (trumpet, horn, trombone, tuba together)
All valid ensemble combinations are used by default; --cap N keeps at most N
per chorale and size (fixed random seed) if you want to limit chorales with
many recordings. Mixes are built like the official toolbox: recordings
were made in sync, so tracks are summed (trimmed to the shortest). Every item
is then peak-normalized to 0.8, inside the level range of the CocoChorales
mixes, and resampled to 16 kHz mono.

Answer key per item: the dataset's note annotations (version 1.1: start, end,
sounding MIDI pitch; version 1.0: start, duration, mean frequency -> nearest
MIDI pitch at A4 = 440 Hz), with our instrument numbers
0=trumpet 1=horn 2=trombone 3=tuba. Velocity is fixed at 64 (not scored).

Reads:  the extracted 01_AudioAndAnnotations folder under R:\datasets\choralebricks
Writes: R:\datasets\choralebricks\prepared\test\<item>\mix.wav
        R:\datasets\choralebricks\labels\test\<item>.tsv
        R:\datasets\choralebricks\prepared\items.csv     (what is in each item)
        R:\splits\choralebricks_brass\solo.txt, duet.txt, trio.txt, quartet.txt, all.txt

Run from the Miniforge Prompt (oaf environment):
    python R:\code\data_prep\prepare_choralebricks.py --inventory    (counts only)
    python R:\code\data_prep\prepare_choralebricks.py                (build, all combinations)
    python R:\code\data_prep\prepare_choralebricks.py --cap 4        (at most 4 per chorale and size)
"""
import argparse
import csv
import math
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

CB_BASE = Path(r'R:\datasets\choralebricks')
OUT_AUDIO = CB_BASE / 'prepared' / 'test'
OUT_LABELS = CB_BASE / 'labels' / 'test'
ITEMS_CSV = CB_BASE / 'prepared' / 'items.csv'
SPLITS = Path(r'R:\splits\choralebricks_brass')

OURS = {'tp': 0, 'fho': 1, 'tb': 2, 'tba': 3}          # short code -> our number
NAMES = ['trumpet', 'horn', 'trombone', 'tuba']
# instrument labels: version 1.0 uses codes, version 1.1 uses full names
TO_CODE = {'tp': 'tp', 'fho': 'fho', 'tb': 'tb', 'tba': 'tba',
           'trumpet': 'tp', 'french horn': 'fho', 'trombone': 'tb', 'tuba': 'tba'}
OTHER_BRASS = {'fh': 'flugelhorn', 'bar': 'baritone',
               'flugelhorn': 'flugelhorn', 'baritone': 'baritone'}  # reported, not used
PARTS = {'S': 1, 'A': 2, 'T': 3, 'B': 4}                # version 1.1 'part' column
SIZES = {1: 'solo', 2: 'duet', 3: 'trio', 4: 'quartet'}
TARGET_SR = 16000
PEAK = 0.8
SEED = 0
HEADER = 'onset\toffset\tpitch\tvelocity\tinstrument\n'


def find_root():
    hits = sorted(CB_BASE.rglob('metadata_tracks.csv'))
    if not hits:
        sys.exit(f'metadata_tracks.csv not found under {CB_BASE}. Is the zip extracted there?')
    return hits[0].parent


def read_notes(path):
    """List of (onset, offset, midi pitch, tuning error in semitones).
    Version 1.1: start_sec;end_sec;...;pitch;...;pitch_dev_cents (sounding MIDI pitch).
    Version 1.0: TIME,VALUE,DURATION,... (VALUE = mean frequency in Hz)."""
    notes = []
    with open(path, newline='', encoding='utf-8') as f:
        text = f.read()
    delim = ';' if text.split('\n', 1)[0].count(';') else ','
    for row in csv.DictReader(text.splitlines(), delimiter=delim):
        if 'start_sec' in row:
            t, end = float(row['start_sec']), float(row['end_sec'])
            p, err = int(row['pitch']), abs(float(row.get('pitch_dev_cents') or 0)) / 100
        else:
            hz, t, d = float(row['VALUE']), float(row['TIME']), float(row['DURATION'])
            if hz <= 0:
                continue
            exact = 12 * math.log2(hz / 440.0) + 69
            p, err, end = int(round(exact)), abs(exact - round(exact)), t + d
        if end > t:
            notes.append((t, end, p, err))
    return notes


def load_tracks(root):
    tracks, skipped = [], Counter()
    with open(root / 'metadata_tracks.csv', newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter=';'):
            label = row['instrument'].strip().lower()
            code = TO_CODE.get(label)
            if code is None:
                skipped[OTHER_BRASS.get(label, 'non-brass')] += 1
                continue
            voice = PARTS[row['part'].strip().upper()] if 'part' in row else int(row['voice'])
            song = row['song_id'].strip()
            audio = root / song / 'tracks_normalized' / row['path_audio'].strip()
            notes = root / song / 'annotations' / str(row.get('path_notes', '')).strip()
            if not audio.is_file() or not notes.is_file():
                skipped['missing audio or note annotations'] += 1
                continue
            tracks.append(dict(song=song, voice=voice, code=code,
                               inst=OURS[code], audio=audio, notes=notes,
                               performer=row.get('performer', '')))
    return tracks, skipped


def ensembles(tracks, cap=None):
    """{size: [tuple of tracks]}: one track per voice, all instruments different."""
    by_song = defaultdict(lambda: defaultdict(list))
    for t in tracks:
        by_song[t['song']][t['voice']].append(t)
    rng = random.Random(SEED)
    out, possible = defaultdict(list), Counter()
    for song in sorted(by_song):
        voices = by_song[song]
        for k in (2, 3, 4):
            combos = []
            for vs in combinations(sorted(voices), k):
                for choice in product(*[voices[v] for v in vs]):
                    if len({t['inst'] for t in choice}) == k:
                        combos.append(choice)
            possible[k] += len(combos)
            out[k] += combos if cap is None else rng.sample(combos, min(cap, len(combos)))
    out[1] = [(t,) for t in tracks]
    possible[1] = len(tracks)
    return out, possible


def item_id(group):
    parts = '-'.join(f"{t['voice']}{t['code']}" for t in sorted(group, key=lambda t: t['voice']))
    perf = '-'.join(str(t['performer']).replace(' ', '')[:6] for t in sorted(group, key=lambda t: t['voice']))
    return f"cb_{group[0]['song']}_{parts}_{perf}".replace('/', '_')


def build(group, iid):
    audios, srs = [], set()
    for t in group:
        a, sr = sf.read(t['audio'], dtype='float32', always_2d=True)
        audios.append(a.mean(axis=1))
        srs.add(sr)
    assert len(srs) == 1, f'{iid}: different sample rates {srs}'
    sr = srs.pop()
    n = min(len(a) for a in audios)
    mix = np.sum([a[:n] for a in audios], axis=0)
    g = math.gcd(TARGET_SR, sr)
    mix = resample_poly(mix, TARGET_SR // g, sr // g).astype(np.float32)
    peak = float(np.abs(mix).max())
    if peak > 0:
        mix *= PEAK / peak
    length = n / sr
    d = OUT_AUDIO / iid
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / 'mix.wav', mix, TARGET_SR, subtype='PCM_16')

    rows = []
    for t in group:
        for on, off, p, _ in read_notes(t['notes']):
            if on < length:
                rows.append((on, min(off, length), p, 64, t['inst']))
    rows.sort()
    with open(OUT_LABELS / f'{iid}.tsv', 'w') as f:
        f.write(HEADER)
        for on, off, p, v, inst in rows:
            f.write(f'{on:.4f}\t{off:.4f}\t{p}\t{v}\t{inst}\n')
    return length, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inventory', action='store_true', help='print counts only, build nothing')
    ap.add_argument('--cap', type=int, help='max ensembles per chorale and size (default: all)')
    a = ap.parse_args()

    root = find_root()
    tracks, skipped = load_tracks(root)
    print(f'ChoraleBricks folder: {root}')
    print(f'Usable recordings of our four instruments: {len(tracks)}')
    for code, i in OURS.items():
        print(f'  {NAMES[i]:<9}{sum(t["code"] == code for t in tracks):>4}')
    print('Not used: ' + ', '.join(f'{k} {v}' for k, v in skipped.most_common()))

    print('\nPer chorale (voice: instruments available):')
    by_song = defaultdict(lambda: defaultdict(list))
    for t in tracks:
        by_song[t['song']][t['voice']].append(t['code'])
    for song in sorted(by_song):
        vs = '   '.join(f"{v}:{','.join(sorted(by_song[song][v]))}" for v in sorted(by_song[song]))
        print(f'  {song[:38]:<38} {vs}')

    groups, possible = ensembles(tracks, a.cap)
    cap_text = f'cap {a.cap} per chorale' if a.cap else 'all used'
    print(f'\nCombinations (possible -> used, {cap_text}):')
    for k, name in SIZES.items():
        print(f'  {name:<8}{possible[k]:>6} -> {len(groups[k])}')

    errs = [e for t in tracks for *_, e in read_notes(t['notes'])]
    sr_ch = Counter(sf.info(t['audio']).samplerate for t in tracks)
    print(f'\nAudio sample rates: {dict(sr_ch)}')
    print(f'Annotated notes: {len(errs)}; pitch off by more than 0.3 semitone from '
          f'the nearest note: {sum(e > 0.3 for e in errs)} ({np.mean(np.array(errs) > 0.3):.1%})')
    if a.inventory:
        print('\nINVENTORY DONE (nothing written)')
        return

    OUT_LABELS.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    lists, items = defaultdict(list), []
    total = sum(len(g) for g in groups.values())
    print(f'\nBuilding {total} items...')
    k_done = 0
    for k, name in SIZES.items():
        for group in groups[k]:
            iid = item_id(group)
            length, n_notes = build(group, iid)
            lists[name].append(iid)
            items.append([iid, name, group[0]['song'],
                          ' '.join(f"{t['voice']}:{NAMES[t['inst']]}" for t in group),
                          ' '.join(str(t['performer']) for t in group), round(length, 1), n_notes])
            k_done += 1
            if k_done % 25 == 0 or k_done == total:
                print(f'  {k_done}/{total}')
    for name in SIZES.values():
        (SPLITS / f'{name}.txt').write_text('\n'.join(lists[name]))
    (SPLITS / 'all.txt').write_text('\n'.join(i[0] for i in items))
    with open(ITEMS_CSV, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['item', 'size', 'song', 'voices', 'performers', 'seconds', 'notes'])
        w.writerows(items)
    print(f'\nTotal audio: {sum(i[5] for i in items) / 60:.1f} min in {len(items)} items')
    print(f'Lists in {SPLITS}; item details in {ITEMS_CSV}')
    print('PREP DONE')


if __name__ == '__main__':
    main()
