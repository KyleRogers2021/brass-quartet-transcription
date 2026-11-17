r"""
Build the detector's training data: augmented CocoChorales clips -> 3-channel HCQT
images (PNG) + YOLO label files, in the folder layout Ultralytics expects.

Clips come from the same augmented loader the Onsets and Frames experiments used
(CocoChoralesBrassAug: stem subsets + pitch shift by resampling), so the
augmentation is identical. Training clips are augmented with the settings in
settings.py; validation clips are plain mixes (no augmentation), as for O&F.

Each image: width = 640 columns of 16 ms (10.24 s), height = MIDI MIDI_LO-MIDI_HI at
4 CQT bins per semitone, stretched 2x to 640 rows (highest pitch at the top);
channels = HCQT harmonics 1, 2, 3; log magnitude, 80 dB range, 0-255.

Runs on the CPU (several processes), so the GPU stays free.

The pitch-shift range and HCQT harmonics default to settings.py and can be set
per dataset with --shift-up, --shift-down and --harmonics. They are added to the
folder name automatically, so the name always says what is inside:
    --name medium                          -> medium_p9m9
    --name medium --shift-up 3 --shift-down 12          -> medium_p3m12
    --name medium --harmonics 0.5 1 2                    -> medium_p9m9_h0.5-1-2
Low-frequency attenuation (training clips only; off by default):
    --lowcut-prob 0.5     share of training clips whose bass is turned down
    --lowcut-freq 60 250  corner frequency range in Hz (random per clip)
    --lowcut-db 6 20      how far below the corner is turned down, in dB (random per clip)
    --name medium --lowcut-prob 0.5                      -> medium_p9m9_lf
It is a low-shelf filter applied to the audio before the HCQT; the note labels do
not change. Its random draws use their own random state, so every clip's music,
shift and stem choice are exactly those of the same dataset without it.
These settings are also saved in the dataset's settings.json, which predict_detector.py
reads so a model is always run with the same HCQT it was trained on.

Run from the onsets-and-frames folder, oaf environment active:
  small sample (minutes), with preview pictures to check the boxes:
    python R:\code\detector\make_dataset.py --name sample --train-tracks 200 --val-tracks 50 --preview 5
  medium (10,000 clips from 5,000 tracks, about 15 minutes):
    python R:\code\detector\make_dataset.py --name medium --train-tracks 5000
  full dataset (a few hours):
    python R:\code\detector\make_dataset.py --name full
The same --seed (default 0) picks the same training tracks, so datasets that differ
only in shift or harmonics are built from identical tracks.
Output: R:\datasets\detector\<name + tag>\ with images\, labels\, data.yaml, settings.json.
"""
import argparse
import json
import os
import sys
import time
import zlib
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, r'R:\code\baselines\onsets-and-frames')
import settings as S                                  # noqa: E402
from boxes import notes_to_boxes, write_yolo          # noqa: E402

OUT_ROOT = r'R:\datasets\detector'
COCO_LABELS = r'R:\datasets\cocochorales\labels'
COLORS = [(246, 105, 0), (246, 189, 0), (0, 156, 189), (125, 84, 199)]
DB_RANGE = 80.0

_dataset = None
_seed = 0
_harmonics = S.HARMONICS
_lowcut = None            # (prob, f_lo, f_hi, db_lo, db_hi) or None


def low_shelf(audio, corner_hz, cut_db, sr=S.SAMPLE_RATE):
    """Turn down everything below corner_hz by cut_db (RBJ audio-EQ low-shelf, slope 1)."""
    from scipy.signal import lfilter
    A = 10 ** (-cut_db / 40)
    w0 = 2 * np.pi * corner_hz / sr
    c, alpha = np.cos(w0), np.sin(w0) / 2 * np.sqrt(2)
    sa = 2 * np.sqrt(A) * alpha
    b = [A * ((A + 1) - (A - 1) * c + sa), 2 * A * ((A - 1) - (A + 1) * c), A * ((A + 1) - (A - 1) * c - sa)]
    a = [(A + 1) + (A - 1) * c + sa, -2 * ((A - 1) + (A + 1) * c), (A + 1) + (A - 1) * c - sa]
    out = lfilter(np.array(b) / a[0], np.array(a) / a[0], audio.astype(np.float64))
    return np.clip(out, -1.0, 1.0).astype(np.float32)


def hcqt_image(audio, harmonics=None):
    """640 x 640 x 3 uint8 image of a 10.24 s clip (column k covers [16k, 16k + 16) ms).
    harmonics: the three HCQT harmonics (default settings.HARMONICS)."""
    harmonics = S.HARMONICS if harmonics is None else tuple(harmonics)
    import librosa
    # librosa centers frame k on sample k*HOP; drop half a hop so it is centered on (k + 0.5)*HOP
    y = np.concatenate([audio[S.HOP // 2:], np.zeros(S.HOP // 2, dtype=audio.dtype)])
    fmin = librosa.midi_to_hz(S.MIDI_LO - 0.5 + 0.5 / S.BINS_PER_SEMITONE)   # bins centered within each semitone
    n_bins = S.N_SEMITONES * S.BINS_PER_SEMITONE
    chans = []
    for h in harmonics:
        c = np.abs(librosa.cqt(y, sr=S.SAMPLE_RATE, hop_length=S.HOP, fmin=fmin * h, n_bins=n_bins,
                               bins_per_octave=12 * S.BINS_PER_SEMITONE))[:, :S.FRAMES]
        chans.append(c)
    mag = np.stack(chans)                                          # (3, bins, frames)
    db = 20 * np.log10(np.maximum(mag, 1e-10))
    db = np.clip(db - db.max(), -DB_RANGE, 0)                      # same reference for all channels
    img = ((db + DB_RANGE) / DB_RANGE * 255).round().astype(np.uint8)
    img = img[:, ::-1, :]                                          # highest pitch at the top
    img = np.repeat(img, S.IMAGE_HEIGHT // n_bins, axis=1)         # 320 -> 640 rows
    return np.transpose(img, (1, 2, 0))                            # (640, 640, 3)


def clip_notes(track, split, info):
    """The clip's notes (in clip time, after pitch shift and stem choice)."""
    from onsets_and_frames.loader_augment import shift_ratio
    rows = np.loadtxt(os.path.join(COCO_LABELS, split, f'{track}.tsv'), delimiter='\t',
                      skiprows=1, ndmin=2)
    up, down = shift_ratio(info['shift'])
    scale = up / down
    keep = {S.INSTRUMENTS.index(n) for n in info['instruments']}
    notes = []
    for on, off, pitch, _, inst in rows:
        if int(inst) in keep:
            notes.append(((on - info['begin_sec']) * scale, (off - info['begin_sec']) * scale,
                          int(round(pitch)) + info['shift'], int(inst)))
    return notes


def init_worker(split, augment, seed, shift_range=(-S.PITCH_SHIFT_MAX, S.PITCH_SHIFT_MAX),
                harmonics=S.HARMONICS, lowcut=None):
    global _dataset, _seed, _harmonics, _lowcut
    _seed, _harmonics = seed, tuple(harmonics)
    _lowcut = lowcut if augment else None
    from onsets_and_frames.loader_augment import CocoChoralesBrassAug
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        _dataset = CocoChoralesBrassAug(
            split=split, sequence_length=int(S.CLIP_SECONDS * S.SAMPLE_RATE),
            pitch_shift=0, shift_range=tuple(shift_range) if augment else (0, 0),
            stem_prob=S.STEM_SUBSET_PROB if augment else 0.0, seed=seed, device='cpu')


def make_one(job):
    track, split, k, img_dir, lab_dir = job
    name = f'{track}_{k}'
    img_path, lab_path = os.path.join(img_dir, name + '.png'), os.path.join(lab_dir, name + '.txt')
    if os.path.exists(img_path) and os.path.exists(lab_path):
        return name, None                                          # already done (resuming)
    # a reproducible random state per clip, whatever the process or order
    _dataset.random = np.random.RandomState(zlib.crc32(f'{track}_{k}_{_seed}'.encode()))
    audio, _, _, info = _dataset.make_clip(track)
    if _lowcut and _lowcut[0] > 0:
        r = np.random.RandomState(zlib.crc32(f'{track}_{k}_{_seed}_lowcut'.encode()))
        if r.rand() < _lowcut[0]:
            info['lowcut'] = (float(r.uniform(_lowcut[1], _lowcut[2])), float(r.uniform(_lowcut[3], _lowcut[4])))
            audio = low_shelf(audio, *info['lowcut'])
    boxes = notes_to_boxes(clip_notes(track, split, info), 0.0, min_width_s=S.CLIP_SECONDS / S.FRAMES)
    Image.fromarray(hcqt_image(audio, _harmonics)).save(img_path)
    write_yolo(lab_path, boxes)
    return name, info


def preview(img_dir, lab_dir, names, out_dir, harmonics=S.HARMONICS):
    """Save a few images with their boxes drawn on, to check alignment by eye.
    Drawn on the harmonic-1 channel, where each note's fundamental sits inside its box."""
    os.makedirs(out_dir, exist_ok=True)
    ch = list(harmonics).index(1)
    for name in names:
        img = Image.open(os.path.join(img_dir, name + '.png')).convert('RGB')
        gray = Image.fromarray(255 - np.asarray(img)[:, :, ch]).convert('RGB')   # harmonic 1, dark = loud
        d = ImageDraw.Draw(gray)
        with open(os.path.join(lab_dir, name + '.txt')) as f:
            for line in f:
                c, xc, yc, w, h = line.split()
                c, xc, yc, w, h = int(c), *(float(v) for v in (xc, yc, w, h))
                x0, x1 = (xc - w / 2) * S.IMAGE_WIDTH, (xc + w / 2) * S.IMAGE_WIDTH
                y0, y1 = (yc - h / 2) * S.IMAGE_HEIGHT, (yc + h / 2) * S.IMAGE_HEIGHT
                d.rectangle([x0, y0, x1, y1], outline=COLORS[c], width=2)
        gray.save(os.path.join(out_dir, name + '_boxes.png'))


def build(split, yolo_split, tracks, clips_per_track, augment, root, workers, seed, shift_range, harmonics,
          lowcut=None):
    img_dir = os.path.join(root, 'images', yolo_split)
    lab_dir = os.path.join(root, 'labels', yolo_split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    jobs = [(t, split, k, img_dir, lab_dir) for t in tracks for k in range(clips_per_track)]
    # make one clip here first, so a setup problem shows as an error instead of a
    # pool of worker processes that keeps failing and restarting silently
    init_worker(split, augment, seed, shift_range, harmonics, lowcut)
    make_one(jobs[0])
    print(f'{split}: {len(jobs)} clips from {len(tracks)} tracks '
          f'({"augmented" if augment else "plain mixes"}) -> {img_dir}')
    t0, done = time.time(), []
    with Pool(workers, initializer=init_worker,
              initargs=(split, augment, seed, shift_range, harmonics, lowcut)) as pool:
        for i, (name, info) in enumerate(pool.imap_unordered(make_one, jobs, chunksize=8), 1):
            done.append(name)
            if i % 500 == 0 or i == len(jobs):
                rate = i / (time.time() - t0)
                print(f'  {i}/{len(jobs)} clips, {rate:.1f}/s, about '
                      f'{(len(jobs) - i) / rate / 60:.0f} min left')
    return img_dir, lab_dir, sorted(done)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='dataset folder name under R:\\datasets\\detector')
    ap.add_argument('--train-tracks', type=int, help='use only this many training tracks (default all)')
    ap.add_argument('--val-tracks', type=int, default=500, help='validation tracks (default 500)')
    ap.add_argument('--clips-per-track', type=int, default=2)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--preview', type=int, default=0, help='save this many pictures with boxes drawn on')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--shift-up', type=int, default=S.PITCH_SHIFT_MAX,
                    help=f'largest upward pitch shift in semitones (default {S.PITCH_SHIFT_MAX})')
    ap.add_argument('--shift-down', type=int, default=S.PITCH_SHIFT_MAX,
                    help=f'largest downward pitch shift in semitones (default {S.PITCH_SHIFT_MAX})')
    ap.add_argument('--harmonics', type=float, nargs=3, default=list(S.HARMONICS),
                    help=f'the three HCQT harmonics (default {" ".join(map(str, S.HARMONICS))})')
    ap.add_argument('--lowcut-prob', type=float, default=0.0,
                    help='share of training clips with low-frequency attenuation (default 0 = off)')
    ap.add_argument('--lowcut-freq', type=float, nargs=2, default=[60, 250], help='corner range, Hz')
    ap.add_argument('--lowcut-db', type=float, nargs=2, default=[6, 20], help='attenuation range, dB')
    a = ap.parse_args()

    lowcut = (a.lowcut_prob, *a.lowcut_freq, *a.lowcut_db) if a.lowcut_prob > 0 else None
    harmonics = tuple(int(h) if float(h).is_integer() else h for h in a.harmonics)
    shift_range = (-a.shift_down, a.shift_up)
    tag = f'_p{a.shift_up}m{a.shift_down}'
    if harmonics != tuple(S.HARMONICS):
        tag += '_h' + '-'.join(f'{h:g}' for h in harmonics)
    if lowcut:
        tag += '_lf'
    name = a.name if a.name.endswith(tag) else a.name + tag
    root = os.path.join(OUT_ROOT, name)
    print(f'Dataset name: {name}')
    rng = np.random.RandomState(a.seed)
    with open(os.path.join(r'R:\splits\cocochorales_brass', 'train.txt')) as f:
        train = f.read().split()
    with open(os.path.join(r'R:\splits\cocochorales_brass', 'valid.txt')) as f:
        valid = f.read().split()
    if a.train_tracks:
        train = list(rng.choice(train, a.train_tracks, replace=False))
    valid = list(rng.choice(valid, min(a.val_tracks, len(valid)), replace=False))

    print(f'Settings: MIDI {S.MIDI_LO}-{S.MIDI_HI}, harmonics {harmonics}, '
          f'shift {-a.shift_down:+d} to {a.shift_up:+d}, '
          f'stem subsets {S.STEM_SUBSET_PROB:.0%}, {a.workers} worker processes')
    if lowcut:
        print(f'Low-frequency attenuation: {lowcut[0]:.0%} of training clips, corner {lowcut[1]:g}-{lowcut[2]:g} Hz, '
              f'{lowcut[3]:g}-{lowcut[4]:g} dB')
    tr_img, tr_lab, tr_names = build('train', 'train', train, a.clips_per_track, True, root, a.workers,
                                     a.seed, shift_range, harmonics, lowcut)
    va_img, va_lab, _ = build('valid', 'val', valid, 1, False, root, a.workers, a.seed,
                              shift_range, harmonics)

    with open(os.path.join(root, 'data.yaml'), 'w') as f:
        f.write(f'path: {root}\ntrain: images/train\nval: images/val\nnames:\n')
        for i, n in enumerate(S.INSTRUMENTS):
            f.write(f'  {i}: {n}\n')
        f.write(f'# settings: MIDI {S.MIDI_LO}-{S.MIDI_HI}, harmonics {harmonics}, '
                f'shift {-a.shift_down:+d} to {a.shift_up:+d}, stem subsets {S.STEM_SUBSET_PROB}'
                + (f', lowcut {lowcut}' if lowcut else '') + '\n')
    with open(os.path.join(root, 'settings.json'), 'w') as f:
        json.dump(dict(midi_lo=S.MIDI_LO, midi_hi=S.MIDI_HI, harmonics=list(harmonics),
                       shift_down=a.shift_down, shift_up=a.shift_up,
                       stem_subset_prob=S.STEM_SUBSET_PROB, seed=a.seed,
                       lowcut=dict(prob=lowcut[0], freq_hz=list(lowcut[1:3]), db=list(lowcut[3:5])) if lowcut else None,
                       train_tracks=len(train), clips_per_track=a.clips_per_track), f, indent=1)
    if a.preview:
        preview(tr_img, tr_lab, tr_names[:a.preview], os.path.join(root, 'preview'), harmonics)
        print(f'Preview pictures: {os.path.join(root, "preview")}')
    print(f'Dataset ready: {os.path.join(root, "data.yaml")}')
    print('DATASET DONE')


if __name__ == '__main__':
    main()
