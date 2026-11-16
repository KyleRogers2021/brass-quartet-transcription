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

Run from the onsets-and-frames folder, oaf environment active:
  small sample (minutes), with preview pictures to check the boxes:
    python R:\code\detector\make_dataset.py --name sample --train-tracks 200 --val-tracks 50 --preview 5
  full dataset (a few hours):
    python R:\code\detector\make_dataset.py --name full
Output: R:\datasets\detector\<name>\ with images\, labels\ and data.yaml.
"""
import argparse
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


def hcqt_image(audio):
    """640 x 640 x 3 uint8 image of a 10.24 s clip (column k covers [16k, 16k + 16) ms)."""
    import librosa
    # librosa centers frame k on sample k*HOP; drop half a hop so it is centered on (k + 0.5)*HOP
    y = np.concatenate([audio[S.HOP // 2:], np.zeros(S.HOP // 2, dtype=audio.dtype)])
    fmin = librosa.midi_to_hz(S.MIDI_LO - 0.5 + 0.5 / S.BINS_PER_SEMITONE)   # bins centered within each semitone
    n_bins = S.N_SEMITONES * S.BINS_PER_SEMITONE
    chans = []
    for h in S.HARMONICS:
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


def init_worker(split, augment, seed):
    global _dataset, _seed
    _seed = seed
    from onsets_and_frames.loader_augment import CocoChoralesBrassAug
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        _dataset = CocoChoralesBrassAug(
            split=split, sequence_length=int(S.CLIP_SECONDS * S.SAMPLE_RATE),
            pitch_shift=S.PITCH_SHIFT_MAX if augment else 0,
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
    boxes = notes_to_boxes(clip_notes(track, split, info), 0.0, min_width_s=S.CLIP_SECONDS / S.FRAMES)
    Image.fromarray(hcqt_image(audio)).save(img_path)
    write_yolo(lab_path, boxes)
    return name, info


def preview(img_dir, lab_dir, names, out_dir):
    """Save a few images with their boxes drawn on, to check alignment by eye."""
    os.makedirs(out_dir, exist_ok=True)
    for name in names:
        img = Image.open(os.path.join(img_dir, name + '.png')).convert('RGB')
        gray = Image.fromarray(255 - np.asarray(img)[:, :, 0]).convert('RGB')   # harmonic 1, dark = loud
        d = ImageDraw.Draw(gray)
        with open(os.path.join(lab_dir, name + '.txt')) as f:
            for line in f:
                c, xc, yc, w, h = line.split()
                c, xc, yc, w, h = int(c), *(float(v) for v in (xc, yc, w, h))
                x0, x1 = (xc - w / 2) * S.IMAGE_WIDTH, (xc + w / 2) * S.IMAGE_WIDTH
                y0, y1 = (yc - h / 2) * S.IMAGE_HEIGHT, (yc + h / 2) * S.IMAGE_HEIGHT
                d.rectangle([x0, y0, x1, y1], outline=COLORS[c], width=2)
        gray.save(os.path.join(out_dir, name + '_boxes.png'))


def build(split, yolo_split, tracks, clips_per_track, augment, root, workers, seed):
    img_dir = os.path.join(root, 'images', yolo_split)
    lab_dir = os.path.join(root, 'labels', yolo_split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)
    jobs = [(t, split, k, img_dir, lab_dir) for t in tracks for k in range(clips_per_track)]
    # make one clip here first, so a setup problem shows as an error instead of a
    # pool of worker processes that keeps failing and restarting silently
    init_worker(split, augment, seed)
    make_one(jobs[0])
    print(f'{split}: {len(jobs)} clips from {len(tracks)} tracks '
          f'({"augmented" if augment else "plain mixes"}) -> {img_dir}')
    t0, done = time.time(), []
    with Pool(workers, initializer=init_worker, initargs=(split, augment, seed)) as pool:
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
    a = ap.parse_args()

    root = os.path.join(OUT_ROOT, a.name)
    rng = np.random.RandomState(a.seed)
    with open(os.path.join(r'R:\splits\cocochorales_brass', 'train.txt')) as f:
        train = f.read().split()
    with open(os.path.join(r'R:\splits\cocochorales_brass', 'valid.txt')) as f:
        valid = f.read().split()
    if a.train_tracks:
        train = list(rng.choice(train, a.train_tracks, replace=False))
    valid = list(rng.choice(valid, min(a.val_tracks, len(valid)), replace=False))

    print(f'Settings: MIDI {S.MIDI_LO}-{S.MIDI_HI}, harmonics {S.HARMONICS}, shift +/-{S.PITCH_SHIFT_MAX}, '
          f'stem subsets {S.STEM_SUBSET_PROB:.0%}, {a.workers} worker processes')
    tr_img, tr_lab, tr_names = build('train', 'train', train, a.clips_per_track, True, root, a.workers, a.seed)
    va_img, va_lab, _ = build('valid', 'val', valid, 1, False, root, a.workers, a.seed)

    with open(os.path.join(root, 'data.yaml'), 'w') as f:
        f.write(f'path: {root}\ntrain: images/train\nval: images/val\nnames:\n')
        for i, n in enumerate(S.INSTRUMENTS):
            f.write(f'  {i}: {n}\n')
        f.write(f'# settings: MIDI {S.MIDI_LO}-{S.MIDI_HI}, harmonics {S.HARMONICS}, '
                f'shift +/-{S.PITCH_SHIFT_MAX}, stem subsets {S.STEM_SUBSET_PROB}\n')
    if a.preview:
        preview(tr_img, tr_lab, tr_names[:a.preview], os.path.join(root, 'preview'))
        print(f'Preview pictures: {os.path.join(root, "preview")}')
    print(f'Dataset ready: {os.path.join(root, "data.yaml")}')
    print('DATASET DONE')


if __name__ == '__main__':
    main()
