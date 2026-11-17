r"""
Run a trained detector on the ChoraleBricks recordings and write its notes in the
same format as the baselines' predictions, so the existing scoring code scores
it exactly like them.

Each recording is split into overlapping 10.24 s windows (every 5.12 s), each
window becomes the same 3-channel HCQT image as in training, the detector finds
the note boxes, and decode_windows (verified exact in check_roundtrip.py) joins
them into notes. Boxes below the confidence cutoff (default 0.25, Ultralytics'
standard, not tuned on ChoraleBricks) are dropped.

Writes:
  R:\experiments\predictions\detector_<run>\choralebricks\<item>.tsv   notes (for scoring)
  R:\experiments\predictions\detector_<run>\choralebricks_boxes\<item>.json
      every box down to confidence 0.05 with its confidence, so other cutoffs can be
      tried later without running the detector again

The HCQT harmonics are read from the training dataset's settings.json (found via
the run's args.yaml), so a model is always run with the HCQT it was trained on.
Datasets made before settings.json existed use harmonics (1, 2, 3).

Run in the yolo environment (needs: pip install librosa soundfile):
    python R:\code\detector\predict_detector.py --run medium_v1
    add --weights epoch5 (or last, epoch10, ...) to use another saved checkpoint
Then score in the oaf environment (the command is printed at the end).
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings as S                                                  # noqa: E402
from boxes import window_starts, decode_windows                        # noqa: E402
from make_dataset import hcqt_image                                   # noqa: E402

RUNS = r'R:\experiments\detector_runs'
PREDICTIONS = r'R:\experiments\predictions'
CB_AUDIO = r'R:\datasets\choralebricks\prepared\test'
CB_LIST = r'R:\splits\choralebricks_brass\all.txt'
CB_REF = r'R:\datasets\choralebricks\labels\test'
HEADER = 'onset\toffset\tpitch\tvelocity\tinstrument\n'
SAVE_CONF = 0.05          # raw boxes kept down to this confidence


def load_audio(path):
    audio, sr = soundfile.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != S.SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=S.SAMPLE_RATE)
    return audio


def run_harmonics(run_dir):
    """The HCQT harmonics of the dataset this run was trained on."""
    data_yaml = None
    with open(os.path.join(run_dir, 'args.yaml')) as f:
        for line in f:
            if line.startswith('data:'):
                data_yaml = line.split(':', 1)[1].strip().strip('\'"')
    settings = os.path.join(os.path.dirname(data_yaml), 'settings.json') if data_yaml else ''
    if settings and os.path.isfile(settings):
        with open(settings) as f:
            return tuple(json.load(f)['harmonics']), data_yaml
    return (1, 2, 3), data_yaml


def window_images(audio, harmonics):
    """(t0, image) for every window; the image channels are reversed to BGR order, which
    is how Ultralytics expects numpy input (training images were read from PNG the same way)."""
    n = int(S.CLIP_SECONDS * S.SAMPLE_RATE)
    out = []
    for t0 in window_starts(len(audio) / S.SAMPLE_RATE):
        start = int(round(t0 * S.SAMPLE_RATE))
        seg = audio[start:start + n]
        if len(seg) < n:
            seg = np.pad(seg, (0, n - len(seg)))
        out.append((t0, np.ascontiguousarray(hcqt_image(seg, harmonics)[:, :, ::-1])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='training run name under R:\\experiments\\detector_runs')
    ap.add_argument('--weights', default='best',
                    help='best, last, or a saved checkpoint such as epoch5 (from --save-period)')
    ap.add_argument('--conf', type=float, default=0.25, help='confidence cutoff (default 0.25, not tuned)')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(os.path.join(RUNS, a.run, 'weights', f'{a.weights}.pt'))
    harmonics, data_yaml = run_harmonics(os.path.join(RUNS, a.run))
    print(f'Trained on {data_yaml}; HCQT harmonics {harmonics}')
    name = f'detector_{a.run}' + ('' if a.weights == 'best' else f'_{a.weights}') + \
           ('' if a.conf == 0.25 else f'_conf{a.conf:g}')
    out_dir = os.path.join(PREDICTIONS, name, 'choralebricks')
    box_dir = os.path.join(PREDICTIONS, name, 'choralebricks_boxes')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(box_dir, exist_ok=True)

    with open(CB_LIST) as f:
        items = f.read().split()
    print(f'{len(items)} ChoraleBricks items, weights {a.weights}.pt, confidence cutoff {a.conf}')
    total_notes = 0
    for i, item in enumerate(items, 1):
        audio = load_audio(os.path.join(CB_AUDIO, item, 'mix.wav'))
        windows = window_images(audio, harmonics)
        results = model.predict([img for _, img in windows], imgsz=640, conf=SAVE_CONF,
                                verbose=False)
        raw, kept = [], []
        for (t0, _), r in zip(windows, results):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            xywh = r.boxes.xywhn.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            raw.append(dict(t0=t0, boxes=[[int(c), *map(float, b), float(p)]
                                           for c, b, p in zip(cls, xywh, conf)]))
            kept.append((t0, [(int(c), *b) for c, b, p in zip(cls, xywh, conf) if p >= a.conf]))
        notes = decode_windows(kept)
        duration = len(audio) / S.SAMPLE_RATE
        notes = [(on, min(off, duration), p, inst) for on, off, p, inst in notes if on < duration]
        total_notes += len(notes)

        tmp = os.path.join(out_dir, f'{item}.tsv.part')
        with open(tmp, 'w') as f:
            f.write(HEADER)
            for on, off, p, inst in notes:
                f.write(f'{on:.4f}\t{off:.4f}\t{p}\t64\t{inst}\n')
        os.replace(tmp, os.path.join(out_dir, f'{item}.tsv'))
        with open(os.path.join(box_dir, f'{item}.json'), 'w') as f:
            json.dump(raw, f)
        if i % 20 == 0 or i == len(items):
            print(f'  {i}/{len(items)} items, {total_notes} notes so far')

    print(f'\nPredictions: {out_dir}')
    print('Score them in the oaf environment with:')
    print(f'  python R:\\code\\evaluation\\score_transcriptions.py --pred {out_dir} --name {name}_choralebricks '
          f'--ref {CB_REF} --list {CB_LIST} --out R:\\experiments\\results\\{name}_choralebricks')
    print('PREDICTIONS DONE')


if __name__ == '__main__':
    main()
