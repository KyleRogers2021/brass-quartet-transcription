r"""
Run Spotify's BasicPitch (published model, default settings) on every track of
a test set and write its predictions as note lists for score_transcriptions.py.

BasicPitch doesn't identify instruments, so every note gets instrument -1,
and it receives instrument-agnostic scores only.

Settings are BasicPitch's published defaults, unchanged (not tuned on the
test set): onset threshold 0.5, frame threshold 0.3, minimum note length
127.70 ms.

Writes R:\experiments\predictions\basicpitch\<dataset>\<track>.tsv:
    onset   offset   pitch   velocity   instrument(-1)
If interrupted, run it again: finished tracks are skipped.

Run from the Miniforge Prompt (basicpitch environment):
    python R:\code\baselines\basic-pitch\predict_basicpitch.py --dataset cocochorales
    python R:\code\baselines\basic-pitch\predict_basicpitch.py --dataset choralebricks
"""
import argparse
import contextlib
import io
import os
import time

from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict

AUDIO_ROOT = r'R:\datasets\cocochorales\extracted\main_dataset'
TRACK_LIST = r'R:\splits\cocochorales_brass\test.txt'
RUN_NAME = 'basicpitch'
DATASETS = {   # name: (audio root containing <split folder>\<track>\mix.wav, split folder, track list)
    'cocochorales': (AUDIO_ROOT, 'test', TRACK_LIST),
    'choralebricks': (r'R:\datasets\choralebricks\prepared', 'test',
                      r'R:\splits\choralebricks_brass\all.txt'),
}
HEADER = 'onset\toffset\tpitch\tvelocity\tinstrument\n'


def load_model():
    """Load the model once (newer versions), or fall back to passing the path."""
    try:
        from basic_pitch.inference import Model
        return Model(ICASSP_2022_MODEL_PATH)
    except ImportError:
        return ICASSP_2022_MODEL_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    a = ap.parse_args()
    audio_root, split, track_list = DATASETS[a.dataset]
    OUT_DIR = os.path.join(r'R:\experiments\predictions', RUN_NAME, a.dataset)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(track_list) as f:
        tracks = f.read().split()
    todo = [t for t in tracks if not os.path.exists(os.path.join(OUT_DIR, f'{t}.tsv'))]
    print(f'Model:  {ICASSP_2022_MODEL_PATH}')
    print(f'Output: {OUT_DIR}')
    print(f'{len(tracks)} tracks ({a.dataset}), {len(tracks) - len(todo)} already done, {len(todo)} to go')
    if not todo:
        print('PREDICTIONS DONE')
        return

    model = load_model()
    t0 = time.time()
    total = 0
    for k, track in enumerate(todo, 1):
        wav = os.path.join(audio_root, split, track, 'mix.wav')
        with contextlib.redirect_stdout(io.StringIO()):       # BasicPitch prints per file
            _, _, note_events = predict(wav, model)
        notes = sorted((float(s), float(e), int(p), int(round(min(max(a, 0), 1) * 127)))
                       for s, e, p, a, *_ in note_events)
        total += len(notes)
        tmp = os.path.join(OUT_DIR, f'{track}.tsv.part')
        with open(tmp, 'w') as f:
            f.write(HEADER)
            for on, off, p, v in notes:
                f.write(f'{on:.4f}\t{off:.4f}\t{p}\t{v}\t-1\n')
        os.replace(tmp, os.path.join(OUT_DIR, f'{track}.tsv'))
        if k in (10, 50) or k % 250 == 0 or k == len(todo):
            el = time.time() - t0
            print(f'  {k}/{len(todo)} tracks  ({el / 60:.1f} min, '
                  f'about {el / k * (len(todo) - k) / 60:.0f} min left)')

    print(f'\nPredicted notes (this session): {total}')
    if a.dataset == 'cocochorales':
        print('\nNext, score them (Miniforge Prompt, oaf environment):')
        print(f'  python R:\\code\\evaluation\\score_transcriptions.py --pred {OUT_DIR} '
              f'--name {RUN_NAME}_cocochorales')
    else:
        print('\nNext: python R:\\code\\evaluation\\score_choralebricks.py')
    print('PREDICTIONS DONE')


if __name__ == '__main__':
    main()
