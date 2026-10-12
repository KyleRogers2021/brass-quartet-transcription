r"""
Run YourMT3+ (published checkpoint, unchanged) on every track of a test set and
write its predictions as note lists for score_transcriptions.py.

Model: YourMT3+ "YPTF.MoE+Multi (noPS)" checkpoint from the authors' demo
(mimbres/YourMT3), run through the mt3-infer package.

By default mt3-infer rewrites every program through a simplified General MIDI
family table when writing MIDI (all brass 56-63 -> 56, i.e. everything brass
becomes "trumpet"). That table is for nicer playback, not evaluation, so this
script replaces it with an identity table: the model's own program numbers are
kept. YourMT3+'s training vocabulary keeps trumpet (56), trombone (57),
tuba (58), French horn (60) and brass section (61) as separate classes.

YourMT3+ names instruments by General MIDI program. Mapping to our labels:
    56, 59 trumpet -> 0     60 French horn -> 1
    57 trombone    -> 2     58 tuba        -> 3
    anything else (incl. "brass section" and drums) -> 4 (other): counts in
    instrument-agnostic scoring, always wrong in instrument-aware scoring.

Writes R:\experiments\predictions\yourmt3\<dataset>\<track>.tsv
If interrupted, run it again: finished tracks are skipped.

Run from the Miniforge Prompt (mt3 environment):
    python R:\code\baselines\yourmt3\predict_yourmt3.py --dataset cocochorales
    python R:\code\baselines\yourmt3\predict_yourmt3.py --dataset choralebricks
    add --list <track list> to run only some tracks
"""
import argparse
import contextlib
import io
import os
import time
from collections import Counter

os.environ.setdefault('MT3_CHECKPOINT_DIR', r'R:\models\mt3_checkpoints')

import pretty_midi
import soundfile
from mt3_infer import load_model

AUDIO_ROOT = r'R:\datasets\cocochorales\extracted\main_dataset'
DEFAULT_LIST = r'R:\splits\cocochorales_brass\test.txt'
RUN_NAME = 'yourmt3'
DATASETS = {   # name: (audio root containing <split folder>\<track>\mix.wav, split folder, track list)
    'cocochorales': (AUDIO_ROOT, 'test', DEFAULT_LIST),
    'choralebricks': (r'R:\datasets\choralebricks\prepared', 'test',
                      r'R:\splits\choralebricks_brass\all.txt'),
}
PROGRAM_TO_INST = {56: 0, 59: 0, 60: 1, 57: 2, 58: 3}
OTHER = 4
NAMES = ['trumpet', 'horn', 'trombone', 'tuba', 'other']
HEADER = 'onset\toffset\tpitch\tvelocity\tinstrument\n'


def to_notes(midi):
    """mido.MidiFile -> list of (onset, offset, pitch, velocity, instrument), plus
    the GM instrument names of notes mapped to 'other'."""
    buf = io.BytesIO()
    midi.save(file=buf)
    buf.seek(0)
    pm = pretty_midi.PrettyMIDI(buf)
    notes, others = [], Counter()
    for ins in pm.instruments:
        inst = OTHER if ins.is_drum else PROGRAM_TO_INST.get(ins.program, OTHER)
        if inst == OTHER:
            name = 'drums' if ins.is_drum else pretty_midi.program_to_instrument_name(ins.program)
            others[name] += len(ins.notes)
        for n in ins.notes:
            if n.end > n.start:
                notes.append((n.start, n.end, n.pitch, n.velocity, inst))
    return sorted(notes), others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    ap.add_argument('--list', help='track list (default: the dataset\'s own list)')
    a = ap.parse_args()
    audio_root, split, default_list = DATASETS[a.dataset]
    a.list = a.list or default_list
    OUT_DIR = os.path.join(r'R:\experiments\predictions', RUN_NAME, a.dataset)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(a.list) as f:
        tracks = f.read().split()
    todo = [t for t in tracks if not os.path.exists(os.path.join(OUT_DIR, f'{t}.tsv'))]
    print(f'Checkpoints: {os.environ["MT3_CHECKPOINT_DIR"]}')
    print(f'Output:      {OUT_DIR}')
    print(f'{len(tracks)} tracks in list, {len(tracks) - len(todo)} already done, {len(todo)} to go')
    if not todo:
        print('PREDICTIONS DONE')
        return

    model = load_model('yourmt3', device='cuda')   # downloads the checkpoint the first time
    # keep the model's own program numbers (see top of file)
    assert hasattr(model.model, 'midi_output_inverse_vocab'), 'mt3-infer internals changed'
    model.model.midi_output_inverse_vocab = {
        p: (p, pretty_midi.program_to_instrument_name(p)) for p in range(128)}
    t0 = time.time()
    per_inst, others, empty = Counter(), Counter(), 0
    for k, track in enumerate(todo, 1):
        audio, sr = soundfile.read(os.path.join(audio_root, split, track, 'mix.wav'),
                                   dtype='float32')
        assert sr == 16000, f'{track}: sample rate {sr}'
        with contextlib.redirect_stdout(io.StringIO()):   # hide mt3-infer's per-track messages
            midi = model.transcribe(audio, sr=sr)
        notes, oth = to_notes(midi)
        others.update(oth)
        empty += not notes
        tmp = os.path.join(OUT_DIR, f'{track}.tsv.part')
        with open(tmp, 'w') as f:
            f.write(HEADER)
            for on, off, p, v, inst in notes:
                f.write(f'{on:.4f}\t{off:.4f}\t{p}\t{v}\t{inst}\n')
                per_inst[inst] += 1
        os.replace(tmp, os.path.join(OUT_DIR, f'{track}.tsv'))
        if k in (5, 20, 50) or k % 250 == 0 or k == len(todo):
            el = time.time() - t0
            print(f'  {k}/{len(todo)} tracks  ({el / 60:.1f} min, '
                  f'about {el / k * (len(todo) - k) / 60:.0f} min left)')

    print('\nPredicted notes per instrument (this session):')
    for i, name in enumerate(NAMES):
        print(f'  {name:<9}{per_inst[i]:>9}')
    if others:
        print('  "other" was made up of:')
        for name, n in others.most_common(8):
            print(f'    {name:<28}{n:>8}')
    if empty:
        print(f'  tracks with no notes at all: {empty}')
    if a.dataset == 'cocochorales':
        print('\nNext, score them (Miniforge Prompt, oaf environment):')
        lst = '' if a.list == DEFAULT_LIST else f' --list {a.list}'
        print(f'  python R:\\code\\evaluation\\score_transcriptions.py --pred {OUT_DIR} '
              f'--name {RUN_NAME}_cocochorales{lst}')
    else:
        print('\nNext: python R:\\code\\evaluation\\score_choralebricks.py')
    print('PREDICTIONS DONE')


if __name__ == '__main__':
    main()
