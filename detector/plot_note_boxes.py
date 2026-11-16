r"""
Draw the planned YOLO note boxes on a real CocoChorales spectrogram (for slides).

Takes one CocoChorales brass test track, computes a CQT spectrogram of a short
excerpt (4 bins per semitone, A0 upward), and draws every answer-key note as a
box: onset to offset in time, one semitone tall at its pitch, colored by
instrument. These are the planned training labels, not model predictions.

Saves two pictures to R:\experiments\figures:
    note_boxes_<track>_<start>s.png        spectrogram with boxes
    spectrogram_<track>_<start>s.png       the same spectrogram without boxes

Run from the onsets-and-frames folder, oaf environment active:
    python R:\code\detector\plot_note_boxes.py
        picks the first test track whose excerpt has all four instruments
    add --track <name> and/or --start <seconds> to choose the excerpt yourself,
        --length <seconds> to change its length (default 8)
"""
import argparse
import os

import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                    # noqa: E402
from matplotlib.patches import Patch, Rectangle    # noqa: E402
import numpy as np                                 # noqa: E402
import soundfile                                   # noqa: E402

AUDIO_ROOT = r'R:\datasets\cocochorales\extracted\main_dataset'
LABEL_ROOT = r'R:\datasets\cocochorales\labels'
TEST_LIST = r'R:\splits\cocochorales_brass\test.txt'
OUT = r'R:\experiments\figures'
INSTRUMENTS = ['Trumpet', 'Horn', 'Trombone', 'Tuba']
COLORS = ['#F66900', '#F6BD00', '#009CBD', '#7D54C7']   # same as the slides
BINS_PER_SEMITONE = 4
MIN_MIDI = 21                                            # A0
N_SEMITONES = 88
HOP = 256


def read_notes(track):
    """(onset, offset, pitch, instrument) for every note of the track's answer key."""
    path = os.path.join(LABEL_ROOT, 'test', f'{track}.tsv')
    rows = np.loadtxt(path, delimiter='\t', skiprows=1, ndmin=2)
    return [(r[0], r[1], int(r[2]), int(r[4])) for r in rows]


def in_window(notes, start, length):
    return [n for n in notes if n[1] > start and n[0] < start + length]


def pick_excerpt(length, start=None, max_tracks=300):
    """First test track (and start time) whose excerpt contains all four instruments."""
    with open(TEST_LIST) as f:
        tracks = f.read().split()
    for track in tracks[:max_tracks]:
        notes = read_notes(track)
        end = max(n[1] for n in notes)
        starts = [start] if start is not None else np.arange(1.0, max(end - length, 1.0) + 0.01, 1.0)
        for s in starts:
            if len({n[3] for n in in_window(notes, s, length)}) == 4:
                return track, float(s)
    raise SystemExit('No excerpt with all four instruments found; try --track and --start.')


def note_name(midi):
    return librosa.midi_to_note(midi, unicode=False)


def draw(spec_db, notes, start, length, track, with_boxes, path, pitch_lo, pitch_hi):
    fig, ax = plt.subplots(figsize=(12, 6.2))
    extent = [0, spec_db.shape[1] * HOP / 16000, MIN_MIDI - 0.5, MIN_MIDI - 0.5 + N_SEMITONES]
    ax.imshow(spec_db, origin='lower', aspect='auto', cmap='gray_r', extent=extent,
              vmin=-70, vmax=0, interpolation='nearest')
    if with_boxes:
        for on, off, pitch, inst in in_window(notes, start, length):
            x0, x1 = max(on, start) - start, min(off, start + length) - start
            ax.add_patch(Rectangle((x0, pitch - 0.5), x1 - x0, 1.0, fill=False,
                                   edgecolor=COLORS[inst], linewidth=2.0))
        ax.legend(handles=[Patch(facecolor='none', edgecolor=c, linewidth=2, label=n)
                           for n, c in zip(INSTRUMENTS, COLORS)],
                  loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=4, fontsize=12,
                  frameon=False)
    ax.set_xlim(0, length)
    ax.set_ylim(pitch_lo, pitch_hi)
    ticks = [m for m in range(int(pitch_lo) + 1, int(pitch_hi)) if m % 12 == 0]   # every C
    ax.set_yticks(ticks)
    ax.set_yticklabels([note_name(m) for m in ticks], fontsize=11)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Pitch', fontsize=12)
    ax.tick_params(axis='x', labelsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--track')
    ap.add_argument('--start', type=float)
    ap.add_argument('--length', type=float, default=8.0)
    a = ap.parse_args()

    if a.track:
        track, start = a.track, (a.start if a.start is not None else pick_start(a.track, a.length))
    else:
        track, start = pick_excerpt(a.length, a.start)
    notes = read_notes(track)

    audio, sr = soundfile.read(os.path.join(AUDIO_ROOT, 'test', track, 'mix.wav'), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    seg = audio[int(start * sr):int((start + a.length) * sr)]
    cqt = np.abs(librosa.cqt(seg, sr=sr, hop_length=HOP, fmin=librosa.midi_to_hz(MIN_MIDI),
                             n_bins=N_SEMITONES * BINS_PER_SEMITONE,
                             bins_per_octave=12 * BINS_PER_SEMITONE))
    spec_db = librosa.amplitude_to_db(cqt, ref=np.max)

    shown = in_window(notes, start, a.length)
    lo = min(n[2] for n in shown) - 4                     # a little room below the tuba
    hi = min(max(n[2] for n in shown) + 20, MIN_MIDI + N_SEMITONES - 1)   # show some harmonics
    os.makedirs(OUT, exist_ok=True)
    tag = f'{track}_{start:g}s'
    for with_boxes, name in ((True, f'note_boxes_{tag}.png'), (False, f'spectrogram_{tag}.png')):
        draw(spec_db, notes, start, a.length, track, with_boxes, os.path.join(OUT, name), lo, hi)
        print(f'Saved {os.path.join(OUT, name)}')
    counts = {INSTRUMENTS[i]: sum(n[3] == i for n in shown) for i in range(4)}
    print(f'Track {track}, {start:g}-{start + a.length:g} s, notes shown: '
          + ', '.join(f'{k} {v}' for k, v in counts.items()))


def pick_start(track, length):
    notes = read_notes(track)
    end = max(n[1] for n in notes)
    for s in np.arange(1.0, max(end - length, 1.0) + 0.01, 1.0):
        if len({n[3] for n in in_window(notes, s, length)}) == 4:
            return float(s)
    return 0.0


if __name__ == '__main__':
    main()
