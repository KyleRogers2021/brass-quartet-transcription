r"""
Transcribe a brass recording into a score: audio file -> trained note detector ->
beat-aligned rhythm -> MusicXML (opens in Sibelius, MuseScore, Finale, Dorico).

Steps
  1. Notes: the detector runs over the recording in overlapping 10.24 s windows,
     exactly as in predict_detector.py (same HCQT, windowing, decoding, confidence
     cutoff), giving notes as (start s, end s, MIDI pitch, instrument).
  2. Tempo: estimated with madmom and shown for confirmation; type another value if
     it is wrong (estimators often give half or double the true tempo). --bpm skips
     the estimate, --yes accepts it without asking.
  3. Beats and bar lines: madmom tracks the beats through the recording at that tempo
     (so the grid follows the players if they speed up or slow down) and finds where
     the bars start for the given time signature. --pickup sets the upbeat instead.
  4. Rhythm: every note start and end is placed on its beat position and rounded to
     the nearest 16th note, so durations come from the beats around each note and
     small tempo errors do not add up over the piece.
  5. Score: one part per instrument, one note at a time (brass): notes starting
     together in one part keep the longest; a note still sounding when the next one
     starts is cut there. Gaps become rests, notes across bar lines are tied, and an
     upbeat becomes a pickup bar (bar 0). No key signature (accidentals are written
     as needed). Trumpet is written in B-flat and horn in F (transposed parts);
     trombone and tuba at concert pitch. Written with music21.

Tested on a synthetic recording with a known score (3/4, one-beat pickup, tempo
slowing from 84 to 70 BPM, given the true note times): every note came out with
the right start, length, bar and pitch, and the pickup was found.

The time signature is always given by the user (--time-sig); it is not detected.

Run in the yolo environment (needs once: pip install music21 madmom-prebuilt):
    python R:\code\detector\transcribe.py recording.wav --time-sig 4/4
    python R:\code\detector\transcribe.py recording.wav --time-sig 3/4 --bpm 72 --pickup 1
    python R:\code\detector\transcribe.py recording.wav --time-sig 4/4 --instruments trumpet tuba
Options
    --time-sig   required, e.g. 4/4, 3/4, 2/2, 6/8 (6/8, 9/8, 12/8 are counted in dotted quarters)
    --bpm        tempo in beats per minute (beat = the time signature's beat); skips the estimate
    --yes        accept the estimated tempo without asking
    --pickup     beats before the first bar line (0 = starts on the downbeat); default: detected
    --instruments  which parts to write, e.g. trumpet horn (default: every instrument the model knows)
    --run / --weights  trained detector (default DEFAULT_RUN below, best.pt);
    --model      or a path to any .pt file (then give --harmonics)
    --harmonics  HCQT harmonics the model was trained on (default: read from the run)
    --conf       detector confidence cutoff (default 0.25, as in the evaluation)
    --out        output file (default: next to the audio, <name>.musicxml)
    --save-notes also write the raw detected notes (.tsv, in seconds)
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import settings as S                                                       # noqa: E402
from boxes import decode_windows                                            # noqa: E402
from predict_detector import RUNS, SAVE_CONF, load_audio, run_harmonics, window_images   # noqa: E402

DEFAULT_RUN = 'full_lf_v1'
GRID_QL = 0.25                     # rhythm grid in quarter notes (0.25 = 16th notes)
MADMOM_SR = 44100                  # madmom's beat models expect 44.1 kHz audio
TEMPO_MARGIN = 0.15                # beat tracking may follow the players within +/-15% of the tempo

# How each detector class is written. Unknown names get a plain part at concert pitch,
# so a future model with other instruments still works.
PART_STYLE = {
    'trumpet': ('Trumpet', 'Trumpet in B\u266d'),     # music21 Trumpet = B-flat, sounds a M2 lower
    'horn': ('Horn', 'Horn in F'),                    # music21 Horn = F, sounds a P5 lower
    'trombone': ('Trombone', 'Trombone'),
    'tuba': ('Tuba', 'Tuba'),
}


# ---------------------------------------------------------------- 1. notes

def detect_notes(audio, weights, harmonics, conf=0.25, batch=16):
    """Notes (onset_s, offset_s, midi, class_id) and the model's class names.
    Same steps as predict_detector.py, so the notes match what was evaluated."""
    from ultralytics import YOLO
    model = YOLO(weights)
    windows = window_images(audio, harmonics)
    kept = []
    for i in range(0, len(windows), batch):
        chunk = windows[i:i + batch]
        results = model.predict([img for _, img in chunk], imgsz=640, conf=SAVE_CONF, verbose=False)
        for (t0, _), r in zip(chunk, results):
            cls = r.boxes.cls.cpu().numpy().astype(int)
            xywh = r.boxes.xywhn.cpu().numpy()
            p = r.boxes.conf.cpu().numpy()
            kept.append((t0, [(int(c), *b) for c, b, q in zip(cls, xywh, p) if q >= conf]))
    duration = len(audio) / S.SAMPLE_RATE
    notes = [(on, min(off, duration), pitch, inst) for on, off, pitch, inst in decode_windows(kept)
             if on < duration]
    return sorted(notes), {int(k): v for k, v in model.names.items()}


# ---------------------------------------------------------------- 2-3. tempo, beats, bars

def import_madmom():
    """Import madmom. The 'madmom-prebuilt' package (Windows wheels for current Python)
    looks up its version under the name 'madmom' and fails; point that lookup at it."""
    import importlib.metadata as md
    orig = md.distribution

    def distribution(name):
        try:
            return orig(name)
        except md.PackageNotFoundError:
            if name == 'madmom':
                return orig('madmom-prebuilt')
            raise
    md.distribution = distribution
    try:
        import madmom
    finally:
        md.distribution = orig
    return madmom


def madmom_signal(path):
    """The recording as a mono 44.1 kHz madmom Signal."""
    import soundfile
    madmom = import_madmom()
    audio, sr = soundfile.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != MADMOM_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=MADMOM_SR)
    return madmom.audio.signal.Signal(audio, sample_rate=MADMOM_SR)


def estimate_tempo(signal):
    """Tempo candidates [(bpm, strength), ...], strongest first."""
    import_madmom()
    from madmom.features.beats import RNNBeatProcessor
    from madmom.features.tempo import TempoEstimationProcessor
    act = RNNBeatProcessor()(signal)
    return [(float(b), float(s)) for b, s in TempoEstimationProcessor(fps=100)(act)]


def track_beats(signal, bpm, beats_per_bar):
    """Beat times (s) and their position in the bar (1 = downbeat), tracked at bpm +/- 15%."""
    import_madmom()
    from madmom.features.downbeats import RNNDownBeatProcessor, DBNDownBeatTrackingProcessor
    act = RNNDownBeatProcessor()(signal)
    proc = DBNDownBeatTrackingProcessor(beats_per_bar=[beats_per_bar], fps=100,
                                        min_bpm=bpm * (1 - TEMPO_MARGIN), max_bpm=bpm * (1 + TEMPO_MARGIN))
    out = proc(act)
    return out[:, 0], out[:, 1].astype(int)


def confirm_tempo(candidates):
    """Show the estimate; Enter accepts it, or type another tempo."""
    best = candidates[0][0]
    print(f'\nEstimated tempo: {best:.0f} BPM  (half: {best / 2:.0f}, double: {best * 2:.0f})')
    others = ', '.join(f'{b:.0f}' for b, _ in candidates[1:4])
    if others:
        print(f'Other candidates: {others}')
    while True:
        answer = input('Press Enter to use it, or type the tempo in BPM: ').strip()
        if not answer:
            return best
        try:
            bpm = float(answer)
            if bpm > 0:
                return bpm
        except ValueError:
            pass
        print('Please type a number, e.g. 72.')


# ---------------------------------------------------------------- 4. rhythm

def meter_info(time_sig):
    """(numerator, denominator, beat length in quarter notes, beats per bar).
    6/8, 9/8, 12/8 (and /16) are counted in dotted beats, like most performers."""
    num, den = (int(x) for x in time_sig.split('/'))
    if den >= 8 and num % 3 == 0 and num > 3:
        return num, den, 3 * 4 / den, num // 3
    return num, den, 4 / den, num


def beat_positions(times, beat_times):
    """Fractional beat index of each time (linear between tracked beats, extended past the ends)."""
    idx = np.arange(len(beat_times), dtype=float)
    pos = np.interp(times, beat_times, idx)
    first = (beat_times[1] - beat_times[0]) if len(beat_times) > 1 else 1.0
    last = (beat_times[-1] - beat_times[-2]) if len(beat_times) > 1 else 1.0
    pos = np.where(times < beat_times[0], (times - beat_times[0]) / first, pos)
    pos = np.where(times > beat_times[-1], idx[-1] + (times - beat_times[-1]) / last, pos)
    return pos


def snap(ql, grid=GRID_QL):
    return np.round(np.asarray(ql) / grid) * grid


def quantize(notes, beat_times, beat_numbers, beat_ql, beats_per_bar, pickup=None):
    """Notes -> (start_ql, end_ql, midi, class_id) from the start of the first bar,
    plus the pickup used (beats before the first bar line)."""
    on = beat_positions(np.array([n[0] for n in notes]), beat_times)
    off = beat_positions(np.array([n[1] for n in notes]), beat_times)
    first = float(snap(on.min() * beat_ql)) / beat_ql          # first note, in beats, on the grid
    if pickup is None:
        downbeats = np.flatnonzero(beat_numbers == 1)
        phase = downbeats[0] if len(downbeats) else 0
        bar_line = phase + math.ceil((first - phase) / beats_per_bar - 1e-6) * beats_per_bar
        pickup = (bar_line - first) % beats_per_bar
    start = first + pickup - beats_per_bar if pickup > 0 else first   # beat where bar 1 (written) starts
    q_on = snap((on - start) * beat_ql)
    q_off = np.maximum(snap((off - start) * beat_ql), q_on + GRID_QL)   # at least one grid step
    return [(float(a), float(b), int(n[2]), int(n[3])) for a, b, n in zip(q_on, q_off, notes)], pickup


def one_line(notes):
    """One note at a time: same start keeps the longest, earlier notes end where the next starts."""
    out = []
    for a, b, p in sorted(notes, key=lambda n: (n[0], -(n[1] - n[0]))):
        if out and a == out[-1][0]:
            continue
        if out and out[-1][1] > a:
            out[-1] = (out[-1][0], a, out[-1][2])
        out.append((a, b, p))
    return [n for n in out if n[1] > n[0]]


# ---------------------------------------------------------------- 5. score

def make_instrument(name):
    from music21 import instrument
    cls_name, label = PART_STYLE.get(name.lower(), (None, name.capitalize()))
    inst = getattr(instrument, cls_name)() if cls_name else instrument.Instrument()
    inst.partName = inst.instrumentName = label
    return inst


def as_pickup_bar(part, pad_ql):
    """Turn the first bar into a pickup bar (bar 0) by dropping its first pad_ql of rests."""
    m1 = part.getElementsByClass('Measure')[0]
    for e in list(m1.notesAndRests):
        if e.offset >= pad_ql:
            e.offset -= pad_ql
        elif e.isRest and e.offset + e.quarterLength <= pad_ql + 1e-9:
            m1.remove(e)
        elif e.isRest:                                    # rest running past the pickup start
            e.quarterLength = e.offset + e.quarterLength - pad_ql
            e.offset = 0.0
    m1.paddingLeft = pad_ql
    for k, m in enumerate(part.getElementsByClass('Measure')):
        m.number = k


def write_score(score, path):
    """MusicXML; a pickup bar (number 0) is marked implicit so it is not counted as bar 1."""
    import re
    score.write('musicxml', fp=path)
    with open(path, encoding='utf-8') as f:
        xml = f.read()
    xml = re.sub(r'<measure implicit="no" number="0"', '<measure implicit="yes" number="0"', xml)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(xml)


def build_score(parts, time_sig, bpm, beat_ql, title=None, pickup_ql=0.0):
    """parts: [(instrument name, [(start_ql, end_ql, midi), ...]), ...] at sounding pitch,
    measured from the start of a full first bar; pickup_ql > 0 makes that bar a pickup bar."""
    from music21 import duration, meter, metadata, note, stream, tempo
    bar_ql = meter.TimeSignature(time_sig).barDuration.quarterLength
    end = max([n[1] for _, ns in parts for n in ns] + [bar_ql])
    total = math.ceil(end / bar_ql - 1e-9) * bar_ql
    score = stream.Score()
    if title:
        score.insert(0, metadata.Metadata(title=title))
    for k, (name, notes) in enumerate(parts):
        part = stream.Part()
        part.insert(0, make_instrument(name))
        part.insert(0, meter.TimeSignature(time_sig))
        if k == 0:
            part.insert(0, tempo.MetronomeMark(number=round(bpm), referent=duration.Duration(beat_ql)))
        for a, b, p in notes:
            part.insert(a, note.Note(midi=p, quarterLength=b - a))
        part.makeRests(refStreamOrTimeRange=[0.0, total], fillGaps=True, inPlace=True)
        part.makeMeasures(inPlace=True)
        part.makeTies(inPlace=True)
        if 0 < pickup_ql < bar_ql:
            as_pickup_bar(part, bar_ql - pickup_ql)
        part.atSoundingPitch = True          # written as transposed parts on export
        score.insert(0, part)
    return score


# ---------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('audio')
    ap.add_argument('--time-sig', required=True, help='e.g. 4/4, 3/4, 6/8 (always set by the user)')
    ap.add_argument('--bpm', type=float, help='tempo; skips the estimate')
    ap.add_argument('--yes', action='store_true', help='accept the estimated tempo without asking')
    ap.add_argument('--pickup', type=float, help='beats before the first bar line (default: detected)')
    ap.add_argument('--instruments', nargs='+', help='parts to write (default: all the model knows)')
    ap.add_argument('--run', default=DEFAULT_RUN, help=f'training run under {RUNS} (default {DEFAULT_RUN})')
    ap.add_argument('--weights', default='best', help='checkpoint in the run (default best)')
    ap.add_argument('--model', help='path to a .pt file instead of --run (give --harmonics too)')
    ap.add_argument('--harmonics', type=float, nargs='+', help='HCQT harmonics (default: from the run)')
    ap.add_argument('--conf', type=float, default=0.25, help='confidence cutoff (default 0.25)')
    ap.add_argument('--out', help='output .musicxml (default: next to the audio)')
    ap.add_argument('--title', help='score title (default: the audio file name)')
    ap.add_argument('--save-notes', action='store_true', help='also write the detected notes as .tsv')
    a = ap.parse_args()

    try:
        num, den, beat_ql, beats_per_bar = meter_info(a.time_sig)
    except ValueError:
        ap.error('--time-sig must look like 4/4')
    if a.model:
        weights = a.model
        if not a.harmonics:
            ap.error('--model needs --harmonics (the HCQT the model was trained on)')
    else:
        weights = os.path.join(RUNS, a.run, 'weights', f'{a.weights}.pt')
    harmonics = tuple(int(h) if float(h).is_integer() else h for h in a.harmonics) if a.harmonics \
        else run_harmonics(os.path.join(RUNS, a.run))[0]
    stem = os.path.splitext(a.audio)[0]
    out = a.out or stem + '.musicxml'

    print(f'Detecting notes with {weights} (HCQT harmonics {harmonics})...')
    audio = load_audio(a.audio)
    notes, names = detect_notes(audio, weights, harmonics, a.conf)
    print(f'  {len(notes)} notes')
    if a.save_notes:
        with open(stem + '_notes.tsv', 'w') as f:
            f.write('onset\toffset\tpitch\tinstrument\n')
            for on, off, p, i in notes:
                f.write(f'{on:.4f}\t{off:.4f}\t{p}\t{names[i]}\n')
    wanted = [n.lower() for n in a.instruments] if a.instruments else [names[i] for i in sorted(names)]
    unknown = [n for n in wanted if n not in names.values()]
    if unknown:
        ap.error(f'unknown instrument(s) {unknown}; this model knows {list(names.values())}')
    order = [i for i in sorted(names) if names[i] in wanted]           # score order of the model
    notes = [n for n in notes if n[3] in order]
    if not notes:
        sys.exit('No notes detected for the chosen instruments.')

    print('Analyzing the beat (madmom)...')
    signal = madmom_signal(a.audio)
    if a.bpm:
        bpm = a.bpm
    else:
        candidates = estimate_tempo(signal)
        bpm = candidates[0][0] if a.yes else confirm_tempo(candidates)
    beat_times, beat_numbers = track_beats(signal, bpm, beats_per_bar)
    tracked = 60 / np.median(np.diff(beat_times)) if len(beat_times) > 1 else bpm
    q, pickup = quantize(notes, beat_times, beat_numbers, beat_ql, beats_per_bar, a.pickup)
    print(f'  tempo {bpm:.0f} BPM (beats tracked at about {tracked:.0f}), {len(beat_times)} beats, '
          f'pickup {pickup:g} beat(s){"" if a.pickup is not None else " (detected; set with --pickup)"}')

    parts = [(names[i], one_line([(s, e, p) for s, e, p, c in q if c == i])) for i in order]
    score = build_score(parts, a.time_sig, bpm, beat_ql, a.title or os.path.basename(stem), pickup * beat_ql)
    write_score(score, out)
    for name, ns in parts:
        print(f'  {name}: {len(ns)} notes')
    print(f'Score: {out}')
    print('TRANSCRIPTION DONE')


if __name__ == '__main__':
    main()
