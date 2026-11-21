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
  4. Clean-up in beats, before rounding (in this order):
     - beat offset: note starts sit slightly off madmom's beats (detector onsets ~22 ms
       early on real recordings, players ahead of the beat, the tracker settling after a
       hold); the local median offset of the starts near a beat (within 4 beats) is
       removed, and bar 1 starts on a tracked beat (--no-beat-offset keeps the offset).
     - phrase ends (--no-phrase-ends to switch off): a gap where no part starts a note
       for clearly longer than the usual note spacing (at least 2.5 beats and 2.5x the
       median spacing) and all parts stop for a breath (a quarter beat with nothing
       sounding; notes under half a beat, e.g. stray detector notes, are ignored).
       Players usually hold the last chord longer than written there (an unwritten
       fermata) and breathe; the beat tracker counts those extra beats, so every later
       bar line drifts. The next phrase is moved back by the smallest number of whole
       beats that puts its first note at the bar position where phrases start (default:
       the first note's, i.e. the pickup; --phrase-pickup sets it), if the hold is long
       enough to contain them (written rests played in time stay). Notes held into the
       gap end where the next phrase starts (a written half note + rest comes out as a
       dotted half); --fermatas marks them with a fermata (off by default: the
       original scores have none, the holds are the players'). The last chord ends where a next phrase would
       start (the final bar completes the pickup bar). Own heuristic, not a published
       method; madmom's own bar positions (downbeat tracking and its raw downbeat
       activation) were checked for this and do not recover after such holds.
     - short gaps: a note is extended to the next note in the same part when the
       silence between them is shorter than --gap-fill beats (default 0.5). Players
       release early to breathe; written out, that gives many short rests.
  5. Rhythm: every note start and end is placed on its beat position and rounded to
     the --grid (16 = 16th notes, the default; 8 = 8th notes, better for hymns and
     chorales), so durations come from the beats around each note and small tempo
     errors do not add up over the piece.
  6. Score: one part per instrument, one note at a time (brass): notes starting
     together in one part keep the longest; a note still sounding when the next one
     starts is cut there. Gaps become rests, notes across bar lines are tied, and an
     upbeat becomes a pickup bar (bar 0). No key signature (accidentals are written
     as needed). Trumpet is written in B-flat and horn in F (transposed parts);
     trombone and tuba at concert pitch. Written with music21.

Tested on a synthetic recording with a known score (3/4, one-beat pickup, tempo
slowing from 84 to 70 BPM, given the true note times): every note came out with
the right start, length, bar and pitch, and the pickup was found. Tested on two
ChoraleBricks recordings against their original scores (11/21). Befiehl du deine
Wege (4/4, quarter pickup, 7 phrase ends held 2-3 beats longer than written), --grid 8:
true notes (--notes): 17 bars as in the score, 98% of notes at the right time and
pitch (before these steps: 22 bars, 3%); detector notes: 17 bars, 89% of note starts
and 52% of notes (with pitch) right (before: 1%). Du Friedensfuerst (4/4 with 6/4
bars, which one time signature cannot write): note values improve, bar lines differ.

Limits: one time signature and one tempo marking; triplets are not written; two players
of the same instrument are merged; phrase ends assume all phrases start at the same
place in the bar (chorales, hymns); a written half note + rest at a phrase end is
written as a dotted half.

The time signature is always given by the user (--time-sig); it is not detected.

Run in the yolo environment (needs once: pip install music21 madmom-prebuilt):
    python R:\code\detector\transcribe.py recording.wav --time-sig 4/4
    python R:\code\detector\transcribe.py recording.wav --time-sig 3/4 --bpm 72 --pickup 1
    python R:\code\detector\transcribe.py recording.wav --time-sig 4/4 --instruments trumpet tuba
    python R:\code\detector\transcribe.py mix.wav --time-sig 4/4 --grid 8 --notes labels.tsv
Options
    --time-sig   required, e.g. 4/4, 3/4, 2/2, 6/8 (6/8, 9/8, 12/8 are counted in dotted quarters)
    --bpm        tempo in beats per minute (beat = the time signature's beat); skips the estimate
    --yes        accept the estimated tempo without asking
    --pickup     beats before the first bar line (0 = starts on the downbeat); default: detected
    --grid       8 or 16: shortest note value (default 16)
    --gap-fill   fill silences shorter than this many beats (default 0.5; 0 = off)
    --no-phrase-ends  switch off the phrase-end handling
    --fermatas   mark notes held longer than written at phrase ends with a fermata (default off)
    --no-beat-offset  keep note starts where they fall against the tracked beats
    --phrase-pickup   beats before the bar line where phrases start (default: the pickup)
    --notes      use a note list (.tsv) instead of the detector: ChoraleBricks labels
                 (onset, offset, pitch, velocity, instrument id 0-3) or a --save-notes file;
                 shows what the rhythm steps do with perfect notes
    --instruments  which parts to write, e.g. trumpet horn (default: every instrument with notes)
    --run / --weights  trained detector (default DEFAULT_RUN below, best.pt);
    --model      or a path to any .pt file (then give --harmonics)
    --harmonics  HCQT harmonics the model was trained on (default: read from the run)
    --conf       detector confidence cutoff (default 0.25, as in the evaluation)
    --out        output file (default: next to the audio, <name>.musicxml; a recording called
                 mix.wav is named after its folder; an existing file is never overwritten,
                 _2, _3, ... are added instead. --out writes exactly the file given.)
    --save-notes also write the raw detected notes (<output name>_notes.tsv, in seconds)
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
DEFAULT_GRID = 16                  # rhythm grid: 16th notes (8 = 8th notes)
GAP_FILL = 0.5                     # fill silences shorter than this many beats
PHRASE_GAP = 2.5                   # phrase end: no new note for >= this many beats and x median spacing
ONSET_CLUSTER = 0.25               # onsets closer than this (beats) count as one chord
BREATH = 0.25                      # a phrase end needs a moment (beats) where no part sounds
MIN_BREAK_NOTE = 0.5               # notes shorter than this (beats) do not end a phrase break
CHORD_SPREAD = 0.5                 # first/last chord: onsets within this many beats (players spread out)
FERMATA_HOLD = 1.0                 # held this many beats past the cut point -> fermata
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


def snap(ql, grid):
    return np.round(np.asarray(ql) / grid) * grid


def fill_gaps(on, off, cls, max_gap):
    """Extend a note to the next note in the same part when the silence between them is
    shorter than max_gap beats. Returns the new offsets and how many notes were extended."""
    off = off.copy()
    filled = 0
    for c in np.unique(cls):
        idx = np.flatnonzero(cls == c)
        starts = np.sort(on[idx])
        for i in idx:
            later = starts[starts > on[i] + 1e-6]
            if len(later) and 0 < later[0] - off[i] < max_gap:
                off[i] = later[0]
                filled += 1
    return off, filled


def beat_offsets(on, window=4.0, max_dev=0.3):
    """How far (beats) note starts sit from the tracked beats, locally: for each note, the
    median deviation of the starts within +/-window beats that lie within max_dev of a beat.
    Note starts sit a little before madmom's beats (detector onsets ~22 ms early on real
    recordings, players ahead of the beat, the tracker settling after a hold), which rounds
    notes to the wrong 16th; this offset is removed before rounding."""
    dev = on - np.round(on)
    near = np.abs(dev) < max_dev
    out = np.zeros(len(on))
    for i, t in enumerate(on):
        sel = near & (np.abs(on - t) <= window)
        if sel.any():
            out[i] = np.median(dev[sel])
    return out


def bar_position(beat, bar1, beats_per_bar):
    """Position in the bar (beats after the bar line) of a beat; bar1 = beat where bar 1 starts."""
    return (beat - bar1) % beats_per_bar


def longest_silence(on, off, t0, t1):
    """Longest stretch (beats) between t0 and t1 where no note sounds."""
    edges = sorted((max(a, t0), min(b, t1)) for a, b in zip(on, off) if b > t0 and a < t1)
    longest, t = 0.0, t0
    for a, b in edges:
        longest = max(longest, a - t)
        t = max(t, b)
    return max(longest, t1 - t)


def phrase_ends(on, off, cls, bar1, pickup, beats_per_bar, phrase_pickup=None, min_gap=PHRASE_GAP):
    """Take out the beats players add at phrase ends (held last chord + breath).
    on/off: note starts/ends in beats as tracked; bar1: beat where bar 1 starts; pickup: beats
    before the first bar line. Phrases are assumed to start at the bar position
    phrase_pickup beats before a bar line (default: the pickup).
    Returns new on, off, fermata flags (one per note) and [(time_index, beats removed, ...)]."""
    n = len(on)
    on2, off2, fermata = on.copy(), off.copy(), np.zeros(n, bool)
    starts = []                                            # chord onsets (blips under half a beat ignored:
    for t in np.sort(on[off - on >= MIN_BREAK_NOTE]):      # stray detector notes inside a hold)
        if not starts or t - starts[-1] > ONSET_CLUSTER:
            starts.append(t)
    if len(starts) < 3:
        return on2, off2, fermata, []
    starts = np.array(starts)
    gaps = np.diff(starts)
    threshold = max(min_gap, min_gap * float(np.median(gaps)))
    target = (beats_per_bar - (pickup if phrase_pickup is None else phrase_pickup)) % beats_per_bar
    removed = 0.0                                          # beats taken out so far
    breaks = []
    long_notes = off - on >= MIN_BREAK_NOTE
    for j in np.flatnonzero(gaps >= threshold):
        last, nxt = starts[j], starts[j + 1]
        if longest_silence(on[long_notes], off[long_notes], last, nxt) < BREATH:
            continue                                       # everyone holds on: long notes, not a phrase end
        pos = bar_position(nxt - removed, bar1, beats_per_bar)
        k = int(round((pos - target) % beats_per_bar)) % beats_per_bar
        if k > round(nxt - last) - 1:                      # the hold is too short: written rest, keep
            k = 0
        breaks.append((last, nxt, k))
        removed += k
    # shift every note by the beats removed before it; cut notes held into a shortened break
    for i in range(n):
        shift = sum(k for _, nxt, k in breaks if nxt <= on[i] + ONSET_CLUSTER)
        on2[i], off2[i] = on[i] - shift, off[i] - shift
        for last, nxt, k in breaks:
            if k and on[i] < nxt - ONSET_CLUSTER and off[i] > last + ONSET_CLUSTER:
                off2[i] = nxt - shift - k                  # held into the break: ends where the next phrase starts
                if off[i] >= last + FERMATA_HOLD:          # still sounding a beat after the last chord
                    fermata[i] = True
                break
    # last chord: cut where a next phrase would start (the final bar completes the pickup bar)
    last_on = float(np.median(on2[on2 >= on2.max() - CHORD_SPREAD]))   # players enter a little apart
    end = last_on + 1e-6
    end += (target - bar_position(end, bar1, beats_per_bar)) % beats_per_bar
    if end - last_on < 1 - 1e-6:                           # at least one beat after the last chord
        end += beats_per_bar
    for i in np.flatnonzero(on2 >= on2.max() - CHORD_SPREAD):
        if off2[i] > end + 1e-6:
            fermata[i] = off2[i] >= end + FERMATA_HOLD
            off2[i] = end
    return on2, off2, fermata, breaks


def first_bar_line(first_on, beat_numbers, beats_per_bar, pickup=None, tol=0.25):
    """Beat (a tracked beat, whole number) where bar 1 starts: pickup beats after the first note
    if given, else madmom's first downbeat phase, at or after the first note (within tol)."""
    if pickup is not None:
        return int(round(first_on + pickup))
    downbeats = np.flatnonzero(beat_numbers == 1)
    phase = downbeats[0] if len(downbeats) else 0
    return int(phase + math.ceil((first_on - tol - phase) / beats_per_bar) * beats_per_bar)


def quantize(on, off, bar1, pickup, beat_ql, beats_per_bar, grid_ql):
    """Starts/ends in beats -> quarter notes from the start of the first (full) bar, on the grid."""
    start = bar1 - beats_per_bar if pickup > 0 else bar1      # a pickup bar is written as a full bar 0
    q_on = snap((on - start) * beat_ql, grid_ql)
    q_off = np.maximum(snap((off - start) * beat_ql, grid_ql), q_on + grid_ql)   # at least one grid step
    return q_on, q_off


def one_line(notes):
    """One note at a time: same start keeps the longest, earlier notes end where the next starts.
    notes: (start, end, midi, fermata)."""
    out = []
    for a, b, p, f in sorted(notes, key=lambda n: (n[0], -(n[1] - n[0]))):
        if out and a == out[-1][0]:
            continue
        if out and out[-1][1] > a:
            out[-1] = (out[-1][0], a, out[-1][2], out[-1][3])
        out.append((a, b, p, f))
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


def fermata_on_last_tied_note(part):
    """A note tied across a bar line keeps its fermata on the last tied note only."""
    from music21 import expressions
    notes = list(part.recurse().notes)
    for i, n in enumerate(notes):
        fs = [e for e in n.expressions if isinstance(e, expressions.Fermata)]
        if not fs or n.tie is None or n.tie.type == 'stop':
            continue
        for e in fs:
            n.expressions.remove(e)
        j = i
        while j + 1 < len(notes) and notes[j].tie is not None and notes[j].tie.type != 'stop':
            j += 1
        if not any(isinstance(e, expressions.Fermata) for e in notes[j].expressions):
            notes[j].expressions.append(expressions.Fermata())


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
    """parts: [(instrument name, [(start_ql, end_ql, midi, fermata), ...]), ...] at sounding
    pitch, measured from the start of a full first bar; pickup_ql > 0 makes that bar a pickup bar."""
    from music21 import duration, expressions, meter, metadata, note, stream, tempo
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
        for a, b, p, f in notes:
            n = note.Note(midi=p, quarterLength=b - a)
            if f:
                n.expressions.append(expressions.Fermata())
            part.insert(a, n)
        part.makeRests(refStreamOrTimeRange=[0.0, total], fillGaps=True, inPlace=True)
        part.makeMeasures(inPlace=True)
        part.makeTies(inPlace=True)
        fermata_on_last_tied_note(part)
        if 0 < pickup_ql < bar_ql:
            as_pickup_bar(part, bar_ql - pickup_ql)
        part.atSoundingPitch = True          # written as transposed parts on export
        score.insert(0, part)
    return score


# ---------------------------------------------------------------- input and output files

def read_notes(path, names):
    """Notes (onset_s, offset_s, midi, class_id) from a .tsv: ChoraleBricks labels
    (onset, offset, pitch, velocity, instrument id) or a --save-notes file
    (onset, offset, pitch, instrument name). Header lines are skipped."""
    ids = {v: k for k, v in names.items()}
    notes = []
    with open(path) as f:
        for line in f:
            row = line.strip().split('\t')
            try:
                on, off, pitch = float(row[0]), float(row[1]), int(round(float(row[2])))
            except (ValueError, IndexError):
                continue
            inst = row[-1].strip().lower()
            try:
                cid = int(float(inst))
            except ValueError:
                cid = ids.get(inst)
            if cid not in names:
                raise SystemExit(f'Unknown instrument {row[-1]!r} in {path}; known: {list(names.values())}')
            notes.append((on, off, pitch, cid))
    return sorted(notes)


def default_out(audio):
    """<audio name>.musicxml next to the audio; a file called mix.* is named after its folder
    (ChoraleBricks); an existing file is never overwritten (_2, _3, ... are added)."""
    folder, base = os.path.split(os.path.abspath(audio))
    name = os.path.splitext(base)[0]
    if name.lower() == 'mix' and os.path.basename(folder):
        name = os.path.basename(folder)
    path, k = os.path.join(folder, name + '.musicxml'), 2
    while os.path.exists(path):
        path, k = os.path.join(folder, f'{name}_{k}.musicxml'), k + 1
    return path


# ---------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('audio')
    ap.add_argument('--time-sig', required=True, help='e.g. 4/4, 3/4, 6/8 (always set by the user)')
    ap.add_argument('--bpm', type=float, help='tempo; skips the estimate')
    ap.add_argument('--yes', action='store_true', help='accept the estimated tempo without asking')
    ap.add_argument('--pickup', type=float, help='beats before the first bar line (default: detected)')
    ap.add_argument('--grid', type=int, choices=(8, 16), default=DEFAULT_GRID,
                    help=f'shortest note value: 8 or 16 (default {DEFAULT_GRID})')
    ap.add_argument('--gap-fill', type=float, default=GAP_FILL,
                    help=f'fill silences shorter than this many beats (default {GAP_FILL}; 0 = off)')
    ap.add_argument('--no-phrase-ends', action='store_true', help='switch off the phrase-end handling')
    ap.add_argument('--fermatas', action='store_true',
                    help='mark notes held longer than written at phrase ends with a fermata')
    ap.add_argument('--phrase-pickup', type=float,
                    help='beats before the bar line where phrases start (default: the pickup)')
    ap.add_argument('--no-beat-offset', action='store_true',
                    help='do not correct for note starts sitting before/after the tracked beats')
    ap.add_argument('--notes', help='note list (.tsv) to use instead of the detector')
    ap.add_argument('--instruments', nargs='+', help='parts to write (default: every instrument with notes)')
    ap.add_argument('--run', default=DEFAULT_RUN, help=f'training run under {RUNS} (default {DEFAULT_RUN})')
    ap.add_argument('--weights', default='best', help='checkpoint in the run (default best)')
    ap.add_argument('--model', help='path to a .pt file instead of --run (give --harmonics too)')
    ap.add_argument('--harmonics', type=float, nargs='+', help='HCQT harmonics (default: from the run)')
    ap.add_argument('--conf', type=float, default=0.25, help='confidence cutoff (default 0.25)')
    ap.add_argument('--out', help='output .musicxml (default: next to the audio, never overwritten)')
    ap.add_argument('--title', help='score title (default: the output file name)')
    ap.add_argument('--save-notes', action='store_true', help='also write the detected notes as .tsv')
    a = ap.parse_args()

    try:
        num, den, beat_ql, beats_per_bar = meter_info(a.time_sig)
    except ValueError:
        ap.error('--time-sig must look like 4/4')
    grid_ql = 4 / a.grid
    out = a.out or default_out(a.audio)
    stem = os.path.splitext(out)[0]

    if a.notes:
        names = dict(enumerate(S.INSTRUMENTS))
        notes = read_notes(a.notes, names)
        print(f'Notes from {a.notes}: {len(notes)}')
    else:
        if a.model:
            weights = a.model
            if not a.harmonics:
                ap.error('--model needs --harmonics (the HCQT the model was trained on)')
        else:
            weights = os.path.join(RUNS, a.run, 'weights', f'{a.weights}.pt')
        harmonics = tuple(int(h) if float(h).is_integer() else h for h in a.harmonics) if a.harmonics \
            else run_harmonics(os.path.join(RUNS, a.run))[0]
        print(f'Detecting notes with {weights} (HCQT harmonics {harmonics})...')
        notes, names = detect_notes(load_audio(a.audio), weights, harmonics, a.conf)
        print(f'  {len(notes)} notes')
        if a.save_notes:
            with open(stem + '_notes.tsv', 'w') as f:
                f.write('onset\toffset\tpitch\tinstrument\n')
                for on, off, p, i in notes:
                    f.write(f'{on:.4f}\t{off:.4f}\t{p}\t{names[i]}\n')
    present = sorted({n[3] for n in notes})
    wanted = [n.lower() for n in a.instruments] if a.instruments else [names[i] for i in present]
    unknown = [n for n in wanted if n not in names.values()]
    if unknown:
        ap.error(f'unknown instrument(s) {unknown}; known: {list(names.values())}')
    order = [i for i in sorted(names) if names[i] in wanted]           # score order of the model
    notes = [n for n in notes if n[3] in order]
    if not notes:
        sys.exit('No notes for the chosen instruments.')

    print('Analyzing the beat (madmom)...')
    signal = madmom_signal(a.audio)
    if a.bpm:
        bpm = a.bpm
    else:
        candidates = estimate_tempo(signal)
        bpm = candidates[0][0] if a.yes else confirm_tempo(candidates)
    beat_times, beat_numbers = track_beats(signal, bpm, beats_per_bar)
    tracked = 60 / np.median(np.diff(beat_times)) if len(beat_times) > 1 else bpm

    # rhythm, in beats: gaps, phrase ends, then the grid
    on = beat_positions(np.array([n[0] for n in notes]), beat_times)
    off = beat_positions(np.array([n[1] for n in notes]), beat_times)
    cls = np.array([n[3] for n in notes])
    shift = np.zeros(len(on)) if a.no_beat_offset else beat_offsets(on)
    on, off = on - shift, off - shift
    first_on = float(np.median(on[on <= on.min() + CHORD_SPREAD]))              # first chord (players spread out)
    bar1 = first_bar_line(first_on, beat_numbers, beats_per_bar, a.pickup, tol=grid_ql / beat_ql)
    if a.pickup is not None:
        pickup = a.pickup
    else:                                                               # beats before bar 1, on the grid
        pickup = float(bar1 - snap(first_on * beat_ql, grid_ql) / beat_ql) % beats_per_bar
    print(f'  tempo {bpm:.0f} BPM (beats tracked at about {tracked:.0f}), {len(beat_times)} beats, '
          f'pickup {pickup:g} beat(s){"" if a.pickup is not None else " (detected; set with --pickup)"}')
    fermata = np.zeros(len(notes), bool)
    breaks = []
    if not a.no_phrase_ends:
        on, off, fermata, breaks = phrase_ends(on, off, cls, bar1, pickup, beats_per_bar, a.phrase_pickup)
        if not a.fermatas:
            fermata[:] = False
    filled = 0
    if a.gap_fill > 0:                                    # after the phrase ends: also closes gaps left there
        off, filled = fill_gaps(on, off, cls, a.gap_fill)
    removed = [k for _, _, k in breaks if k]
    print(f'  note starts vs beats: median {np.median(shift):+.2f}, range {shift.min():+.2f} to {shift.max():+.2f} '
          f'beats (removed)' if not a.no_beat_offset else '  beat offset: kept')
    print(f'  short gaps filled: {filled}; phrase ends: {len(breaks)}'
          + (f', beats taken out at {len(removed)} of them: {"+".join(map(str, removed))} '
             f'= {sum(removed)}' if removed else '') + f'; grid: {a.grid}ths')
    q_on, q_off = quantize(on, off, bar1, pickup, beat_ql, beats_per_bar, grid_ql)
    if breaks:
        start = bar1 - beats_per_bar if pickup > 0 else bar1
        first_bar = 0 if pickup > 0 else 1
        done, where = 0, []
        for last, nxt, k in breaks:
            done += k
            b = int((nxt - done - start) // beats_per_bar) + first_bar
            where.append(f'{b}' + (f' (-{k})' if k else ''))
        print(f'  next phrase starts in bar: {", ".join(where)}  (-k: beats taken out before it)')

    parts = [(names[i], one_line([(float(s), float(e), int(n[2]), bool(f))
                                  for s, e, n, f in zip(q_on, q_off, notes, fermata) if n[3] == i]))
             for i in order]
    title = a.title or os.path.basename(stem)
    score = build_score(parts, a.time_sig, bpm, beat_ql, title, pickup * beat_ql)
    write_score(score, out)
    for name, ns in parts:
        print(f'  {name}: {len(ns)} notes')
    print(f'Score: {out}')
    print('TRANSCRIPTION DONE')


if __name__ == '__main__':
    main()
