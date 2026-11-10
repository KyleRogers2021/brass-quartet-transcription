r"""
Notes <-> YOLO boxes, and windowing for long recordings.

A note is (onset_s, offset_s, midi_pitch, instrument_id). In an image covering
[t0, t0 + CLIP_SECONDS) it becomes a YOLO box (class, x_center, y_center, width,
height), all normalized to 0-1: x is time, y is pitch with high pitches at the top
(image row 0), and the box is BOX_SEMITONES tall, centered on the pitch.
"""
import settings as S


def pitch_to_y(pitch):
    """Normalized y of a pitch's center (0 = top of image = highest pitch)."""
    return (S.MIDI_HI + 0.5 - pitch) / S.N_SEMITONES


def y_to_pitch(y):
    return int(round(S.MIDI_HI + 0.5 - y * S.N_SEMITONES))


def notes_to_boxes(notes, t0, min_width_s=0.0):
    """Boxes for every note that overlaps the clip starting at t0 (clipped at its edges)."""
    t1 = t0 + S.CLIP_SECONDS
    h = S.BOX_SEMITONES / S.N_SEMITONES
    boxes = []
    for on, off, pitch, inst in notes:
        a, b = max(on, t0), min(off, t1)
        if b - a <= min_width_s or not (S.MIDI_LO <= pitch <= S.MIDI_HI):
            continue
        boxes.append((inst, ((a + b) / 2 - t0) / S.CLIP_SECONDS, pitch_to_y(pitch),
                      (b - a) / S.CLIP_SECONDS, h))
    return boxes


def boxes_to_notes(boxes, t0):
    """Notes from boxes (class, xc, yc, w, h[, confidence]) of the clip starting at t0."""
    notes = []
    for box in boxes:
        inst, xc, yc, w = box[0], box[1], box[2], box[3]
        on = t0 + (xc - w / 2) * S.CLIP_SECONDS
        off = t0 + (xc + w / 2) * S.CLIP_SECONDS
        notes.append((on, off, y_to_pitch(yc), int(inst)))
    return notes


def window_starts(duration):
    """Start times of overlapping clips covering a recording of this length."""
    starts, t = [], 0.0
    while True:
        starts.append(t)
        if t + S.CLIP_SECONDS >= duration:
            return starts
        t += S.WINDOW_HOP_SECONDS


def merge_notes(notes, min_overlap=0.001):
    """Join pieces of the same note found in overlapping windows.

    Each instrument plays one note at a time, so two pieces with the same instrument
    and pitch that overlap in time are the same note. Pieces that only touch are kept
    apart, since a repeated note starts exactly where the previous one ends.
    """
    merged = []
    groups = {}
    for n in notes:
        groups.setdefault((n[3], n[2]), []).append(n)
    for (inst, pitch), group in groups.items():
        group.sort()
        cur_on, cur_off = group[0][0], group[0][1]
        for on, off, _, _ in group[1:]:
            if on < cur_off - min_overlap:
                cur_off = max(cur_off, off)
            else:
                merged.append((cur_on, cur_off, pitch, inst))
                cur_on, cur_off = on, off
        merged.append((cur_on, cur_off, pitch, inst))
    return sorted(merged)


def write_yolo(path, boxes):
    with open(path, 'w') as f:
        for c, xc, yc, w, h in boxes:
            f.write(f'{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')


def read_yolo(path):
    with open(path) as f:
        return [(int(p[0]), *map(float, p[1:5])) for p in (line.split() for line in f) if p]
