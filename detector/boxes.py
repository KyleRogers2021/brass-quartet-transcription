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


def decode_windows(window_boxes, eps=1e-4):
    """Notes of a whole recording from the boxes of its overlapping windows.

    window_boxes: list of (t0, boxes) for every window, in time order.
    Each window is trusted only for the middle half of its span (the first window
    from the start, the last to the end), so the trusted parts tile the recording
    without overlap and a note cut at a window's edge is never used. A note that
    crosses the border between two trusted parts is cut there into two pieces,
    which are joined only if both windows see it continuing across the border.
    A repeated note that starts exactly where the previous one ends is therefore
    never joined to it.
    """
    n = len(window_boxes)
    quarter = S.CLIP_SECONDS / 4
    pieces = {}
    for i, (t0, boxes) in enumerate(window_boxes):
        lo = float('-inf') if i == 0 else t0 + quarter
        hi = float('inf') if i == n - 1 else t0 + 3 * quarter
        for on, off, pitch, inst in boxes_to_notes(boxes, t0):
            a, b = max(on, lo), min(off, hi)
            if b - a <= eps:          # nothing left (or only rounding) in the trusted part
                continue
            pieces.setdefault((inst, pitch), []).append([a, b, on < lo - eps, off > hi + eps])
    notes = []
    for (inst, pitch), group in pieces.items():
        group.sort()
        cur = group[0]
        for nxt in group[1:]:
            joins = abs(nxt[0] - cur[1]) <= eps and cur[3] and nxt[2]
            if joins:
                cur = [cur[0], nxt[1], cur[2], nxt[3]]
            else:
                notes.append((cur[0], cur[1], pitch, inst))
                cur = nxt
        notes.append((cur[0], cur[1], pitch, inst))
    return sorted(notes)


def write_yolo(path, boxes):
    with open(path, 'w') as f:
        for c, xc, yc, w, h in boxes:
            f.write(f'{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n')


def read_yolo(path):
    with open(path) as f:
        return [(int(p[0]), *map(float, p[1:5])) for p in (line.split() for line in f) if p]
