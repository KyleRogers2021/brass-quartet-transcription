r"""
Shared settings for the brass note detector. Dataset generation, training,
decoding and the checks all read these, so changing the pitch window or the
pitch-shift range is a change in one place.
"""
# audio and time axis
SAMPLE_RATE = 16000
HOP = 256                         # samples per image column: 16 ms
FRAMES = 640                      # image width in columns
CLIP_SECONDS = FRAMES * HOP / SAMPLE_RATE      # 10.24 s, same clip length as the O&F runs
WINDOW_HOP_SECONDS = CLIP_SECONDS / 2          # long recordings: windows overlap by half

# pitch axis: MIDI MIDI_LO..MIDI_HI (inclusive), BINS_PER_SEMITONE CQT bins each,
# stretched vertically to IMAGE_HEIGHT pixels
MIDI_LO = 20
MIDI_HI = 99
N_SEMITONES = MIDI_HI - MIDI_LO + 1            # 80
BINS_PER_SEMITONE = 4
IMAGE_HEIGHT = 640                             # 8 pixels per semitone
IMAGE_WIDTH = FRAMES

# HCQT channels (harmonic multiples of the base CQT); 3 so pretrained YOLO loads directly
HARMONICS = (1, 2, 3)

# boxes: one per note, BOX_SEMITONES tall, centered on the note's pitch
BOX_SEMITONES = 2

# training data augmentation (as in the O&F experiments)
PITCH_SHIFT_MAX = 9
STEM_SUBSET_PROB = 0.5

INSTRUMENTS = ['trumpet', 'horn', 'trombone', 'tuba']   # YOLO class ids 0-3
