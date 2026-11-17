# -*- coding: utf-8 -*-
"""Score for the postcard: the same plucked-string voice, half as busy.

The wind-film score played eight eighths per chord. On a still card with a line
of small type writing itself on, that reads as background noise competing with
the letter, so this version drops to four notes per chord, moves the melody up
an octave as a music-box line, and lengthens the room. 60 BPM, two chords per 2 s
bar, four bars to the 8 s loop, two loops to the 16 s video.

    python tri_card_music.py      -> card_music.wav (16 s, 44.1 kHz stereo)
"""
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

from tri_music import SR, EIGHTH, note, add

BPM = 60.0
PHRASE = 8 * EIGHTH * 2          # 8.0 s
TOTAL = 2 * PHRASE
HERE = os.path.dirname(os.path.abspath(__file__))

CHORDS = [
    ("D", [146.83, 220.00, 293.66, 369.99]),
    ("A/C#", [138.59, 220.00, 277.18, 329.63]),
    ("Bm", [123.47, 185.00, 246.94, 293.66]),
    ("G", [98.00, 196.00, 246.94, 293.66]),
]
# one note per beat: root, fifth, third -- the arpeggio states the chord and
# then gets out of the way of the writing
BEATS = [0, 2, 1, 3]


def build():
    n = int(TOTAL * SR)
    arp = np.zeros(n)
    bass = np.zeros(n)
    pad = np.zeros(n)
    bell = np.zeros(n)
    t = np.arange(int(2.6 * SR)) / SR

    for ph in range(2):
        for c, (name, tones) in enumerate(CHORDS):
            t0 = ph * PHRASE + c * 2.0
            f0 = tones[0] / 2
            env = np.minimum(1.0, t / 0.45) * np.exp(-t / 1.9)
            add(bass, 0.24 * env * np.sin(2 * np.pi * f0 * t), t0)
            for k, f in enumerate(tones[1:]):
                env2 = np.minimum(1.0, t / 0.6) * np.exp(-t / 2.4)
                add(pad, 0.050 * env2 * np.sin(2 * np.pi * f * t + k), t0)
            for e, k in enumerate(BEATS):
                add(arp, note(tones[k], 2.4, gain=0.30 if e % 2 == 0 else 0.22,
                              seed=100 * ph + 10 * c + e), t0 + e * 2 * EIGHTH)
            # one high note per bar, very quiet: it is the part that makes the
            # card feel lit rather than merely accompanied
            if ph == 1 or c % 2 == 0:
                add(bell, note(tones[2] * 2, 2.8, gain=0.085, seed=900 + 10 * ph + c),
                    t0 + 3 * EIGHTH)

    dry = arp + bass + pad + bell

    rng = np.random.default_rng(11)
    ir_t = np.arange(int(1.05 * SR)) / SR
    ir = rng.normal(0, 1, len(ir_t)) * np.exp(-ir_t / 0.34)
    ir[: int(0.010 * SR)] = 0.0
    ir /= np.abs(ir).max()
    wet = fftconvolve(dry, ir)[:n] * 0.16

    mix = dry + wet
    mix = np.convolve(mix, np.array([0.25, 0.5, 0.25]), mode="same")

    fi, fo = int(0.45 * SR), int(1.6 * SR)
    mix[:fi] *= np.linspace(0, 1, fi) ** 1.3
    mix[-fo:] *= np.linspace(1, 0, fo) ** 1.5
    mix /= max(1e-9, np.abs(mix).max())
    mix *= 0.74

    d = int(0.007 * SR)
    right = np.concatenate([np.zeros(d), mix[:-d]])
    stereo = np.stack([mix, 0.72 * mix + 0.28 * right], axis=1)
    return np.clip(stereo, -1, 1)


def main():
    audio = build()
    out = os.path.join(HERE, "card_music.wav")
    wavfile.write(out, SR, (audio * 32767).astype(np.int16))
    print(f"wrote card_music.wav  {audio.shape[0]/SR:.2f} s  "
          f"peak {np.abs(audio).max():.2f}  rms {np.sqrt((audio**2).mean()):.3f}")


if __name__ == "__main__":
    main()
