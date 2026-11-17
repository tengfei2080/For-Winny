# -*- coding: utf-8 -*-
"""Score for the ranunculus card: slower and emptier than the bougainvillea one.

The earlier score played an arpeggio through every chord. This card is a single
still bloom with three lines of type, so the music leaves room: one chord every
2.2 s, four notes per chord, a music-box line above it, and a longer reverb tail.
F major, 8.8 s -- the loop length of the card, so the video can hold it twice.

    python ran_card_music.py      -> card_music.wav
"""
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "4_三角梅")))
from tri_music import SR, note, add  # noqa: E402

LOOP = 8.8
BAR = LOOP / 4.0                 # 2.2 s per chord
TOTAL = 2 * LOOP                 # 17.6 s, two passes for the video

CHORDS = [
    ("F", [174.61, 220.00, 261.63, 349.23]),
    ("C/E", [164.81, 196.00, 261.63, 329.63]),
    ("Dm", [146.83, 174.61, 220.00, 293.66]),
    ("Bb", [116.54, 146.83, 174.61, 233.08]),
]
# the music-box line, one or two notes a bar -- sparse on purpose
BOX = [[698.46, 1046.50], [659.25], [587.33, 698.46], [587.33]]
ARP = [0, 2, 1, 3]


def reverb(x, tail=2.6, mix=0.30, seed=3):
    rng = np.random.default_rng(seed)
    n = int(tail * SR)
    t = np.arange(n) / SR
    ir = rng.normal(0, 1, n) * np.exp(-t / 0.85)
    ir[: int(0.02 * SR)] *= np.linspace(0, 1, int(0.02 * SR))
    ir /= np.sqrt((ir ** 2).sum())
    wet = np.stack([fftconvolve(x[:, c], ir)[: len(x)] for c in range(x.shape[1])], 1)
    return (1 - mix) * x + mix * wet * 1.6


def build():
    n = int(TOTAL * SR)
    arp = np.zeros(n)
    pad = np.zeros(n)
    bass = np.zeros(n)
    box = np.zeros(n)

    for ph in range(2):
        for c, (name, tones) in enumerate(CHORDS):
            t0 = ph * LOOP + c * BAR
            f0 = tones[0] / 2
            # bass: a slow swell, no attack to speak of
            tt = np.arange(int(3.0 * SR)) / SR
            env = np.minimum(1.0, tt / 0.55) * np.exp(-tt / 1.9)
            add(bass, 0.26 * env * np.sin(2 * np.pi * f0 * tt), t0)
            # pad: the chord, quiet, breathing across the whole bar
            for k, f in enumerate(tones[1:]):
                add(pad, 0.050 * env * np.sin(2 * np.pi * f * tt + k * 0.7), t0)
            # arpeggio: four notes, each left to ring for most of the bar
            for i, k in enumerate(ARP):
                add(arp, note(tones[k], 2.4, gain=0.20, seed=i),
                    t0 + i * BAR / 4.4)
            # music box an octave or two up
            for i, f in enumerate(BOX[c]):
                add(box, note(f, 2.0, gain=0.085 if i else 0.10, seed=10 + i),
                    t0 + 0.16 + i * 0.62)

    dry = np.stack([arp + 0.9 * pad + bass + box] * 2, 1)
    dry[:, 1] *= 0.96
    out = reverb(dry)
    peak = np.abs(out).max()
    out *= 0.14 / max(peak, 1e-9)          # about -17 dBFS, a background level
    # soft fade at both ends so a repeated pass has no click
    a = int(0.05 * SR)
    out[:a] *= np.linspace(0, 1, a)[:, None]
    out[-a:] *= np.linspace(1, 0, a)[:, None]
    return out


if __name__ == "__main__":
    x = build()
    p = os.path.join(HERE, "card_music.wav")
    wavfile.write(p, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
    print("wrote", p, f"{len(x)/SR:.2f} s", "peak", f"{np.abs(x).max():.3f}")
