# -*- coding: utf-8 -*-
"""A quiet score for the bougainvillea loop: plucked strings over a low pad.

Nothing is sampled -- the plucks are Karplus-Strong (a burst of noise recirculated
through a one-pole lowpass), which is why they sound like a string and not like a
sine. 60 BPM, one chord per two beats, so 16 notes fill exactly the 8 s loop and
two phrases fill the 16 s video.

    python tri_music.py            -> bougainvillea.wav (16 s, 44.1 kHz stereo)
"""
import os

import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

SR = 44100
BPM = 60.0
EIGHTH = 60.0 / BPM / 2          # 0.5 s
PHRASE = 16 * EIGHTH             # 8.0 s
TOTAL = 2 * PHRASE
HERE = os.path.dirname(os.path.abspath(__file__))

# D major, the four chords of the phrase, two beats each
CHORDS = [
    ("D", [146.83, 220.00, 293.66, 369.99]),
    ("A/C#", [138.59, 220.00, 277.18, 329.63]),
    ("Bm", [123.47, 185.00, 246.94, 293.66]),
    ("G", [98.00, 196.00, 246.94, 293.66]),
]
ARP = [0, 2, 1, 3, 2, 1, 3, 2]   # which chord tone per eighth


def note(freq, dur, gain=1.0, damp=0.9975, seed=0):
    """A plucked string, additive.

    Karplus-Strong was the first choice and it was wrong here: its loop filter is
    a two-point average, which barely touches partials at 900 Hz, so the spectrum
    came out brightest at the 6th harmonic (measured 885 Hz on a 146.8 Hz note) --
    a twang, not a harp. Four partials with faster decays give a fundamental that
    actually dominates, and the timbre is fully determined by numbers I can check.
    """
    t = np.arange(int(round(dur * SR))) / SR
    env = np.exp(-t / (dur * 0.42))
    sig = np.zeros_like(t)
    for k, (amp, rel) in enumerate(((1.00, 1.00), (0.34, 0.62), (0.16, 0.42), (0.07, 0.30)), start=1):
        sig += amp * np.exp(-t / (dur * 0.42 * rel)) * np.sin(2 * np.pi * freq * k * t + 0.3 * k)
    sig *= env
    a = int(0.008 * SR)
    if a < len(sig):
        sig[:a] *= np.linspace(0, 1, a) ** 1.5
    return gain * sig


def add(track, sig, at):
    i = int(round(at * SR))
    j = min(len(track), i + len(sig))
    if j > i:
        track[i:j] += sig[:j - i]


def build():
    n = int(TOTAL * SR)
    arp = np.zeros(n)
    bass = np.zeros(n)
    pad = np.zeros(n)
    t = np.arange(int(2.4 * SR)) / SR

    for ph in range(2):
        for c, (name, tones) in enumerate(CHORDS):
            t0 = ph * PHRASE + c * 2 * EIGHTH
            # bass: root, two octaves down, slow swell
            f0 = tones[0] / 2
            env = np.minimum(1.0, t / 0.35) * np.exp(-t / 1.6)
            add(bass, 0.30 * env * np.sin(2 * np.pi * f0 * t), t0)
            # pad: the chord, very quiet, breathing with the phrase
            for k, f in enumerate(tones[1:]):
                env2 = np.minimum(1.0, t / 0.5) * np.exp(-t / 2.2)
                add(pad, 0.055 * env2 * np.sin(2 * np.pi * f * t + k), t0)
            # arpeggio: eight eighths per chord
            for e in range(8):
                k = ARP[e]
                f = tones[k]
                octave = 2.0 if (ph == 1 and e >= 4) else 1.0
                g = 0.36 if e % 2 == 0 else 0.27
                add(arp, note(f * octave, 1.8,
                              gain=g, seed=100 * ph + 10 * c + e),
                    t0 + e * EIGHTH)

    dry = arp + bass + pad

    # a small room: decaying noise as the impulse response
    rng = np.random.default_rng(7)
    ir_t = np.arange(int(0.85 * SR)) / SR
    ir = rng.normal(0, 1, len(ir_t)) * np.exp(-ir_t / 0.28)
    ir[: int(0.008 * SR)] = 0.0
    ir /= np.abs(ir).max()
    wet = fftconvolve(dry, ir)[:n] * 0.12

    mix = dry + wet
    # gentle top-end smoothing: the KS buffer has a little digital fizz
    mix = np.convolve(mix, np.array([0.25, 0.5, 0.25]), mode="same")

    # fades and level
    fi, fo = int(0.35 * SR), int(1.4 * SR)
    mix[:fi] *= np.linspace(0, 1, fi)
    mix[-fo:] *= np.linspace(1, 0, fo) ** 1.4
    mix /= max(1e-9, np.abs(mix).max())
    mix *= 0.80

    # a touch of width: 6 ms delay on the right, which is all a small room needs
    d = int(0.006 * SR)
    right = np.concatenate([np.zeros(d), mix[:-d]])
    stereo = np.stack([mix, 0.75 * mix + 0.25 * right], axis=1)
    return np.clip(stereo, -1, 1)


def main():
    audio = build()
    out = os.path.join(HERE, "bougainvillea.wav")
    wavfile.write(out, SR, (audio * 32767).astype(np.int16))
    print(f"wrote bougainvillea.wav  {audio.shape[0]/SR:.2f} s  "
          f"peak {np.abs(audio).max():.2f}  rms {np.sqrt((audio**2).mean()):.3f}")


if __name__ == "__main__":
    main()
