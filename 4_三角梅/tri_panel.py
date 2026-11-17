# -*- coding: utf-8 -*-
"""Bougainvillea in wind -- full-bleed loop, no matte.

Why this is not the camellia/chrysanthemum motion: nothing rotates around a
pivot and nothing falls off the plant. The frame itself is displaced by a
travelling wave whose amplitude is weighted by height, so the crown moves and
the trunk base, the gate and the birds do not. A soft shade band crosses the
frame once per loop, as if a cloud passed.

No matte is cut on purpose. The wall in this photo is flat but in focus, and the
gate and lamp carry strong edges -- `layers.py` would either swallow them into
the foreground (and then move them, which reads as a glitch) or leave a ghost.
A whole-frame displacement field avoids the problem instead of repairing it.

    python tri_panel.py                 -> GIF + PNG frames for the video
    python tri_panel.py --still 2.0     -> one frame, for looking
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "src", "bougainvillea.jpg")

LOOP, FPS = 8.0, 12.5
W, H = 600, 800                       # 3:4, the source's own shape
PAD = 10                              # warp margin: no frame edge is ever sampled

A_WAVE, A_SWAY, A_FLUT = 4.2, 2.2, 0.9     # px at 600 wide
SHADE, SHADE_SIGMA = 0.045, 0.20
BODY_KEEP = 0.75                      # below this height the picture is locked


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def height_weight(h):
    """1 through the crown, 0 by the trunk base. The wall stays still."""
    yn = (np.arange(h, dtype=np.float32) + 0.5) / h
    return smoothstep((BODY_KEEP - yn) / 0.50)[:, None]


def load_base():
    """Cover-fit to panel + warp margin in one resize, then centre-crop exactly.

    The margin has to be *built* here, not cropped from a slightly bigger image:
    a crop box that reaches past the resized picture is padded black, and the
    warp then samples those zeros along the right and bottom edges.
    """
    im = Image.open(SRC).convert("RGB")
    tw, th = W + 2 * PAD, H + 2 * PAD
    s = max(tw / im.width, th / im.height)
    nw, nh = max(tw, int(np.ceil(im.width * s))), max(th, int(np.ceil(im.height * s)))
    im = im.resize((nw, nh), Image.LANCZOS)
    cx, cy = (nw - tw) // 2, (nh - th) // 2
    return im.crop((cx, cy, cx + tw, cy + th))


def fields(w, h, t):
    """Displacement and light for one instant. Every period divides LOOP."""
    ph = 2 * np.pi * t / LOOP
    yn = (np.arange(h, dtype=np.float32) + 0.5) / h
    wn = ((np.arange(w, dtype=np.float32) + 0.5) / w)[None, :]
    wgt = height_weight(h)

    # gust: one wavelength across ~1.1 frames, travelling left to right once
    # per loop -- so the pattern at t=LOOP is the pattern at t=0
    kx = 2 * np.pi * 0.9
    wave = np.sin(kx * wn - ph)
    dx = (A_WAVE * wave
          + A_SWAY * np.sin(2 * ph + 0.4)          # LOOP/2
          + A_FLUT * np.sin(2 * np.pi * 3.1 * wn + 4 * ph))   # LOOP/4
    dy = (0.34 * A_WAVE * np.cos(kx * wn - ph + 0.7)
          + 0.6 * A_SWAY * np.sin(2 * ph + 1.1))

    k = w / 600.0                     # amplitudes are tuned at 600 px wide
    dx = dx * wgt * k
    dy = dy * wgt * k

    # a cloud passes once per loop: a wide, shallow dimming
    xc = w * (-0.6 + 2.2 * (t / LOOP))
    xcol = np.arange(w, dtype=np.float32)
    shade = 1.0 - SHADE * np.exp(-0.5 * ((xcol - xc) / (SHADE_SIGMA * w)) ** 2)
    breath = 1.0 + 0.010 * np.sin(ph + 0.9)
    return dx.astype(np.float32), dy.astype(np.float32), (shade * breath).astype(np.float32)


def frame(base, t):
    """Render t. Coordinates are taken around the pad margin, so dx=0 keeps the
    same crop as the still."""
    a = np.asarray(base, np.float32)
    h, w = a.shape[:2]
    dx, dy, light = fields(w, h, t)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ys = np.clip(yy + dy + PAD, 0, h - 1.001)
    xs = np.clip(xx + dx + PAD, 0, w - 1.001)
    out = np.empty((H, W, 3), np.float32)
    for c in range(3):
        ch = ndimage.map_coordinates(a[..., c], [ys, xs], order=1, mode="nearest")
        out[..., c] = ch[PAD:PAD + H, PAD:PAD + W]
    out *= light[PAD:PAD + W][None, :, None]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB")


# ------------------------------------------------------------- quantisation --
def bayer(n=8):
    if n == 1:
        return np.zeros((1, 1), np.float32)
    b = bayer(n // 2)
    return np.block([[4 * b, 4 * b + 2], [4 * b + 3, 4 * b + 1]])


THRESH = (bayer(8) / 64.0 - 0.5).astype(np.float32)


def orderly(img, pal, amp):
    a = np.asarray(img, np.float32)
    t = np.tile(THRESH, (a.shape[0] // 8 + 1, a.shape[1] // 8 + 1))[:a.shape[0], :a.shape[1]]
    a = np.clip(a + t[..., None] * amp, 0, 255).astype(np.uint8)
    return Image.fromarray(a).quantize(palette=pal, dither=Image.NONE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--frames", default="frames")
    ap.add_argument("--gif", default="bougainvillea_wind.gif")
    ap.add_argument("--size", default="600x800", help="WxH; motion scales with W")
    ap.add_argument("--fps", type=float, default=12.5)
    a = ap.parse_args()

    global W, H, FPS
    W, H = (int(v) for v in a.size.lower().split("x"))
    FPS = a.fps

    base = load_base()
    n = int(round(LOOP * FPS))
    ts = [i * LOOP / n for i in range(n)]

    if a.still is not None:
        frame(base, a.still).save("_still.png")
        print(f"wrote _still.png  {W}x{H}  t={a.still:.2f}")
        return

    os.makedirs(os.path.join(HERE, a.frames), exist_ok=True)
    rgb = []
    for i, t in enumerate(ts):
        im = frame(base, t)
        rgb.append(im)
        im.save(os.path.join(HERE, a.frames, f"f_{i:03d}.png"))
    print(f"rendered {n} frames  {W}x{H}")

    probe = Image.new("RGB", (W * 4, H))
    for k, say in enumerate((0.15, 0.4, 0.65, 0.9)):
        probe.paste(rgb[min(n - 1, int(round(say * n)))], (k * W, 0))
    pal = probe.quantize(colors=160, method=Image.MEDIANCUT)

    # The bracts are the point of the picture and they are 1 % of its pixels, so
    # MEDIANCUT never spends an entry on the magenta: it quantises to the nearest
    # beige and the tree reads brown. Reserve two slots -- the vivid pink found by
    # looking at the most saturated pixels, and a deeper one for the shadowed
    # bracts -- and orderly() maps them exactly.
    px = np.asarray(base, np.float32).reshape(-1, 3)
    mx, mn = px.max(1), px.min(1)
    score = (mx - mn) / np.maximum(1.0, mx) * (mx / 255.0)
    pink = np.median(px[score >= np.percentile(score, 99.6)], axis=0)
    reserved = [pink, pink * 0.82]
    keep = 160 - len(reserved)
    pl = list(pal.getpalette()[:keep * 3])
    for c in reserved:
        pl += [int(v) for v in np.clip(c, 0, 255)]
    pl += [0] * (768 - len(pl))
    pal.putpalette(pl)
    print("reserved pink", [int(v) for v in pink])

    out = [orderly(im, pal, 2.0) for im in rgb]
    delay = int(round(1000 * LOOP / n))
    path = os.path.join(HERE, a.gif)
    out[0].save(path, save_all=True, append_images=out[1:],
                duration=delay, loop=0, optimize=True)
    print(f"wrote {a.gif}  {os.path.getsize(path)/1e6:.2f} MB  {W}x{H}  {n} frames")


if __name__ == "__main__":
    main()
