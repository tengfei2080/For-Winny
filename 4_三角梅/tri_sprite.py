"""One falling bract.

Shape is solved rather than guessed: a bougainvillea bract is a broad ovule,
1.4 times longer than wide, widest just above the middle, rounded at the base
with a shallow fold where the stalk was, and closing to a small blunt tip.
The width profile below is exactly that; the tip taper is applied on top of it
-- an oval without the taper is the shape that read as a sticker.

Surface: one half turns away from the light (a fold along the midrib), the base
sits in the cluster's shade, and the rim is both lighter and more transparent
because paper transmits. Mottle is kept coarse: at 18 px finer texture averages
out to a flat tone, and finer noise becomes sand.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE = os.path.dirname(os.path.abspath(__file__))
OV = 4
CREAM = np.array([239, 227, 205], np.float32)
PINK = np.array([202, 122, 146], np.float32)


def outline(n, half=0.300, tip_taper=0.74, notch=0.030, wave=0.012, bend=0.012):
    """Mask + lateral coordinate. v: 0 at the base, 1 at the tip."""
    m = Image.new("L", (n, n), 0)
    dr = ImageDraw.Draw(m)
    y_base, y_tip = 0.955 * n, 0.050 * n
    rows = int(n * 1.6)
    left, right = [], []
    for i in range(rows + 1):
        t = i / rows
        # oval: widest at 46% of the length, broad at the base, closing to a
        # blunt tip; a last taper gives the bract its small point
        w = np.sqrt(max(0.0, 1 - ((t - 0.50) / 0.58) ** 2))
        w *= 1 - tip_taper * np.clip((t - 0.78) / 0.22, 0, 1) ** 1.35
        w *= 1 + wave * np.sin(2 * np.pi * 2.1 * t + 0.7)
        hw = half * n * w
        y = y_base + (y_tip - y_base) * t
        cx = 0.5 * n + bend * n * np.sin(np.pi * t)
        left.append((cx - hw * 0.97, y))
        right.append((cx + hw * 1.03, y))
    dr.polygon(left + right[::-1], fill=255)
    a = np.asarray(m, np.float32) / 255

    # shallow fold at the base: the bract was attached to a stalk
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32)
    cx = 0.5 * n + bend * n * np.sin(np.pi * 0.995)
    dxn = np.abs(xx - cx) / (0.130 * n)
    cutoff = y_base - notch * n * (1 - np.clip(dxn, 0, 1) ** 0.8)
    a *= np.clip((cutoff - yy) / (0.010 * n) + 0.5, 0, 1)
    a = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                   .filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255

    u = np.zeros((n, n), np.float32)
    for y in range(n):
        row = np.nonzero(a[y] > 0.02)[0]
        if row.size < 2:
            continue
        x0, x1 = row.min(), row.max()
        u[y, x0:x1 + 1] = ((np.arange(x0, x1 + 1) - 0.5 * (x0 + x1))
                           / max(1.0, 0.5 * (x1 - x0)))
    return a, u


def surface(n, u, blush, seed, base_col=CREAM):
    yy = np.arange(n)[:, None] / n
    rng = np.random.default_rng(seed)
    # coarse mottle only: broad tonal unevenness, no grain
    mot = rng.normal(0, 1, (n // 3 + 2, n // 3 + 2)).astype(np.float32)
    mot = np.asarray(Image.fromarray(((mot * 0.4 + 0.5) * 255).clip(0, 255).astype(np.uint8))
                     .resize((n, n), Image.BICUBIC), np.float32) / 255.0 - 0.5

    val = np.ones_like(u)
    val *= 1 - 0.165 * u                                # the fold
    val *= 1 - 0.050 * np.exp(-(u / 0.10) ** 2)         # crease
    val *= 0.90 + 0.14 * np.clip((0.955 - yy) / 0.90, 0, 1) ** 1.3   # base in shade
    val *= 1 + 0.055 * mot
    val *= 1 - 0.15 * np.clip((np.abs(u) - 0.78) / 0.22, 0, 1) ** 1.5

    col = np.minimum(base_col[None, None, :] * val[..., None], 249.0)

    # pigment: flushes from the tip in patches, never a flat fill
    tt = np.clip((0.955 - yy) / 0.900, 0, 1)
    w = np.clip((tt - 0.42) / 0.58, 0, 1) ** 1.6 * blush
    w = w * np.ones((1, n), np.float32)
    w *= 0.74 + 0.50 * np.clip(mot + 0.5, 0, 1)
    w = np.clip(w, 0, 1)[..., None]
    return np.clip(col * (1 - w) + PINK[None, None, :] * val[..., None] * w, 0, 255)


def build(size=96, blush=(0.05, 0.55, 0.9), angles=(-10, 4, 14)):
    n = 96 * OV
    out = []
    for k, b in enumerate(blush):
        a, u = outline(n)
        t = surface(n, u, b, seed=31 + 13 * k)
        # the rim is where the paper turns under: a thin dark line, then a
        # touch of transmission. Bleaching it white -- the usual "soft edge"
        # trick -- makes the bract vanish on a pale wall, which is exactly
        # what a falling petal must not do.
        rim = np.clip((np.abs(u) - 0.86) / 0.14, 0, 1) ** 1.6
        t = t * (1 - 0.10 * rim[..., None]) + 246 * (0.10 * rim[..., None])
        al = np.clip(a * (1 - 0.08 * rim), 0, 1)
        im = Image.fromarray(np.dstack([t.astype(np.uint8),
                                       (al * 255).astype(np.uint8)]), "RGBA")
        im = im.resize((size, size), Image.LANCZOS)
        im = im.rotate(angles[k], resample=Image.BICUBIC)
        out.append(im)
    return out


if __name__ == "__main__":
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    outdir = os.path.join(BASE, "sp6")
    os.makedirs(outdir, exist_ok=True)
    ims = build(S)
    for i, im in enumerate(ims):
        im.save(os.path.join(outdir, f"bract_{i}.png"))
    U = 4
    sheet = Image.new("RGB", (S * U * 3 + 60, S * U + 210), (240, 236, 228))
    sheet.paste(Image.new("RGB", (S * U * 3 + 60, 80), (197, 195, 192)), (0, 0))
    sheet.paste(Image.new("RGB", (S * U * 3 + 60, 120), (62, 60, 57)), (0, S * U + 80))
    for k, im in enumerate(ims):
        big = im.resize((S * U, S * U), Image.LANCZOS)
        sheet.paste(big, (k * S * U + 10, 80), big)
        for j, px in enumerate((14, 18, 24, 32)):
            sm = im.resize((px, px), Image.LANCZOS)
            sheet.paste(sm, (k * S * U + 16 + j * 30, 28), sm)
            sheet.paste(sm, (k * S * U + 16 + j * 30, S * U + 130), sm)
    sheet.save(os.path.join(outdir, "_sheet.png"))
    print("wrote", os.path.join(outdir, "_sheet.png"))
