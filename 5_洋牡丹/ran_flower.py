"""Ranunculus head: petals placed on the phyllotactic spiral, not in rings.

The ring version read as folded paper: each layer was the same size, evenly
spaced, so the eye found a lattice instead of a flower. Petals on the golden
angle spiral have no rings at all -- neighbours are 137.5 deg apart and their
lengths differ only slightly, so they overlap like real petals do, and the
silhouette comes out irregular without any jitter being added.

    python ran_flower.py --size 1120 --sheet
"""
import argparse
import os

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))

# a white ranunculus is cream, blushes pink towards the tip and goes honey at
# the base; the pink is a blush, not a colour
CREAM = np.array([252, 247, 241], np.float32)
PINK = np.array([236, 201, 206], np.float32)
HONEY = np.array([240, 222, 190], np.float32)
CORE = np.array([241, 212, 196], np.float32)
SHADOW = np.array([205, 166, 172], np.float32)
LINE = np.array([166, 104, 106], np.float32)

N_PETAL = 48
GOLDEN = np.deg2rad(137.507)


def draw_petal(rgb, alpha, cx, cy, phi, length, width, rng, light, blush=1.0, lift=0.0):
    """One petal, base at (cx, cy), pointing at angle phi (0 = up)."""
    h, w = alpha.shape
    ext = int(length + width)
    x0, x1 = max(0, int(cx) - ext), min(w, int(cx) + ext + 1)
    y0, y1 = max(0, int(cy) - ext), min(h, int(cy) + ext + 1)
    if x1 <= x0 or y1 <= y0:
        return

    s_a, c_a = np.sin(phi), np.cos(phi)
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    along = dx * s_a - dy * c_a          # 0 at the base, +length at the tip
    lat = dx * c_a + dy * s_a

    t = along / length
    tc = np.clip(t, 0.0, 1.0)
    bend = rng.uniform(0.10, 0.26) * length * rng.choice([-1.0, 1.0])

    # pinched base, broad belly, blunt tip: a ranunculus petal is a wide spoon
    prof = (np.maximum(tc, 1e-4) ** 0.44) * ((1.0 - tc) ** 0.30 + 1e-4)
    peak = (0.44 / 0.74) ** 0.44 * (0.30 / 0.74) ** 0.30
    hw = 0.5 * width * (prof / peak)
    hw *= 1.0 + 0.026 * np.sin(2 * np.pi * 2.1 * tc + rng.uniform(0, 6))
    centre = bend * (tc ** 1.7)
    u = (lat - centre) / np.maximum(hw, 1e-3)

    t_edge = 1.0 - 0.018 * (1.0 - np.clip(np.abs(u), 0, 1) ** 1.6)
    aa = 2.4
    m = np.clip((hw - np.abs(lat - centre)) / aa + 0.5, 0, 1)
    m *= np.clip((t_edge - t) * length / aa + 0.5, 0, 1)
    m *= np.clip(tc * length / aa + 0.5, 0, 1)
    m *= np.clip(t * length / aa + 0.5, 0, 1)
    if m.max() <= 0.01:
        return

    # --- colour ---------------------------------------------------------
    # A light wash, not rendered plastic: honey at the base fading to cream,
    # a blush of pink towards the tip, and nothing else. The form is carried by
    # the outline, which is how a 工笔 flower is built.
    w_pink = (np.clip((tc - 0.25) / 0.75, 0, 1) ** 1.2
              * rng.uniform(0.55, 1.00) * blush)
    body = HONEY + (CREAM - HONEY) * np.clip(tc / 0.60, 0, 1)[..., None]
    body = body + (PINK - body) * (w_pink * 0.62)[..., None]
    # 分染: the petal holds a little more pigment low down and along its rim
    body *= (1.0 - 0.055 * np.clip((0.50 - tc) / 0.50, 0, 1))[..., None]
    body *= (1.0 - 0.030 * np.clip((np.abs(u) - 0.55) / 0.45, 0, 1) ** 1.5)[..., None]
    body *= (0.985 + 0.040 * light + 0.030 * lift)[..., None]
    occ = (0.16 * np.clip(u, 0, 1) ** 1.2
           + 0.13 * np.clip((0.38 - tc) / 0.38, 0, 1) ** 1.3
           + 0.07 * lift)
    body = body + (SHADOW - body) * np.clip(occ, 0, 0.55)[..., None]

    # --- the outline ------------------------------------------------------
    # one hairline, drawn at its own width: heavy enough to read at 40 px,
    # light enough to disappear when the petals are seen from across the room
    d_lat = hw - np.abs(lat - centre)
    d_top = (t_edge - t) * length
    edge = np.minimum(d_lat, d_top)
    line = np.exp(-((edge - 0.85) / 1.45) ** 2) * m
    line *= 0.86 - 0.40 * lift
    col = body * (1.0 - line[..., None]) + LINE * line[..., None]
    col = np.minimum(col, 253.0)

    a3 = m[..., None]
    sub_rgb = rgb[y0:y1, x0:x1]
    sub_a = alpha[y0:y1, x0:x1]
    sub_rgb *= (1.0 - a3)
    sub_rgb += col * a3
    sub_a *= (1.0 - m)
    sub_a += m


def build_head(size=1120, seed=5, radius=0.335, cy=0.427, n_petal=N_PETAL):
    rng = np.random.default_rng(seed)
    rgb = np.zeros((size, size, 3), np.float32)
    alpha = np.zeros((size, size), np.float32)
    cx, cyy = 0.5 * size, cy * size
    R = radius * size

    # a soft core disc under everything, so the gaps between petal bases show
    # the inside of the flower and never the paper
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cyy) ** 2) / (R * 0.30)
    core_a = np.clip(1.30 - rr, 0, 1) ** 0.9 * 0.98
    rgb += CORE[None, None, :] * core_a[..., None]
    alpha += core_a

    for k in range(n_petal):
        # 0 at the outside, 1 at the centre
        f = k / max(1, n_petal - 1)
        phi = k * GOLDEN + np.deg2rad(rng.uniform(-5, 5))
        length = R * (1.0 - f) ** 1.05 * rng.uniform(0.97, 1.03)
        # inner petals are relatively wider, so the centre packs solid
        width = length * (0.88 + 0.16 * f) * rng.uniform(0.94, 1.06)
        # a shallow cup: the further in, the higher the base
        oy = cyy - f * R * 0.075
        ox = cx + f * R * 0.012
        draw_petal(rgb, alpha, ox, oy, phi, length, width, rng,
                   light=np.cos(phi + np.deg2rad(38)) * 0.9, blush=0.62 + 0.38 * f,
                   lift=min(1.0, f * 1.7))

    return Image.fromarray(np.dstack([np.clip(rgb, 0, 255),
                                      np.clip(alpha * 255, 0, 255)]).astype(np.uint8), "RGBA")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1120)
    ap.add_argument("--out", default="sp/head.png")
    ap.add_argument("--radius", type=float, default=0.335)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--sheet", action="store_true")
    a = ap.parse_args()

    os.makedirs(os.path.join(HERE, os.path.dirname(a.out) or "."), exist_ok=True)
    img = build_head(a.size, a.seed, a.radius)
    img.save(os.path.join(HERE, a.out))
    print("wrote", a.out, img.size)

    if a.sheet:
        S = 560
        prev = img.resize((S, S), Image.LANCZOS)
        paper = Image.new("RGB", (S * 2 + 30, S), (250, 246, 239))
        paper.paste(prev, (0, 0), prev)
        dark = Image.new("RGB", (S, S), (86, 74, 68))
        dark.paste(prev, (0, 0), prev)
        paper.paste(dark, (S + 30, 0))
        paper.save(os.path.join(HERE, os.path.dirname(a.out) or "sp", "_head_prev.png"))
        print("wrote sp/_head_prev.png")


if __name__ == "__main__":
    main()
