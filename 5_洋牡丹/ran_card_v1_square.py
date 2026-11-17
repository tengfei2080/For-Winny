"""Ranunculus postcard: a square, full-bleed sheet -- no photograph frame.

The three earlier cards all framed a photograph on paper. This one is the paper:
the bloom sits high and centred, the type is small, evenly tracked and centred
under it, and nothing has a border. What moves is light and breath -- the head
swells by one percent and a warm wash drifts across the sheet -- so there is no
falling petal, no displacement field, nothing that can read as fake.

    python ran_card.py --still 4.2     # one frame
    python ran_card.py                 # GIF
    python ran_card.py --print card.png --scale 2
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\wang_\.workbuddy\skills\handwriting-postcard\scripts")
import postcard as P  # noqa: E402

SRC = os.path.join(HERE, "gen", "head_clean.png")

# ------------------------------------------------------------------ design --
W = H = 560
LOOP = 8.8
FPS = 12.5
GIF_COLORS = 256
DITHER = 1.1

PAPER = (247, 241, 231)
INK = (58, 46, 40)
INK_SOFT = (126, 108, 92)
ACCENT = (150, 38, 44)
PLATE = (166, 40, 34)

F_TITLE = r"C:\Windows\Fonts\STSONG.TTF"
F_SMALL = r"C:\Windows\Fonts\STKAITI.TTF"
F_LATIN = r"C:\Windows\Fonts\georgia.ttf"

# the bloom: centred on x, its top edge 34 px below the sheet edge
FLOWER_CY = 236
FLOWER_D = 380
CROP_R = 480          # radius taken from the source, in source pixels
FEATHER = 7.0         # px, in destination scale

TITLE = dict(text="洋牡丹", size=30, tracking=10, y=480, weight=0.4,
             t0=0.55, step=0.55, dur=0.55, colour=INK, font=F_TITLE)
RULE = dict(x0=W / 2 - 26, x1=W / 2 + 26, y=505,
            t0=1.65, dur=0.60, colour=(190, 172, 150))
LANG = dict(text="花语  迷人的魅力", size=11, tracking=3.4, y=527,
            t0=2.05, step=0.20, dur=0.30, colour=(120, 96, 78), font=F_SMALL)
LATIN = dict(text="RANUNCULUS ASIATICUS   ·   NOV 2026", size=7.4, tracking=2.6,
             y=549, t0=2.75, step=0.10, dur=0.26, colour=(136, 118, 100), font=F_LATIN)
SEAL = dict(image=os.path.join(HERE, "..", "3_印章", "seal_baiwen_tall.png"),
            h=28, x=508, y=531, ink=PLATE, t0=3.55, dur=0.40)
SEAL["image"] = os.path.normpath(SEAL["image"])

FADE_IN = 0.60
FADE_OUT = 7.70


# ------------------------------------------------------------------- pieces --
def font(path, px):
    return ImageFont.truetype(path, max(1, int(round(px))))


def grade(a):
    """Lift the plate to a paper tone, holding the highlights with a soft knee.

    The source arrives dark: its paper sits near 170, which on a card reads as
    dirty rather than quiet. A plain gain would blow every petal tip, so the top
    of the range is compressed instead of clipped.
    """
    x = np.clip(a, 0, 255) / 255.0 * 1.235
    knee = 0.80
    y = np.where(x < knee, x, knee + (1.0 - knee) * (1.0 - np.exp(-(x - knee) / 0.20)))
    y = np.clip(y, 0, 1) * 255.0
    y[..., 0] *= 1.010                       # a touch of warmth
    y[..., 2] *= 0.990
    # whatever is left of the leaf: pull the green out, it is the one cold note
    # in an otherwise warm frame
    green = np.clip((y[..., 1] - np.maximum(y[..., 0], y[..., 2])) / 8.0, 0, 1)
    y[..., 0] += 6.0 * green
    y[..., 1] -= 5.0 * green
    # warm the shadows: the shaded rim of the bloom was reading cold next to a
    # warm sheet, and a cold grey rim is all it takes for a cut-out to look cut
    w = np.clip(1.0 - y.mean(2) / 255.0, 0, 1) ** 1.35
    y[..., 0] += 12.0 * w
    y[..., 1] += 3.5 * w
    y[..., 2] -= 7.5 * w
    # the blush is barely visible once the plate is lifted; a little
    # saturation brings it back without touching the paper
    y = 128.0 + (y - 128.0) * 1.06
    return np.clip(y, 0, 252)


def load_flower(d):
    """The bloom on its own, cut along the outline of its petals.

    A disc crop left a visible plate: the disc's own darker background ringed
    the flower and the leaf behind it ended in a straight cut. The mask is a
    GrabCut run on the plate (see gen/fgmask.npy) -- the background is flat
    enough that it separates cleanly, and the cut then follows the petals.
    """
    im = Image.open(SRC).convert("RGB")
    a = grade(np.asarray(im, np.float32))
    m = np.load(os.path.join(HERE, "gen", "fgmask.npy")).astype(np.float32) / 255.0
    mi = Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.MinFilter(5))
    m = np.asarray(mi.filter(ImageFilter.GaussianBlur(1.2)), np.float32) / 255.0
    # The mask cuts through the shaded back of the outer petals, and that dark
    # rim read as a saw edge on a pale sheet. A thin rim light -- the last thing
    # daylight does to a thin petal -- blends it into the paper.
    core = np.asarray(mi.filter(ImageFilter.MinFilter(5)), np.float32) / 255.0
    rim = np.clip((m - core) / 0.30, 0, 1)[..., None]

    ys, xs = np.nonzero(m > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    half = int(max(y1 - y0, x1 - x0) * 0.5 * 1.015)
    box = (cx - half, cy - half, cx + half, cy + half)

    a = a + (252.0 - a) * 0.13 * rim
    rgba = np.dstack([np.clip(a, 0, 255), m * 255]).astype(np.uint8)
    tile = Image.fromarray(rgba, "RGBA").crop(box).resize((d, d), Image.LANCZOS)
    rgb = tile.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1.6, percent=52, threshold=2))
    out = rgb.convert("RGBA")
    out.putalpha(tile.split()[3])
    return out


def edge_colour(src=SRC):
    """The paper tone the bloom was shot on -- taken from just outside it."""
    im = np.asarray(Image.open(src).convert("RGB"), np.float32)
    n = im.shape[0]
    cx = cy = n / 2
    vals = []
    for k in range(96):
        th = 2 * np.pi * k / 96
        for rr in (CROP_R - 12, CROP_R - 6, CROP_R - 1):
            x, y = int(cx + rr * np.cos(th)), int(cy + rr * np.sin(th))
            if 0 <= x < n and 0 <= y < n:
                vals.append(im[max(0, y - 3):y + 4, max(0, x - 3):x + 4].reshape(-1, 3).mean(0))
    return np.median(np.array(vals), axis=0)


def make_background(size=W, s=1.0):
    """Paper: a warm wash that falls away from the bloom, plus a faint floor
    shadow under it so the flower sits on the sheet instead of floating."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    cx, cy = size * 0.5, FLOWER_CY * s
    r = np.sqrt(((xx - cx) / size) ** 2 + ((yy - cy) / size) ** 2)
    fall = np.clip((r - 0.26) / 0.62, 0, 1) ** 1.5
    base = np.array([246, 240, 230], np.float32)
    rgb = base[None, None, :] * (1.0 - 0.048 * fall[..., None])
    rgb -= np.array([2.0, 2.5, 3.0])[None, None, :] * (fall[..., None] ** 2)
    return rgb.astype(np.float32)


def paper_grain(size, seed=4):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, (size // 2, size // 2)).astype(np.float32)
    g = np.asarray(Image.fromarray(((g * 0.4 + 0.5) * 255).clip(0, 255).astype(np.uint8))
                   .resize((size, size), Image.BICUBIC), np.float32) / 255.0 - 0.5
    f = rng.normal(0, 1, (size, size)).astype(np.float32)
    return g * 0.010 + f * 0.004


# ------------------------------------------------------------------ drawing --
def tracked(layer, spec, t, g, scale=1.0):
    r = (t - spec["t0"]) / spec["dur"]
    if r <= 0 or g <= 0.002:
        return
    r = min(1.0, r)
    f = font(spec["font"], spec["size"] * scale)
    dr = ImageDraw.Draw(layer)
    step = 0.80 / max(1, len(spec["text"]))
    widths = [dr.textlength(ch, font=f) for ch in spec["text"]]
    total = sum(widths) + spec["tracking"] * scale * max(0, len(spec["text"]) - 1)
    x = W * scale / 2 - total / 2
    y = spec["y"] * scale
    for i, ch in enumerate(spec["text"]):
        c = int(255 * P.smooth(min(1.0, max(0.0, (r - i * step) * 6.0))) * g)
        if c > 2:
            dr.text((x, y), ch, font=f, fill=tuple(spec["colour"]) + (c,), anchor="ls",
                    stroke_width=spec.get("weight", 0), stroke_fill=tuple(spec["colour"]) + (c,))
        x += widths[i] + spec["tracking"] * scale


def draw_rule(layer, t, g, scale=1.0):
    r = (t - RULE["t0"]) / RULE["dur"]
    if r <= 0 or g <= 0.002:
        return
    p = P.smooth(min(1.0, r))
    cx = W * scale / 2
    half = (RULE["x1"] - RULE["x0"]) / 2 * scale * p
    ImageDraw.Draw(layer).line(
        [(cx - half, RULE["y"] * scale), (cx + half, RULE["y"] * scale)],
        fill=tuple(RULE["colour"]) + (int(190 * g),), width=max(1, int(scale)))


def draw_seal(layer, t, g, scale=1.0):
    r = (t - SEAL["t0"]) / SEAL["dur"]
    if r <= 0 or g <= 0.002:
        return
    p = P.smooth(min(1.0, r))
    alpha = g * min(1.0, r * 8.0)
    k = 1.0 + 0.24 * (1.0 - p)
    art = Image.open(SEAL["image"]).convert("RGBA")
    h = SEAL["h"] * scale
    w = h * art.width / art.height
    tile = art.resize((max(1, int(w * k)), max(1, int(h * k))), Image.LANCZOS)
    ink = np.array(SEAL["ink"], np.float32)
    a = np.asarray(tile.split()[3], np.float32)[..., None] / 255.0
    rgb = np.asarray(tile.convert("RGB"), np.float32)
    # keep the paper-white carving of the seal, recolour only the ink
    rgb = np.where(rgb[..., :1] > 210, rgb, ink[None, None, :] * (0.55 + 0.45 * rgb / 255.0))
    tile = Image.fromarray(np.dstack([np.clip(rgb, 0, 255),
                                      a[..., 0] * 255 * alpha]).astype(np.uint8), "RGBA")
    cx = SEAL["x"] * scale + (SEAL["h"] * scale * art.width / art.height) / 2
    cy = SEAL["y"] * scale + SEAL["h"] * scale / 2
    layer.alpha_composite(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)))


# --------------------------------------------------------------- composition --
def light_field(t, size, scale=1.0):
    ph = 2 * np.pi * t / LOOP
    gx = 0.40 + 0.035 * np.cos(ph)
    gy = 0.24 + 0.022 * np.sin(ph)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    d = ((xx / size - gx) ** 2 + ((yy / size - gy) * 1.1) ** 2) / 0.16
    wash = 0.026 * np.exp(-d)[..., None]
    breath = 1.0 + 0.0055 * np.sin(ph)          # the whole sheet breathes
    return (1.0 + wash) * breath


def compose(bg, flower, t, scale=1.0):
    cw = int(round(W * scale))
    ph = 2 * np.pi * t / LOOP
    # one percent of swell, centred on the bloom: read as breath, not as zoom
    k = 1.0 + 0.010 * (0.5 - 0.5 * np.cos(ph))
    d = int(round(FLOWER_D * scale * k))
    spr = flower.resize((d, d), Image.LANCZOS)

    bgr = Image.fromarray(np.clip(bg, 0, 255).astype(np.uint8))
    if bgr.size != (cw, cw):
        bgr = bgr.resize((cw, cw), Image.LANCZOS)
    canvas = Image.new("RGB", (cw, cw))
    canvas.paste(bgr, (0, 0))
    cx = cw / 2
    cy = FLOWER_CY * scale + (FLOWER_D * scale - d) / 2
    canvas.paste(spr, (int(cx - d / 2), int(cy - d / 2)), spr)
    canvas = canvas.convert("RGBA")

    lay = Image.new("RGBA", (cw, cw), (0, 0, 0, 0))
    g = 1.0
    if t < FADE_IN:
        g = P.smooth(t / FADE_IN)
    elif t > FADE_OUT:
        g = P.smooth(max(0.0, 1.0 - (t - FADE_OUT) / (LOOP - 1.0 / FPS - FADE_OUT)))
    if g > 0.002:
        tracked(lay, TITLE, t, g, scale)
        draw_rule(lay, t, g, scale)
        tracked(lay, LANG, t, g, scale)
        tracked(lay, LATIN, t, g, scale)
        draw_seal(lay, t, g, scale)
    canvas.alpha_composite(lay)

    a = np.asarray(canvas.convert("RGB"), np.float32) * light_field(t, cw)
    a = np.clip(a, 0, 255)
    # the fade lands on the paper itself, so a loop restart is invisible
    if g < 0.999:
        paper = np.clip(bg, 0, 255)
        a = paper + (a - paper) * g
    return Image.fromarray(a.astype(np.uint8))


def encode(bg, flower, out_path, scale=1.0, fps=FPS):
    n = int(round(LOOP * fps))
    ts = [i * LOOP / n for i in range(n)]
    frames = [compose(bg, flower, t) for t in ts]

    probe = Image.new("RGB", (W * 4, H))
    for k, say in enumerate((0.15, 0.42, 0.70, 0.95)):
        probe.paste(frames[min(n - 1, int(round(say * n)))], (k * W, 0))
    pal = probe.quantize(colors=GIF_COLORS - 2, method=Image.MEDIANCUT)
    pl = list(pal.getpalette()[: (GIF_COLORS - 2) * 3])
    for c in (ACCENT, PLATE):          # the seal is 30 px: it needs its own slots
        pl += [int(v) for v in c]
    pl += [0] * (768 - len(pl))
    pal.putpalette(pl)

    q = [P.orderly(f, pal, DITHER) for f in frames]
    q[0].save(out_path, save_all=True, append_images=q[1:],
              duration=int(round(1000 / fps)), loop=0, optimize=True, disposal=1)
    print("wrote", out_path, f"{n} frames, {os.path.getsize(out_path)/1e6:.2f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float)
    ap.add_argument("--print", dest="pr")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--frames", default=None, help="directory for a video sequence")
    a = ap.parse_args()

    bg = make_background(W)
    bg = bg * (1.0 + paper_grain(W)[..., None])
    flower = load_flower(FLOWER_D)

    if a.still is not None:
        im = compose(bg, flower, a.still)
        im.save(os.path.join(HERE, f"_card_{a.still:.2f}.png"))
        print("still ok", im.size)
        return
    if a.frames:
        scl = a.scale
        fbg = make_background(int(round(W * scl)), scl)
        fbg = fbg * (1.0 + paper_grain(int(round(W * scl)))[..., None])
        ff = load_flower(int(round(FLOWER_D * scl)))
        os.makedirs(a.frames, exist_ok=True)
        n = int(round(LOOP * a.fps))
        for i in range(n):
            compose(fbg, ff, i * LOOP / n, scl).save(
                os.path.join(a.frames, f"f_{i:03d}.png"))
        print("wrote", n, "frames to", a.frames)
        return
    if a.pr:
        s = a.scale
        big_bg = make_background(int(round(W * s)), s)
        big_bg = big_bg * (1.0 + paper_grain(int(round(W * s)))[..., None])
        big_flower = load_flower(int(round(FLOWER_D * s)))
        im = compose(big_bg, big_flower, 4.2, s)
        im.save(os.path.join(HERE, a.pr))
        print("print ok", im.size)
        return

    if a.frames:
        scl = a.scale
        fbg = make_background(int(round(W * scl)), scl)
        fbg = fbg * (1.0 + paper_grain(int(round(W * scl)))[..., None])
        ff = load_flower(int(round(FLOWER_D * scl)))
        os.makedirs(a.frames, exist_ok=True)
        n = int(round(LOOP * a.fps))
        for i in range(n):
            compose(fbg, ff, i * LOOP / n, scl).save(
                os.path.join(a.frames, f"f_{i:03d}.png"))
        print("wrote", n, "frames to", a.frames)
        return

    encode(bg, flower, os.path.join(HERE, "ranunculus_postcard.gif"), 1.0, a.fps)


if __name__ == "__main__":
    main()
