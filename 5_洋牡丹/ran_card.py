"""Ranunculus postcard -- a horizontal A6 sheet, full bleed.

The earlier cards framed a photograph on paper. This one is the paper itself:
the bouquet runs off all four edges, the type sits in the quiet band the
lighting left at the top of the frame, and there is no border anywhere.

The bouquet is not cut out. Cutting it produced a pale ring of plate around
every leaf -- the plate and the white petals sit within a few levels of each
other, so any mask has to run through the fuzzy transition and leaves a shell.
Full bleed sidesteps that entirely: the plate is the card.

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

SRC = os.path.join(HERE, "gen", "bouquet_clean.png")

# ------------------------------------------------------------------ design --
# A6, landscape: 148 x 105 mm. The art runs to all four edges.
W, H = 740, 525
RATIO = W / H

LOOP = 8.8
FPS = 12.5
GIF_COLORS = 256
DITHER = 1.0

INK = (56, 44, 36)
INK_SOFT = (104, 86, 70)
ACCENT = (150, 38, 44)
PLATE = (152, 46, 40)

F_TITLE = r"C:\Windows\Fonts\STSONG.TTF"
F_SMALL = r"C:\Windows\Fonts\STKAITI.TTF"
F_LATIN = r"C:\Windows\Fonts\georgia.ttf"

# The frame is cropped up from the plate, then the breathing is done by
# cropping further in -- so nothing ever runs out of picture.
ZOOM = 1.20

TITLE = dict(text="洋牡丹", size=31, tracking=19, y=74, weight=0.4,
             t0=0.55, step=0.55, dur=0.60, colour=INK, font=F_TITLE)
RULE = dict(x0=W / 2 - 33, x1=W / 2 + 33, y=91,
            t0=1.70, dur=0.60, colour=(176, 158, 136))
LANG = dict(text="花语  迷人的魅力", size=12, tracking=4.6, y=114,
            t0=2.05, step=0.20, dur=0.30, colour=(84, 68, 56), font=F_SMALL)
SEAL = dict(image=os.path.join(HERE, "..", "3_印章", "seal_baiwen_tall.png"),
            h=33, x=690, y=33, ink=PLATE, t0=3.60, dur=0.42)
SEAL["image"] = os.path.normpath(SEAL["image"])

FADE_IN = 0.50
FADE_OUT = 7.60


# ------------------------------------------------------------------- pieces --
def font(path, px):
    return ImageFont.truetype(path, max(1, int(round(px))))


def flatten_light(a, p=1.0):
    """Even out the lighting across the plate.

    The bouquet was shot with the key coming from the right: the left of the
    sheet is a full stop darker and a good deal yellower, which on a card reads
    as a dirty corner rather than as light. A quadratic surface is fitted to the
    sheet's own margins, so it describes the lighting and not the flowers, and
    the picture is divided by it -- at 80 %, since a perfectly flat sheet looks
    like a scan rather than a photograph.
    """
    h, w = a.shape[:2]
    yn, xn = (np.arange(h, dtype=np.float32) / h) * 2 - 1, (np.arange(w, dtype=np.float32) / w) * 2 - 1
    X, Y = np.meshgrid(xn, yn)

    def terms(yy, xx):
        # cubic: the plate is graded from one corner, and a quadratic leaves a
        # broad residue that reads as a stain once the sheet is lifted
        cols = []
        for d in range(4):
            for i in range(d + 1):
                cols.append((xx ** (d - i)) * (yy ** i))
        return np.stack(cols, -1)

    plate = np.zeros((h, w), bool)
    plate[: int(0.17 * h), :] = True
    plate[:, : int(0.03 * w)] = True
    plate[:, int(0.97 * w):] = True
    ys, xs = np.nonzero(plate)
    A = terms(Y[ys, xs], X[ys, xs])
    coef = np.linalg.lstsq(A, a[ys, xs], rcond=None)[0]
    field = (terms(Y.ravel(), X.ravel()) @ coef).reshape(h, w, 3)
    target = np.median(a[ys, xs], axis=0)
    # a cubic extrapolated to the very edges can dive well below the sheet it
    # was fitted to, and dividing by it there leaves a dark stain down the
    # margin -- hold the surface within reach of the paper it describes
    field = np.clip(field, target * 0.90, target * 1.14)
    return a * (target[None, None, :] / np.maximum(field, 1.0)) ** p


def grade(a):
    """Bring the plate up to paper, without touching the foliage.

    The source was shot on a mid-grey card: its paper reads near 180, which on
    a postcard looks like dirt. A straight gain would blow the cream petals, so
    the top of the range is compressed instead of clipped. The greens stay as
    they are -- the leaves are half the subject here.
    """
    x = np.clip(a, 0, 255) / 255.0 * 1.135
    knee = 0.845
    y = np.where(x < knee, x, knee + (1.0 - knee) * (1.0 - np.exp(-(x - knee) / 0.18)))
    y = np.clip(y, 0, 1) * 255.0
    y[..., 0] *= 1.012                        # a touch of warmth
    y[..., 2] *= 1.002
    y = 128.0 + (y - 128.0) * 1.045           # the blush needed a little back
    return np.clip(y, 0, 253)


def leaf_mask(a):
    """Where the greenery is, as a soft 0..1 field.

    Green, not too bright, not too pale: the blooms are cream and blush and the
    plate is warm beige, so a hue/saturation gate separates them cleanly. The
    mask is deliberately soft -- it is used to blend texture in, and a hard edge
    would show as a cut-out around every leaf.
    """
    hsv = np.asarray(Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).convert("HSV"))
    hue = hsv[..., 0].astype(np.float32) * 360.0 / 255.0
    sat = hsv[..., 1].astype(np.float32) / 255.0
    val = hsv[..., 2].astype(np.float32) / 255.0
    # hue alone cannot tell a pale leaf from a cream petal -- but green always
    # carries more green than red, and cream and blush never do
    greenish = a[..., 1] > a[..., 0] * 1.05
    leaf = (hue > 40) & (hue < 185) & (sat > 0.13) & (val < 0.93) & greenish
    m = Image.fromarray((leaf * 255).astype(np.uint8))
    m = m.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.MinFilter(3))
    return np.asarray(m.filter(ImageFilter.GaussianBlur(2.2)), np.float32) / 255.0


def leaf_finish(a):
    """Put back the imperfections the generator left out.

    Generated foliage is airbrushed: one flat green, no surface relief, and an
    edge sharp to the last pixel. On a shallow-focus photograph that reads as
    plastic. Three small things fix most of it -- a slow swell of light and
    shade across each leaf, a breath of grain, and an edge that sits a shade
    out of focus like the rest of the image.
    """
    h, w = a.shape[:2]
    m = leaf_mask(a)[..., None]
    rng = np.random.default_rng(11)

    # broad undulation: what a leaf does when it is not a flat card
    small = rng.normal(0, 1, (max(2, h // 26), max(2, w // 26))).astype(np.float32)
    swell = np.asarray(
        Image.fromarray(((small * 0.22 + 0.5) * 255).clip(0, 255).astype(np.uint8))
        .resize((w, h), Image.BICUBIC), np.float32) / 255.0 - 0.5
    # fine grain, barely there
    grain = rng.normal(0, 1, (h, w)).astype(np.float32)

    # structure runs lengthwise: sampled coarse across the blade and fine along
    # it, the field becomes a soft brush of veins -- which is what an eye looks
    # for and never finds on a generated leaf
    veins = np.asarray(
        Image.fromarray(((rng.normal(0, 1, (h, max(2, w // 72))) * 0.24 + 0.5) * 255)
                        .clip(0, 255).astype(np.uint8))
        .resize((w, h), Image.BICUBIC), np.float32) / 255.0 - 0.5

    tex = 1.0 + swell * 0.115 + grain * 0.016 + veins * 0.070
    out = a * (1.0 + (tex[..., None] - 1.0) * m)

    # a little less saturated, and a touch lighter in the highlights -- digitial
    # green is the tell
    grey = out.mean(axis=2, keepdims=True)
    out = out * (1.0 - 0.020 * m) + grey * (0.020 * m)

    # edge that is one step softer than the pixel grid
    soft = np.asarray(
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(1.5)), np.float32)
    inner = np.asarray(
        Image.fromarray((m[..., 0] * 255).astype(np.uint8))
        .filter(ImageFilter.MinFilter(5)), np.float32) / 255.0
    rim = (m * (1.0 - inner[..., None]))
    out = out * (1.0 - 0.55 * rim) + soft * (0.55 * rim)
    return np.clip(out, 0, 255)


def leaf_depth(a):
    """Sink the foliage so the flowers carry the card.

    Square-on, the leaves fan out in a near-regular arc -- every blade whole,
    every blade the same size. That regularity is what reads as a prop. In a
    bouquet photograph the foliage is the dark that the blooms are lit against,
    so the fix is not more detail but less attention: take the shaded leaves
    down and warm, leave the single backlit leaf on the right alone, and let the
    range run from near-black green to one glowing tip.
    """
    m = leaf_mask(a)[..., None]
    lum = a.mean(axis=2, keepdims=True)
    deep = np.clip((132.0 - lum) / 132.0, 0, 1) ** 0.85        # 1 in shadow, 0 in the glow
    gain = 1.0 - 0.13 * deep * m
    out = a * gain
    out[..., 0:1] *= 1.0 + 0.030 * deep * m                    # less blue in the shade,
    out[..., 2:3] *= 1.0 - 0.055 * deep * m                    # so the green sits warm

    # the out-of-focus leaf across the bottom is the one thing that reads as a
    # thumb in front of the lens -- it only needs to be a dark edge
    h = a.shape[0]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    out *= 1.0 - 0.10 * np.clip((yy - 0.80) / 0.20, 0, 1) ** 1.4
    return np.clip(out, 0, 255)


def despeckle(a):
    """Take the pale blotches off the leaves.

    The generator read "dewy" as a scatter of milky patches, and on a leaf that
    reads as spray residue or mildew. A real leaf is never paler *and* less
    saturated than its own surface, so both tests are used, and the patch is
    refilled from the median of the leaf around it. Veins survive because they
    are lighter but still green.
    """
    src = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    med = np.asarray(src.filter(ImageFilter.MedianFilter(11)), np.float32)
    hsv = np.asarray(src.convert("HSV"), np.float32)
    sat = hsv[..., 1]
    m = leaf_mask(a)
    pale = (a.mean(axis=2) - med.mean(axis=2)) > 9.0
    washed = sat < 52.0
    hit = ((pale | washed) & (m > 0.45))
    out = a.copy()
    out[hit] = med[hit]

    # a few blades came out the colour of skimmed milk -- leaf underside, or
    # just the model losing the plot. No leaf in a fresh bouquet is that pale,
    # so give the foliage a floor on saturation and let it read as one plant.
    src2 = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    arr = np.asarray(src2.convert("HSV")).astype(np.float32)
    arr[..., 1] = np.where(m > 0.40, np.maximum(arr[..., 1], 96.0), arr[..., 1])
    out = np.asarray(Image.fromarray(arr.astype(np.uint8), "HSV").convert("RGB"), np.float32)
    return out


def leaf_light(a):
    """Relight the foliage from the upper right, where the key already is.

    The bouquet was lit by one soft source high on the right -- the blooms say
    so, their shadows fall down and left onto the plate. The generated foliage
    does not agree: it is evenly lit, so it reads as a flat cut-out lying on the
    paper rather than as leaves standing in the same light as the flowers.
    Here the shade is a function of position. Where the light reaches, the leaf
    gains and warms toward yellow-green; where it does not, it goes down and
    cooler, and the leaf edges that face the source pick up a translucent rim.
    """
    h, w = a.shape[:2]
    m = leaf_mask(a)[..., None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :, None]
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]

    # distance from the key, in frame widths: 0 at the upper right, ~1 far left
    d = np.sqrt(((1.0 - xx) * 0.92) ** 2 + ((yy - 0.04) * 0.62) ** 2)
    shade = np.clip(d / 1.02, 0.0, 1.0) ** 0.92

    gain = 0.80 - 0.14 * shade                       # fresh but not neon
    out = a * (1.0 + (gain - 1.0) * m)

    deep = (shade ** 1.25) * m
    out[..., 0:1] *= 1.0 + 0.055 * deep              # warm the shade, it was going
    out[..., 2:3] *= 1.0 - 0.105 * deep              # steel-blue on a beige sheet

    # the stem end of a leaf sits in the flowers' own shadow and the tip reaches
    # out into the light; without that run from dark to light a leaf has no
    # thickness, whatever its outline does
    rad = np.sqrt(((xx - 0.54) * 0.90) ** 2 + ((yy - 0.50) * 0.72) ** 2)
    at_base = np.clip((0.62 - rad) / 0.62, 0.0, 1.0) ** 1.3 * m
    out *= 1.0 - 0.10 * at_base

    # the source green is acid-yellow; real foliage carries blue in it
    out[..., 0:1] *= 1.0 - 0.045 * m
    out[..., 2:3] *= 1.0 + 0.20 * m

    # the leaves nearest the key thin out against it: a lit membrane, not paint
    lit = np.clip((1.0 - shade * 1.9), 0.0, 1.0) * m
    out = out * (1.0 + 0.045 * lit) + np.array([2.0, 5.0, -2.5], np.float32) * lit

    # the shade had gone olive-grey; a leaf in shadow is still a saturated green
    grey = out.mean(axis=2, keepdims=True)
    out = grey + (out - grey) * (1.0 + 0.40 * m)
    out[..., 1:2] *= 1.0 + 0.025 * m

    # and the lowest band of foliage only needs to be a dark hem, not a thicket
    hem = np.clip((yy - 0.70) / 0.30, 0.0, 1.0) ** 1.25
    out *= 1.0 - 0.06 * hem * m

    # a fine leaf lit from behind is not a dark thing with a light edge: the
    # whole membrane goes thin and lets the sheet through. Without this the
    # foliage stays opaque and, however correct its shape, reads as dead.
    lum = out.mean(axis=2, keepdims=True) / 255.0
    trans = np.clip((lum - 0.34) / 0.45, 0.0, 1.0) ** 1.7 * m * 0.08
    paper = np.array([238.0, 229.0, 203.0], np.float32)
    out = out * (1.0 - trans) + paper * trans
    out[..., 0:1] *= 1.0 + 0.020 * trans                     # and warms as it thins
    out[..., 2:3] *= 1.0 - 0.050 * trans
    g2 = out.mean(axis=2, keepdims=True)
    out = g2 + (out - g2) * (1.0 + 0.24 * m)     # thinning must not bleach the green

    # a rim of light where a leaf edge turns into the key -- one pixel wide is
    # all it takes for the foliage to sit in the same light as the blooms
    inner = np.asarray(
        Image.fromarray((m[..., 0] * 255).astype(np.uint8))
        .filter(ImageFilter.MinFilter(5)), np.float32)[..., None] / 255.0
    edge = np.clip(m - inner, 0, 1) * np.clip(1.0 - shade * 0.85, 0, 1)
    out = out + np.array([2.6, 4.0, 0.4], np.float32) * edge

    # the foliage nearest the lens was shot at the same aperture as the rest,
    # so it cannot all be in focus: let the bottom third drift out of it. This
    # is also what stops a hundred fine leaflets from reading as a thicket.
    plane = np.clip((yy - 0.50) / 0.50, 0.0, 1.0) ** 1.15 * m
    blur = np.asarray(
        Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
        .filter(ImageFilter.GaussianBlur(3.2)), np.float32)
    # every leaf sits behind the flowers, so none of them can be as crisp as a
    # petal -- and the lower the leaf, the further out of the plane it is
    soft = np.clip(0.22 * m + 0.50 * plane, 0.0, 1.0)
    out = out * (1.0 - soft) + blur * soft

    return np.clip(out, 0, 255)


def load_plate(w, h, src=SRC):
    """The whole bouquet, cropped to A6 and lifted to paper."""
    im = Image.open(src).convert("RGB")
    a = grade(flatten_light(np.asarray(im, np.float32)))
    a = leaf_finish(leaf_light(despeckle(a)))

    # centre crop to the card's aspect, with a margin for the breathing zoom
    sw, sh = im.size
    tw, th = sw / ZOOM, sh / ZOOM
    if tw / th > RATIO:
        tw = th * RATIO
    else:
        th = tw / RATIO
    cx, cy = sw / 2, sh / 2
    box = (cx - tw / 2, cy - th / 2, cx + tw / 2, cy + th / 2)
    tile = Image.fromarray(a.astype(np.uint8)).crop(
        tuple(int(round(v)) for v in box)).resize((w, h), Image.LANCZOS)
    # the source is soft-focus by nature; a little local contrast keeps the
    # petals from reading as fog once the plate is lifted
    return tile.filter(ImageFilter.UnsharpMask(radius=1.2, percent=34, threshold=2))


def paper_grain(size, seed=4):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, 1, (size // 2, size // 2)).astype(np.float32)
    g = np.asarray(Image.fromarray(((g * 0.4 + 0.5) * 255).clip(0, 255).astype(np.uint8))
                   .resize((size, size), Image.BICUBIC), np.float32) / 255.0 - 0.5
    f = rng.normal(0, 1, (size, size)).astype(np.float32)
    return g * 0.008 + f * 0.003


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
        fill=tuple(RULE["colour"]) + (int(190 * g),), width=max(1, int(round(scale))))


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
    cx = SEAL["x"] * scale + w * scale / 2
    cy = SEAL["y"] * scale + SEAL["h"] * scale / 2
    layer.alpha_composite(tile, (int(cx - tile.width / 2), int(cy - tile.height / 2)))


# --------------------------------------------------------------- composition --
def light_field(t, w, h):
    """A slow warm wash crossing the frame, plus the faintest swell.

    Nothing translates: a bouquet laid on paper does not drift, and the one
    thing that made an earlier card uncomfortable to watch was motion the eye
    could follow. What moves here is only the light.
    """
    ph = 2 * np.pi * t / LOOP
    gx = 0.36 + 0.030 * np.cos(ph)
    gy = 0.28 + 0.020 * np.sin(ph)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    aspect = w / h
    d = ((xx / w - gx) * aspect) ** 2 + (yy / h - gy) ** 2
    wash = 0.022 * np.exp(-d / 0.14)[..., None]
    breath = 1.0 + 0.0035 * np.sin(ph)
    return (1.0 + wash) * breath


def compose(plate, t, scale=1.0):
    cw, ch = int(round(W * scale)), int(round(H * scale))
    canvas = plate if plate.size == (cw, ch) else plate.resize((cw, ch), Image.LANCZOS)

    ph = 2 * np.pi * t / LOOP
    k = ZOOM - 0.006 * (0.5 - 0.5 * np.cos(ph))          # < 0.6 % of breath
    bw, bh = cw / k, ch / k
    canvas = canvas.crop((int((cw - bw) / 2), int((ch - bh) / 2),
                          int((cw + bw) / 2), int((ch + bh) / 2)))
    canvas = canvas.resize((cw, ch), Image.LANCZOS).convert("RGBA")

    lay = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    g = 1.0
    if t < FADE_IN:
        g = P.smooth(t / FADE_IN)
    elif t > FADE_OUT:
        g = P.smooth(max(0.0, 1.0 - (t - FADE_OUT) / (LOOP - 1.0 / FPS - FADE_OUT)))
    if g > 0.002:
        tracked(lay, TITLE, t, g, scale)
        draw_rule(lay, t, g, scale)
        tracked(lay, LANG, t, g, scale)
        draw_seal(lay, t, g, scale)
    canvas.alpha_composite(lay)

    a = np.asarray(canvas.convert("RGB"), np.float32) * light_field(t, cw, ch)
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def encode(plate, out_path, scale=1.0, fps=FPS):
    n = int(round(LOOP * fps))
    ts = [i * LOOP / n for i in range(n)]
    frames = [compose(plate, t) for t in ts]

    probe = Image.new("RGB", (W * 4, H))
    for k, say in enumerate((0.15, 0.42, 0.70, 0.95)):
        probe.paste(frames[min(n - 1, int(round(say * n)))], (k * W, 0))
    pal = probe.quantize(colors=GIF_COLORS - 2, method=Image.MEDIANCUT)
    pl = list(pal.getpalette()[: (GIF_COLORS - 2) * 3])
    for c in (ACCENT, PLATE):          # the seal is 26 px: it needs its own slots
        pl += [int(v) for v in c]
    pl += [0] * (768 - len(pl))
    pal.putpalette(pl)

    q = [P.orderly(f, pal, DITHER) for f in frames]
    q[0].save(out_path, save_all=True, append_images=q[1:],
              duration=int(round(1000 / fps)), loop=0, optimize=True, disposal=1)
    print("wrote", out_path, f"{n} frames, {os.path.getsize(out_path)/1e6:.2f} MB")


# ---------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float, default=None, help="write one frame at t")
    ap.add_argument("--print", dest="pr", default=None, help="print resolution PNG")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=FPS)
    ap.add_argument("--frames", default=None, help="directory for a video sequence")
    ap.add_argument("--src", default=SRC, help="plate to use (default: the first one)")
    a = ap.parse_args()

    if a.still is not None:
        s = max(a.scale, 1.0)
        p = load_plate(int(round(W * s)), int(round(H * s)), a.src)
        compose(p, a.still, s).save(os.path.join(HERE, f"_card_{a.still:.2f}.png"))
        return

    if a.frames:
        scl = a.scale
        p = load_plate(int(round(W * scl)), int(round(H * scl)), a.src)
        os.makedirs(a.frames, exist_ok=True)
        n = int(round(LOOP * a.fps))
        for i in range(n):
            compose(p, i * LOOP / n, scl).save(
                os.path.join(a.frames, f"f_{i:03d}.png"))
        print("wrote", n, "frames to", a.frames)
        return

    if a.pr:
        p = load_plate(int(round(W * a.scale)), int(round(H * a.scale)), a.src)
        compose(p, 4.2, a.scale).save(os.path.join(HERE, a.pr))
        print("wrote", a.pr, p.size)
        return

    p = load_plate(W, H, a.src)
    encode(p, os.path.join(HERE, "ranunculus_postcard.gif"))


if __name__ == "__main__":
    main()
