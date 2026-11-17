"""Bougainvillea postcard: still photograph, drifting bracts, vertical dedication.

The wind loop was dropped -- a full-frame displacement reads as motion sickness
on a hand-held-looking photo. What moves now is only the bracts falling, plus the
writing. Everything else is grain and paper.

Layout is a hanging-scroll card rather than the horizontal signature of the two
earlier cards: the photo sits square on the sheet, the dedication runs down in
vertical columns at the lower right (title, then the flower's language, then the
seal), and a column of small latin type anchors the lower left.

    python tri_card.py --still 4.2      # one frame, for looking
    python tri_card.py                  # GIF
    python tri_card.py --print card.png --scale 2
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, r"C:\Users\wang_\.workbuddy\skills\handwriting-postcard\scripts")
import postcard as P  # noqa: E402  (smooth / draw_glyph / orderly, all verified)

# ------------------------------------------------------------------ layout --
W, H = 540, 880
M = 30
PW, PH = 480, 640            # the photo is 0.7506 wide; 480x640 is 0.7500
FOOT = H - M - PH            # 210

PAPER = (250, 246, 238)
INK = (54, 42, 36)
INK_SOFT = (122, 105, 88)
ACCENT = (150, 38, 44)
PLATE = (168, 40, 34)

LOOP = 8.0
FPS = 12.5
GIF_COLORS = 256
DITHER = 6.0

F_TITLE = r"C:\Windows\Fonts\STSONG.TTF"
F_SMALL = r"C:\Windows\Fonts\STKAITI.TTF"
F_LATIN = r"C:\Windows\Fonts\georgia.ttf"

TITLE = dict(text="三角梅", size=34, gap=10, x=462, dy=20,
             t0=0.55, step=0.42, dur=0.36, colour=tuple(INK), font=F_TITLE)
SMALL = dict(text="热情坚韧", size=16, gap=8, x=408, dy=22,
             t0=2.30, step=0.17, dur=0.22, colour=(126, 96, 78), font=F_SMALL)
# a pair of short columns between the date and the dedication: three vertical
# runs at three depths is what makes the band read as a scroll rather than as
# one ragged signature pushed to the right
POEM_A = dict(text="风过墙头", size=15, gap=9, x=306, dy=24,
              t0=3.05, step=0.20, dur=0.28, colour=(104, 92, 80), font=F_SMALL)
POEM_B = dict(text="花自安然", size=15, gap=9, x=278, dy=24,
              t0=3.80, step=0.20, dur=0.28, colour=(104, 92, 80), font=F_SMALL)
LATIN = dict(text="BOUGAINVILLEA  ·  NOV 2026", size=8, tracking=1.9,
             x=M + 4, dy=168, t0=4.60, dur=0.50, colour=(150, 134, 116))
MONTH = dict(text="十一月", size=15, tracking=4.0,
             x=M + 5, dy=146, t0=3.35, dur=0.45, colour=(126, 108, 90))
# the flower's language, spelled out -- the two characters beside the seal say
# what it means, this line says what it is
LANG = dict(text="花语  热情 · 坚韧", size=12, tracking=1.4,
            x=M + 5, dy=100, t0=2.68, dur=0.55, colour=(140, 120, 100))
BYNAME = dict(text="又名  簕杜鹃 · 叶子花", size=10, tracking=1.0,
              x=M + 5, dy=122, t0=3.55, dur=0.55, colour=(162, 144, 124))
# dx centres the seal on the column above it: the glyphs start at x=408 and are
# 16 px wide (centre 416) and dx is measured from the paper margin, so the
# plate -- 27 wide -- needs dx = 416 - 13 - 30
SEAL = dict(image=os.path.join(HERE, "..", "3_印章", "seal_baiwen_tall.png"),
            h=46, dx=373, dy=140, ink=PLATE, t0=4.05, dur=0.42)
SEAL["image"] = os.path.normpath(SEAL["image"])

FADE_OUT = 6.75
FEATHER = 3.0

# falling bracts
N_PETAL = 26
# a bract in this photograph is 10-16 px across; the earlier 22-34 px sprites
# were nearly twice the size of the flowers they fell from, which is half of
# why they read as stickers
PETAL_MIN, PETAL_MAX = 21, 31
FALL = (H + 2 * PETAL_MAX) / LOOP          # px/s, so every bract clears the card


# ------------------------------------------------------------------ pieces --
def geometry(scale=1.0):
    return dict(cw=int(round(W * scale)), ch=int(round(H * scale)),
                m=int(round(M * scale)), pw=int(round(PW * scale)),
                ph=int(round(PH * scale)), top=int(round((M + PH) * scale)))


def font(path, px):
    return ImageFont.truetype(path, max(1, int(round(px))))


def vertical_layout(scale=1.0):
    """Each column is a run of single glyphs stacked downwards.

    draw_glyph() wipes from the left edge of the glyph, which on a vertical
    column still reads as a pen stroke -- the character fills in from its left
    side rather than from its top, and at 30 px nobody can tell the difference.
    """
    glyphs = []
    for col in (TITLE, SMALL, POEM_A, POEM_B):
        f = font(col["font"], col["size"] * scale)
        step = (col["size"] + col["gap"]) * scale
        x = col["x"] * scale
        y = (M + PH) * scale + col["dy"] * scale + col["size"] * scale * 0.86
        for i, ch in enumerate(col["text"]):
            if ch.strip():
                glyphs.append(dict(ch=ch, x=x, y=y, font=f, colour=tuple(col["colour"]),
                                   t0=col["t0"] + i * col["step"], dur=col["dur"], mode="wipe"))
            y += step
    return glyphs


def load_photo(size=(PW, PH)):
    im = Image.open(os.path.join(HERE, "src", "bougainvillea.jpg")).convert("RGB")
    im = im.resize(size, Image.LANCZOS)
    # the source is a reposted screenshot: its jpeg speckle sits exactly where a
    # 256-colour palette has no entries, and comes out as confetti in the flower
    # heads. Blur the speckle away, then put the edges back.
    im = im.filter(ImageFilter.GaussianBlur(0.55))
    return im.filter(ImageFilter.UnsharpMask(radius=1.1, percent=62, threshold=2))


class Bract:
    """One falling bract: a fixed fall speed, its own sway, spin and flip."""

    def __init__(self, idx, rng, art):
        self.size = rng.uniform(PETAL_MIN, PETAL_MAX)
        self.art = art[[0, 1, 2, 0][idx % 4]]   # one in four carries the pink
        # start anywhere above the card, evenly spread, so the column of fallers
        # is continuous from t=0 and every one of them is off-frame at both ends
        p = (idx + rng.uniform(0.0, 0.9)) / N_PETAL
        self.y0 = -self.size - (1.0 - p) * (H + 2 * PETAL_MAX)
        self.x0 = rng.uniform(52, W - 52)
        self.sway = rng.uniform(9, 21)
        self.freq = rng.uniform(0.17, 0.33)
        self.ph = rng.uniform(0, 6.283)
        self.spin = rng.uniform(-26, 26)
        self.spin0 = rng.uniform(0, 360)
        self.flip_f = rng.uniform(0.28, 0.48)
        self.flip_p = rng.uniform(0, 6.283)
        # a bract is thin enough to let the wall show through it; an opaque
        # pale shape is what makes a falling petal read as pasted card
        self.alpha = rng.uniform(0.52, 0.78)
        self._sh = {}

    def shadow(self, scale=1.0):
        """The drop shadow, blurred in the size it is actually used at.

        Blurring the 1x alpha and then enlarging it leaves a soft grey halo that
        is wider than the bract -- on the print and video frames the shadow read
        as a smudge detached from the sprite. It is kept weak: a bract is thin
        paper, and a hard shadow is what makes it look pasted on.
        """
        key = round(scale, 2)
        if key not in self._sh:
            a = self.art.split()[3].filter(ImageFilter.GaussianBlur(1.9 * scale))
            sh = Image.new("RGBA", self.art.size, (74, 58, 46, 0))
            sh.putalpha(a.point(lambda v: int(v * 0.30)))
            self._sh[key] = sh
        return self._sh[key]

    def place(self, t):
        y = self.y0 + FALL * t
        x = self.x0 + self.sway * np.sin(2 * np.pi * self.freq * t + self.ph)
        spin = self.spin0 + self.spin * t
        # a bract turning edge-on never goes to a zero-width sliver -- and when
        # it is close to edge-on you see the edge of the paper, not its face,
        # so the sprite is darkened then instead of staying bright white
        sx = 0.46 + 0.54 * abs(np.cos(2 * np.pi * self.flip_f * t + self.flip_p))
        return x, y, spin, sx


def petal_art():
    """Sprites padded to a square so a spin never clips a corner."""
    out = []
    for k in range(3):
        src = Image.open(os.path.join(HERE, "sp", f"bract_{k}.png")).convert("RGBA")
        n = int(src.width * 1.45)
        tile = Image.new("RGBA", (n, n), (0, 0, 0, 0))
        tile.paste(src, ((n - src.width) // 2, (n - src.height) // 2), src)
        out.append(tile)
    return out


def draw_bracts(card, bracts, t, scale=1.0):
    # the bracts have to leave with the letter, or the last frame still carries
    # a handful mid-air and the restart pops them off
    g = P.global_alpha(dict(FADE_OUT=FADE_OUT, LOOP=LOOP), t)
    if g <= 0.004:
        return
    H_s = H * scale
    edge = (M + PH) * scale
    for b in bracts:
        x, y, spin, sx = b.place(t)
        y *= scale
        x *= scale
        if y < -PETAL_MAX * 2 * scale or y > H_s + PETAL_MAX * 2 * scale:
            continue
        w = max(2, int(round(b.size * scale * sx)))
        h = max(2, int(round(b.size * scale)))
        spr = b.art.resize((w, h), Image.LANCZOS)
        # a bract falls ~9 px between frames: at 12.5 fps that is a smear, not
        # a crisp cut-out, and the crisp version is what reads as a sticker.
        hs = max(2, int(round(h * 1.05)))
        spr = spr.resize((w, hs), Image.LANCZOS).filter(
            ImageFilter.GaussianBlur(0.55 * scale))
        # edge-on, the face is no longer what you see
        face = 0.86 + 0.14 * (sx - 0.46) / 0.54
        if face < 0.995:
            sh_a = spr.split()[3]
            spr = spr.point(lambda v: int(v * face))
            spr.putalpha(sh_a)
        if abs(b.spin) > 0.5:
            spr = spr.rotate(b.spin, resample=Image.BICUBIC)
        sh = b.shadow(scale).resize(spr.size, Image.LANCZOS)
        if abs(b.spin) > 0.5:
            sh = sh.rotate(b.spin, resample=Image.BICUBIC)
        px, py = int(x - spr.width / 2), int(y - spr.height / 2)
        # A bract that reaches the bottom of the photo leaves the picture: it
        # fades out over the last 48 px instead of printing a mark on top of the
        # dedication. The shadow has to fade with it -- fading only the sprite
        # left a grey blot sitting alone on the paper.
        # The same at the top edge: the bracts fall in from above the sheet, and
        # without a ramp they printed on the blank paper margin as pale ovals
        # floating outside the photograph.
        fade = 1.0
        top = M * scale
        if y < top + 40 * scale:
            fade = max(0.0, (y - top) / (40 * scale))
        if y > edge - 48 * scale:
            fade = min(fade, max(0.0, 1.0 - (y - (edge - 48 * scale)) / (48 * scale)))
        a = b.alpha * g * fade
        if a <= 0.02:
            continue
        if a < 0.999:
            spr = spr.copy()
            spr.putalpha(spr.split()[3].point(lambda v: int(v * a)))
            sh = sh.copy()
            sh.putalpha(sh.split()[3].point(lambda v: int(v * a / max(b.alpha, 1e-6))))
        card.alpha_composite(sh, (px + int(2 * scale), py + int(3 * scale)))
        card.alpha_composite(spr, (px, py))


# ----------------------------------------------------------------- drawing --
def foot_layer(geo, layout, t, scale=1.0):
    layer = Image.new("RGBA", (geo["cw"], geo["ch"]), (0, 0, 0, 0))
    g = P.global_alpha(dict(FADE_OUT=FADE_OUT, LOOP=LOOP), t)
    if g <= 0.002:
        return layer
    for it in layout:
        P.draw_glyph(layer, it, t, g, FEATHER * scale)
    P.draw_seal(layer, geo, SEAL, t, g, scale)
    draw_track(layer, LANG, t, g, F_SMALL, scale)
    draw_track(layer, BYNAME, t, g, F_SMALL, scale)
    draw_track(layer, MONTH, t, g, F_SMALL, scale)
    draw_track(layer, LATIN, t, g, F_LATIN, scale)
    return layer


def draw_track(layer, spec, t, g, path, scale=1.0):
    """A tracked run revealed letter by letter."""
    r = (t - spec["t0"]) / spec["dur"]
    if r <= 0:
        return
    r = min(1.0, r)
    f = font(path, spec["size"] * scale)
    dr = ImageDraw.Draw(layer)
    x = spec["x"] * scale
    y = (M + PH) * scale + spec["dy"] * scale
    # the step has to divide the run: with 0.045 the 23rd letter needs r > 1 and
    # never arrives, so the date loses its last four characters
    step = 0.80 / max(1, len(spec["text"]))
    for i, ch in enumerate(spec["text"]):
        c = int(255 * P.smooth(min(1.0, max(0.0, (r - i * step) * 6.0))) * g)
        if c > 2:
            dr.text((x, y), ch, font=f, fill=tuple(spec["colour"]) + (c,), anchor="ls")
        x += dr.textlength(ch, font=f) + spec["tracking"] * scale


def compose(photo, layout, bracts, t, scale=1.0):
    geo = geometry(scale)
    card = Image.new("RGB", (geo["cw"], geo["ch"]), PAPER)
    card.paste(photo.resize((geo["pw"], geo["ph"]), Image.LANCZOS) if photo.size !=
               (geo["pw"], geo["ph"]) else photo, (geo["m"], geo["m"]))
    # the hairline stays one pixel at 1x and scales with the card, otherwise the
    # print and the video lose the edge that separates paper from photograph
    ImageDraw.Draw(card).rectangle(
        (geo["m"], geo["m"], geo["m"] + geo["pw"] - 1, geo["m"] + geo["ph"] - 1),
        outline=(208, 198, 182), width=max(1, int(round(scale))))
    card = card.convert("RGBA")
    lay = foot_layer(geo, layout, t, scale)
    card.alpha_composite(lay)
    draw_bracts(card, bracts, t, scale)
    return card.convert("RGB")


# ------------------------------------------------------------ quantisation --
def encode(photo, layout, bracts, out_path, scale=1.0):
    n = int(round(LOOP * FPS))
    ts = [i * LOOP / n for i in range(n)]
    frames = [compose(photo, layout, bracts, t) for t in ts]
    print(f"composed {n} frames  {W}x{H}")

    probe = Image.new("RGB", (W * 4, H))
    for k, say in enumerate((0.20, 0.45, 0.72, 0.93)):
        probe.paste(frames[min(n - 1, int(round(say * n)))], (k * W, 0))
    pal = probe.quantize(colors=GIF_COLORS - 2, method=Image.MEDIANCUT)

    # MEDIANCUT counts pixels; a 46 px seal is a rounding error on a 2.1 Mpx
    # probe, so both reds need a reserved slot or they land on brown.
    pl = list(pal.getpalette()[: (GIF_COLORS - 2) * 3])
    for c in (ACCENT, PLATE):
        pl += [int(v) for v in c]
    pl += [0] * (768 - len(pl))
    pal.putpalette(pl)

    base = P.orderly(Image.new("RGB", (W, H), PAPER), pal, 0.0)
    photo_box = (M, M, M + PW, M + PH)
    out = []
    for card in frames:
        f = base.copy()
        # Ordered dither on the photo, none on the foot. Floyd-Steinberg reads
        # sharper but it costs 7 MB (it destroys the long runs a GIF compressor
        # lives on) and it turns the source's speckle into coloured confetti.
        f.paste(P.orderly(card.crop(photo_box), pal, DITHER), (M, M))
        f.paste(P.orderly(card.crop((0, M + PH, W, H)), pal, 0.0), (0, M + PH))
        out.append(f)

    delay = int(round(1000 * LOOP / n))
    out[0].save(out_path, save_all=True, append_images=out[1:],
                duration=delay, loop=0, optimize=True)
    print(f"wrote {os.path.basename(out_path)}  {os.path.getsize(out_path)/1e6:.2f} MB  "
          f"{n} frames  {delay} ms")
    return out_path


def fully_written():
    t = max([it["t0"] + it["dur"] for it in vertical_layout()] +
            [SEAL["t0"] + SEAL["dur"], LATIN["t0"] + LATIN["dur"],
             MONTH["t0"] + MONTH["dur"], LANG["t0"] + LANG["dur"],
             BYNAME["t0"] + BYNAME["dur"]])
    return min(t + 0.10, FADE_OUT - 0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--still", type=float, default=None)
    ap.add_argument("--gif", default=os.path.join(HERE, "bougainvillea_postcard.gif"))
    ap.add_argument("--print", dest="print_png", default=None)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--frames", default=None, help="directory for a video sequence")
    ap.add_argument("--fps", type=float, default=FPS)
    a = ap.parse_args()

    photo = load_photo()
    rng = np.random.default_rng(20261116)
    bracts = [Bract(i, rng, petal_art()) for i in range(N_PETAL)]

    if a.still is not None:
        compose(photo, vertical_layout(), bracts, a.still).save(
            os.path.join(HERE, f"_card_still_{a.still:.2f}.png"))
        print("wrote still")
        return

    if a.sheet:
        n = int(round(LOOP * FPS))
        sheet = Image.new("RGB", (270 * 4, (440 + 6) * 2), (18, 18, 18))
        for i, say in enumerate((0.04, 0.22, 0.38, 0.52, 0.66, 0.80, 0.92, 0.99)):
            idx = min(n - 1, int(round(say * n)))
            im = compose(photo, vertical_layout(), bracts, idx * LOOP / n)
            sheet.paste(im.resize((270, 440), Image.LANCZOS),
                        ((i % 4) * 276, (i // 4) * 446))
        sheet.save(os.path.join(HERE, "_card_sheet.png"))
        print("wrote _card_sheet.png")
        return

    if a.print_png:
        sc = a.scale
        # the print card takes its photo from the original, not from the 480 px
        # GIF panel -- upscaling the panel is the one blur that cannot be undone
        big = load_photo((int(PW * sc), int(PH * sc)))
        im = compose(big, vertical_layout(sc), [], fully_written(), scale=sc)
        im.save(a.print_png)
        print(f"wrote {os.path.basename(a.print_png)}  {im.size}")
        return

    if a.frames:
        sc = a.scale
        os.makedirs(a.frames, exist_ok=True)
        big = photo if sc == 1.0 else load_photo((int(PW * sc), int(PH * sc)))
        n = int(round(LOOP * a.fps))
        for i in range(n):
            im = compose(big, vertical_layout(sc), bracts, i * LOOP / n, scale=sc)
            im.save(os.path.join(a.frames, f"f_{i:03d}.png"))
        print(f"wrote {n} frames to {a.frames}  {im.size}")
        return

    encode(photo, vertical_layout(), bracts, a.gif)


if __name__ == "__main__":
    main()
