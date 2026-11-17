"""Clear the leaves that stand up into the type band.

On a flat-lay bouquet the foliage lies low; here the generator pushed a rank of
leaves up to the top of the frame, which collides with the title. A real
arrangement would not do that, so they are taken out and the paper heals behind
them. The cut is limited to green pixels above a line well clear of the blooms,
and the mask is grown by a couple of pixels so no rim of leaf is left behind.

    python clear_crown.py <in.png> <out.png> [--line 300]
"""
import sys

import cv2
import numpy as np


def main(src, dst, line):
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        sys.exit("cannot read " + src)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].astype(np.int16), hsv[..., 1], hsv[..., 2]
    green = ((hue > 28) & (hue < 100) & (sat > 55) & (val > 40)).astype(np.uint8) * 255

    band = np.zeros_like(green)
    band[:line] = 255
    mask = cv2.bitwise_and(green, band)
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = (mask > 90).astype(np.uint8) * 255

    out = cv2.inpaint(im, mask, 10, cv2.INPAINT_TELEA)

    # the patch is smoother than the sheet around it; give the grain back
    ys, xs = np.nonzero(mask)
    if len(ys):
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        rng = np.random.default_rng(3)
        patch = out[y0:y1, x0:x1].astype(np.float32)
        patch += rng.normal(0, 2.0, patch.shape).astype(np.float32)
        out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)

    cv2.imwrite(dst, out)
    print("cleared", int(mask.sum() / 255), "px above y =", line, "->", dst)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ln = 300
    for a in sys.argv[1:]:
        if a.startswith("--line"):
            ln = int(a.split("=")[-1])
    main(args[0], args[1], ln)
