"""Prepare a generated bouquet for the card: remove the platform mark.

The generator stamps "AI生成 / WORKBUDDY" into the bottom-right corner. On this
plate it always lands on bare paper, so a small TELEA inpaint heals it without
touching a single petal -- which is why no mask work is needed here.

    python prep_bouquet.py <in.png> <out.png>
"""
import sys

import cv2
import numpy as np

BOX = (1388, 948, 1536, 1024)          # x0, y0, x1, y1 -- paper only
RADIUS = 8


def main(src, dst):
    im = cv2.imread(src, cv2.IMREAD_COLOR)
    if im is None:
        sys.exit("cannot read " + src)
    x0, y0, x1, y1 = BOX
    x1, y1 = min(x1, im.shape[1]), min(y1, im.shape[0])
    mask = np.zeros(im.shape[:2], np.uint8)
    mask[y0:y1, x0:x1] = 255
    out = cv2.inpaint(im, mask, RADIUS, cv2.INPAINT_TELEA)
    # inpaint returns a slightly smoother patch than the surrounding grain;
    # put a whisper of noise back so the plate stays continuous
    patch = out[y0:y1, x0:x1].astype(np.float32)
    rng = np.random.default_rng(7)
    patch += rng.normal(0, 2.1, patch.shape).astype(np.float32)
    out[y0:y1, x0:x1] = np.clip(patch, 0, 255).astype(np.uint8)
    cv2.imwrite(dst, out)
    print("wrote", dst, out.shape[1], "x", out.shape[0])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
