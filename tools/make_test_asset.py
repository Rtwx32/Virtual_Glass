"""
make_test_asset.py — يولّد صورة نظارة بنمط صورة منتج (خلفية بيضاء) لاختبار
المسار كاملاً بلا الحاجة لتنزيل صور من الإنترنت. للاختبار فقط، ليس أصلاً حقيقيًا.
"""
from pathlib import Path
import argparse

import cv2
import numpy as np


def draw(style: str = "rect", w: int = 900, h: int = 360) -> np.ndarray:
    img = np.full((h, w, 3), 245, np.uint8)          # خلفية استوديو
    col = (38, 34, 30)
    t = 12
    cy = int(h * 0.52)
    lens_w, lens_h = int(w * 0.31), int(h * 0.46)
    gap = int(w * 0.07)
    lx = int(w * 0.5 - gap / 2 - lens_w)
    rx = int(w * 0.5 + gap / 2)

    for x0 in (lx, rx):
        box = (x0, cy - lens_h // 2, lens_w, lens_h)
        if style == "round":
            cv2.ellipse(img, (x0 + lens_w // 2, cy), (lens_w // 2, lens_h // 2),
                        0, 0, 360, col, t, cv2.LINE_AA)
        else:
            _rounded(img, box, 26, col, t)

    # الجسر
    bx1, bx2 = lx + lens_w, rx
    cv2.line(img, (bx1, cy - lens_h // 6), (bx2, cy - lens_h // 6), col, t, cv2.LINE_AA)
    # نتوءا المفصلة
    cv2.line(img, (lx - 14, cy - lens_h // 5), (lx, cy - lens_h // 5), col, t + 4, cv2.LINE_AA)
    cv2.line(img, (rx + lens_w, cy - lens_h // 5), (rx + lens_w + 14, cy - lens_h // 5),
             col, t + 4, cv2.LINE_AA)
    return img


def _rounded(img, box, r, col, t):
    x, y, bw, bh = box
    cv2.line(img, (x + r, y), (x + bw - r, y), col, t, cv2.LINE_AA)
    cv2.line(img, (x + r, y + bh), (x + bw - r, y + bh), col, t, cv2.LINE_AA)
    cv2.line(img, (x, y + r), (x, y + bh - r), col, t, cv2.LINE_AA)
    cv2.line(img, (x + bw, y + r), (x + bw, y + bh - r), col, t, cv2.LINE_AA)
    for cx, cy_, a in ((x + r, y + r, 180), (x + bw - r, y + r, 270),
                       (x + bw - r, y + bh - r, 0), (x + r, y + bh - r, 90)):
        cv2.ellipse(img, (cx, cy_), (r, r), a, 0, 90, col, t, cv2.LINE_AA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/test_glasses.jpg")
    ap.add_argument("--style", default="rect", choices=["rect", "round"])
    ns = ap.parse_args()
    p = Path(ns.out); p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), draw(ns.style), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("كُتبت:", p)
