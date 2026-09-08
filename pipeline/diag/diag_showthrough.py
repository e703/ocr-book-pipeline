#!/usr/bin/env python3
"""Diagnose show-through (透色) and skew on scanned book pages. Read-only, no OCR."""
import cv2
import numpy as np
import pymupdf
import sys

PDF = "/book/sources/Word的排版艺术_By_侯捷.pdf"
SAMPLE = [1, 30, 60, 120, 180, 250, 320, 400]


def page_image(doc, pno):
    page = doc[pno - 1]
    imgs = page.get_images(full=True)
    if not imgs:
        pix = page.get_pixmap(dpi=200)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    xref = imgs[0][0]
    info = doc.extract_image(xref)
    img = cv2.imdecode(np.frombuffer(info["image"], np.uint8), cv2.IMREAD_COLOR)
    pw, ph = page.rect.width, page.rect.height
    iw, ih = img.shape[1], img.shape[0]
    if abs(iw / ih - pw / ph) > 0.15 and abs(iw / ih - ph / pw) < 0.15:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if page.rotation in (90, 270):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if page.rotation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def skew_est(gray):
    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-6.0, 6.01, 0.25):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        s = float(np.var(rot.sum(axis=1, dtype=np.float32)))
        if s > best_s:
            best_s, best_a = s, a
    return best_a


def main():
    doc = pymupdf.open(PDF)
    print(f"total pages: {doc.page_count}")
    print(f"{'page':>5} {'size':>12} {'deskew°':>8} {'otsu':>5} {'text<100':>8} "
          f"{'gray>100':>8} {'marginDark%':>10} {'bgMean':>6}")
    for pno in SAMPLE:
        img = page_image(doc, pno)
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ang = skew_est(gray)
        otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu = int(otsu)
        text_frac = float(np.mean(gray < 100))
        gray_frac = float(np.mean((gray >= 100) & (gray < 200)))
        # show-through proxy: dark pixels in left/right margin strips (no text there normally)
        strip_w = max(8, int(w * 0.08))
        strips = np.concatenate([gray[:, :strip_w].ravel(), gray[:, -strip_w:].ravel()])
        margin_dark = float(np.mean(strips < 200))
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
        bg_mean = float(np.mean(bg))
        print(f"{pno:>5} {w}x{h:>6} {ang:>+8.2f} {otsu:>5} {text_frac:>8.4f} {gray_frac:>8.4f} "
              f"{margin_dark:>10.4f} {bg_mean:>6.1f}")
    doc.close()


if __name__ == "__main__":
    main()