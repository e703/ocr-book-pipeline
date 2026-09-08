#!/usr/bin/env python3
"""Calibrate: darkness distribution of show-through (margins) vs real text."""
import cv2
import numpy as np
import pymupdf

PDF = "/book/sources/Word的排版艺术_By_侯捷.pdf"
TEST = [30, 120, 180, 250]


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


def main():
    doc = pymupdf.open(PDF)
    print(f"{'page':>5} {'margin d: p50 p90 p99':>26} {'text d: p10 p50 p90 max':>26} "
          f"{'otsu(d)':>8} {'gap':>6}")
    for pno in TEST:
        img = page_image(doc, pno)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
        d = (bg.astype(np.int16) - gray.astype(np.int16)).clip(0, 255)
        strip_w = max(8, int(w * 0.08))
        margin = np.concatenate([d[:, :strip_w].ravel(), d[:, -strip_w:].ravel()])
        # real text pixels: gray < 150 (generous; show-through overlaps a bit)
        text = d[gray < 150]
        mp = np.percentile(margin, [50, 90, 99])
        tp = np.percentile(text, [10, 50, 90, 100]) if len(text) else [0, 0, 0, 0]
        otsu_d = int(cv2.threshold(d.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        print(f"{pno:>5} {str(np.round(mp,1)):>26} {str(np.round(tp,1)):>26} {otsu_d:>8} "
              f"{float(tp[0]-mp[2]):>6.1f}")
    doc.close()


if __name__ == "__main__":
    main()