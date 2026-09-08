#!/usr/bin/env python3
"""Prototype show-through removal, quantify on worst pages. Saves before/after PNGs for eyeballing."""
import os
import cv2
import numpy as np
import pymupdf

PDF = "/book/sources/Word的排版艺术_By_侯捷.pdf"
TEST = [30, 120, 180]   # clean control + two heavy show-through pages
OUTDIR = "/book/pilot_proto"
os.makedirs(OUTDIR, exist_ok=True)


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


def metrics(gray):
    h, w = gray.shape
    strip_w = max(8, int(w * 0.08))
    strips = np.concatenate([gray[:, :strip_w].ravel(), gray[:, -strip_w:].ravel()])
    return {
        "marginDark%": round(float(np.mean(strips < 200)) * 100, 2),
        "text<100%": round(float(np.mean(gray < 100)) * 100, 3),
        "gray100-200%": round(float(np.mean((gray >= 100) & (gray < 200))) * 100, 2),
    }


def chain(gray, clean=None):
    """Pipeline preprocessing as in book_real.py (blur3 + CLAHE2), optionally after clean()."""
    if clean:
        gray = clean(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    return gray


def clean_detail(gray, eps):
    """Edge-amplitude threshold: keep sharp detail (text), suppress soft show-through."""
    base = cv2.GaussianBlur(gray, (0, 0), 5)
    hp = gray.astype(np.int16) - base.astype(np.int16)
    out = base.astype(np.int16) + np.sign(hp) * np.maximum(np.abs(hp) - eps, 0)
    return np.clip(out, 0, 255).astype(np.uint8)


def clean_bgsub(gray, tau):
    """Darkness soft-clip: bg estimate via close(51), keep only darkness > tau."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    d = bg.astype(np.int16) - gray.astype(np.int16)
    out = 255 - np.clip((d - tau) * (255.0 / (255 - tau)), 0, 255)
    return out.astype(np.uint8)


def main():
    doc = pymupdf.open(PDF)
    print(f"{'page':>5} {'stage':>26} {'marginDark%':>11} {'text<100%':>9} {'gray100-200%':>12}")
    for pno in TEST:
        img = page_image(doc, pno)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw = chain(gray)                       # current pipeline (baseline)
        d1 = chain(gray, lambda g: clean_detail(g, 25))
        d2 = chain(gray, lambda g: clean_detail(g, 40))
        bs = chain(gray, lambda g: clean_bgsub(g, 70))
        for name, g in [("raw(现状)", raw), ("detail eps25", d1), ("detail eps40", d2), ("bgsub tau70", bs)]:
            m = metrics(g)
            print(f"{pno:>5} {name:>26} {m['marginDark%']:>11.2f} {m['text<100%']:>9.3f} {m['gray100-200%']:>12.2f}")
        cv2.imwrite(f"{OUTDIR}/p{pno:03d}_raw.png", raw)
        cv2.imwrite(f"{OUTDIR}/p{pno:03d}_detail25.png", d1)
        cv2.imwrite(f"{OUTDIR}/p{pno:03d}_detail40.png", d2)
        cv2.imwrite(f"{OUTDIR}/p{pno:03d}_bgsub70.png", bs)
    doc.close()
    print("PNGs saved to", OUTDIR)


if __name__ == "__main__":
    main()