#!/usr/bin/env python3
"""Sharpen visible-bg: compare output-chain variants on worst pages. Laplacian var = sharpness."""
import cv2
import numpy as np
import pymupdf

PDF = "/book/sources/Word的排版艺术_By_侯捷.pdf"
TEST = [30, 120, 180]


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


def clean_flat(gray, tau, open3=True):
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    d = (bg.astype(np.int16) - gray.astype(np.int16)).clip(0, 255)
    out = 255 - np.clip((d - tau) * 4.0, 0, 255)
    out = out.astype(np.uint8)
    if open3:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return out, d


def tau_of(gray):
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    d = (bg.astype(np.int16) - gray.astype(np.int16)).clip(0, 255)
    otsu_d = int(cv2.threshold(d.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    return max(55, int(otsu_d * 1.15))


def sharp(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def metrics(gray):
    h, w = gray.shape
    sw = max(8, int(w * 0.08))
    strips = np.concatenate([gray[:, :sw].ravel(), gray[:, -sw:].ravel()])
    return (round(float(np.mean(strips < 200)) * 100, 2),
            round(float(np.mean(gray < 100)) * 100, 3),
            round(sharp(gray), 1))


def main():
    doc = pymupdf.open(PDF)
    for pno in TEST:
        img = page_image(doc, pno)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tau = tau_of(gray)
        flat_open, d = clean_flat(gray, tau, open3=True)
        flat_noopen, _ = clean_flat(gray, tau, open3=False)
        # composite: sharp native strokes over cleaned flat bg (feathered mask)
        mask = (d > tau + 35).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        comp = (flat_noopen.astype(np.float32) * (1 - mask) + gray.astype(np.float32) * mask).astype(np.uint8)
        # unsharp on flat (no open, no blur)
        flat_us = cv2.addWeighted(flat_noopen, 1.5, cv2.GaussianBlur(flat_noopen, (0, 0), 2), -0.5, 0)
        variants = {
            "raw(现状OCR链)": gray,
            "flat+open3(现输出)": flat_open,
            "flat(无open)": flat_noopen,
            "flat+unsharp": flat_us,
            "composite(原字叠净底)": comp,
        }
        print(f"\n=== p{pno} τ={tau} ===")
        print(f"{'variant':>22} {'mDark%':>7} {'t<100%':>7} {'lapVar':>8}")
        for name, g in variants.items():
            m, t, s = metrics(g)
            print(f"{name:>22} {m:>7.2f} {t:>7.3f} {s:>8.1f}")
            cv2.imwrite(f"/book/pilot_proto/p{pno:03d}_sh_{name.split('(')[0].strip().replace('+','_').replace(' ','_')}.png", g)
    doc.close()


if __name__ == "__main__":
    main()