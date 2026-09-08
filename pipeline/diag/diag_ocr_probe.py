#!/usr/bin/env python3
"""OCR probe: raw chain vs show-through-cleaned chain on worst pages (real acceptance test)."""
import cv2
import numpy as np
import pymupdf
from paddleocr import PaddleOCR

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


def clean_bgsub(gray, tau):
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    d = bg.astype(np.int16) - gray.astype(np.int16)
    out = 255 - np.clip((d - tau) * (255.0 / 100.0), 0, 255)
    return out.astype(np.uint8)


def tau_for(gray):
    h, w = gray.shape
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    d = (bg.astype(np.int16) - gray.astype(np.int16)).clip(0, 255)
    strip_w = max(8, int(w * 0.08))
    margin = np.concatenate([d[:, :strip_w].ravel(), d[:, -strip_w:].ravel()])
    mp99 = float(np.percentile(margin, 99))
    return int(np.clip(mp99 * 1.15 + 8, 55, 150))


def main():
    kw = dict(ocr_version="PP-OCRv4", lang="ch",
              use_doc_orientation_classify=True, use_doc_unwarping=False,
              use_textline_orientation=True,
              text_det_limit_side_len=960, text_det_limit_type="max", cpu_threads=8)
    ocr = PaddleOCR(**kw)
    doc = pymupdf.open(PDF)
    for pno in TEST:
        img = page_image(doc, pno)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        raw = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(gray, (3, 3), 0))
        tau = tau_for(gray)
        clean = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            cv2.GaussianBlur(clean_bgsub(gray, tau), (3, 3), 0))
        for name, g in [("RAW   ", raw), (f"CLEAN τ={tau}", clean)]:
            res = ocr.predict(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))[0]
            texts = res["rec_texts"]
            print(f"\n=== p{pno} {name}: {len(texts)} lines ===")
            for t in texts[:8]:
                print("   ", t)
            cv2.imwrite(f"/book/pilot_proto/p{pno:03d}_ocr_{'raw' if name.startswith('RAW') else 'clean'}.png", g)
    doc.close()


if __name__ == "__main__":
    main()