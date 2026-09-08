#!/usr/bin/env python3
"""Output-only cleaning: OCR path untouched; visible bg cleaned. Verify metrics + save PNGs."""
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


def clean_out(gray, tau, despeckle=True):
    """Visible-bg cleaning: darkness soft-clip with steep ramp (keeps text dark), optional despeckle."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    d = bg.astype(np.int16) - gray.astype(np.int16)
    out = 255 - np.clip((d - tau) * 4.0, 0, 255)   # d=tau -> 255, d=tau+64 -> 0 (dark text)
    out = out.astype(np.uint8)
    if despeckle:
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return out


def metrics(gray):
    h, w = gray.shape
    strip_w = max(8, int(w * 0.08))
    strips = np.concatenate([gray[:, :strip_w].ravel(), gray[:, -strip_w:].ravel()])
    return (round(float(np.mean(strips < 200)) * 100, 2),
            round(float(np.mean(gray < 100)) * 100, 3),
            round(float(np.mean((gray >= 100) & (gray < 200))) * 100, 2))


def main():
    doc = pymupdf.open(PDF)
    print(f"{'page':>5} | {'raw mD/tT':>14} | {'clean mD/tT/gG':>18}")
    for pno in TEST:
        img = page_image(doc, pno)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
        d = (bg.astype(np.int16) - gray.astype(np.int16)).clip(0, 255)
        otsu_d = int(cv2.threshold(d.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
        tau = max(55, int(otsu_d * 1.15))
        raw = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(gray, (3, 3), 0))
        out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(clean_out(gray, tau), (3, 3), 0))
        mr, tr = metrics(raw)[:2]
        mc, tc, gc = metrics(out)
        print(f"{pno:>5} τ={tau:>3} | {mr:>5.1f}/{tr:>5.2f} | {mc:>5.1f}/{tc:>5.2f}/{gc:>5.1f}")
        cv2.imwrite(f"/book/pilot_proto/p{pno:03d}_outbg_clean.png", out)
    doc.close()


if __name__ == "__main__":
    main()