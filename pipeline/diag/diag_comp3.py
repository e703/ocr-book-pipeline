#!/usr/bin/env python3
"""Prototype v3: composite native strokes (masked by OCR boxes) over cleaned flat bg.
Fixes eaten stroke edges — the ramp currently erases anti-aliased edge pixels."""
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


def clean_flat(gray, tau):
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    d = (bg.astype(np.int16) - gray.astype(np.int16)).clip(0, 255)
    return (255 - np.clip((d - tau) * 4.0, 0, 255)).astype(np.uint8), d


def tau_of(gray):
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    d = (bg.astype(np.int16) - gray.astype(np.int16)).clip(0, 255)
    otsu_d = int(cv2.threshold(d.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    return max(55, int(otsu_d * 1.15))


def unsharp(gray, amount=1.3):
    return cv2.addWeighted(gray, amount, cv2.GaussianBlur(gray, (0, 0), 2.0), 1 - amount, 0)


def box_mask(shape, boxes, dilate_iter=3):
    """Mask: OCR text boxes (scaled to native res), dilated to cover stroke edges."""
    mask = np.zeros(shape, np.uint8)
    for _, box in boxes:
        pts = np.asarray(box, dtype=np.int32).reshape(-1, 2)
        cv2.fillPoly(mask, [pts], 1)
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=dilate_iter)


def metrics(g, raw):
    h, w = g.shape
    sw = max(8, int(w * 0.08))
    strips = np.concatenate([g[:, :sw].ravel(), g[:, -sw:].ravel()])
    md = float(np.mean(strips < 200)) * 100
    t100 = float(np.mean(g < 100)) * 100
    surv = float(np.mean(g[raw < 120] < 200)) * 100   # % of raw core pixels still visible
    ring = float(np.mean(g[(raw >= 120) & (raw < 200)] < 235)) * 100  # % of raw edge-ring pixels kept darkish
    return (round(md, 2), round(t100, 3), round(surv, 1), round(ring, 1), round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1))


def main():
    from paddleocr import PaddleOCR
    kw = dict(ocr_version="PP-OCRv4", lang="ch", use_doc_orientation_classify=True,
              use_doc_unwarping=False, use_textline_orientation=True,
              text_det_limit_side_len=960, text_det_limit_type="max", cpu_threads=8)
    ocr = PaddleOCR(**kw)
    doc = pymupdf.open(PDF)
    print(f"{'page':>5} {'variant':>22} {'mDark%':>7} {'t<100%':>7} {'surv%':>6} {'lapVar':>8}")
    for pno in TEST:
        img = page_image(doc, pno)
        raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tau = tau_of(raw)
        # mimic pipeline OCR path (deskew on raw-enhance chain)
        ocr_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(cv2.GaussianBlur(raw, (3, 3), 0))
        res = ocr.predict(cv2.cvtColor(ocr_gray, cv2.COLOR_GRAY2RGB))[0]
        boxes = list(zip(res["rec_texts"], res["rec_boxes"]))
        flat, d = clean_flat(raw, tau)
        flat = _rot(flat, 0)  # no deskew in proto (angles ~0 on these pages); boxes align 1:1
        mask = box_mask(raw.shape, boxes, dilate_iter=3)
        dark = (d > 20).astype(np.uint8)   # inside OCR boxes trust text: lenient darkness floor
        mask = (mask & dark).astype(np.float32)
        mask = cv2.GaussianBlur(mask, (3, 3), 0)   # feather
        comp = (flat.astype(np.float32) * (1 - mask) + raw.astype(np.float32) * mask).astype(np.uint8)
        variants = {
            "flat+unsharp(现版)": unsharp(flat, 1.5),
            "composite+unsharp": unsharp(comp, 1.3),
            "composite(no US)": comp,
        }
        for name, g in variants.items():
            m = metrics(g, raw)
            print(f"{pno:>5} {name:>22} {m[0]:>7.2f} {m[1]:>7.3f} {m[2]:>6.1f} {m[3]:>6.1f} {m[4]:>8.1f}")
            cv2.imwrite(f"/book/pilot_proto/p{pno:03d}_v3_{name.split('(')[0].strip().replace('+','_').replace(' ','_')}.png", g)
    doc.close()


def _rot(g, ang):
    return g


if __name__ == "__main__":
    main()