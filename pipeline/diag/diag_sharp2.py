#!/usr/bin/env python3
"""Final output-chain candidates, measured AFTER JPEG roundtrip (what actually lands in the PDF)."""
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


def jpeg_rt(g, q):
    return cv2.imdecode(cv2.imencode(".jpg", g, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_GRAYSCALE)


def metrics(g):
    h, w = g.shape
    sw = max(8, int(w * 0.08))
    strips = np.concatenate([g[:, :sw].ravel(), g[:, -sw:].ravel()])
    return (round(float(np.mean(strips < 200)) * 100, 2),
            round(float(np.mean(g < 100)) * 100, 3),
            round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1))


def main():
    doc = pymupdf.open(PDF)
    print(f"{'page':>5} {'variant':>22} {'mDark%':>7} {'t<100%':>7} {'lapVar@q88':>10} {'lapVar@q92':>10}")
    for pno in TEST:
        img = page_image(doc, pno)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tau = tau_of(gray)
        flat, _ = clean_flat(gray, tau)
        us = cv2.addWeighted(flat, 1.5, cv2.GaussianBlur(flat, (0, 0), 2), -0.5, 0)
        us_cl = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(us)
        cl_us = cv2.addWeighted(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat), 1.5,
                                cv2.GaussianBlur(cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(flat), (0, 0), 2), -0.5, 0)
        variants = {
            "flat+unsharp": us,
            "us->CLAHE": us_cl,
            "CLAHE->us": cl_us,
        }
        for name, g in variants.items():
            m88, t88, s88 = metrics(jpeg_rt(g, 88))
            m92, t92, s92 = metrics(jpeg_rt(g, 92))
            print(f"{pno:>5} {name:>22} {m88:>7.2f} {t88:>7.3f} {s88:>10.1f} {s92:>10.1f}")
            cv2.imwrite(f"/book/pilot_proto/p{pno:03d}_final_{name.replace('->','_')}.png", g)
    doc.close()


if __name__ == "__main__":
    main()