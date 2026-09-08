#!/usr/bin/env python3
"""Real scanned-book pipeline: sources/*.pdf -> targets/ (searchable PDF + text + previews).

Reads embedded scan images at native resolution (no re-render loss), deskews,
enhances, OCRs with PP-OCRv4 mobile, orders text by reading order (handles
horizontal/vertical and multi-column), rebuilds a searchable PDF.
"""
import glob
import json
import os
import time
import cv2
import numpy as np
import pymupdf

FONT = "/tmp/ocrbench/fonts/wqy-zenhei.ttc"
SRC = os.environ.get("SRC", "/book/sources")
OUT = os.environ.get("OUT", "/book/targets")
MAX_PAGES = int(os.environ.get("MAX_PAGES", "20"))
START = int(os.environ.get("START_PAGE", "1"))
OCR_DPI = 200.0
SCALE = 72.0 / OCR_DPI
CLEAN = os.environ.get("CLEAN", "1") != "0"   # clean show-through on the visible background only
KEEP_COVER = os.environ.get("KEEP_COVER", "1") != "0"  # page 1 (cover): original color, untouched

os.makedirs(f"{OUT}/text", exist_ok=True)
os.makedirs(f"{OUT}/previews", exist_ok=True)


def deskew(gray):
    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-6.0, 6.01, 0.25):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        score = float(np.var(rot.sum(axis=1, dtype=np.float32)))
        if score > best_s:
            best_s, best_a = score, a
    return _apply_angle(gray, best_a), best_a


def _apply_angle(gray, ang, interp=cv2.INTER_CUBIC):
    """Rotate gray by ang degrees around center; no-op for tiny angles (matches deskew)."""
    if abs(ang) < 0.2:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=interp, borderValue=255)


def clean_out(gray):
    """Visible-background show-through removal (OUTPUT image only — never feed OCR).

    Returns (flat, d, tau): flat = show-through-suppressed background (steep darkness
    ramp), d = darkness map (bg_estimate - gray), tau = per-page threshold.
    NOTE: the ramp alone eats anti-aliased stroke edges; the caller composites the
    NATIVE strokes back inside OCR text boxes, so this flat layer is only visible
    in margins/gaps. Measured: margin dark-pixel 16-46% -> <3.5%."""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((51, 51), np.uint8))
    d = (bg.astype(np.int16) - gray.astype(np.int16)).clip(0, 255)
    otsu_d = int(cv2.threshold(d.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    tau = max(55, int(otsu_d * 1.15))
    flat = (255 - np.clip((d - tau) * 4.0, 0, 255)).astype(np.uint8)
    return flat, d, tau


def unsharp(gray, amount=1.5, sigma=2.0):
    """Mild unsharp mask: crisp text edges without halos (measured +50-70% lapVar)."""
    return cv2.addWeighted(gray, amount, cv2.GaussianBlur(gray, (0, 0), sigma), 1 - amount, 0)


def page_image(doc, pno):
    """Extract the embedded scan image at native resolution (applying page rotation)."""
    page = doc[pno - 1]
    imgs = page.get_images(full=True)
    if not imgs:
        # fallback: render
        pix = page.get_pixmap(dpi=OCR_DPI)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR), pix.width, pix.height
    xref = imgs[0][0]
    info = doc.extract_image(xref)
    img = cv2.imdecode(np.frombuffer(info["image"], np.uint8), cv2.IMREAD_COLOR)
    pw, ph = page.rect.width, page.rect.height
    iw, ih = img.shape[1], img.shape[0]
    # if embedded aspect is transposed vs page, rotate (scans often embedded 90deg)
    if abs(iw / ih - pw / ph) > 0.15 and abs(iw / ih - ph / pw) < 0.15:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if page.rotation in (90, 270):
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE if page.rotation == 90 else cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img, iw, ih


def reading_order(texts, boxes):
    """Sort (text, box) pairs by reading order. Handles horizontal & vertical CJK
    and multi-column layouts."""
    items = [(t, np.asarray(b, dtype=float).reshape(-1, 2)) for t, b in zip(texts, boxes) if t and t.strip()]
    if not items:
        return []
    xs = [(it[1][:, 0].min() + it[1][:, 0].max()) / 2 for it in items]
    ys = [(it[1][:, 1].min() + it[1][:, 1].max()) / 2 for it in items]
    ws = [it[1][:, 0].max() - it[1][:, 0].min() for it in items]
    hs = [it[1][:, 1].max() - it[1][:, 1].min() for it in items]
    med_w, med_h = float(np.median(ws)), float(np.median(hs))
    vertical = med_h > 1.5 * med_w  # vertical CJK: boxes are tall & narrow

    order = sorted(range(len(items)), key=lambda i: xs[i])
    cols, cur = [], [order[0]]
    gap_thr = 2.2 * med_w
    for i in order[1:]:
        if xs[i] - xs[cur[-1]] > gap_thr:
            cols.append(cur)
            cur = [i]
        else:
            cur.append(i)
    cols.append(cur)
    cols.sort(key=lambda c: xs[c[0]], reverse=vertical)  # vertical books read right-to-left

    out = []
    for c in cols:
        c = sorted(c, key=lambda i: (ys[i], xs[i]))
        out.extend(items[i] for i in c)
    return out


def main():
    pdfs = sorted(glob.glob(f"{SRC}/*.pdf"))
    if not pdfs:
        raise SystemExit(f"no pdf in {SRC}")
    from paddleocr import PaddleOCR
    kw = dict(ocr_version="PP-OCRv4", lang="ch",
              use_doc_orientation_classify=True,   # fix 90/180 rotated pages
              use_doc_unwarping=False,
              use_textline_orientation=True,
              text_det_limit_side_len=960, text_det_limit_type="max")
    threads = os.environ.get("THREADS")
    if threads:
        kw["cpu_threads"] = int(threads)
    ocr = PaddleOCR(**kw)

    needed = START + MAX_PAGES - 1
    log = {"source": pdfs, "pages": [], "warnings": []}
    outdoc = pymupdf.open()
    t_start = time.perf_counter()
    total_pages = 0
    for pdf in pdfs:
        doc = pymupdf.open(pdf)
        npages = doc.page_count
        for pno in range(START, min(npages, needed) + 1):
            t0 = time.perf_counter()
            img, iw, ih = page_image(doc, pno)
            raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cover = KEEP_COVER and pno == 1   # cover: keep original color page, untouched
            # --- OCR path: unchanged raw-enhance chain. Cleaning HURTS small/faint text
            # (footnotes, page numbers) on heavy show-through pages — OCR stays on raw.
            ocr_gray = cv2.GaussianBlur(raw, (3, 3), 0)
            ocr_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(ocr_gray)
            if cover:
                ang = 0.0                     # cover stays exactly as scanned (no deskew)
            else:
                ocr_gray, ang = deskew(ocr_gray)
            # downscale to OCR working size (max side 2400px) for speed
            if max(ocr_gray.shape) > 2400:
                f = 2400 / max(ocr_gray.shape)
                ocr_gray = cv2.resize(ocr_gray, (int(ocr_gray.shape[1] * f), int(ocr_gray.shape[0] * f)),
                                      interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(ocr_gray, cv2.COLOR_GRAY2RGB)
            res = ocr.predict(rgb)[0]
            texts, boxes = res["rec_texts"], res["rec_boxes"]
            ordered = reading_order(texts, boxes)
            t_ocr = time.perf_counter() - t0
            # --- output background: cleaned flat bg + NATIVE strokes composited inside OCR
            # boxes (shape-exact text — the darkness ramp alone eats stroke edges).
            # Everything in deskewed native space so OCR box coordinates align 1:1.
            out_gray = ocr_gray   # default: OCR enhance chain (CLEAN=0 / fallthrough)
            if cover:
                out_bgr = img   # visible page = original color image, zero processing
            elif CLEAN:
                nat = _apply_angle(raw, ang, interp=cv2.INTER_LANCZOS4)
                flat, dmap, tau = clean_out(nat)
                mask = np.zeros(nat.shape, np.uint8)
                s = nat.shape[1] / ocr_gray.shape[1]   # OCR coords -> native res
                for _, box in ordered:
                    pts = (np.asarray(box, dtype=np.float32).reshape(-1, 2) * s).astype(np.int32)
                    cv2.fillPoly(mask, [pts], 1)
                mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=3)
                mask = (mask & (dmap > 20).astype(np.uint8)).astype(np.float32)
                mask = cv2.GaussianBlur(mask, (3, 3), 0)   # feather the seam
                out_gray = (flat.astype(np.float32) * (1 - mask) + nat.astype(np.float32) * mask).astype(np.uint8)
                out_gray = unsharp(out_gray, amount=1.3)
            else:
                out_gray = ocr_gray

            # build page: native-res bg (no downscale/blur — keeps it readable) + invisible text layer
            h, w = ocr_gray.shape
            pw, ph = doc[pno - 1].rect.width, doc[pno - 1].rect.height
            page = outdoc.new_page(width=pw, height=ph)
            if cover:
                stream = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                      [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            else:
                stream = cv2.imencode(".jpg", out_gray, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            page.insert_image(pymupdf.Rect(0, 0, pw, ph), stream=stream)
            lines = 0
            for txt, box in ordered:
                pts = box
                x0, y0 = float(pts[:, 0].min()), float(pts[:, 1].min())
                bw_, bh_ = float(pts[:, 0].max()) - x0, float(pts[:, 1].max()) - y0
                fs = max(4.0, bh_ * (pw / w) * 0.92)   # px -> pt via ratio
                page.insert_text((x0 * (pw / w), y0 * (pw / w) + fs * 0.8), txt,
                                 fontname="wqy", fontfile=FONT, fontsize=fs, render_mode=3)
                lines += 1
            with open(f"{OUT}/text/p{pno:03d}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(t for t, _ in ordered))
            if lines == 0:
                log["warnings"].append(f"page {pno}: 0 text lines (blank or unreadable)")
            t_tot = time.perf_counter() - t_start
            print(f"p{pno:03d}: lines={lines:3d} deskew={ang:+.2f}° "
                  f"ocr={t_ocr:.1f}s total={t_tot:.1f}s", flush=True)
            log["pages"].append({"pdf": os.path.basename(pdf), "page": pno,
                                 "lines": lines, "deskew": round(ang, 2),
                                 "ocr_s": round(t_ocr, 2)})
            if pno in (START, START + 9) and pno <= needed:
                # before/after preview
                doc[pno - 1].get_pixmap(dpi=100).save(f"{OUT}/previews/before_p{pno:03d}.png")
                outdoc[-1].get_pixmap(dpi=100).save(f"{OUT}/previews/after_p{pno:03d}.png")
            total_pages += 1
            if total_pages >= MAX_PAGES:
                break
        doc.close()
        if total_pages >= MAX_PAGES:
            break

    outdoc.subset_fonts()
    outpdf = f"{OUT}/book_p{START:03d}-{START + total_pages - 1:03d}.pdf"
    outdoc.save(outpdf, deflate=True)
    log["output"] = outpdf
    log["output_size_mb"] = round(os.path.getsize(outpdf) / 1e6, 2)
    log["total_pages"] = total_pages
    log["elapsed_s"] = round(time.perf_counter() - t_start, 1)
    logfile = f"{OUT}/run_log_p{START:03d}-{START + total_pages - 1:03d}.json"
    with open(logfile, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(f"[done] {total_pages} pages -> {outpdf} ({log['output_size_mb']}MB) "
          f"in {log['elapsed_s']}s; warnings={log['warnings']}")


if __name__ == "__main__":
    main()