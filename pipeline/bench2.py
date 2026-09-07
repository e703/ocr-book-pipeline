#!/usr/bin/env python3
"""PaddleOCR in-container benchmark v2: 4 configs, warmup, per-page render+OCR timing, accuracy."""
import json, os, re, sys, time
import pymupdf

PDF = os.environ.get("PDF", "/tmp/ocrbench/sample_mixed.pdf")
GT_FILE = os.environ.get("GT", "/tmp/ocrbench/ground_truth.txt")
WORK = "/tmp/ocrbench/work"
os.makedirs(WORK, exist_ok=True)

def load_gt():
    pages = {}
    cur = None
    for line in open(GT_FILE, encoding="utf-8"):
        if line.startswith("=== PAGE"):
            cur = []
            pages[int(line.split()[2])] = cur
        elif cur is not None:
            cur.append(line.rstrip("\n"))
    return {k: "\n".join(v) for k, v in pages.items()}

def render_pages(dpi=200):
    doc = pymupdf.open(PDF)
    out = []
    for p in doc:
        t0 = time.perf_counter()
        pix = p.get_pixmap(dpi=dpi)
        img = f"{WORK}/page{p.number+1}.png"
        pix.save(img)
        out.append((img, time.perf_counter() - t0))
    return out

def norm(s):
    return re.sub(r"\s+", "", s)

def char_recall(gt, ocr):
    from collections import Counter
    g, o = norm(gt), norm(ocr)
    if not g:
        return 1.0
    gc, oc = Counter(g.lower()), Counter(o.lower())
    return sum(min(gc[k], oc.get(k, 0)) for k in gc) / len(g)

CONFIGS = [
    ("v6", "v6 (out-of-box default)", None, {"enable_mkldnn": False}),
    ("v5", "v5 (default: server rec)", "PP-OCRv5", {}),
    ("v5m", "v5 mobile (lightweight)", "PP-OCRv5",
     {"text_detection_model_name": "PP-OCRv5_mobile_det",
      "text_recognition_model_name": "PP-OCRv5_mobile_rec"}),
    ("v4", "v4 (default: mobile)", "PP-OCRv4", {}),
]

def run(ver, tag, overrides, pages, GT):
    from paddleocr import PaddleOCR
    kw = dict(lang="ch",
              use_doc_orientation_classify=False,
              use_doc_unwarping=False,
              use_textline_orientation=True,
              text_det_limit_side_len=960, text_det_limit_type="max",
              enable_mkldnn=os.environ.get("MKLDNN", "1") == "1")
    kw.update(overrides)
    if ver:
        kw["ocr_version"] = ver
    t0 = time.perf_counter()
    ocr = PaddleOCR(**kw)
    t_init = time.perf_counter() - t0

    # untimed warmup on page 1
    ocr.predict(pages[0][0])

    res = {"tag": tag, "init_s": round(t_init, 2), "pages": []}
    for i, (img, render_s) in enumerate(pages, 1):
        t0 = time.perf_counter()
        out = ocr.predict(img)
        t_ocr = time.perf_counter() - t0
        lines = [t for r in out for t in r.get("rec_texts", [])]
        text = "\n".join(lines)
        res["pages"].append({
            "page": i, "render_s": round(render_s, 3), "ocr_s": round(t_ocr, 2),
            "text_lines": len(lines), "char_recall": round(char_recall(GT[i], text), 4),
        })
        with open(f"{WORK}/ocr_{tag}_p{i}.txt", "w", encoding="utf-8") as f:
            f.write(text)
    return res

def main():
    dpi = int(os.environ.get("DPI", "200"))
    only = set(os.environ.get("ONLY", "").split(",")) - {""}
    configs = [c for c in CONFIGS if not only or c[0] in only]
    pages = render_pages(dpi)
    print(f"[render] {len(pages)} pages @ {dpi}dpi: " +
          ", ".join(f"p{i+1}={s:.2f}s" for i, (_, s) in enumerate(pages)), flush=True)
    results = []
    for stag, tag, ver, ov in configs:
        r = run(ver, stag, ov, pages, GT)
        results.append(r)
        tot = sum(p["total_s"] for p in r["pages"]) if False else None
        ocr_tot = sum(p["ocr_s"] for p in r["pages"])
        rec = sum(p["char_recall"] for p in r["pages"]) / len(r["pages"])
        print(f"[{tag}] init={r['init_s']}s | ocr/page=" +
              ", ".join(f"p{p['page']}={p['ocr_s']}s" for p in r["pages"]) +
              f" | ocr_total={ocr_tot:.1f}s | avg_recall={rec:.3f}", flush=True)
    with open(f"{WORK}/summary2.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("[done]", flush=True)

if __name__ == "__main__":
    GT = load_gt()
    main()