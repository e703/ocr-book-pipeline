#!/usr/bin/env python3
"""Scan book -> enhanced searchable PDF pipeline (demo on this host).

Stages per page: render 300dpi -> denoise+CLAHE -> deskew -> OCR (PP-OCRv4 mobile)
-> rebuild PDF: enhanced image as background + invisible text layer at OCR boxes.
"""
import os
import time
import cv2
import numpy as np
import pymupdf

FONT = "/tmp/ocrbench/fonts/wqy-zenhei.ttc"
BASE = "/tmp/ocrbench/book"
SCAN_PDF = f"{BASE}/scan_book.pdf"
OUT_PDF = f"{BASE}/book_rebuilt.pdf"
GT_FILE = f"{BASE}/gt.txt"

BOOK = [
    ("第一章 绪论 Chapter 1 Introduction", [
        "本书系统介绍光学字符识别（OCR）技术在现代文档处理中的应用，涵盖检测、识别与版面分析。",
        "OCR 技术经历了从传统图像处理到深度学习驱动的范式转变，PaddleOCR 是其中的代表性开源框架。",
        "本书面向工程师与研究人员，假定读者具备 Python 与机器学习基础知识，版本以 PaddleOCR 3.x 为准。",
        "The core pipeline consists of text detection, text recognition, and layout analysis modules."]),
    ("第二章 图像预处理 Chapter 2 Image Preprocessing", [
        "扫描件常见退化包括倾斜、灰度不均、噪点与低对比度，预处理质量直接影响后续识别准确率。",
        "去倾斜（deskew）通过估计文本行主轴角度并反向旋转实现，常用方法有投影轮廓法与 Hough 变换。",
        "对比度增强推荐 CLAHE 算法，其限制对比度直方图均衡可在抑制噪声的同时提升暗部细节。",
        "Denoising with Gaussian blur and adaptive thresholding are two standard techniques in practice."]),
    ("第三章 文本检测 Chapter 3 Text Detection", [
        "PP-OCRv4 检测模型基于可微分二值化（DBNet）架构，输出任意四边形的文本区域。",
        "检测后处理包括阈值化、膨胀与多边形拟合，unclip_ratio 参数控制文本框向外扩展程度。",
        "对于密集排版的书页，建议将 limit_side_len 设为 960 并启用 textline orientation 分类。",
        "The detection model runs on a 960px downscaled image and typically takes 300-500ms on CPU."]),
    ("第四章 文本识别 Chapter 4 Text Recognition", [
        "识别模型将每个文本行图像映射为字符序列，中文场景下字典包含 6623 个常用字符。",
        "PP-OCRv4 mobile 识别模型参数量约 6.7M，在本机四线程 CPU 上单行耗时约 20-40ms。",
        "模型输出不含空格，中英文混排文本的词语切分需要依赖后处理或语言模型完成。",
        "Recognition accuracy on clean scans typically exceeds 98% for Simplified Chinese text."]),
    ("第五章 版面分析 Chapter 5 Layout Analysis", [
        "版面分析将页面划分为标题、正文、图片、表格等区域，是文档理解任务的基础。",
        "PP-StructureV3 产线集成了版面检测、表格识别与公式识别，可直接输出 Markdown 文档。",
        "对纯阅读场景，双图层 PDF（原图背景 + 文字层）比完全重排保留更多原始版面信息。",
        "Layout analysis enables downstream tasks such as document QA, retrieval and translation."]),
    ("第六章 可搜索 PDF Chapter 6 Searchable PDF", [
        "可搜索 PDF 将 OCR 文本以不可见文字层嵌入原图之上，实现全文检索与文本复制。",
        "PyMuPDF 支持以 render_mode=3 写入不可见文本，文字位置可映射回检测框坐标。",
        "文字层字体应嵌入中文字体（如文泉驿正黑），否则复制出的文本可能显示为乱码。",
        "A 300-page scanned book becomes fully searchable within minutes on a 4-core laptop CPU."]),
]

def gen_scan():
    """Render clean book pages, degrade (rotate + low contrast + noise), save as scan PDF."""
    doc = pymupdf.open()
    doc2 = pymupdf.open()
    gt_lines = []
    for ci, (title, paras) in enumerate(BOOK, 1):
        p = doc2.new_page(width=595, height=842)
        gt = [title]
        p.insert_text((50, 70), title, fontsize=14, fontname="wqy", fontfile=FONT)
        y = 110
        for para in paras:
            p.insert_textbox(pymupdf.Rect(50, y, 545, y + 90), para, fontsize=11,
                             fontname="wqy", fontfile=FONT)
            gt.append(para)
            y += 100
        p.insert_text((280, 810), f"— {ci} —", fontsize=10, fontname="wqy", fontfile=FONT)
        gt_lines.append("\n".join(gt))
        # render clean page at 300dpi, then degrade
        pix = p.get_pixmap(dpi=300)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # rotate ~2.5 deg
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.5, 1.0)
        gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
        # low contrast + noise + gray cast
        gray = cv2.convertScaleAbs(gray, alpha=0.75, beta=-25)
        noise = np.random.default_rng(ci).normal(0, 14, gray.shape).astype(np.float32)
        gray = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        page = doc.new_page(width=595, height=842)
        page.insert_image(pymupdf.Rect(0, 0, 595, 842), stream=cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
    doc.save(SCAN_PDF)
    with open(GT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(gt_lines))
    print(f"[gen] {len(BOOK)} pages -> {SCAN_PDF}")

def deskew(gray):
    """Projection-profile deskew: find rotation maximizing row-projection variance."""
    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2),
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape
    best_a, best_s = 0.0, -1.0
    for a in np.arange(-6.0, 6.01, 0.25):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
        rot = cv2.warpAffine(small, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=255)
        score = float(np.var(rot.sum(axis=1, dtype=np.float32)))
        if score > best_s:
            best_s, best_a = score, a
    if abs(best_a) < 0.2:
        return gray, best_a
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_a, 1.0)
    out = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255)
    return out, best_a

def process(ocr, page_no, out, t_all):
    t = {}
    DPI = 200  # OCR working resolution
    scale = 72.0 / DPI
    t0 = time.perf_counter()
    doc = pymupdf.open(SCAN_PDF)
    pix = doc[page_no - 1].get_pixmap(dpi=DPI)
    doc.close()
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    t["render"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray, ang = deskew(gray)
    enhanced = gray
    # NOTE: feed enhanced grayscale directly to OCR. Global Otsu binarization is
    # unstable on noisy scans (threshold flips between 116/208) and kills detection.
    t["preprocess"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    res = ocr.predict(rgb)[0]
    texts = res["rec_texts"]
    boxes = res["rec_boxes"]
    t["ocr"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    OCR_DPI = 200.0
    scale = 72.0 / OCR_DPI
    h, w = enhanced.shape
    # background at 150dpi + median denoise + q80: removes speckle noise
    # (readability) and shrinks file (noise is the #1 JPEG size killer)
    bg = cv2.resize(enhanced, (int(w * 0.75), int(h * 0.75)), interpolation=cv2.INTER_AREA)
    bg = cv2.medianBlur(bg, 5)
    page = out.new_page(width=w * scale, height=h * scale)
    page.insert_image(pymupdf.Rect(0, 0, w * scale, h * scale),
                      stream=cv2.imencode(".jpg", bg, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
    for txt, box in zip(texts, boxes):
        if not txt or not txt.strip():
            continue
        pts = np.asarray(box, dtype=float).reshape(-1, 2)
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        x0, y0 = min(xs), min(ys)
        bw, bh = max(xs) - x0, max(ys) - y0
        fs = max(4.0, bh * scale * 0.92)
        # invisible text layer with real font; subset_fonts() after build keeps size small
        page.insert_text((x0 * scale, y0 * scale + fs * 0.8), txt,
                         fontname="wqy", fontfile=FONT, fontsize=fs, render_mode=3)
    t["build"] = time.perf_counter() - t0
    t_all[page_no] = t
    return len(texts), ang

def main():
    gen_scan()
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(ocr_version="PP-OCRv4", lang="ch",
                    use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=True,
                    text_det_limit_side_len=960, text_det_limit_type="max")
    out = pymupdf.open()
    t_all = {}
    for pn in range(1, len(BOOK) + 1):
        n, ang = process(ocr, pn, out, t_all)
        print(f"  page{pn}: lines={n} deskew_angle={ang:+.2f}deg "
              f"render={t_all[pn]['render']:.2f}s pre={t_all[pn]['preprocess']:.2f}s "
              f"ocr={t_all[pn]['ocr']:.2f}s build={t_all[pn]['build']:.2f}s", flush=True)
    out.subset_fonts()
    out.save(OUT_PDF, deflate=True)
    tot = {k: sum(v[k] for v in t_all.values()) for k in ("render", "preprocess", "ocr", "build")}
    print(f"[out] {OUT_PDF}  size={os.path.getsize(OUT_PDF)/1e6:.1f}MB  "
          f"stage totals: { {k: round(v,1) for k,v in tot.items()} }")

    # verify text layer extractable
    vdoc = pymupdf.open(OUT_PDF)
    extracted = "\n".join(vdoc[i].get_text() for i in range(len(vdoc)))
    gt = open(GT_FILE, encoding="utf-8").read()
    import re
    from collections import Counter
    def norm(s): return re.sub(r"\s+", "", s).lower()
    g, e = norm(gt), norm(extracted)
    gc, ec = Counter(g), Counter(e)
    recall = sum(min(gc[k], ec.get(k, 0)) for k in gc) / len(g)
    print(f"[verify] extracted_chars={len(e)} gt_chars={len(g)} char_recall={recall:.3f}")
    # previews
    for pn in (1, 3):
        sp = pymupdf.open(SCAN_PDF)[pn - 1].get_pixmap(dpi=100)
        sp.save(f"{BASE}/preview_scan_p{pn}.png")
        op = vdoc[pn - 1].get_pixmap(dpi=100)
        op.save(f"{BASE}/preview_rebuilt_p{pn}.png")
    print("[preview] saved preview_scan_p*.png / preview_rebuilt_p*.png")

if __name__ == "__main__":
    main()