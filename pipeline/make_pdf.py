#!/usr/bin/env python3
"""Generate a mixed Chinese/English test PDF for OCR benchmarking."""
import pymupdf

FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

# (heading, [lines...]) - each line contains mixed zh/en terms
SECTIONS = [
    ("设备概述 Device Overview", [
        "本设备采用 Intel Core i7-6650U 处理器，支持 4 线程并行计算，主频 2.20GHz，功耗 15W。",
        "系统基于 Docker 容器化部署，镜像版本 paddlepaddle/paddle:3.0.0，用于文档 OCR 识别。",
        "设备支持 TCP/IP、HTTPS 和 WebSocket 协议，默认监听端口 8080，可配置 TLS 证书。",
        "The device supports both Simplified Chinese and English text recognition with accuracy over 95%.",
    ]),
    ("系统架构 System Architecture", [
        "整体架构分为三层：前端 UI、API 网关和推理服务，推理服务基于 PaddleOCR 3.x 构建。",
        "图像预处理阶段采用 OpenCV 完成自适应二值化、降噪和倾斜校正，然后送入检测模型。",
        "PP-OCRv5 检测模型参数量 3.5M，识别模型 12.9M，FP32 精度下单页推理约需 2.5 秒。",
        "数据流：PDF 页面 -> PyMuPDF 渲染 -> 文本检测 Detection -> 方向分类 -> 文本识别 Recognition。",
        "The batch processing queue uses Redis as message broker, supporting up to 100 concurrent jobs.",
    ]),
    ("性能指标 Performance Metrics", [
        "在 1080P 分辨率测试样本上，检测模型单张耗时 380ms，识别模型耗时 210ms，合计约 590ms。",
        "CPU 占用率峰值 92%，内存占用 1.8GB，GPU 利用率 0%（本机无独立显卡）。",
        "端到端延迟：P95 = 3.4 秒/页，P99 = 5.1 秒/页，吞吐量 17.6 页/分钟。",
        "Device-ID: OCR-2026-0017，固件版本 v3.2.1，校验码 SHA256: 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08。",
        "与 GPU 方案对比：RTX 4090 上单页 0.12 秒，加速比约 21 倍；T4 上 0.35 秒，加速比约 7 倍。",
    ]),
    ("附录：英语术语表 Appendix: English Glossary", [
        "OCR: Optical Character Recognition 光学字符识别；DPI: Dots Per Inch 每英寸点数。",
        "PaddleOCR 提供 PP-OCRv4 与 PP-OCRv5 两代模型，v5 在长文本和表格场景提升明显。",
        "Compatibility: supports Windows 10/11, Ubuntu 20.04/22.04, macOS 12+, and Docker Desktop.",
        "调用示例：paddleocr --image_dir=invoice.pdf --lang=ch --ocr_version=PP-OCRv5，输出 JSON 格式。",
        "联系邮箱 support@paddleocr.example.com，官网 https://www.paddleocr.ai/，版本 3.0.0-alpha。",
    ]),
]

def main():
    doc = pymupdf.open()
    gt_pages = []
    for section_title, lines in SECTIONS:
        page = doc.new_page(width=595, height=842)  # A4
        gt = []
        y = 56
        # title
        page.insert_text((56, y), section_title, fontsize=15, fontname="wqy", fontfile=FONT)
        gt.append(section_title)
        y += 34
        for line in lines:
            page.insert_text((56, y), line, fontsize=10.5, fontname="wqy", fontfile=FONT)
            gt.append(line)
            y += 18
        gt_pages.append("\n".join(gt))
    doc.save("/tmp/ocrbench/sample_mixed.pdf")
    with open("/tmp/ocrbench/ground_truth.txt", "w", encoding="utf-8") as f:
        for i, g in enumerate(gt_pages):
            f.write(f"=== PAGE {i+1} ===\n{g}\n\n")
    print(f"pages={len(gt_pages)} chars_total={sum(len(g.replace(chr(10),'')) for g in gt_pages)}")
    print("saved /tmp/ocrbench/sample_mixed.pdf")

if __name__ == "__main__":
    main()