#!/usr/bin/env python3
"""Generate a mixed image+text PDF (figures with in-image labels, photos, table, scanned page)."""
import random
import pymupdf

random.seed(42)
FONT = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"

def noise_pixmap(w, h, base=130, amp=60, seed=42):
    rnd = random.Random(seed)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, w, h), False)
    for y in range(h):
        for x in range(w):
            v = max(0, min(255, base + rnd.randint(-amp, amp)))
            pix.set_pixel(x, y, (v, v, v))
    return pix

def make_figure(kind):
    """Render a figure (with in-image text) to PNG, return (path, inimage_text)."""
    doc = pymupdf.open()
    p = doc.new_page(width=760, height=420)
    intext = []
    if kind == "photo":
        # noisy photo background
        p.insert_image(pymupdf.Rect(0, 0, 760, 420), pixmap=noise_pixmap(760, 420, 135, 70, seed=7))
        # darker object blob
        blob = noise_pixmap(300, 180, 70, 45, seed=8)
        p.insert_image(pymupdf.Rect(230, 90, 530, 270), pixmap=blob)
        lbl = "设备样机 Prototype"
        p.insert_text((270, 330), lbl, fontsize=22, fontname="wqy", fontfile=FONT, color=(0.05, 0.05, 0.05))
        intext.append(lbl)
        lbl2 = "Serial No. PRT-2026-A1"
        p.insert_text((250, 365), lbl2, fontsize=16, fontname="wqy", fontfile=FONT, color=(0.1, 0.1, 0.1))
        intext.append(lbl2)
    else:  # bar chart
        p.draw_rect(pymupdf.Rect(0, 0, 760, 420), color=(1, 1, 1), fill=(1, 1, 1))
        bars = [(80, 150, "Q1"), (240, 220, "Q2"), (400, 180, "Q3"), (560, 260, "Q4")]
        for x, h, q in bars:
            p.draw_rect(pymupdf.Rect(x, 360 - h, x + 100, 360), color=(0.1, 0.25, 0.6), fill=(0.2, 0.4, 0.8))
            p.insert_text((x + 30, 360 - h - 12), str(h), fontsize=15, fontname="wqy", fontfile=FONT)
            p.insert_text((x + 32, 385), q, fontsize=15, fontname="wqy", fontfile=FONT)
            intext.append(str(h))
            intext.append(q)
        p.draw_line(pymupdf.Point(40, 360), pymupdf.Point(720, 360), color=(0, 0, 0), width=1.5)
        p.draw_line(pymupdf.Point(40, 360), pymupdf.Point(40, 40), color=(0, 0, 0), width=1.5)
        t = "季度出货量（台） Quarterly Shipments"
        p.insert_text((180, 50), t, fontsize=20, fontname="wqy", fontfile=FONT, color=(0, 0, 0))
        intext.append(t)
        ax = "Q1 150，Q2 220，Q3 180，Q4 260；累计 810 台，同比增长 18%"
        p.insert_text((60, 408), ax, fontsize=14, fontname="wqy", fontfile=FONT, color=(0.2, 0.2, 0.2))
        intext.append(ax)
    path = f"/tmp/ocrbench/work/fig_{kind}.png"
    p.get_pixmap(dpi=110).save(path)
    doc.close()
    return path, intext

def main():
    photo, ptxt = make_figure("photo")
    chart, ctxt = make_figure("chart")
    doc = pymupdf.open()
    gt_pages = []

    # ---------- PAGE 1: title + wrapped paragraphs + centered photo ----------
    page = doc.new_page(width=595, height=842)
    gt = []
    t = "图文混排测试文档 Mixed-Media Document"
    page.insert_text((50, 56), t, fontsize=16, fontname="wqy", fontfile=FONT)
    gt.append(t)
    para1 = ("本页同时包含文字段落与嵌入式照片。该设备支持 4K 视频采集、GPU 加速推理"
             "以及 Docker 容器化部署，适用于工业质检、医疗影像 OCR 等场景，"
             "支持 TCP/IP、MQTT 协议，整机功耗 45W。")
    page.insert_textbox(pymupdf.Rect(50, 84, 545, 168), para1, fontsize=10.5,
                        fontname="wqy", fontfile=FONT, align=0)
    gt.append(para1)
    page.insert_image(pymupdf.Rect(122, 180, 473, 342), filename=photo)
    cap = "图 1：设备样机实物照片，序列号 PRT-2026-A1"
    page.insert_text((122, 362), cap, fontsize=10.5, fontname="wqy", fontfile=FONT)
    gt.append(cap)
    para2 = ("产品外壳采用铝合金 CNC 加工，表面阳极氧化处理，防护等级 IP65；"
             "内置 128GB 固态硬盘与 16GB 内存，接口包括 USB 3.2、HDMI 2.1、"
             "千兆以太网，运行温度范围 -10℃ 至 50℃。参考认证：CE、FCC、RoHS。")
    page.insert_textbox(pymupdf.Rect(50, 390, 545, 520), para2, fontsize=10.5,
                        fontname="wqy", fontfile=FONT, align=0)
    gt.append(para2)
    page.insert_text((50, 560), "设备主要技术参数（Table 1）", fontsize=12, fontname="wqy", fontfile=FONT)
    gt.append("设备主要技术参数（Table 1）")
    rows = [("参数 Parameter", "规格 Specification"),
            ("处理器 CPU", "i7-6650U 四线程 2.2GHz"),
            ("内存 RAM", "16GB DDR4 2400MHz"),
            ("存储 Storage", "512GB NVMe SSD"),
            ("网络 Network", "双千兆网口，Wi-Fi 6"),
            ("接口 I/O", "USB-C ×2，RS-485 ×1")]
    y = 590
    for i, (a, b) in enumerate(rows):
        page.draw_rect(pymupdf.Rect(50, y, 545, y + 26), color=(0, 0, 0), width=0.8,
                       fill=(0.92, 0.94, 0.96) if i == 0 else (1, 1, 1))
        page.insert_text((56, y + 17), a, fontsize=10, fontname="wqy", fontfile=FONT)
        page.insert_text((300, y + 17), b, fontsize=10, fontname="wqy", fontfile=FONT)
        gt.append(a)
        gt.append(b)
        y += 26
    gt_pages.append("\n".join(gt))

    # ---------- PAGE 2: chart + photo side by side + bullets ----------
    page = doc.new_page(width=595, height=842)
    gt = []
    h = "销售数据分析 Sales Data Analysis"
    page.insert_text((50, 56), h, fontsize=15, fontname="wqy", fontfile=FONT)
    gt.append(h)
    page.insert_image(pymupdf.Rect(40, 76, 340, 264), filename=chart)
    cap = "图 2：年度出货量柱状图，数据来源 ERP 系统"
    page.insert_text((40, 284), cap, fontsize=10.5, fontname="wqy", fontfile=FONT)
    gt.append(cap)
    page.insert_image(pymupdf.Rect(355, 76, 555, 264), filename=photo)
    cap2 = "图 3：现场安装照片"
    page.insert_text((355, 284), cap2, fontsize=10.5, fontname="wqy", fontfile=FONT)
    gt.append(cap2)
    body = ["总结 Summary：", "1. Q4 出货 260 台为全年最高，Q1 仅 150 台，季节性明显；",
            "2. 华东区贡献 45% 营收，重点客户为 JD Logistics 与 SF Express；",
            "3. 2026 年目标 1400 台，计划新增 2 条产线，扩产 30%；",
            "4. 售后备件库存周转率提升至 6.8 次/年，客户满意度 96.5%。"]
    y = 340
    for line in body:
        page.insert_text((50, y), line, fontsize=10.5, fontname="wqy", fontfile=FONT)
        gt.append(line)
        y += 20
    box = "注意：柱状图数值标注同时包含中文与英文缩写，OCR 需正确识别图内文字。"
    page.draw_rect(pymupdf.Rect(50, y + 10, 545, y + 44), color=(0.7, 0.2, 0.2), width=1, fill=(1, 0.97, 0.94))
    page.insert_text((58, y + 32), box, fontsize=10, fontname="wqy", fontfile=FONT, color=(0.6, 0.1, 0.1))
    gt.append(box)
    gt_pages.append("\n".join(gt))

    # ---------- PAGE 3: full-page "scanned" raster (dense mixed text) ----------
    doc2 = pymupdf.open()
    sp = doc2.new_page(width=595, height=842)
    scan_text = [
        "验收报告 Acceptance Report（扫描件）",
        "项目名称：PaddleOCR 中英混合识别项目 Project Name: Mixed CN-EN OCR",
        "验收日期：2026-08-30 交付方：AI Studio 有限公司（Supplier: AI Studio Co., Ltd.）",
        "验收标准：300dpi 扫描文档字符识别准确率不低于 95%，单页处理时间不超过 5 秒。",
        "测试样本：含印刷体、手写数字、表格、公式（E=mc²）以及嵌入式图片共 120 页。",
        "检测项 Inspection Item | 结果 Result | 结论 Conclusion",
        "1. 中文字符识别率 Chinese character accuracy | 97.8% | 通过 PASS",
        "2. 英文术语与数字识别率 English term & digit accuracy | 98.6% | 通过 PASS",
        "3. 图文混排版面还原 Layout reconstruction | 良好 Good | 通过 PASS",
        "4. 表格结构识别 Table structure | 优 Excellent | 通过 PASS",
        "5. 批量处理吞吐 Batch throughput | 42 页/分钟 PPM | 通过 PASS",
        "存在问题备注：低对比度扫描件（灰度<120）识别率下降约 4%，建议开启图像增强。",
        "签名 Signature：张伟 Zhang Wei（项目主管）日期 Date：2026-08-30",
    ]
    y = 70
    for i, line in enumerate(scan_text):
        size = 13 if i == 0 else 10.5
        sp.insert_text((50, y), line, fontsize=size, fontname="wqy", fontfile=FONT)
        y += 24
    scan_png = "/tmp/ocrbench/work/scan_page.png"
    sp.get_pixmap(dpi=150).save(scan_png)  # 150dpi render = scanned look
    doc2.close()

    page = doc.new_page(width=595, height=842)
    page.insert_image(pymupdf.Rect(0, 0, 595, 842), filename=scan_png)
    gt_pages.append("\n".join(scan_text))

    doc.save("/tmp/ocrbench/sample_mixed_image.pdf")
    with open("/tmp/ocrbench/ground_truth_image.txt", "w", encoding="utf-8") as f:
        for i, g in enumerate(gt_pages):
            f.write(f"=== PAGE {i+1} ===\n{g}\n\n")
    chars = sum(len(g.replace("\n", "")) for g in gt_pages)
    print(f"pages={len(gt_pages)} chars_total={chars} (incl in-image labels {len(ptxt)+len(ctxt)})")
    print("saved /tmp/ocrbench/sample_mixed_image.pdf")

if __name__ == "__main__":
    main()