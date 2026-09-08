# ocr-book-pipeline

基于 PaddleOCR 3.x + Docker 的中英混合文档 OCR 与「扫描书 → 可搜索 PDF」处理管线。

适用于无独立显卡的 CPU 主机（本项目实测机型：Intel i7-6650U 双核四线程 / 15GB 内存）。

## 目录结构

```
ocr-book-pipeline/
├── README.md              本文件
├── PERFORMANCE.md         性能测试报告（基准数据、配置对比、踩坑记录）
├── deploy/                服务化部署（REST API，供其他程序/大模型调用）
│   ├── Dockerfile         paddle:3.0.0 + paddleocr 3.7.0 + serving 插件 + v4 mobile 配置
│   ├── docker-compose.yml 一键启动（8080 端口、模型缓存、重启策略）
│   ├── ocr_v4.yml         OCR 产线配置（钉死 PP-OCRv4 mobile，解除 10 页限制）
│   └── client.py          纯标准库调用示例（PDF/图片自适应 fileType）
└── pipeline/              批处理脚本
    ├── book_real.py       扫描书 → 可搜索 PDF（真实书管线：原图直取、去倾斜、
    │                      CLAHE 增强、读取顺序排序[横排/竖排/多栏]、文字层重建；
    │                      双轨输出：OCR 吃原图增强链，可见背景去透色+框内合成
    │                      原生笔画，封面[第1页]保留原始彩图）
    ├── book_rebuild.py    同管线的最小演示版（自建模拟扫描件闭环验证）
    ├── bench2.py          基准测试脚本（多配置/多 DPI/召回率评估）
    └── make_pdf*.py       测试样本生成器
```

## 快速开始

### 1. 构建镜像

```bash
cd deploy && docker build -t ocrserv:local .
```

### 2. 启动 OCR 服务（REST API）

```bash
cd deploy && docker compose up -d
# 或手动：
docker run -d --name paddleocr -p 8080:8080 \
  -v $PWD/ocr_v4.yml:/opt/ocr_v4.yml \
  -v /root/.paddlex 的宿主目录:/root/.paddlex \
  --restart unless-stopped --memory 2g ocrserv:local
```

服务就绪后：
- 健康检查：`curl http://localhost:8080/health`
- 接口文档：`http://localhost:8080/docs`

调用（PDF 用 `fileType=0`，图片用 `fileType=1`，详见 deploy/client.py）：

```bash
python3 deploy/client.py 某文件.pdf
```

### 3. 扫描书转可搜索 PDF

```bash
# 目录约定：SRC=输入扫描 PDF 目录，OUT=输出目录
# 环境变量：START_PAGE / MAX_PAGES / THREADS(建议=CPU 线程数/进程数)
#           CLEAN=0 关闭可见背景去透色；KEEP_COVER=0 封面也走统一处理
# 中文字体挂载：wqy-zenhei.ttc（可从系统 /usr/share/fonts/truetype/wqy/ 复制）
docker run --rm \
  -v /home/alan/workspace/book:/book \
  -v /home/alan/workspace/ocr-book-pipeline/pipeline:/pipeline \
  -v /path/to/fonts:/tmp/ocrbench/fonts \
  -v <你的模型缓存目录>:/root/.paddlex \
  ocrserv:local \
  bash -c "cd /pipeline && SRC=/book/sources OUT=/book/targets START_PAGE=1 MAX_PAGES=104 THREADS=1 python book_real.py"
```

输出：`<OUT>/book_pNNN-NNN.pdf`（可搜索 PDF）+ `<OUT>/text/pNNN.txt`（逐页识别文本）
+ `<OUT>/previews/`（处理前后对比图）+ `run_log_pNNN-NNN.json`（逐页日志）。

## 关键设计决策（为什么这样配）

| 决策 | 原因 |
|---|---|
| 镜像用 paddle:3.0.0 而非最新 3.3.1 | 3.3.1 的 MKLDNN 对所有模型崩溃（oneDNN 属性转换 bug）；3.0.0 上 MKLDNN 正常且快约 5 倍 |
| 模型钉死 PP-OCRv4 mobile | paddleocr 3.7 默认 PP-OCRv6 在 3.0.0 镜像上无法加载；v5+中文默认 server 识别模型慢 5 倍且精度几乎无提升 |
| 服务配置解除 10 页上限 | 自部署 API 默认单请求只处理 10 页 PDF（静默截断，坑） |
| OCR 喂 CLAHE 增强灰度图 | 噪声扫描件上全局 Otsu 二值化阈值不稳定（116↔208 跳变）会导致检测全灭 |
| OCR 输入保持原图增强链，去透色只作用于可见背景 | 实测清洗图喂 OCR 是负优化：重透色页的脚注/页码等小字被抹掉（6 行→4 行乱码）；双轨后文字层逐字零回归 |
| 可见背景 = 去透色平底 + OCR 框内合成原生笔画 | 单纯暗度斜坡会吃掉笔画边缘过渡像素（文字"缺边"）；用检测框做掩膜把原图笔画原样叠回，形状逐像素保真，页边/行距保持干净 |
| 封面（第 1 页）保持原始彩图 | 封面是彩色设计页，灰度化+去倾斜都会破坏观感；KEEP_COVER=0 可关闭 |
| 文字层用真实字体 + subset_fonts() | 内置 CJK 字体 china-s 对 ASCII 提取不完整；整嵌 TTC 每页 6MB；子集化后约 200KB/页 |
| 背景图用原扫描原生分辨率 + JPEG q88 | 早期 75% 缩放 + 5×5 中值滤波把背景压到实际 90dpi，观感一片糊（锐度 12）；改原生分辨率后锐度提升约 10 倍，代价是体积 84KB/页 → 190KB/页 |

## 已知限制

- 无 GPU 时精度/速度受 CPU 算力限制；本书实测字符召回约 96%（个别字误识属正常）
- 封面艺术字、CIP 版权页多栏小块信息识别质量差（通用 OCR 天然弱点）；封面可见页保持原始彩图不做处理
- 双栏/竖排书：已支持读取顺序排序，但复杂版面仍建议抽检
- 极重透色页（如 p120 型）：页边仍有 ~3% 残留；个别极淡小字视觉上变浅，但文字层不受影响（搜索/复制照常）
- 服务无鉴权，仅限内网或自行加 Nginx 反向代理/token