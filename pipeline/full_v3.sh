#!/bin/bash
# Full 414-page rebuild v3: 4 parallel segments -> merge -> summary.
# Output to /book/targets_v3 (keeps v2 targets/ intact for comparison).
set -u
BOOK=/home/alan/workspace/book
PIPE=/home/alan/workspace/ocr-book-pipeline/pipeline
IMG=ocrserv:local
OUT=/book/targets_v3
LOG=/home/alan/workspace/book/full_v3.log

echo "=== v3 full run start $(date +%H:%M:%S) ===" | tee "$LOG"

run_seg() {
  docker run --rm -v "$BOOK":/book -v "$PIPE":/pipeline \
    -v /tmp/ocrbench/fonts:/tmp/ocrbench/fonts \
    -v /tmp/ocrbench/paddlex_cache:/root/.paddlex \
    "$IMG" bash -c "cd /pipeline && SRC=/book/sources OUT=$OUT START_PAGE=$1 MAX_PAGES=$2 THREADS=1 python book_real.py" \
    > "$BOOK/seg_$1.log" 2>&1
  echo "seg $1 exit=$?" | tee -a "$LOG"
}

run_seg 1 104 &
run_seg 105 104 &
run_seg 209 104 &
run_seg 313 104 &
wait

echo "=== merge $(date +%H:%M:%S) ===" | tee -a "$LOG"
docker run --rm -v "$BOOK":/book "$IMG" python3 -c "
import pymupdf
out = pymupdf.open()
for f in ['book_p001-104.pdf','book_p105-208.pdf','book_p209-312.pdf','book_p313-414.pdf']:
    d = pymupdf.open(f'/book/targets_v3/{f}')
    out.insert_pdf(d); d.close()
out.save('/book/targets_v3/Word的排版艺术_可搜索版.pdf', deflate=True)
print('merged pages:', out.page_count)
" >> "$LOG" 2>&1

echo "=== summary $(date +%H:%M:%S) ===" | tee -a "$LOG"
CHARS=$(cat /home/alan/workspace/book/targets_v3/text/*.txt | wc -m)
SIZE=$(du -sm /home/alan/workspace/book/targets_v3/Word的排版艺术_可搜索版.pdf | cut -f1)
echo "text_chars=$CHARS size_mb=$SIZE" | tee -a "$LOG"
echo "=== ALL DONE $(date +%H:%M:%S) ===" | tee -a "$LOG"
