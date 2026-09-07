#!/usr/bin/env python3
"""Minimal PaddleOCR serving client (stdlib only, works anywhere)."""
import base64
import json
import sys
import time
import urllib.request

API = "http://localhost:8080/ocr"


def call_ocr(file_path, api=API, timeout=300):
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    ext = file_path.lower().rsplit(".", 1)[-1]
    file_type = 0 if ext in ("pdf", "tif", "tiff") else 1  # 0=PDF, 1=IMAGE
    body = json.dumps({"file": b64, "fileType": file_type}).encode()
    req = urllib.request.Request(api, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = json.loads(resp.read())
    dt = time.time() - t0
    return out, dt


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/ocrbench/sample_mixed_image.pdf"
    out, dt = call_ocr(path)
    print(f"[{dt:.1f}s wall] errorCode={out.get('errorCode')} errorMsg={out.get('errorMsg')}")
    if out.get("errorCode") != 0:
        print(json.dumps(out, ensure_ascii=False)[:500])
        return
    for i, page in enumerate(out["result"]["ocrResults"], 1):
        pr = page["prunedResult"]
        texts = pr["rec_texts"]
        print(f"--- page {i} ({len(texts)} lines) ---")
        print("\n".join(texts))


if __name__ == "__main__":
    main()