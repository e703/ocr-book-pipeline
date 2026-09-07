# PaddleOCR serving image for CPU hosts
# Base: official paddle CPU image (3.0.0 has working MKLDNN; 3.3.1's MKLDNN crashes)
# NOT 3.3.1: on 3.3.1 all PP-OCR models crash with oneDNN attribute errors,
# and PP-OCRv6 models cannot even load on 3.0.0 -- that's why this pins v4 mobile.
FROM paddlepaddle/paddle:3.0.0

RUN python -m pip install --no-cache-dir -q "paddleocr>=3.0" pymupdf \
 && paddlex --install serving

COPY ocr_v4.yml /opt/ocr_v4.yml

EXPOSE 8080
CMD ["paddlex", "--serve", "--pipeline", "/opt/ocr_v4.yml", "--host", "0.0.0.0", "--port", "8080"]