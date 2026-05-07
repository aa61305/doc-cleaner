# -*- coding: utf-8 -*-
"""
PDF 處理：把 PDF 每一頁渲染成影像 → 套用 image_processor → 重組回 PDF。

對掃描型 PDF (例如複印機產出，無內嵌文字層) 最為適用。
若 PDF 內含可選取文字，會以渲染影像取代該頁；不過原文字本身不會被修改 ——
只是該頁變成「乾淨的影像版」。

依存：PyMuPDF (fitz)、Pillow、image_processor (本專案)
"""

from __future__ import annotations

import io
import os
from typing import Callable

import fitz  # PyMuPDF
from PIL import Image

from image_processor import CleanConfig, DEFAULT_CONFIG, clean_pil


def process_pdf(
    src_path: str,
    dst_path: str,
    cfg: CleanConfig = DEFAULT_CONFIG,
    dpi: int = 200,
    jpeg_quality: int = 88,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> dict:
    """
    處理單一 PDF。

    參數:
      src_path: 來源 PDF
      dst_path: 輸出 PDF
      cfg:       影像處理設定
      dpi:       渲染解析度，預設 200 對掃描文件足夠且檔案不會過大
      jpeg_quality: JPEG 壓縮品質 (1-95)
      progress_cb: 回呼 (page_idx_1based, total_pages, message)

    回傳: {"pages": N, "angles": [...]}
    """
    src = fitz.open(src_path)
    out = fitz.open()  # 空 PDF

    angles = []
    n = src.page_count

    try:
        for i in range(n):
            page = src[i]

            # 渲染這一頁為 PIL Image
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            mode = "RGB" if pix.n >= 3 else "L"
            img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)

            # 套用清理
            cleaned, info = clean_pil(img, cfg)
            angles.append(info.get("skew_angle", 0.0))

            # 把處理後影像塞進新 PDF (以原頁面尺寸保留版面)
            buf = io.BytesIO()
            # 灰階存 JPEG 也可以；保險起見先轉 RGB 再存
            if cleaned.mode != "RGB":
                cleaned_rgb = cleaned.convert("RGB")
            else:
                cleaned_rgb = cleaned
            cleaned_rgb.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
            buf.seek(0)

            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=buf.getvalue())

            if progress_cb:
                progress_cb(
                    i + 1, n,
                    f"第 {i+1}/{n} 頁 — 轉正 {info.get('skew_angle', 0.0):+.2f}°",
                )

        # 儲存
        out.save(dst_path, garbage=4, deflate=True)
    finally:
        out.close()
        src.close()

    return {"pages": n, "angles": angles}


# 命令列測試
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python pdf_processor.py <in.pdf> <out.pdf> [dpi]")
        sys.exit(1)
    dpi = int(sys.argv[3]) if len(sys.argv) > 3 else 200

    def _cb(i, n, msg):
        print(f"[{i}/{n}] {msg}")

    info = process_pdf(sys.argv[1], sys.argv[2], dpi=dpi, progress_cb=_cb)
    print(info)
