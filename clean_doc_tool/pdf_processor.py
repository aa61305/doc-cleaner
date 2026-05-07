# -*- coding: utf-8 -*-
"""
PDF 處理：把 PDF 每一頁渲染成影像 → 套用 image_processor → 重組回 PDF。

對掃描型 PDF (例如複印機產出，無內嵌文字層) 最為適用。

記憶體最佳化：
  ‧ 每頁處理完強制釋放暫存物件並 gc.collect()
  ‧ 偵測到「實質為黑白」的頁面自動轉灰階 (記憶體用量減 2/3)
  ‧ PyMuPDF pixmap 用完即釋放，避免 hold 住 native memory

依存：PyMuPDF (fitz)、Pillow、image_processor (本專案)
"""

from __future__ import annotations

import gc
import io
import os
from typing import Callable

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from image_processor import CleanConfig, DEFAULT_CONFIG, clean_pil


def _is_effectively_grayscale(img: Image.Image, sample_size: int = 64) -> bool:
    """
    快速判斷一張 RGB 圖是否「實質上」就是灰階 (R≈G≈B)。
    從圖中隨機取點抽樣，若所有取樣點 RGB 三通道幾乎相等，就視為灰階。
    """
    if img.mode != "RGB":
        return img.mode in ("L", "1")
    # 縮小到 64×64 後檢查 RGB 通道差異
    small = img.resize((sample_size, sample_size), Image.NEAREST)
    arr = np.asarray(small, dtype=np.int16)  # int16 避免溢位
    # 三個通道兩兩之差
    rg = np.abs(arr[..., 0] - arr[..., 1])
    gb = np.abs(arr[..., 1] - arr[..., 2])
    rb = np.abs(arr[..., 0] - arr[..., 2])
    max_diff = max(rg.max(), gb.max(), rb.max())
    return max_diff <= 5  # 容忍 JPEG 雜訊


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
      cfg:       影像處理設定。若 cfg.force_grayscale=True 強制以灰階處理。
      dpi:       渲染解析度，預設 200
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

            # ---- 1) 渲染這一頁 ----
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pw, ph = pix.width, pix.height
            mode = "RGB" if pix.n >= 3 else "L"
            img = Image.frombytes(mode, (pw, ph), pix.samples)
            # 立刻釋放 pixmap，省 native memory
            pix = None
            del pix

            # ---- 2) 灰階優化：若實質為黑白文件，轉灰階節省 2/3 記憶體 ----
            grayscale_mode = cfg.force_grayscale
            if not grayscale_mode and img.mode == "RGB":
                if _is_effectively_grayscale(img):
                    grayscale_mode = True

            if grayscale_mode and img.mode != "L":
                img2 = img.convert("L")
                img.close()
                img = img2

            # ---- 3) 套用清理 ----
            cleaned, info = clean_pil(img, cfg)
            angles.append(info.get("skew_angle", 0.0))

            # 原圖用完了
            try:
                img.close()
            except Exception:
                pass
            img = None
            del img

            # ---- 4) 編碼 JPEG 嵌進新 PDF ----
            buf = io.BytesIO()
            if cleaned.mode == "L":
                # 灰階直接存灰階 JPEG，檔案小、記憶體用量低
                cleaned.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
            else:
                if cleaned.mode != "RGB":
                    cleaned_rgb = cleaned.convert("RGB")
                    cleaned.close()
                    cleaned = cleaned_rgb
                cleaned.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)

            jpg_bytes = buf.getvalue()
            buf.close()
            try:
                cleaned.close()
            except Exception:
                pass
            cleaned = None
            buf = None
            del cleaned, buf

            # ---- 5) 加進輸出 PDF ----
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=jpg_bytes)
            jpg_bytes = None
            del jpg_bytes

            # ---- 6) 強制 GC，把這頁的暫存徹底回收 ----
            gc.collect()

            if progress_cb:
                progress_cb(
                    i + 1, n,
                    f"第 {i+1}/{n} 頁 — 轉正 {info.get('skew_angle', 0.0):+.2f}°"
                    + ("（灰階）" if grayscale_mode else ""),
                )

        # 儲存
        out.save(dst_path, garbage=4, deflate=True)
    finally:
        out.close()
        src.close()
        gc.collect()

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
