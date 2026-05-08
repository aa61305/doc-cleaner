# -*- coding: utf-8 -*-
"""
PDF 處理：把 PDF 每一頁渲染成影像 → 套用 image_processor → 重組回 PDF。

對掃描型 PDF (例如複印機產出，無內嵌文字層) 最為適用。

記憶體最佳化：
  ‧ 每頁處理完強制釋放暫存物件並 gc.collect()
  ‧ 偵測到「實質為黑白」的頁面自動轉灰階 (記憶體用量減 2/3)
  ‧ PyMuPDF pixmap 用完即釋放，避免 hold 住 native memory

輸出模式：
  ‧ 預設：JPEG (彩色或灰階，依輸入)
  ‧ output_bitonal=True：用 Otsu 閾值化成 1-bit 黑白，存 PNG
                         適合純文字掃描檔，檔案非常小

依存：PyMuPDF (fitz)、Pillow、numpy、opencv-python、image_processor
"""

from __future__ import annotations

import gc
import io
import os
from typing import Callable

import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from image_processor import CleanConfig, DEFAULT_CONFIG, clean_pil


def _is_effectively_grayscale(img: Image.Image, sample_size: int = 64) -> bool:
    """快速判斷一張 RGB 圖是否「實質上」就是灰階 (R≈G≈B)。"""
    if img.mode != "RGB":
        return img.mode in ("L", "1")
    small = img.resize((sample_size, sample_size), Image.NEAREST)
    arr = np.asarray(small, dtype=np.int16)
    rg = np.abs(arr[..., 0] - arr[..., 1])
    gb = np.abs(arr[..., 1] - arr[..., 2])
    rb = np.abs(arr[..., 0] - arr[..., 2])
    return max(int(rg.max()), int(gb.max()), int(rb.max())) <= 5


def _to_bitonal_png_bytes(img: Image.Image, dilate_iter: int = 1) -> bytes:
    """
    灰階/RGB 影像閾值化成 1-bit 黑白，存 PNG bytes。

    流程：
      1. Otsu 自動閾值化
      2. 膨脹筆畫 (= 侵蝕白色背景) dilate_iter 次，
         補回旋轉造成的筆畫變細

    dilate_iter:
      0 = 不補 (字會略細，但檔案最小)
      1 = 補 1 px (預設，字保持原始厚度)
      2 = 補 2 px (字會比原始略粗)
    """
    if img.mode != "L":
        gray = img.convert("L")
    else:
        gray = img
    arr = np.asarray(gray, dtype=np.uint8)
    _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if dilate_iter > 0:
        # 侵蝕白色 = 黑色筆畫膨脹 1 px / iteration
        kernel = np.ones((3, 3), dtype=np.uint8)
        bw = cv2.erode(bw, kernel, iterations=dilate_iter)

    pil_1bit = Image.fromarray(bw, mode="L").convert("1")
    buf = io.BytesIO()
    pil_1bit.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


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
      cfg:       處理設定
                   ‧ cfg.force_grayscale=True 強制以灰階處理
                   ‧ cfg.output_bitonal=True  輸出 1-bit 黑白 (PNG，超小)
      dpi:       渲染解析度，預設 200
      jpeg_quality: JPEG 品質 (1-95)；bitonal 模式下無關
      progress_cb: 回呼 (page_idx_1based, total_pages, message)

    回傳: {"pages": N, "angles": [...]}
    """
    src = fitz.open(src_path)
    out = fitz.open()

    angles = []
    n = src.page_count

    try:
        for i in range(n):
            page = src[i]

            # ---- 1) 渲染 ----
            mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            pw, ph = pix.width, pix.height
            mode = "RGB" if pix.n >= 3 else "L"
            img = Image.frombytes(mode, (pw, ph), pix.samples)
            pix = None
            del pix

            # ---- 2) 灰階優化 ----
            # output_bitonal 必定要先轉灰階才能 Otsu
            grayscale_mode = cfg.force_grayscale or cfg.output_bitonal
            if not grayscale_mode and img.mode == "RGB":
                if _is_effectively_grayscale(img):
                    grayscale_mode = True

            if grayscale_mode and img.mode != "L":
                img2 = img.convert("L")
                try:
                    img.close()
                except Exception:
                    pass
                img = img2

            # ---- 3) 套用清理 (去髒 + 轉正) ----
            cleaned, info = clean_pil(img, cfg)
            angles.append(info.get("skew_angle", 0.0))

            try:
                img.close()
            except Exception:
                pass
            img = None
            del img

            # ---- 4) 編碼 ----
            if cfg.output_bitonal:
                # 1-bit PNG: 純文字掃描檔最佳
                img_bytes = _to_bitonal_png_bytes(cleaned)
                tag = "1-bit"
            else:
                buf = io.BytesIO()
                if cleaned.mode == "L":
                    cleaned.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
                    tag = "灰階"
                else:
                    if cleaned.mode != "RGB":
                        c2 = cleaned.convert("RGB")
                        try:
                            cleaned.close()
                        except Exception:
                            pass
                        cleaned = c2
                    cleaned.save(buf, format="JPEG", quality=jpeg_quality, optimize=False)
                    tag = "彩色"
                img_bytes = buf.getvalue()
                buf.close()
                buf = None
                del buf

            try:
                cleaned.close()
            except Exception:
                pass
            cleaned = None
            del cleaned

            # ---- 5) 嵌入新 PDF ----
            new_page = out.new_page(width=page.rect.width, height=page.rect.height)
            new_page.insert_image(new_page.rect, stream=img_bytes)
            img_bytes = None
            del img_bytes

            # ---- 6) 強制 GC ----
            gc.collect()

            if progress_cb:
                progress_cb(
                    i + 1, n,
                    f"第 {i+1}/{n} 頁 — 轉正 {info.get('skew_angle', 0.0):+.2f}° ({tag})",
                )

        out.save(dst_path, garbage=4, deflate=True)
    finally:
        out.close()
        src.close()
        gc.collect()

    return {"pages": n, "angles": angles}


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
