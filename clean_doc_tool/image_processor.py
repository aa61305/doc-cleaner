# -*- coding: utf-8 -*-
"""
影像處理核心：去髒點 (despeckle) + 文字轉正 (deskew)。

設計重點：
1. 不做 OCR、不重打字、不改變字體 → 純影像處理。
2. 去髒點採用「面積極小 + 尺寸極小」的雙重條件，避免誤殺中文標點 (、。「」‧)。
3. 文字轉正使用「投影輪廓變異數」法，對掃描文件非常穩定。

依存：opencv-python (或 opencv-python-headless)、numpy、Pillow
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


# ----------------------------- 設定 ----------------------------- #


@dataclass
class CleanConfig:
    """處理設定。預設為「中度」，已調校好不會吃掉中文標點。"""

    # 去髒點：移除面積 ≤ max_speck_area 且 寬高都 ≤ max_speck_dim 的連通元件
    # 在 200 DPI 下，中文最小的標點 (、 。 . ) 面積至少 30+，所以 8/3 很安全
    max_speck_area: int = 8
    max_speck_dim: int = 3

    # 文字轉正：搜尋角度範圍 (±deg)，步進 step
    deskew_range_deg: float = 5.0
    deskew_step_deg: float = 0.1

    # 輕度去雜訊 (邊緣保留型)。0 = 關閉。
    bilateral_d: int = 0  # 預設關閉，因為連通元件清雜已足夠且更安全

    # 是否強制轉灰階輸出 (False = 保留原色彩)
    force_grayscale: bool = False


DEFAULT_CONFIG = CleanConfig()


# --------------------------- 影像 IO --------------------------- #


def pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL Image → OpenCV BGR/灰階 numpy array。"""
    if img.mode == "L":
        return np.array(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def cv_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV array → PIL Image。"""
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB), mode="RGB")


# --------------------------- 去髒點 --------------------------- #


def despeckle(img: np.ndarray, cfg: CleanConfig = DEFAULT_CONFIG) -> np.ndarray:
    """
    去除小於閾值的「孤立小黑點」，輸出與輸入同 shape / dtype。

    流程：
    1. 灰階化
    2. Otsu 二值化找出深色像素 (文字 + 髒點都是深色)
    3. 連通元件分析
    4. 把面積 ≤ max_speck_area 且 寬/高都 ≤ max_speck_dim 的元件
       在原圖對應位置塗成「該區附近的背景色」(白)
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 二值化：白底黑字 → 反轉成「黑底白前景」方便連通元件分析
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 連通元件
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    # 收集要移除的小元件
    mask = np.zeros_like(bw)
    removed = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area <= cfg.max_speck_area and w <= cfg.max_speck_dim and h <= cfg.max_speck_dim:
            mask[labels == i] = 255
            removed += 1

    if removed == 0:
        return img

    # 估計背景色：拿這張圖的眾數亮度當底
    # 多數掃描文件背景接近 245~255。這裡保守用 255 (純白)。
    out = img.copy()
    if out.ndim == 3:
        out[mask > 0] = (255, 255, 255)
    else:
        out[mask > 0] = 255
    return out


# --------------------------- 文字轉正 --------------------------- #


def _projection_score(binary: np.ndarray) -> float:
    """投影輪廓變異數 — 文字水平排列時，每行黑像素總和的變異數最大。"""
    proj = binary.sum(axis=1, dtype=np.float64)
    return float(np.var(proj))


def estimate_skew_angle(img: np.ndarray, cfg: CleanConfig = DEFAULT_CONFIG) -> float:
    """
    估算文字傾斜角度 (度)，正值表示需逆時針旋轉。
    用投影輪廓變異數法在 ±range 內粗掃 + 細掃。
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 為了速度，把影像縮小 (長邊不超過 1000)
    h, w = gray.shape
    scale = 1.0
    if max(h, w) > 1000:
        scale = 1000.0 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # 二值化 (反轉後白色=文字)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 粗掃 (0.5° 步進，更精準)
    coarse_step = 0.5
    best_angle = 0.0
    best_score = _projection_score(bw)
    h2, w2 = bw.shape
    center = (w2 / 2.0, h2 / 2.0)

    rng = cfg.deskew_range_deg
    coarse_angles = np.arange(-rng, rng + coarse_step, coarse_step)
    for a in coarse_angles:
        if a == 0:
            score = best_score
        else:
            M = cv2.getRotationMatrix2D(center, float(a), 1.0)
            rot = cv2.warpAffine(
                bw, M, (w2, h2),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT, borderValue=0,
            )
            score = _projection_score(rot)
        if score > best_score:
            best_score = score
            best_angle = float(a)

    # 細掃 (在最佳粗角度附近)
    fine_step = cfg.deskew_step_deg
    fine_angles = np.arange(best_angle - coarse_step, best_angle + coarse_step + fine_step, fine_step)
    for a in fine_angles:
        M = cv2.getRotationMatrix2D(center, float(a), 1.0)
        rot = cv2.warpAffine(
            bw, M, (w2, h2),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        score = _projection_score(rot)
        if score > best_score:
            best_score = score
            best_angle = float(a)

    return best_angle


def rotate_image(img: np.ndarray, angle_deg: float, bg_value=255) -> np.ndarray:
    """以白色填充背景旋轉影像，保留原始尺寸 (不會放大畫布)。"""
    if abs(angle_deg) < 0.05:  # 小於 0.05 度就不旋轉了
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    border = (bg_value, bg_value, bg_value) if img.ndim == 3 else bg_value
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,  # 雙立方比較不會讓字邊緣鋸齒
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


# --------------------------- 完整管線 --------------------------- #


def clean_image(img: np.ndarray, cfg: CleanConfig = DEFAULT_CONFIG) -> tuple[np.ndarray, dict]:
    """
    主入口：清理單張影像。
    回傳 (處理後影像, 統計資訊 dict)。
    """
    info = {}

    # 可選：強制灰階輸出
    if cfg.force_grayscale and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 去髒點
    cleaned = despeckle(img, cfg)
    info["despeckled"] = True

    # 2) 估算傾斜角度
    angle = estimate_skew_angle(cleaned, cfg)
    info["skew_angle"] = round(angle, 3)

    # 3) 旋轉轉正
    deskewed = rotate_image(cleaned, angle)

    # 4) 旋轉後可能在邊緣帶入小三角形空白，再做一次極輕的去髒 (只清角落)
    final = despeckle(deskewed, cfg)

    return final, info


def clean_pil(img: Image.Image, cfg: CleanConfig = DEFAULT_CONFIG) -> tuple[Image.Image, dict]:
    """PIL Image 版本的入口。"""
    arr = pil_to_cv(img)
    out, info = clean_image(arr, cfg)
    return cv_to_pil(out), info


# --------------------------- 命令列測試 --------------------------- #


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python image_processor.py <input> <output>")
        sys.exit(1)

    src = Image.open(sys.argv[1])
    dst, info = clean_pil(src)
    dst.save(sys.argv[2])
    print(info)
