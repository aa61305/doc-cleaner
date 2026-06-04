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

    # 是否輸出為 1-bit 純黑白 (適合純文字掃描檔，檔案最小)
    # True 時會用 Otsu 閾值法把處理後影像 threshold 成純黑白
    output_bitonal: bool = False

    # 是否把內容水平置中 (掃描歪斜或邊緣留白不對稱時有用)
    center_horizontally: bool = False

    # 強制置中：忽略「位移過小就不動」的門檻，只要有內容就拉到正中央
    # (搭配 center_horizontally=True 使用；單獨設 True 也視為要置中)
    center_force: bool = False

    # 是否執行去髒點 (despeckle + 孤立髒點)。False = 只做轉正/置中，完全不動內容
    do_despeckle: bool = True

    # 是否啟用「離文字遠的孤立髒點清除」第二階段
    # 用 dilate(LARGE 元件) 形成「near-text」區域，落在外面的小元件視為髒點
    remove_isolated_specks: bool = True

    # 第二階段：可疑髒點的尺寸範圍 (200 DPI 基準，會 scale)
    # 同時滿足 area ≤ max_area 「且」 max(w,h) ≤ max_dim 才算候選；
    # 兩條件需同時成立，任一超過就被當成「文字/線」自動保護。
    # 設計原則：max_dim 守護「字一定比 speck 高」(細數字 1 在 200 DPI 下約 15px 高)
    # 設計原則：max_area 守護不會把大塊的真實內容當成髒點
    iso_speck_min_area: int = 8     # 200 DPI 下 8 → 600 DPI 下 ~70
    iso_speck_max_area: int = 80    # 200 DPI 下 80 → 600 DPI 下 ~720 (約 30 px 半徑)
    iso_speck_max_dim: int = 10     # 200 DPI 下 10px → 600 DPI 下 ~30px

    # 第二階段：「靠近文字」的距離閾值 (px，200 DPI 基準，會 scale)
    # 5 = 600 DPI 下 ~15px。較積極清掉緊貼數字邊緣的髒點，
    # 仍能保護真正緊鄰字邊 (5-15 px) 的中文標點。
    iso_near_text_dist: int = 5

    # 基本 despeckle 的 dim 絕對上限 (保護細字細線)
    # 即使 auto-scale 也不會超過這個值
    max_speck_dim_cap: int = 4

    # ★ 第一階段去髒的「尊重上下文」保護 (預設開) ★
    # 解決「細字 / 細線被連同髒點一起清掉」的問題。
    # 原本第一階段只要元件夠小就無條件清；但細字的小筆畫、斷裂的
    # 細線段也很小，旁邊卻一定有其他內容 → 用下列兩道保護避免誤殺：
    protect_near_text: bool = True

    # (a) 鄰近保護：小元件若緊貼「字/線主體」(大元件) 就保留。
    #     200 DPI 基準距離，會隨 DPI scale。4 → 600 DPI 下 ~12px。
    #     設大一點以涵蓋「剛被二值化切斷、離筆畫 2-4px 的字部件/點」，
    #     符合「寧可少清髒點，也別傷字線」的取捨。孤立髒點離內容通常
    #     遠在數十 px 以上，不受影響。
    protect_near_dist: int = 4

    # (b) 細物保護：厚度 ≤ 此值 (px，200 DPI 基準，會 scale) 的細長碎片
    #     視為細線 / 細筆畫，一律保留。1 → 600 DPI 下保護 ≤3px 厚的細物。
    #     真髒點通常 ≥2px 厚且近似方形，所以不太會被這條規則漏放。
    protect_thin_px: int = 1


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
    2. Otsu 二值化找出深色像素 (文字 + 髒點都是深點)
    3. 連通元件分析
    4. 把面積 ≤ eff_max_area 且 寬/高都 ≤ eff_max_dim 的「孤立小元件」
       在原圖對應位置塗白

    ★ 尊重上下文的保護 (cfg.protect_near_text，預設開) ★
    細字的小筆畫、斷裂的細線段也很小，但旁邊一定有其他內容。為避免把它們
    當髒點誤殺，移除前再過兩道保護：
      (a) 鄰近保護：質心落在「字/線主體 (大元件) 膨脹區」內 → 保留。
      (b) 細物保護：厚度 (min(w,h)) ≤ eff_thin → 視為細線/細筆畫 → 保留。
    只有「又小、又離內容遠、又不是細長」的元件才會被當成真髒點清掉。

    ★ 自動 scale ★
    cfg.max_speck_area 與 cfg.max_speck_dim 是以 200 DPI A4 (~2200 px 高)
    為基準的閾值。實際處理時依影像高度自動放大，讓 200/300/600 DPI
    輸入有一致的視覺去髒效果（不會因高 DPI 反而抓不到大髒點）。
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 自動 scale：以 200 DPI A4 (~2200 px) 為基準
    h_img = img.shape[0]
    scale = max(1.0, h_img / 2200.0)
    eff_max_area = int(cfg.max_speck_area * scale * scale)  # 面積跟 scale 平方
    # 邊長 scale 後仍受 max_speck_dim_cap 上限保護 (保護細字細線)
    eff_max_dim = min(int(cfg.max_speck_dim * scale), cfg.max_speck_dim_cap)

    # 二值化：白底黑字 → 反轉成「黑底白前景」方便連通元件分析
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 連通元件
    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    # --- 尊重上下文的保護：先備好「字/線主體」的膨脹區 --- #
    eff_thin = max(1, int(round(cfg.protect_thin_px * scale)))
    near_text = None
    if cfg.protect_near_text:
        # 「主體」= 任一邊大於 eff_max_dim 的前景 (字、線、表格)。
        big_mask = np.zeros_like(bw)
        has_big = False
        for i in range(1, num):
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            if max(w, h) > eff_max_dim:
                big_mask[labels == i] = 255
                has_big = True
        if has_big:
            pad = max(1, int(round(cfg.protect_near_dist * scale)))
            k = pad * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
            near_text = cv2.dilate(big_mask, kernel)

    # 收集要移除的小元件 (套用兩道保護)
    mask = np.zeros_like(bw)
    removed = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        if not (area <= eff_max_area and w <= eff_max_dim and h <= eff_max_dim):
            continue
        # (b) 細物保護：厚度極薄 → 細線/細筆畫，保留
        if cfg.protect_near_text and min(w, h) <= eff_thin:
            continue
        # (a) 鄰近保護：元件「任一像素」觸及字/線主體膨脹區 → 保留。
        #     用元件像素 (而非質心) 判定，邊緣貼字的字部件也救得到。
        if near_text is not None:
            x0 = int(stats[i, cv2.CC_STAT_LEFT])
            y0 = int(stats[i, cv2.CC_STAT_TOP])
            sub_lab = labels[y0:y0 + h, x0:x0 + w]
            sub_near = near_text[y0:y0 + h, x0:x0 + w]
            if np.any((sub_lab == i) & (sub_near > 0)):
                continue
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


# --------------------------- 孤立大髒點 (第二階段) --------------------------- #


def remove_isolated_specks(img: np.ndarray, cfg: CleanConfig = DEFAULT_CONFIG) -> np.ndarray:
    """
    第二階段去髒：清除「離文字遠的孤立小髒點」。

    判定邏輯（用 dim 而不是 area，比較不會把細數字誤殺）：
      ‧ max(w,h) > SPECK_MAX_DIM     → 真正的文字 / 表格線 → 視為「LARGE」
      ‧ max(w,h) ≤ SPECK_MAX_DIM 且
        area ∈ [SPECK_MIN_AREA, SPECK_MAX_AREA]
                                    → 可疑髒點，跑距離檢查
      ‧ 其他 (太小 / 太大)            → 不動

    可疑髒點若質心 *不在* dilate 過的 LARGE 元件附近 → 視為孤立髒點，清掉。
    標點符號 (例如 「、」) 因為一定在字旁邊，所以會被保護。
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    h_img, w_img = gray.shape

    scale = max(1.0, h_img / 2200.0)
    scale_sq = scale * scale

    SPECK_MIN_AREA = int(cfg.iso_speck_min_area * scale_sq)
    SPECK_MAX_AREA = int(cfg.iso_speck_max_area * scale_sq)
    SPECK_MAX_DIM = int(cfg.iso_speck_max_dim * scale)
    NEAR_DIST = int(cfg.iso_near_text_dist * scale)

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    if num <= 1:
        return img

    # 一次走訪：分類成 LARGE 與 SPECK candidate
    large_mask = np.zeros_like(bw)
    speck_candidates = []
    has_large = False
    for i in range(1, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if max(ww, hh) > SPECK_MAX_DIM:
            # 任何一邊比 speck 還大 → 視為文字/線
            large_mask[labels == i] = 255
            has_large = True
        elif SPECK_MIN_AREA <= a <= SPECK_MAX_AREA:
            speck_candidates.append(i)

    if not has_large or not speck_candidates:
        return img

    # dilate 形成「near-text」遮罩
    k = max(3, NEAR_DIST * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    near_text = cv2.dilate(large_mask, kernel)

    # 對每個候選看質心是否在 near-text 內
    remove_mask = np.zeros_like(bw)
    removed = 0
    for i in speck_candidates:
        cx = int(centroids[i][0])
        cy = int(centroids[i][1])
        if 0 <= cy < h_img and 0 <= cx < w_img:
            if near_text[cy, cx] == 0:
                remove_mask[labels == i] = 255
                removed += 1

    if removed == 0:
        return img

    out = img.copy()
    if out.ndim == 3:
        out[remove_mask > 0] = (255, 255, 255)
    else:
        out[remove_mask > 0] = 255
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


def rotate_image(img: np.ndarray, angle_deg: float, bg_value=255,
                 nearest: bool = False) -> np.ndarray:
    """以白色填充背景旋轉影像，保留原始尺寸 (不會放大畫布)。

    nearest=True 時用 INTER_NEAREST：不產生抗鋸齒灰階，後續 Otsu 二值化
    不會把灰邊吃成黑色 → 1-bit 黑白輸出不會因旋轉而「字變粗」。
    彩色 / 灰階 JPEG 輸出則用 LINEAR (預設)，邊緣較平滑。
    """
    if abs(angle_deg) < 0.05:  # 小於 0.05 度就不旋轉了
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    border = (bg_value, bg_value, bg_value) if img.ndim == 3 else bg_value
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


# --------------------------- 水平置中 --------------------------- #


def center_content_horizontally(img: np.ndarray, dark_threshold: int = 200,
                                min_shift: int = 3) -> np.ndarray:
    """
    把影像中的「內容」(非白色像素) 在水平方向上置中。

    流程：
      1. 找出所有「有內容」的欄 (該欄存在像素值 < dark_threshold)
      2. 取最左、最右形成內容 bounding box
      3. 計算把這個 bounding box 平移到頁面正中央所需的水平位移
      4. 用 cv2.warpAffine 平移整張圖 (背景填白)

    若位移量 < min_shift 像素 (差異微小)，則不動。
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    h, w = gray.shape
    # 找有內容的欄
    col_has_content = (gray < dark_threshold).any(axis=0)
    if not col_has_content.any():
        return img  # 全白頁，不動

    left = int(np.argmax(col_has_content))
    right = w - int(np.argmax(col_has_content[::-1]))
    content_w = right - left
    new_left = (w - content_w) // 2
    dx = new_left - left
    if abs(dx) < min_shift:
        return img  # 差異太小不動

    M = np.array([[1, 0, dx], [0, 1, 0]], dtype=np.float32)
    border = (255, 255, 255) if img.ndim == 3 else 255
    return cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_NEAREST,  # 純水平整數平移用 NEAREST 即可，不會糊
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

    # 1) 去髒點 (可關閉：do_despeckle=False 時完全不動內容，只做轉正/置中)
    if cfg.do_despeckle:
        cleaned = despeckle(img, cfg)
        info["despeckled"] = True
    else:
        cleaned = img
        info["despeckled"] = False

    # 2) 估算傾斜角度
    angle = estimate_skew_angle(cleaned, cfg)
    info["skew_angle"] = round(angle, 3)

    # 3) 旋轉轉正 (1-bit 輸出用 NEAREST，避免抗鋸齒讓字變粗)
    deskewed = rotate_image(cleaned, angle, nearest=cfg.output_bitonal)

    # 4) 旋轉後可能在邊緣帶入小三角形空白，再做一次極輕的去髒 (只清角落)
    if cfg.do_despeckle:
        final = despeckle(deskewed, cfg)
        # 5) 第二階段：清「離文字遠的中等大小髒點」
        if cfg.remove_isolated_specks:
            final = remove_isolated_specks(final, cfg)
    else:
        final = deskewed

    # 6) 可選：水平置中 (center_force=True 視同要置中，且忽略最小位移門檻)
    if cfg.center_horizontally or cfg.center_force:
        before = final
        min_shift = 1 if cfg.center_force else 3
        final = center_content_horizontally(final, min_shift=min_shift)
        info["centered"] = (final is not before)

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
