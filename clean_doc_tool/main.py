# -*- coding: utf-8 -*-
"""
書面文件清理小工具 — Tkinter GUI

功能：
  • 批次處理一個資料夾內的 PDF / JPG / PNG
  • 每個檔案：去除小髒點 + 文字轉正
  • 不修改文字本身
  • 輸出到指定資料夾，副檔名與輸入相同

操作：
  1) 選輸入資料夾
  2) 選輸出資料夾 (預設為輸入資料夾下的 cleaned/)
  3) 按「開始處理」

執行：python main.py
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path
from queue import Queue, Empty

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# 確保子模組可被 import 到 (打包成 exe 時也行)
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from PIL import Image  # noqa: E402

from image_processor import CleanConfig, clean_pil  # noqa: E402
from pdf_processor import process_pdf  # noqa: E402


SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
SUPPORTED_PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_PDF_EXTS


# --------------------------- 工作執行緒 --------------------------- #


class Worker(threading.Thread):
    """背景處理執行緒，透過 queue 把進度訊息傳回 GUI。"""

    def __init__(self, in_dir: Path, out_dir: Path, cfg: CleanConfig,
                 dpi: int, jpeg_quality: int, msgs: Queue):
        super().__init__(daemon=True)
        self.in_dir = in_dir
        self.out_dir = out_dir
        self.cfg = cfg
        self.dpi = dpi
        self.jpeg_quality = jpeg_quality
        self.msgs = msgs
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self) -> bool:
        return self._stop.is_set()

    # 訊息協定
    def post(self, kind: str, **payload):
        self.msgs.put({"kind": kind, **payload})

    def run(self):
        try:
            files = self._collect_files()
            self.post("info", text=f"找到 {len(files)} 個可處理檔案。")
            if not files:
                self.post("done", success=True)
                return

            self.out_dir.mkdir(parents=True, exist_ok=True)

            total = len(files)
            for idx, src in enumerate(files, 1):
                if self.stopped():
                    self.post("info", text="已取消。")
                    self.post("done", success=False)
                    return

                rel = src.relative_to(self.in_dir)
                dst = self.out_dir / rel
                dst.parent.mkdir(parents=True, exist_ok=True)

                self.post("file_begin", index=idx, total=total, name=str(rel))
                try:
                    if src.suffix.lower() in SUPPORTED_IMAGE_EXTS:
                        self._process_image(src, dst)
                    elif src.suffix.lower() in SUPPORTED_PDF_EXTS:
                        self._process_pdf(src, dst, idx, total)
                except Exception as e:
                    self.post("error", text=f"  ✗ {rel}: {e}")
                    self.post("error", text=traceback.format_exc())
                    continue

                self.post("file_end", index=idx, total=total, name=str(rel))

            self.post("info", text="全部完成。")
            self.post("done", success=True)
        except Exception as e:
            self.post("error", text=str(e))
            self.post("error", text=traceback.format_exc())
            self.post("done", success=False)

    # --- 處理單一檔案 --- #

    def _process_image(self, src: Path, dst: Path):
        with Image.open(src) as img:
            img.load()
            cleaned, info = clean_pil(img, self.cfg)

        # 確保輸出副檔名與輸入相同；JPEG 需 RGB
        suffix = dst.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            if cleaned.mode != "RGB":
                cleaned = cleaned.convert("RGB")
            cleaned.save(dst, format="JPEG", quality=self.jpeg_quality, optimize=True)
        else:  # png
            cleaned.save(dst, format="PNG", optimize=True)

        self.post("info", text=f"  ✓ 影像 轉正 {info['skew_angle']:+.2f}°  → {dst.name}")

    def _process_pdf(self, src: Path, dst: Path, file_idx: int, total: int):
        rel = src.relative_to(self.in_dir)

        def _cb(page_i, page_n, msg):
            # 進度回傳
            self.post(
                "page_progress",
                file_index=file_idx, file_total=total,
                page=page_i, page_total=page_n,
                file_name=str(rel),
                msg=msg,
            )

        info = process_pdf(
            str(src), str(dst),
            cfg=self.cfg,
            dpi=self.dpi,
            jpeg_quality=self.jpeg_quality,
            progress_cb=_cb,
        )
        avg = sum(info["angles"]) / max(1, len(info["angles"]))
        self.post(
            "info",
            text=f"  ✓ PDF {info['pages']} 頁 (平均轉正 {avg:+.2f}°) → {dst.name}",
        )

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []
        for root, _, names in os.walk(self.in_dir):
            for name in names:
                p = Path(root) / name
                if p.suffix.lower() in SUPPORTED_EXTS:
                    files.append(p)
        files.sort()
        return files


# --------------------------- GUI --------------------------- #


class App(tk.Tk):
    APP_TITLE = "書面文件清理工具 — 去髒點 + 文字轉正"

    def __init__(self):
        super().__init__()
        self.title(self.APP_TITLE)
        self.geometry("780x600")
        self.minsize(640, 480)

        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.dpi_var = tk.IntVar(value=200)
        self.jpeg_var = tk.IntVar(value=88)
        self.strength_var = tk.StringVar(value="中度（推薦）")

        self.worker: Worker | None = None
        self.msgs: Queue = Queue()

        self._build_ui()
        self.after(100, self._poll_msgs)

    # --- 介面 --- #
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        frame_top = ttk.Frame(self)
        frame_top.pack(fill="x", **pad)

        # 輸入資料夾
        ttk.Label(frame_top, text="輸入資料夾：").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame_top, textvariable=self.in_var).grid(
            row=0, column=1, sticky="ew", padx=4
        )
        ttk.Button(frame_top, text="瀏覽…", command=self._pick_input).grid(
            row=0, column=2
        )

        # 輸出資料夾
        ttk.Label(frame_top, text="輸出資料夾：").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(frame_top, textvariable=self.out_var).grid(
            row=1, column=1, sticky="ew", padx=4, pady=(6, 0)
        )
        ttk.Button(frame_top, text="瀏覽…", command=self._pick_output).grid(
            row=1, column=2, pady=(6, 0)
        )

        frame_top.columnconfigure(1, weight=1)

        # 選項
        frame_opts = ttk.LabelFrame(self, text="處理選項")
        frame_opts.pack(fill="x", **pad)

        ttk.Label(frame_opts, text="去髒強度：").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        cb = ttk.Combobox(
            frame_opts, textvariable=self.strength_var, state="readonly", width=18,
            values=["輕度（保守）", "中度（推薦）", "較強"],
        )
        cb.grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(frame_opts, text="PDF 解析度 (DPI)：").grid(row=0, column=2, sticky="w", padx=(20, 6))
        ttk.Spinbox(
            frame_opts, from_=100, to=400, increment=50,
            textvariable=self.dpi_var, width=6,
        ).grid(row=0, column=3, sticky="w")

        ttk.Label(frame_opts, text="JPEG 品質：").grid(row=0, column=4, sticky="w", padx=(20, 6))
        ttk.Spinbox(
            frame_opts, from_=60, to=95, increment=1,
            textvariable=self.jpeg_var, width=5,
        ).grid(row=0, column=5, sticky="w")

        # 按鈕
        frame_btn = ttk.Frame(self)
        frame_btn.pack(fill="x", **pad)
        self.btn_start = ttk.Button(frame_btn, text="開始處理", command=self._start)
        self.btn_start.pack(side="left")
        self.btn_stop = ttk.Button(frame_btn, text="停止", command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=8)

        # 進度條
        frame_prog = ttk.Frame(self)
        frame_prog.pack(fill="x", **pad)
        ttk.Label(frame_prog, text="總進度：").grid(row=0, column=0, sticky="w")
        self.pb_total = ttk.Progressbar(frame_prog, length=200, mode="determinate")
        self.pb_total.grid(row=0, column=1, sticky="ew", padx=4)
        self.lbl_total = ttk.Label(frame_prog, text="0 / 0")
        self.lbl_total.grid(row=0, column=2, sticky="w")

        ttk.Label(frame_prog, text="目前頁面：").grid(row=1, column=0, sticky="w", pady=(4, 0))
        self.pb_page = ttk.Progressbar(frame_prog, length=200, mode="determinate")
        self.pb_page.grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
        self.lbl_page = ttk.Label(frame_prog, text="0 / 0")
        self.lbl_page.grid(row=1, column=2, sticky="w", pady=(4, 0))

        frame_prog.columnconfigure(1, weight=1)

        # Log 視窗
        frame_log = ttk.LabelFrame(self, text="處理紀錄")
        frame_log.pack(fill="both", expand=True, **pad)
        self.log = tk.Text(frame_log, height=12, wrap="none")
        self.log.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame_log, orient="vertical", command=self.log.yview)
        sb.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=sb.set)

        self._log("歡迎使用書面文件清理工具。請選擇輸入資料夾後按「開始處理」。\n"
                 "支援格式：PDF / JPG / PNG。輸出檔名與原檔相同。\n")

    # --- UI 互動 --- #

    def _pick_input(self):
        d = filedialog.askdirectory(title="選擇輸入資料夾")
        if d:
            self.in_var.set(d)
            # 預設輸出 = 輸入/cleaned
            if not self.out_var.get():
                self.out_var.set(str(Path(d) / "cleaned"))

    def _pick_output(self):
        d = filedialog.askdirectory(title="選擇輸出資料夾")
        if d:
            self.out_var.set(d)

    def _start(self):
        in_path = Path(self.in_var.get().strip())
        out_path = Path(self.out_var.get().strip())

        if not in_path.is_dir():
            messagebox.showerror("錯誤", "請選擇有效的輸入資料夾。")
            return
        if not out_path:
            messagebox.showerror("錯誤", "請選擇輸出資料夾。")
            return
        if out_path.resolve() == in_path.resolve():
            messagebox.showerror(
                "錯誤", "輸出資料夾不能與輸入資料夾相同 (避免覆蓋原檔)。"
            )
            return

        cfg = self._make_config()

        self.pb_total["value"] = 0
        self.pb_page["value"] = 0
        self.lbl_total.configure(text="0 / 0")
        self.lbl_page.configure(text="0 / 0")
        self._log(f"\n=== 開始處理 ===\n  輸入：{in_path}\n  輸出：{out_path}\n"
                  f"  強度：{self.strength_var.get()}  DPI：{self.dpi_var.get()}\n")

        self.worker = Worker(
            in_path, out_path, cfg,
            dpi=int(self.dpi_var.get()),
            jpeg_quality=int(self.jpeg_var.get()),
            msgs=self.msgs,
        )
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        self.worker.start()

    def _stop(self):
        if self.worker and self.worker.is_alive():
            self.worker.stop()
            self._log("正在停止…\n")

    def _make_config(self) -> CleanConfig:
        s = self.strength_var.get()
        if s.startswith("輕度"):
            return CleanConfig(max_speck_area=4, max_speck_dim=2)
        if s.startswith("較強"):
            return CleanConfig(max_speck_area=18, max_speck_dim=4)
        return CleanConfig()  # 中度

    # --- 訊息處理 --- #

    def _poll_msgs(self):
        try:
            while True:
                msg = self.msgs.get_nowait()
                self._handle(msg)
        except Empty:
            pass
        self.after(80, self._poll_msgs)

    def _handle(self, msg: dict):
        kind = msg["kind"]
        if kind == "info":
            self._log(msg["text"] + "\n")
        elif kind == "error":
            self._log(msg["text"] + "\n")
        elif kind == "file_begin":
            self.lbl_total.configure(text=f"{msg['index']} / {msg['total']}")
            self.pb_total["maximum"] = msg["total"]
            self.pb_total["value"] = msg["index"] - 1
            self.pb_page["value"] = 0
            self._log(f"[{msg['index']}/{msg['total']}] {msg['name']}\n")
        elif kind == "file_end":
            self.pb_total["value"] = msg["index"]
        elif kind == "page_progress":
            self.pb_page["maximum"] = msg["page_total"]
            self.pb_page["value"] = msg["page"]
            self.lbl_page.configure(text=f"{msg['page']} / {msg['page_total']}")
        elif kind == "done":
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            if msg.get("success"):
                self._log("=== 處理結束 ✓ ===\n")
                messagebox.showinfo("完成", "所有檔案處理完成。")
            else:
                self._log("=== 已停止或發生錯誤 ===\n")

    def _log(self, text: str):
        self.log.insert("end", text)
        self.log.see("end")


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
