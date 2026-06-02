# -*- coding: utf-8 -*-
"""
書面文件清理 — 網頁版伺服器

開啟後在同 WiFi 內的 iPhone / iPad 用瀏覽器即可使用：
  1) 在電腦執行本程式
  2) 終端會印出「電腦在區網內的 IP」+ 連結 (例: http://192.168.1.10:8765/)
  3) iPhone 開 Safari 輸入該網址 → 上傳檔案 → 自動下載清理後的檔案

依存：Flask, PyMuPDF, opencv-python, numpy, Pillow
"""

from __future__ import annotations

import io
import os
import socket
import sys
import tempfile
import threading
import uuid
import webbrowser
from pathlib import Path

from flask import (
    Flask, request, send_file, render_template_string,
    jsonify, abort, url_for,
)

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from PIL import Image  # noqa: E402
from image_processor import CleanConfig, clean_pil  # noqa: E402
from pdf_processor import process_pdf  # noqa: E402


PORT = int(os.environ.get("DOCCLEANER_PORT", "8765"))
MAX_UPLOAD_MB = 100
SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# 用暫存資料夾儲存處理結果，重啟即清空
_TMP_DIR = Path(tempfile.gettempdir()) / "doccleaner_web"
_TMP_DIR.mkdir(parents=True, exist_ok=True)

# 簡單記憶體狀態：job_id -> {"status": "...", "out_path": "...", "msg": "..."}
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# --------------------------- HTML --------------------------- #

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <meta name="apple-mobile-web-app-capable" content="yes">
  <title>書面文件清理</title>
  <style>
    :root { color-scheme: light dark; }
    * { box-sizing: border-box; }
    html, body { margin: 0; padding: 0; -webkit-text-size-adjust: 100%; }
    body {
      font-family: -apple-system, "PingFang TC", "Microsoft JhengHei", sans-serif;
      background: #f5f5f7;
      color: #1d1d1f;
      padding: 16px;
      max-width: 540px;
      margin: 0 auto;
    }
    @media (prefers-color-scheme: dark) {
      body { background: #1c1c1e; color: #f2f2f7; }
      .card { background: #2c2c2e !important; }
      .help { color: #a1a1a6 !important; }
      input, select { background:#3a3a3c !important; color:#fff !important; border-color:#48484a !important; }
    }
    h1 { font-size: 22px; margin: 8px 0 4px; }
    .sub { color: #6e6e73; font-size: 13px; margin-bottom: 16px; }
    .card {
      background: #fff;
      border-radius: 14px;
      padding: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      margin-bottom: 14px;
    }
    label { display: block; font-size: 14px; margin: 6px 0; }
    input[type=file], select {
      width: 100%;
      padding: 12px;
      font-size: 16px;
      border-radius: 10px;
      border: 1px solid #d2d2d7;
      background: #fff;
      margin-top: 4px;
    }
    button {
      width: 100%;
      padding: 14px;
      font-size: 17px;
      font-weight: 600;
      border-radius: 12px;
      border: 0;
      background: #007aff;
      color: white;
      cursor: pointer;
      -webkit-tap-highlight-color: transparent;
    }
    button:disabled { opacity: 0.5; }
    .help { color: #6e6e73; font-size: 12px; margin-top: 4px; }
    .progress {
      display: none;
      margin-top: 12px;
      font-size: 14px;
    }
    .bar {
      width: 100%;
      height: 8px;
      background: #e5e5ea;
      border-radius: 4px;
      overflow: hidden;
      margin-top: 8px;
    }
    .bar > div {
      height: 100%;
      background: #007aff;
      width: 0%;
      transition: width 0.3s;
    }
    .result {
      display: none;
      margin-top: 12px;
      padding: 12px;
      background: #e8f5e9;
      border-radius: 10px;
      color: #1b5e20;
    }
    @media (prefers-color-scheme: dark) {
      .result { background: #1f3a1f; color: #c8e6c9; }
      .bar { background: #48484a; }
    }
    .err {
      display: none;
      margin-top: 12px;
      padding: 12px;
      background: #ffebee;
      border-radius: 10px;
      color: #b71c1c;
      font-size: 14px;
    }
    a.dl {
      display: block;
      margin-top: 8px;
      padding: 12px;
      background: #34c759;
      color: white;
      text-align: center;
      border-radius: 10px;
      text-decoration: none;
      font-weight: 600;
    }
  </style>
</head>
<body>
  <h1>📄 書面文件清理</h1>
  <div class="sub">去髒點 + 文字轉正 ｜ 不修改原文字</div>

  <div class="card">
    <form id="f">
      <label>選擇檔案 (PDF / JPG / PNG)
        <input type="file" id="file" name="file"
               accept=".pdf,.jpg,.jpeg,.png,application/pdf,image/jpeg,image/png" required>
      </label>

      <label>去髒強度
        <select id="strength" name="strength">
          <option value="none">不去髒 (只轉正／置中)</option>
          <option value="light">輕度 (保守)</option>
          <option value="medium" selected>中度 (推薦)</option>
          <option value="strong">較強</option>
        </select>
      </label>

      <label style="display:flex; align-items:center; gap:8px;">
        <input type="checkbox" id="center" name="center" value="1"
               style="width:auto; margin:0; transform:scale(1.3);">
        強制頁面左右置中
      </label>

      <label>PDF 解析度 (DPI)
        <select id="dpi" name="dpi">
          <option value="150">150 (檔案較小)</option>
          <option value="200" selected>200 (推薦)</option>
          <option value="300">300 (高品質)</option>
        </select>
      </label>

      <button type="submit" id="btn">開始處理</button>
      <div class="help">處理時間視檔案大小而定，多頁 PDF 可能需要 30 秒以上。</div>
    </form>

    <div class="progress" id="progress">
      <div id="msg">處理中…</div>
      <div class="bar"><div id="bar"></div></div>
    </div>

    <div class="result" id="result">
      ✓ 處理完成
      <a class="dl" id="dl" href="#" download>下載清理後的檔案</a>
    </div>

    <div class="err" id="err"></div>
  </div>

  <div class="sub" style="text-align:center;">
    伺服器於 {{ server_addr }} ｜ 拷貝整個資料夾、再雙擊啟動腳本即可重開
  </div>

<script>
const f = document.getElementById('f');
const fileEl = document.getElementById('file');
const btn = document.getElementById('btn');
const prog = document.getElementById('progress');
const bar = document.getElementById('bar');
const msg = document.getElementById('msg');
const result = document.getElementById('result');
const dl = document.getElementById('dl');
const err = document.getElementById('err');

f.addEventListener('submit', async (e) => {
  e.preventDefault();
  err.style.display = 'none';
  result.style.display = 'none';
  prog.style.display = 'block';
  btn.disabled = true;
  bar.style.width = '5%';
  msg.textContent = '上傳中…';

  if (!fileEl.files.length) { btn.disabled=false; return; }

  const fd = new FormData();
  fd.append('file', fileEl.files[0]);
  fd.append('strength', document.getElementById('strength').value);
  fd.append('center', document.getElementById('center').checked ? '1' : '0');
  fd.append('dpi', document.getElementById('dpi').value);

  // 1) 上傳並啟動 job
  let resp;
  try {
    resp = await fetch('/upload', { method: 'POST', body: fd });
  } catch (ex) {
    showErr('上傳失敗：' + ex);
    return;
  }
  if (!resp.ok) {
    showErr('上傳失敗：' + (await resp.text()));
    return;
  }
  const data = await resp.json();
  const job = data.job_id;
  bar.style.width = '15%';

  // 2) 輪詢進度
  const poll = setInterval(async () => {
    let r;
    try { r = await fetch('/status/' + job); }
    catch (ex) { return; }
    if (!r.ok) return;
    const s = await r.json();
    if (s.percent !== undefined) {
      bar.style.width = Math.max(15, s.percent) + '%';
    }
    if (s.message) msg.textContent = s.message;
    if (s.status === 'done') {
      clearInterval(poll);
      bar.style.width = '100%';
      msg.textContent = '完成 ✓';
      dl.href = '/download/' + job;
      dl.setAttribute('download', s.filename || 'cleaned');
      result.style.display = 'block';
      btn.disabled = false;
    } else if (s.status === 'error') {
      clearInterval(poll);
      showErr(s.message || '處理失敗');
    }
  }, 800);
});

function showErr(m) {
  prog.style.display = 'none';
  err.textContent = m;
  err.style.display = 'block';
  btn.disabled = false;
}
</script>
</body>
</html>
"""


# --------------------------- 工具 --------------------------- #


def _local_ip() -> str:
    """猜本機在區網內的 IP (給手機連用)。"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        # 不會真的送出
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _make_config(strength: str, center: bool = False) -> CleanConfig:
    # center=True → 強制左右置中 (忽略最小位移門檻)
    kwargs = dict(center_horizontally=center, center_force=center)
    if strength == "none":
        # 不去髒：只做轉正 (＋可選置中)，完全不動內容
        return CleanConfig(do_despeckle=False, **kwargs)
    if strength == "light":
        return CleanConfig(max_speck_area=4, max_speck_dim=2, **kwargs)
    if strength == "strong":
        return CleanConfig(max_speck_area=18, max_speck_dim=4, **kwargs)
    return CleanConfig(**kwargs)  # medium


def _set_status(job: str, **kw):
    with JOBS_LOCK:
        if job in JOBS:
            JOBS[job].update(kw)


def _process_in_bg(job: str, src_path: Path, out_path: Path, ext: str,
                   cfg: CleanConfig, dpi: int):
    try:
        if ext == ".pdf":
            def cb(i, n, msg):
                _set_status(job,
                    percent=int(20 + 75 * i / max(n, 1)),
                    message=f"處理第 {i}/{n} 頁",
                )
            process_pdf(str(src_path), str(out_path), cfg=cfg,
                        dpi=dpi, jpeg_quality=88, progress_cb=cb)
        else:
            with Image.open(src_path) as img:
                img.load()
                _set_status(job, percent=40, message="處理中…")
                cleaned, info = clean_pil(img, cfg)
            if ext in (".jpg", ".jpeg"):
                if cleaned.mode != "RGB":
                    cleaned = cleaned.convert("RGB")
                cleaned.save(out_path, format="JPEG", quality=90, optimize=True)
            else:
                cleaned.save(out_path, format="PNG", optimize=True)
            _set_status(job, percent=95, message=f"轉正 {info['skew_angle']:+.2f}°")

        _set_status(job, status="done", percent=100, message="完成",
                    out_path=str(out_path))
    except Exception as e:
        _set_status(job, status="error", message=f"錯誤：{e}")
    finally:
        try:
            src_path.unlink(missing_ok=True)
        except Exception:
            pass


# --------------------------- 路由 --------------------------- #


@app.route("/")
def index():
    return render_template_string(
        INDEX_HTML,
        server_addr=f"http://{_local_ip()}:{PORT}/",
    )


@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "缺少 file", 400
    f = request.files["file"]
    if not f.filename:
        return "空的檔名", 400

    ext = Path(f.filename).suffix.lower()
    if ext not in SUPPORTED_EXTS:
        return f"不支援的副檔名：{ext}", 400

    strength = request.form.get("strength", "medium")
    center = request.form.get("center", "0") == "1"
    try:
        dpi = int(request.form.get("dpi", "200"))
    except ValueError:
        dpi = 200
    dpi = max(100, min(400, dpi))

    job = uuid.uuid4().hex
    src_path = _TMP_DIR / f"{job}_in{ext}"
    out_name = Path(f.filename).stem + "_cleaned" + ext
    out_path = _TMP_DIR / f"{job}_out{ext}"
    f.save(src_path)

    with JOBS_LOCK:
        JOBS[job] = {
            "status": "processing",
            "percent": 5,
            "message": "排隊中…",
            "filename": out_name,
            "ext": ext,
        }

    cfg = _make_config(strength, center=center)
    th = threading.Thread(
        target=_process_in_bg,
        args=(job, src_path, out_path, ext, cfg, dpi),
        daemon=True,
    )
    th.start()
    return jsonify({"job_id": job})


@app.route("/status/<job>")
def status(job):
    with JOBS_LOCK:
        s = JOBS.get(job)
    if not s:
        return jsonify({"status": "error", "message": "找不到工作"}), 404
    return jsonify(s)


@app.route("/download/<job>")
def download(job):
    with JOBS_LOCK:
        s = JOBS.get(job)
    if not s or s.get("status") != "done":
        abort(404)
    out_path = Path(s["out_path"])
    if not out_path.exists():
        abort(404)
    mime = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(s["ext"], "application/octet-stream")
    return send_file(out_path, mimetype=mime,
                     as_attachment=True, download_name=s["filename"])


# --------------------------- 啟動 --------------------------- #


def main():
    ip = _local_ip()
    print("=" * 60)
    print(" 書面文件清理 — 網頁版")
    print("=" * 60)
    print()
    print("  本機開啟：")
    print(f"    http://127.0.0.1:{PORT}/")
    print()
    print("  同 WiFi 的 iPhone / iPad 開啟：")
    print(f"    http://{ip}:{PORT}/")
    print()
    print("  按 Ctrl+C 結束伺服器")
    print("=" * 60)

    # 嘗試在電腦本機開瀏覽器
    if "--no-browser" not in sys.argv:
        try:
            threading.Timer(1.0, lambda: webbrowser.open(f"http://127.0.0.1:{PORT}/")).start()
        except Exception:
            pass

    # 0.0.0.0 讓區網其他裝置可以連
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
