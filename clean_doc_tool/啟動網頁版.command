#!/bin/bash
# ============================================================
# 書面文件清理 - 網頁版啟動 (macOS)
#
# 雙擊本檔即可在 Mac 上啟動網頁伺服器，
# 同 WiFi 內任何裝置 (iPhone / 舊 Win 電腦 / iPad) 用瀏覽器即可使用。
# ============================================================

cd "$(dirname "$0")"

VENV_DIR=".venv"
PY="${VENV_DIR}/bin/python3"
REQ="requirements.txt"

clear
echo "════════════════════════════════════════════════════════════"
echo "  書面文件清理 — 網頁版伺服器"
echo "════════════════════════════════════════════════════════════"
echo

# ----- 確認有 python3 -----
if ! command -v python3 >/dev/null 2>&1; then
  echo "[錯誤] 找不到 python3。"
  echo "       請開啟 Terminal 執行：xcode-select --install"
  echo "       或到 https://www.python.org/downloads/macos/ 下載安裝。"
  echo
  read -n 1 -s -r -p "按任意鍵關閉..."
  exit 1
fi

# ----- 第一次：建 venv 裝套件 -----
if [ ! -x "${PY}" ]; then
  echo "[初始化] 第一次啟動，正在建立 Python 環境..."
  echo "         （約 1-3 分鐘，需要網路）"
  echo
  if ! python3 -m venv "${VENV_DIR}"; then
    echo
    echo "[錯誤] 建立虛擬環境失敗。"
    read -n 1 -s -r -p "按任意鍵關閉..."
    exit 1
  fi

  if ! "${PY}" -m pip install --upgrade pip --quiet; then
    echo "[錯誤] pip 升級失敗。"
    read -n 1 -s -r -p "按任意鍵關閉..."
    exit 1
  fi

  if ! "${PY}" -m pip install -r "${REQ}"; then
    echo "[錯誤] 套件安裝失敗。可能是網路問題。"
    read -n 1 -s -r -p "按任意鍵關閉..."
    exit 1
  fi
  echo
  echo "[完成] 環境建立成功。"
  echo
fi

# ----- 列出本機 IP，方便手機/其他電腦連線 -----
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "127.0.0.1")

echo
echo "════════════════════════════════════════════════════════════"
echo "  伺服器啟動中..."
echo "════════════════════════════════════════════════════════════"
echo
echo "  ★ 本機網址 (在這台 Mac 上用):"
echo "        http://127.0.0.1:8765/"
echo
echo "  ★ 區網網址 (給手機 / 其他電腦用):"
echo "        http://${LOCAL_IP}:8765/"
echo
echo "  把上面那個區網網址用 LINE / 簡訊傳給使用者，"
echo "  使用者用 Safari / Chrome 等瀏覽器打開即可上傳檔案。"
echo
echo "  ※ 使用者必須跟這台 Mac 連同一個 WiFi。"
echo "  ※ 按 Ctrl+C 或關掉本視窗即停止伺服器。"
echo
echo "════════════════════════════════════════════════════════════"
echo

# ----- 啟動 Flask -----
"${PY}" web_server.py
EC=$?

echo
echo "伺服器已關閉 (結束碼: ${EC})"
read -n 1 -s -r -p "按任意鍵關閉視窗..."
