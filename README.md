# Qwen3-ASR OpenWhispr Bridge

這個專案會在 Linux 本機啟動一個 OpenAI-compatible 語音轉文字 API，讓
[OpenWhispr](https://github.com/OpenWhispr/openwhispr) 的 `Self-Hosted` 模式可以使用
[Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 模型。

目標是做一個可免費自架、可用快捷鍵語音輸入到 Terminal 的 ASR pipeline：OpenWhispr
負責快捷鍵、錄音與貼上，本專案負責把 OpenWhispr 傳來的音檔交給 Qwen3-ASR 轉文字。

## 系統需求

- Linux 桌面環境
- Python 3.12，由 `uv` 管理
- `ffmpeg`
- NVIDIA GPU 建議使用，CPU 也可嘗試但延遲會明顯變高
- OpenWhispr 桌面版

Ubuntu 範例：

```bash
sudo apt update
sudo apt install -y ffmpeg curl
curl -LsSf https://astral.sh/uv/install.sh | sh
```

確認 GPU：

```bash
nvidia-smi
```

## 從 GitHub 下載後直接啟動

```bash
git clone https://github.com/Patrick-zhuyanxun/qwen-openwhispr-asr.git
cd qwen-openwhispr-asr
./run_qwen_asr.sh --model-size 0.6B
```

第一次執行時，`uv` 會自動建立 `.venv` 並依照 `uv.lock` 安裝依賴；Qwen3-ASR 權重也會在
第一次載入模型時下載。

## OpenWhispr 設定

在 OpenWhispr 的 `Speech-to-Text` 設定中選 `Self-Hosted`，Server URL 填：

```text
http://127.0.0.1:8179/v1
```

## 可選：OpenWhispr 文字清理模型

如果你想讓 OpenWhispr 在語音轉文字後再做 AI 修正，例如移除口語 filler words、
修正標點、整理語法，可以到 OpenWhispr 的 `Language Models` 頁面設定。

這一段是文字後處理，不是 ASR 本體；ASR 仍然由本機 Qwen3-ASR bridge 負責。

設定方式：

1. 開啟 `Language Models`
2. 選 `Dictation Cleanup`
3. 打開 `Enable text cleanup`
4. 選 `Cloud Providers`
5. 下方 provider 選 `Custom`
6. `Endpoint URL` 填：

```text
https://generativelanguage.googleapis.com/v1beta/openai
```

7. `API Key` 填你的 Google Gemini API key
8. 如果畫面有 `Model` 或 `Model ID` 欄位，填：

```text
gemma-4-31b-it
```

這個 URL 是 Google Gemini 的 OpenAI-compatible endpoint，因此 OpenWhispr 可以把它當成
custom OpenAI-compatible language model provider 使用。這會呼叫 Google API；是否免費、
是否有額度限制，取決於你的 Google API key 與帳號方案。

目前建議用 `gemma-4-31b-it`，也就是 Gemma 4 31B，作為 `Dictation Cleanup`
的文字清理模型。選它的原因是：在目前這組 Google API 額度下，每日請求處理上限約可到
1,500 次，對快捷語音輸入的短句修正比較夠用。實際可用模型與每日請求上限仍以
Google AI Studio / Gemini API 後台顯示為準。

## 啟動或更換模型

只需要執行同一個腳本。每次執行時，它都會先停止舊的
`qwen-openwhispr-asr.service`，清掉殘留的 Qwen ASR process，然後重新用
`systemd --user` 啟動新的背景服務。

預設使用 `Qwen/Qwen3-ASR-0.6B`：

```bash
./run_qwen_asr.sh
```

指定模型大小：

```bash
./run_qwen_asr.sh --model-size 0.6B
./run_qwen_asr.sh --model-size 1.7B
```

切換模型時不需要手動停止舊服務，直接執行：

```bash
./run_qwen_asr.sh --model-size 1.7B
```

腳本會自動處理停止舊服務、清理殘留 process、啟動新服務、等待 health check。

也可以用環境變數：

```bash
QWEN_ASR_MODEL_SIZE=1.7B ./run_qwen_asr.sh
QWEN_ASR_DTYPE=float16 ./run_qwen_asr.sh
QWEN_ASR_PORT=8179 ./run_qwen_asr.sh
```

## 模型選擇建議

`0.6B` 是日常快捷輸入的建議預設：啟動較快、顯存壓力較低、延遲較小，適合在
Terminal 與 AI 對話時頻繁短句輸入。

`1.7B` 適合你更重視準確度，或要處理較長語音、混合中英文、專有名詞較多的內容。
代價是第一次下載較久、載入較慢、顯存與推理時間都會增加。

目前這台機器有 NVIDIA RTX A5000，兩個模型理論上都能跑；如果主要目標是「快捷輸入」，
先用 `0.6B` 比較實際。需要更高準確度時再切到 `1.7B`。

## 檢查是否開啟

服務狀態：

```bash
systemctl --user is-active qwen-openwhispr-asr.service
systemctl --user status qwen-openwhispr-asr.service --no-pager
journalctl --user-unit=qwen-openwhispr-asr.service -n 80 --no-pager
```

HTTP health check：

```bash
curl http://127.0.0.1:8179/health
```

正常會看到類似：

```json
{"ok":true,"model":"Qwen/Qwen3-ASR-0.6B","loaded":true,"cuda_available":true}
```

## API

OpenWhispr 會呼叫這個 endpoint：

```text
POST /v1/audio/transcriptions
```

本服務也支援：

- `POST /audio/transcriptions`
- `GET /health`
- `GET /v1/models`

## 手動測試轉錄

把任意音檔傳到 endpoint：

```bash
curl -X POST http://127.0.0.1:8179/v1/audio/transcriptions \
  -F file=@/path/to/audio.webm \
  -F model=Qwen/Qwen3-ASR-0.6B \
  -F language=auto
```

正常會回傳：

```json
{"text":"..."}
```

## 常見問題

### Port 8179 被佔用

直接重新執行腳本即可，它會先停掉舊服務與殘留 process：

```bash
./run_qwen_asr.sh --model-size 0.6B
```

### 想停止服務

```bash
systemctl --user stop qwen-openwhispr-asr.service
```

### 查看錯誤日誌

```bash
journalctl --user-unit=qwen-openwhispr-asr.service -n 120 --no-pager
```

### 第一次啟動很久

第一次會下載 Python 依賴與 Qwen3-ASR 權重，屬於正常現象。後續啟動會快很多。

## Rust 替代方案

[second-state/qwen3_asr_rs](https://github.com/second-state/qwen3_asr_rs) 也提供
OpenAI-compatible 的 `asr-server`，endpoint 同樣是 `POST /v1/audio/transcriptions`。
如果之後想改成單一 Rust binary，可以把它當成這個 Python bridge 的替代方案。
