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
http://127.0.0.1:8179/v1
```

7. `API Key` 填你的 Google Gemini API key
8. 如果畫面有 `Model` 或 `Model ID` 欄位，可以選或填：

```text
gemma-4-31b-it
```

這裡填的是本機 proxy，不是 Google 官方 endpoint。原因是 OpenWhispr 目前會把
`generativelanguage.googleapis.com` 這類已知非 OpenAI provider 的 Custom URL 拒絕掉，
然後 fallback 到 OpenAI endpoint，造成 Google API key 被拿去打 OpenAI。

本服務會把 OpenWhispr 送來的 OpenAI-compatible `chat/completions` 或 `responses`
請求轉成 Google Gemini API `generateContent` 請求，所以可以在 OpenWhispr 的
`Custom` provider 裡使用 Google Gemini / Gemma 模型。

目前建議用 `gemma-4-31b-it`，也就是 Gemma 4 31B，作為 `Dictation Cleanup`
的文字清理模型。選它的原因是：在目前這組 Google API 額度下，每日請求處理上限約可到
1,500 次，對快捷語音輸入的短句修正比較夠用。實際可用模型與每日請求上限仍以
Google AI Studio / Gemini API 後台顯示為準。

本機 `/v1/models` 會列出 Google 目前支援的文字模型 fallback 清單；如果 OpenWhispr
會讀取 Custom provider 的 models endpoint，就會看到這些選項。若 UI 沒有自動顯示，
也可以直接手動輸入 model id。

目前 fallback 清單包含：

```text
gemma-4-31b-it
gemma-4-26b-a4b-it
gemini-3.5-flash
gemini-3.1-pro-preview
gemini-3.1-pro-preview-customtools
gemini-3-flash-preview
gemini-3.1-flash-lite
gemini-3.1-flash-lite-preview
gemini-2.5-flash
gemini-2.5-flash-lite
gemini-2.5-pro
```

如果你的 Google API key 可用，本服務也可以透過 Google `models.list` 讀取即時模型清單，
並保留 `supportedGenerationMethods` 包含 `generateContent` 的文字生成模型。
OpenWhispr 有時會把 Gemma 存成 `models/gemma-4-31b-it`，本服務會自動正規化成
`gemma-4-31b-it`。

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
GEMINI_CLEANUP_MODEL=gemma-4-31b-it ./run_qwen_asr.sh
```

通常不需要把 Google API key 放在環境變數；直接填在 OpenWhispr 的 `API Key` 欄位即可。
如果你想讓服務啟動後即使 request 沒帶 key 也能呼叫 Google，可以用：

```bash
GEMINI_API_KEY=你的_google_api_key ./run_qwen_asr.sh
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
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /health`
- `GET /v1/models`
- `GET /models`

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

測試文字清理 proxy：

```bash
curl -X POST http://127.0.0.1:8179/v1/chat/completions \
  -H "Authorization: Bearer $GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-31b-it",
    "messages": [
      {"role": "system", "content": "Clean up dictation text. Return only the corrected text."},
      {"role": "user", "content": "呃 幫我 修正 這段話 的 標點"}
    ]
  }'
```

查看本機提供給 OpenWhispr 的模型選項：

```bash
curl http://127.0.0.1:8179/v1/models
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

### 文字清理回傳整段 prompt

如果 OpenWhispr 的 cleaned text 變成 `Input:`、`Role:`、`Task:` 這類提示詞原文，代表
language model 把 OpenWhispr 的 cleanup prompt 當成一般文字處理了。本服務會偵測這種
OpenWhispr cleanup prompt，只把真正的 `Input` 內容送給 Gemini / Gemma，並在 proxy 層加上
嚴格的 cleanup system instruction。

更新後重新執行腳本即可：

```bash
./run_qwen_asr.sh --model-size 0.6B
```

### 第一次啟動很久

第一次會下載 Python 依賴與 Qwen3-ASR 權重，屬於正常現象。後續啟動會快很多。

## Rust 替代方案

[second-state/qwen3_asr_rs](https://github.com/second-state/qwen3_asr_rs) 也提供
OpenAI-compatible 的 `asr-server`，endpoint 同樣是 `POST /v1/audio/transcriptions`。
如果之後想改成單一 Rust binary，可以把它當成這個 Python bridge 的替代方案。
