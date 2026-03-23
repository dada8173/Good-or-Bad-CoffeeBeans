# ☕ Cofe_log 專案技術手冊 (claude.md)

## 1. 專案概覽
本專案為一套基於深度學習的專業級咖啡豆 AI 分類系統，旨在透過即時影像處理識別咖啡豆品質。

- **核心技術**：Python, Flask, PyTorch, OpenCV, WebRTC
- **UI 風格**：現代化 Glassmorphism (毛玻璃) 質感

---

## 2. 系統架構

### 後端 (Flask API)
- **進入點**：`app.py`
- **關鍵功能**：
  - 自動掃描 `models/` 目錄並動態加載多個豆種模型。
  - 提供 `/api/predict_frame` RESTful 接口，接收 Base64 影像幀並返回預測機率。
  - 使用 `PIL` 與 `torchvision.transforms` 對輸入影像進行標準化處理。

### 前端 (WebRTC & Vanilla JS)
- **進入點**：`templates/index.html`, `static/app.js`
- **關鍵功能**：
  - 透過瀏覽器獲取攝影機串流。
  - 定時捕捉影像幀並異步傳輸至後端推論處理。
  - 結合動態數據面板，實效呈現「Good」與「Bad」特徵分析。

---

## 3. 目錄結構與關鍵組件

```text
Good-or-Bad-CoffeeBeans/
├── app.py              # 核心後端逻辑與模型管理
├── models/             # 存放訓練完成之權重檔 (.pth) 與中繼資料 (.json)
├── static/
│   ├── app.js          # 前端 WebRTC 控制與 API 請求邏輯
│   └── styles.css      # Glassmorphism UI 樣式定義
├── templates/
│   └── index.html      # 主介面 HTML5 模板
└── train_all.py        # 離線訓練腳本，生成特定豆種之模型
```

---

## 4. 模型規範

### 設定檔格式 (.json)
模型必須配備對應的 JSON 設定檔，包含以下欄位：
- `model_name`: 模型顯示名稱。
- `model_path`: 權重檔案路徑。
- `classes`: 類別索引對應 (e.g., {"0": "Bad", "1": "Good"})。

### 推論預處理
影像縮放比例需與訓練時一致（通常為 128x128 像素），並執行 ImageNet 標準化：
```python
transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

---

## 5. 開發規範 (AI 助手行為)

### 協作守則
- **UI 修改**：優先保持細膩的毛玻璃質感。禁止移除 `styles.css` 中的變數定義。
- **邏輯變更**：若涉及 `app.py` 修改，需確保動態掃描 `models/` 的邏輯不被破壞。
- **新增功能**：如需新增豆種，應在 `models/` 下建立新的權重檔與對應 JSON。

---

## 6. 使用流程
1. 啟動環境：`pip install -r requirements.txt`
2. 執行後端：`python app.py`
3. 存取入口：`http://localhost:5000`
4. 操作步驟：選擇豆種 -> 開啟相機 -> 點擊「即時偵測」。
