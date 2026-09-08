<p align="center">
  <img src="./samplePhoto/logo.png" width="160" alt="Cofe_log logo"/>
</p>

<h1 align="center">☕ Cofe_log：咖啡豆 AI 分類器</h1>
<p align="center"><b>Coffee Bean Classifier</b></p>

<p align="center">
  <a href="https://huggingface.co/spaces/dada8173/coffee-bean-classifier">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces">
  </a>
  <a href="https://huggingface.co/dada8173/coffee-bean-classifier-models">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Repository-yellow" alt="Hugging Face model repository">
  </a>
</p>

<p align="center">
  使用 PyTorch CNN 與瀏覽器相機輸入，透過 Flask Web 介面辨識咖啡豆的良品與瑕疵。<br>
  <i>A Flask web application that classifies coffee beans as good or defective using a PyTorch CNN.</i>
</p>

## 🌟 核心功能 | Core Features

| 功能 | 描述 |
|---|---|
| 📹 **即時推論** | 定時擷取瀏覽器相機畫面並送至 Flask API 進行 near-real-time inference。 |
| 🧠 **Custom CNN / PyTorch** | 使用三層卷積區塊的二元影像分類模型。 |
| 🖼️ **影像前處理** | 使用 OpenCV 裁切單顆咖啡豆，推論時執行等比例補邊、縮放與標準化。 |
| 📷 **瀏覽器相機** | 使用 MediaDevices `getUserMedia()` 取得相機影像，也支援靜態圖片上傳。 |
| 🚀 **Web 部署** | Flask、Gunicorn 與 Docker，可部署至 Hugging Face Spaces。 |

## 🧠 模型與資料集

### 資料集

- 原始影像由作者自行拍攝，再使用 OpenCV 依輪廓裁切成單顆咖啡豆圖片。
- 每張裁切圖片皆由作者依影像內容人工分類：
  - `good`：外觀完整，依本專案的標註準則判定為良品。
  - `bad`：可觀察到明顯異常，依本專案的標註準則判定為瑕疵豆。
  - `back`：拍攝到咖啡豆背面，僅憑該角度不易判斷是否具有瑕疵。
  - `idontknow`：影像無法清楚辨識、咖啡豆不完整，或資訊不足以可靠標註。
- 已部署模型只使用可明確判定的 `good` 與 `bad` 類別；`back` 與 `idontknow` 不納入目前的二元分類訓練。
- 本專案目前公開程式碼、模型權重與評估紀錄；自行拍攝的訓練影像尚未對外發布。
- 上述標籤是本專案使用的人工判定，不代表專業咖啡分級認證。
- Ethiopia Washed 與 Honduras Natural 已完成訓練；Kenya Natural 仍在資料整理與標註階段，暫不發布模型。

### 模型

| 項目 | 設定 |
|---|---|
| 框架 | PyTorch |
| 架構 | Custom CNN：3 個 Conv/BatchNorm/ReLU/MaxPool 區塊，加上 256-unit 全連接層 |
| 任務 | 二元分類（`bad`、`good`） |
| 輸入 | RGB，128 × 128 |
| 前處理 | 補邊成正方形、縮放、轉換為 tensor，並以 mean/std 0.5 標準化 |
| 最佳化器 | Adam，learning rate `5e-5` |
| 損失函數 | 依訓練子集計算權重的交叉熵 |
| 訓練 | 最多 40 epochs，early stopping patience 10 |
| 切分 | 沿用既有 80/20 隨機訓練／驗證切分 |

### 資料增強

離線資料增強由 `data_augment.ipynb` 執行，每張來源 crop 產生兩個版本，包含 color jitter、random grayscale、Gaussian blur、sharpness adjustment 與 Gaussian noise。

## 📊 驗證結果

以下結果由修正後的 best-checkpoint 保存流程重新訓練取得。數值來自現有 augmented dataset 的 80/20 random validation split。

| 咖啡豆類型 | 驗證樣本數 | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Ethiopia Washed | 143 | 79.02% | 78.57% | 79.16% | 78.72% | 79.13% |
| Honduras Natural | 114 | 78.07% | 69.30% | 73.89% | 70.76% | 79.22% |

### 各類別結果

| 咖啡豆類型 | 類別 | Precision | Recall | F1 | 樣本數 |
|---|---|---:|---:|---:|---:|
| Ethiopia Washed | bad | 84.42% | 78.31% | 81.25% | 83 |
| Ethiopia Washed | good | 72.73% | 80.00% | 76.19% | 60 |
| Honduras Natural | bad | 48.48% | 66.67% | 56.14% | 24 |
| Honduras Natural | good | 90.12% | 81.11% | 85.38% | 90 |

混淆矩陣以實際標籤為列、預測標籤為欄，順序皆為 `[bad, good]`：

```text
Ethiopia Washed: [[65, 18], [12, 48]]
Honduras Natural: [[16, 8], [17, 73]]
```

> **評估限制：** 為保留目前專案流程，資料是在離線增強後才進行切分，因此同一來源 crop 的增強版本可能同時出現在訓練集與驗證集。驗證集也用於 early stopping，目前沒有獨立測試集。這些數字只代表本次驗證結果，不應視為完全排除資料洩漏的 benchmark。

Honduras Natural 對 `bad` 類別的表現仍較弱，尤其是 precision。單看 accuracy 會掩蓋此問題，因此同時呈現 macro 與各類別指標。

機器可讀的結果位於 `models/*_metrics.json`，各 epoch 紀錄位於 `models/*_history.json`。

## 🚀 快速開始

### Web 應用程式

建議使用 Python 3.10 或 3.11。

```bash
conda create -n coffee-beans-env python=3.10
conda activate coffee-beans-env
pip install -r requirements.txt
python app.py
```

服務啟動後，開啟 <http://localhost:5000>。

應用程式啟動時，會從 `dada8173/coffee-bean-classifier-models` 的固定 revision `efc47d2ec57916c5ea152333bd801a07a6977f59` 下載權重。Hugging Face cache 會避免重複下載未變更的檔案；若 Hub 暫時無法使用，程式才會改用本機 `models/*.pth` 備援。需要強制使用本機模型時可設定 `USE_LOCAL_MODELS=1`；只有在刻意測試其他已發布版本時才應設定 `HF_MODEL_REVISION`。下載公開模型不需要 access token。

### 訓練與 notebooks

```bash
pip install -r requirements-dev.txt
python train_all.py
```

`train_all.py` 預設只訓練 Ethiopia Washed 與 Honduras Natural；Kenya Natural 仍在開發中。

## 📖 使用方式

1. 選擇對應咖啡豆類型的可用模型。
2. 開啟瀏覽器相機，或上傳支援格式的圖片。
3. 使用相機時可開啟即時辨識。
4. 查看預測標籤與模型輸出分數。

瀏覽器相機功能需要 HTTPS 或 localhost，並須取得使用者授權。

## 📂 專案結構

```text
Good-or-Bad-CoffeeBeans/
├── models/             # 模型設定、訓練紀錄與驗證指標
├── static/             # CSS、JavaScript 與範例圖片
├── templates/          # Flask HTML 模板
├── app.py              # Web 推論與模型載入
├── train_all.py        # Ethiopia／Honduras 訓練與評估
├── requirements.txt    # Web 推論相依套件
├── requirements-dev.txt# 訓練與 notebook 相依套件
└── coffee_beans_data/  # 私人本機資料集；Git 會忽略此目錄
```

## ⚠️ 限制

- Labels are manually assigned and are not a professional coffee-grading certification.
- Each deployed model is specific to one bean type.
- The current validation protocol may contain augmented-sibling leakage as described above.
- Softmax output scores are not calibrated confidence intervals.
- Kenya Natural is under development and has no published model.

## 👨‍💻 開發者與授權

- Author: dachen8173
- Stack: Python / PyTorch / Flask / OpenCV / JavaScript
- Contact: op.dada.op@gmail.com
- Instagram: [da_chen_527](https://www.instagram.com/da_chen_527) / [cofe_log](https://www.instagram.com/cofe_log)

原始碼採用 MIT License；資料集維持私人狀態，不隨本專案發布。
