---
title: Coffee Bean Classifier
emoji: ☕
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

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
| 🎨 **介面設計** | 暗色 Glassmorphism 儀表板，顯示模型輸出機率與架構資訊。 |

## 🧠 Model & Dataset

### Dataset

- 圖片由開發者自行拍攝，使用 OpenCV 依輪廓裁切成單顆咖啡豆。
- 裁切圖片由開發者人工標示為 `good`、`bad`、`back` 或 `idontknow`。
- 已部署模型只使用 `good` 與 `bad` 類別；人工判定並非專業咖啡分級認證。
- Dataset 為私人資料，**不隨 GitHub repository 或 Hugging Face Model repository 發布**。
- Ethiopia Washed 與 Honduras Natural 已完成訓練；Kenya Natural 仍在資料整理與標註階段，暫不發布模型。

### Model

| 項目 | 設定 |
|---|---|
| Framework | PyTorch |
| Architecture | Custom CNN：3 個 Conv/BatchNorm/ReLU/MaxPool blocks + 256-unit fully connected layer |
| Task | Binary classification (`bad`, `good`) |
| Input | RGB, 128 × 128 |
| Preprocessing | Pad to square, resize, tensor conversion, normalize with mean/std 0.5 |
| Optimizer | Adam, learning rate `5e-5` |
| Loss | Class-weighted cross entropy based on the training subset |
| Training | Up to 40 epochs, early stopping patience 10 |
| Split | Existing 80/20 random train/validation split |

### Data augmentation

離線資料增強由 `data_augment.ipynb` 執行，每張來源 crop 產生兩個版本，包含 color jitter、random grayscale、Gaussian blur、sharpness adjustment 與 Gaussian noise。

## 📊 Validation Results

以下結果由修正後的 best-checkpoint 保存流程重新訓練取得。數值來自現有 augmented dataset 的 80/20 random validation split。

| Bean type | Validation samples | Accuracy | Macro precision | Macro recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|
| Ethiopia Washed | 143 | 79.02% | 78.57% | 79.16% | 78.72% | 79.13% |
| Honduras Natural | 114 | 78.07% | 69.30% | 73.89% | 70.76% | 79.22% |

### Per-class results

| Bean type | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Ethiopia Washed | bad | 84.42% | 78.31% | 81.25% | 83 |
| Ethiopia Washed | good | 72.73% | 80.00% | 76.19% | 60 |
| Honduras Natural | bad | 48.48% | 66.67% | 56.14% | 24 |
| Honduras Natural | good | 90.12% | 81.11% | 85.38% | 90 |

Confusion matrices use rows as actual labels and columns as predicted labels in `[bad, good]` order:

```text
Ethiopia Washed: [[65, 18], [12, 48]]
Honduras Natural: [[16, 8], [17, 73]]
```

> **Evaluation limitation:** To preserve the current project workflow, the split is performed after offline augmentation. Augmented variants of the same source crop may therefore appear in both training and validation sets. The validation set is also used for early stopping, and no independent test set is reported. These figures describe the current validation run and should not be treated as a leakage-free benchmark.

Honduras Natural remains weaker on the `bad` class, particularly precision. Accuracy alone would hide this behavior, so macro and per-class metrics are reported together.

Machine-readable results are available in `models/*_metrics.json`; epoch histories are stored in `models/*_history.json`.

## 🚀 Quick Start

### Web application

Python 3.10 or 3.11 is recommended.

```bash
conda create -n coffee-beans-env python=3.10
conda activate coffee-beans-env
pip install -r requirements.txt
python app.py
```

Open <http://localhost:5000> after the service starts.

At startup, the application downloads its weights from `dada8173/coffee-bean-classifier-models` at pinned revision `c859bc6049e24649ca86bf4f5d2600d0fbec6198`. Hugging Face's cache prevents unchanged weights from being downloaded again. If the Hub is unavailable, the application falls back to matching local `models/*.pth` files. Set `USE_LOCAL_MODELS=1` for explicitly local development, or set `HF_MODEL_REVISION` only when intentionally testing another published revision. Public model downloads do not require an access token.

### Training and notebooks

```bash
pip install -r requirements-dev.txt
python train_all.py
```

By default, `train_all.py` trains Ethiopia Washed and Honduras Natural only. Kenya Natural remains work in progress.

## 📖 Usage

1. Select an available bean-specific model.
2. Open the browser camera or upload a supported image.
3. Enable live detection when using the camera.
4. Review the predicted label and model output scores.

Browser camera access requires HTTPS or localhost and user permission.

## 📂 Project Structure

```text
Good-or-Bad-CoffeeBeans/
├── models/             # Model metadata, logs, histories and validation metrics
├── static/             # CSS, JavaScript and example assets
├── templates/          # Flask HTML template
├── app.py              # Web inference and model loading
├── train_all.py        # Ethiopia/Honduras training and evaluation
├── requirements.txt    # Web inference dependencies
├── requirements-dev.txt# Training/notebook dependencies
└── coffee_beans_data/  # Private local dataset; ignored by Git
```

## ⚠️ Limitations

- Labels are manually assigned and are not a professional coffee-grading certification.
- Each deployed model is specific to one bean type.
- The current validation protocol may contain augmented-sibling leakage as described above.
- Softmax output scores are not calibrated confidence intervals.
- Kenya Natural is under development and has no published model.

## 👨‍💻 Developer & License

- Author: dachen8173
- Stack: Python / PyTorch / Flask / OpenCV / JavaScript
- Contact: op.dada.op@gmail.com
- Instagram: [da_chen_527](https://www.instagram.com/da_chen_527) / [cofe_log](https://www.instagram.com/cofe_log)

The source code is released under the MIT License. The dataset is private and is not distributed with this project.
