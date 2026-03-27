---
title: Coffee Bean Classifier
emoji: ☕
colorFrom: brown
colorTo: orange
sdk: docker
pinned: false
---

<p align="center">
  <img src="./samplePhoto/logo.png" width="160"/>
</p>

<h1 align="center">☕ Cofe_log：咖啡豆 AI 分類器</h1>
<p align="center"><b>Coffee Bean Classifier</b></p>

<p align="center">
  <a href="https://huggingface.co/spaces/dada8173/coffee-bean-classifier" target="_blank">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces">
  </a>
</p>

<p align="center">
  利用深度學習模型與即時影像處理技術，辨識咖啡豆的良品與瑕疵。<br>
  <i>Advanced Real-time Deep Learning System for Coffee Bean Quality Inspection.</i>
</p>

---

## 🌟 核心亮點 | Core Features

| 功能 | 描述 |
|------|------|
| 🎨 **Glassmorphism UI** | 專業級毛玻璃質感儀表板，支援暗色模式與動態視覺效果。 |
| 📹 **即時影像偵測** | 整合 WebRTC，支援攝影機現場即時分類，無需手動上傳。 |
| 🚀 **快速推論 API** | 專為即時幀處理設計的 `/api/predict_frame` 接口。 |
| 📊 **智能數據面板** | 即時顯示預測標籤、詳細置信度百分比與模型架構資訊。 |
| 📂 **自動化部署** | 掃描 `models/` 資料夾自動載入可用模型與對應設定。 |

---

## 🚀 快速上手 | Quick Start

### 1. 環境準備
建議使用 Conda 建立環境（Python 3.10+）：
```bash
conda create -n coffee-beans-env python=3.10
conda activate coffee-beans-env
pip install -r requirements.txt
```

### 2. 啟動服務
執行以下指令啟動 Flask 伺服器：
```bash
python app.py
```
啟動後，請開啟瀏覽器連線至 http://localhost:5000。

---

## 📖 操作指南 | Usage Guide

1. **選擇模型**：在右側面板選擇要分析的豆種（如：宏都拉斯日曬、衣索比亞水洗）。
2. **啟動相機**：點擊「開啟相機」按鈕獲取影像流。
3. **即時偵測**：點擊「即時偵測」啟動 AI。系統將不斷分析視窗中的影像。
4. **檢查結果**：
   - <span style="color:green">**Good**</span>：代表良品豆，外觀完整。
   - <span style="color:red">**Bad**</span>：代表瑕疵豆，包含破裂、霉變或色澤異常。

---

## 🧠 技術細節 | Technical Deep Dive

### 模型架構
- **框架**：PyTorch
- **架構**：Custom CNN (針對咖啡豆特徵進行優化，包含多層卷積與池化)
- **已部署豆種**：
  - 衣索比亞水洗豆 (Ethiopia Washed)
  - 宏都拉斯日曬豆 (Honduras Natural)

### 資料集類別
由開發者親自拍攝並透過 OpenCV 裁切後手工挑選：
- **Good (良品)**：外觀完整，顏色自然。
- **Bad (瑕疵)**：破裂、霉變、色澤異常。
  *(註：手工挑選，可能與絕對專業標準有異)*

---

## 🛠️ 複現與訓練指南 | Reproduction & Training Guide

想要從零開始訓練您自己的咖啡豆分類模型？請遵循以下步驟：

### 1. 資料準備 (Data Preparation)
由於資料集較大，原始圖片並未上傳至 GitHub。請按照以下結構放置您的圖片：
- **路徑**：`coffee_beans_data/{bean_type}/crop/classByhands/`
- **子目錄**：`good/` (良品) 與 `bad/` (瑕疵)。
- **支援豆種**：`ethiopia_washed`, `kenya_natural`, `honduras_natural`。

### 2. 資料增強 (Data Augmentation)
為了提高模型的泛化能力與平衡樣本量，請執行：
- **檔案**：`data_augment.ipynb`
- **操作**：依序執行 Notebook 中的所有儲存格。它會讀取原始圖片並套用隨機旋轉、亮度調整、雜訊注入等技術。
- **產出**：增強後的圖片將儲存在 `{bean_type}/corp_augmented_dataNoback/` 中。

### 3. 模型訓練 (Model Training)
準備好資料後，即可執行自動化訓練腳本：
```bash
python train_all.py
```
- **流程**：該腳本會遍歷所有豆種，自動執行：80/20 資料分割 -> Custom CNN 架構初始化 -> Adam 優化器訓練 -> EarlyStopping 監控。
- **模型儲存**：訓練完成且驗證損失最低的模型將以 `.pth` 格式儲存在 `models/` 資料夾中。

### 4. 運行應用程式 (Run App)
確保 `models/` 下有對應的 `.pth` 與 `.json` 設定檔後，執行 `python app.py` 即可在本地或 Hugging Face 上啟動即時偵測！

---

## 📂 專案結構 | Project Structure

```text
Good-or-Bad-CoffeeBeans/
├── models/             # 已訓練模型與 JSON 設定
├── static/             # UI 素材與前端邏輯 (Glassmorphism)
├── templates/          # 網頁模板 (index.html)
├── app.py              # Flask 後端與推論 API
├── train_all.py        # 自動化多豆種訓練腳本
├── coffee_beans_data/  # 原始與增強資料集
└── samplePhoto/        # 範例圖片與 Logo
```

---

## 🖼️ 展示空間 | Showcase

### 範例對比
<p align="center">
  <img src="./samplePhoto/goodsample.jpg" width="300" alt="Good Bean"/>
  <img src="./samplePhoto/badsample.jpg" width="300" alt="Bad Bean"/>
  <br>
  <em>左：良品豆 (Good) | 右：瑕疵豆 (Bad)</em>
</p>

---

## 🎯 未來規劃 | Future Plans

- [ ] 支援多視角融合偵測（雙鏡頭技術）
- [ ] 擴展至更多咖啡豆處理方式（如：蜜處理）
- [ ] 優化手機瀏覽器端的執行效能

---

## 👨‍💻 開發者與授權 | About & License

- **作者**：dachen8173
- **技術棧**：Python / PyTorch / Flask / WebRTC / Vanilla CSS
- **聯絡方式**：op.dada.op@gmail.com
- **Instagram**：[da_chen_527](https://www.instagram.com/da_chen_527) / [cofe_log](https://www.instagram.com/cofe_log)

本專案採用 **MIT License** 授權 – 歡迎開發者使用、改進並回饋社群！
