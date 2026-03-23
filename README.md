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

<h1 align="center">☕ Cofe_log：專業級咖啡豆 AI 分類器 | Coffee Bean Classifier</h1>

<p align="center">
  利用深度學習模型與即時影像處理技術，精準辨識咖啡豆的良品與瑕疵。<br>
  Advanced Real-time Deep Learning System for Coffee Bean Quality Inspection.
</p>

---

## 📂 專案結構 | Project Structure

```
Good-or-Bad-CoffeeBeans/
├── models/             # 已訓練模型與 JSON 設定 | Trained models & configs
├── static/             # 全新 Glassmorphism UI 素材 | Modern UI assets
├── templates/          # 網頁模板 (index.html) | Web templates
├── app.py              # Flask 後端 (支援即時偵測 API) | Backend with Real-time API
├── train_all.py        # 自動化多豆種訓練腳本 | Automated training script
├── coffee_beans_data/  # 原始與增強資料集 | Datasets
├── use_cnn_train.ipynb # 訓練腳本原始稿 | Training source
└── README.md
```

---

## 🧠 模型資訊 | Model Info

- **框架 | Framework**：PyTorch
- **架構 | Architecture**：Custom CNN (針對咖啡豆特徵優化)
- **已部署豆種**：
  - 衣索比亞水洗豆 (Ethiopia Washed) - `Custom CNN`
  - 宏都拉斯日曬豆 (Honduras Natural) - `Custom CNN`
- **訓練策略**：
  - 80/20 資料分割，使用 Adam 優化器。
  - 導入 EarlyStopping 確保最佳泛化能力。

---

## 🧪 核心功能 | Core Features

| 功能 | 描述 |
|------|------|
| 🎨 **Glassmorphism UI** | 專業級毛玻璃質感儀表板，支援暗色模式與動態視覺效果。 |
| 📹 **即時影像偵測** | 整合 WebRTC，支援攝影機現場即時分類，無需手動上傳。 |
| 🚀 **快速推論 API** | 專為即時幀處理設計的 `/api/predict_frame` 接口。 |
| 📊 **智能數據面板** | 即時顯示預測標籤、詳細置信度百分比與模型架構資訊。 |
| 📂 **自動化部署** | 掃描 `models/` 資料夾自動載入可用模型與對應設定。 |

---

## 🚀 啟動與使用 | Getting Started

### 環境準備
1. 建議使用 Conda 建立環境：
   ```bash
   conda create -n coffee-beans-env python=3.10
   conda activate coffee-beans-env
   pip install -r requirements.txt
   ```

### 啟動服務
1. 執行 `python app.py` 啟動 Flask 伺服器。
2. 開啟瀏覽器連線至 <http://localhost:5000>。

### 操作指南
1. **選擇模型**：在右側面板選擇預計分析的豆種。
2. **即時監測**：點擊「開啟相機」後，啟動「即時偵測」。AI 將不斷分析視窗中的影像。
3. **結果反饋**：系統將即時在預測面板更新結果；「Good」標示為綠色，「Bad」標示為紅色。

---

## 🖼️ 介面展示 | UI Showcase

### 專業儀表板 (Professional Dashboard)
![Dashboard Screenshot](file:///C:/Users/dachen/.gemini/antigravity/brain/ffbbab78-2eac-4a9a-9e93-34db6a862fac/verify_new_ui_and_realtime_1774097174040.webp)
*利用毛玻璃效果與暗色系配色，打造極具科技感的 AI 分類體驗。*

---

## 📊 資料集類別 | Classes

由我本人拍攝，透過 OpenCV 裁切後手工挑選。
- `good`：外觀完整，顏色自然的好豆。
- `bad`：破裂、霉變、色澤異常的瑕疵豆。

---

## 👨‍💻 開發者資訊 | About the Developer

- 作者 | Author：dachen8173
- 技術棧 | Stack：Python / PyTorch / Flask / WebRTC / Vanilla CSS
- 聯絡方式 | Contact：op.dada.op@gmail.com
- Instagram | [da_chen_527](https://www.instagram.com/da_chen_527) / [cofe_log](https://www.instagram.com/cofe_log)

---

## 📜 授權 | License

MIT License – 歡迎使用、改進並回饋社群！
要時新增對應的 JSON 設定檔（詳見 `models/README.md`）。
3. 執行 `python app.py` 啟動 Flask 伺服器。
4. 開啟瀏覽器連線至 <http://localhost:5000>。
5. 前端頁面提供模型清單、圖片預覽、分類結果與各類別機率，協助快速驗證模型效果。

---

## 📊 資料集類別 | Classes
由我本人拍攝，透過opencv裁切後手工挑選。
- `good`：外觀完整，顏色自然的好豆(我本人挑選的，非絕對專業!)
  *(good beans with intact and natural appearance)*
  *(Note: Manually selected; may not be professionally certified)*  
- `bad`：破裂、霉變、色澤異常的瑕疵豆(我本人挑選的，非絕對專業!)
  *(defective beans with damage or discoloration)*
  *(Note: Manually selected; may not be professionally certified)*

## 🌱 咖啡豆類型 | Coffee Bean Types

| 類型 | 處理方式 | 特色 |
|------|----------|------|
| 衣索比亞水洗豆 | Ethiopia Washed | 明亮酸度，花香調性 |
| 肯亞日曬豆 | Kenya Natural | 濃郁果香，甜度豐富 |
| 宏都拉斯日曬豆 | Honduras Natural | 平衡風味，巧克力調性 |

---

## 🖼️ 圖示展示 | Sample Images

### Good Bean
<img src="./samplePhoto/goodsample.jpg" width="200"/>

### Bad Bean
<img src="./samplePhoto/badsample.jpg" width="200"/>

---

## 🎯 未來規劃 | Future Plans

- [ ] 加入多視角融合（雙鏡頭）
- [ ] 模型部署為 Web 或 Mobile 應用
- [ ] 支援更多咖啡豆品種和處理方式

---

## 👨‍💻 開發者資訊 | About the Developer

- 作者 | Author：dachen8173
- 技術棧 | Stack：Python / PyTorch / OpenCV / Jupyter
- 聯絡方式 | Contact：op.dada.op@gmail.com
- instagram | da_chen_527 / cofe_log

---

## 📜 授權 | License

MIT License – 歡迎使用、改進並回饋社群！  
MIT License – Feel free to use, modify, and contribute.
