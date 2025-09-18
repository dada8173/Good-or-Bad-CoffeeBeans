# 模型與設定檔說明

將推論所需的模型權重（`.pth`）與對應設定檔（`.json`，可選）放置於此資料夾。

## 1. 權重檔命名規則

若沒有額外的設定檔，系統會嘗試從檔名推斷資訊。建議遵循訓練腳本的預設格式：

```
{豆種}_{模型架構}[_Noback]_best_model.pth
```

例如：

- `ethiopia_washed_custom_best_model.pth`
- `honduras_natural_resnet18_Noback_best_model.pth`

沒有 JSON 設定時，系統會假設影像尺寸為 128、類別為 `bad / good`。

## 2. JSON 設定檔（建議）

建立與權重同名的 JSON（或自訂 `model_file` 欄位）可以明確指定模型資訊：

```json
{
  "key": "ethiopia_washed_custom_best_model",
  "display_name": "衣索比亞水洗豆 - Custom CNN (去背)",
  "bean_type": "ethiopia_washed",
  "architecture": "custom",
  "img_size": 128,
  "classes": ["bad", "good"],
  "model_file": "ethiopia_washed_custom_best_model.pth",
  "description": "使用去背資料訓練的 Custom CNN 模型"
}
```

欄位說明：

- `key`：模型唯一識別碼（未填時預設為檔名）。
- `display_name`：前端顯示名稱。
- `bean_type`：豆種資料夾名稱，例如 `ethiopia_washed`。
- `architecture`：支援 `resnet18`、`convnext_tiny`、`custom`、`ultrafast`。
- `img_size`：訓練時的輸入尺寸。
- `classes`：分類標籤，順序需與訓練時一致。
- `model_file`（選填）：權重檔檔名，預設為同名 `.pth`。
- `description`（選填）：顯示在前端的補充說明。

## 3. 常見問題

- 權重檔不存在：前端會顯示警告，推論時會回傳錯誤訊息。
- 類別數不符合：請確認 JSON `classes` 與模型輸出維度一致。
- 需要更新模型列表：新增檔案後重新整理網頁即可載入。
