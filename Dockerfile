# 使用官方 Python 輕量版作為基底
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴 (如果需要的話，目前 Torch/Flask 不需要額外系統庫)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案原始碼與模型
COPY . .

# 設定環境變數 (Hugging Face 會覆蓋 PORT，但我們預設 7860)
ENV PORT=7860

# 暴露端口
EXPOSE 7860

# 啟動命令 (使用 gunicorn 提高穩定性)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
