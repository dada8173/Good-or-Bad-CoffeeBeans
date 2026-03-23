# 使用官方 Python 輕量版作為基底
FROM python:3.9-slim

# 設定環境變數
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user

# 創建一個非 root 用戶 (Hugging Face 預設使用 UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

# 設定工作目錄
WORKDIR $HOME/app

# 複製 requirements.txt 並安裝 Python 依賴
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 複製專案原始碼與模型，並確保權限正確
COPY --chown=user . .

# 暴露端口
EXPOSE 7860

# 啟動命令 (使用 gunicorn)
CMD ["gunicorn", "--bind", "0.0.0.0:7860", "app:app"]
