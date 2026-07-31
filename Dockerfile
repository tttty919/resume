FROM python:3.12

WORKDIR /app

# 系统依赖（chromadb 需要 libgomp1，pymupdf 1.24+ 自带 MuPDF）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY backend/ ./backend/
COPY skill-tester/ ./skill-tester/

# 持久化数据目录
RUN mkdir -p /app/data /app/uploads /app/chroma_data

# HuggingFace 缓存（避免重启重复下载 embedding 模型）
ENV HF_HOME=/app/hf_cache
RUN mkdir -p /app/hf_cache

# 容器内对外监听所有网卡；Railway 会注入 PORT，本地 docker 回落到 8766
ENV HOST=0.0.0.0
EXPOSE 8766

CMD ["python", "-m", "backend.dev_server"]
