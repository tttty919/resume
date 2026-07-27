FROM python:3.12-slim

WORKDIR /app

# System deps for PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY backend/ ./backend/
COPY skill-tester/ ./skill-tester/

# Persistent data directories (mounted as volumes in production)
RUN mkdir -p /app/data /app/uploads /app/chroma_data

# HuggingFace cache (avoid re-download on restart)
ENV HF_HOME=/app/hf_cache
RUN mkdir -p /app/hf_cache

# Railway sets PORT env var; fallback to 8766
EXPOSE 8766

CMD ["sh", "-c", "python -m backend.dev_server"]
