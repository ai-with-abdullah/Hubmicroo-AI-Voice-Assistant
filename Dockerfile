# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps: ffmpeg (Whisper audio decode), curl (health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg curl wget tar \
    && rm -rf /var/lib/apt/lists/*

# Install Piper TTS binary
RUN mkdir -p /usr/local/bin && \
    wget -q https://github.com/rhasspy/piper/releases/download/v1.2.0/piper_linux_x86_64.tar.gz \
    -O /tmp/piper.tar.gz && \
    tar -xzf /tmp/piper.tar.gz -C /usr/local/bin --strip-components=1 piper/piper && \
    rm /tmp/piper.tar.gz || echo "Piper download skipped (non-fatal)"

COPY --from=builder /install /usr/local

WORKDIR /app

# App source
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY backend/data/ ./data/

# Piper voice models directory (mounted or pre-downloaded)
RUN mkdir -p /app/piper_voices

ENV PYTHONPATH=/app \
    DATA_DIR=/app/data \
    PRODUCTS_FILE=/app/data/products.json \
    PAGES_FILE=/app/data/pages.json \
    PIPER_VOICES_DIR=/app/piper_voices \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
