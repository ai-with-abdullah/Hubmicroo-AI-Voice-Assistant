FROM python:3.11-slim

# ── System libraries ──────────────────────────────────────────────────────────
# ffmpeg: audio decode for Whisper
# libgomp1: required by CTranslate2 (faster-whisper)
# curl / wget / tar: downloads + health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgomp1 curl wget tar \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
# Install CPU-only torch first (much smaller than CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# ── Optional: Piper TTS binary ─────────────────────────────────────────────────
# Piper releases: https://github.com/rhasspy/piper/releases
# Download the binary if available; skip gracefully if URL changes.
RUN mkdir -p /usr/local/bin /usr/lib/piper && \
    PIPER_URL="https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_x86_64.tar.gz" && \
    wget -q "$PIPER_URL" -O /tmp/piper.tar.gz 2>/dev/null && \
    tar -xzf /tmp/piper.tar.gz -C /tmp/ && \
    cp /tmp/piper/piper /usr/local/bin/piper && \
    cp -r /tmp/piper/espeak-ng-data /usr/lib/piper/ && \
    rm -rf /tmp/piper /tmp/piper.tar.gz || echo "Piper TTS not installed (non-fatal, TTS_ENABLED=false will skip it)"

# ── App source ────────────────────────────────────────────────────────────────
COPY backend/ ./backend/
COPY frontend/ ./frontend/
# Seed data — can be overridden by volume mount in docker-compose
COPY backend/data/ ./data/

# ── Piper voice models directory (mount voice .onnx files here) ───────────────
RUN mkdir -p /app/piper_voices

# ── Environment (Docker-specific overrides of config.py defaults) ─────────────
ENV PYTHONPATH=/app \
    DATA_DIR=/app/data \
    PRODUCTS_FILE=/app/data/products.json \
    PAGES_FILE=/app/data/pages.json \
    FRONTEND_DIR=/app/frontend \
    PIPER_VOICES_DIR=/app/piper_voices \
    TTS_ENABLED=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s \
    CMD curl -f http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
