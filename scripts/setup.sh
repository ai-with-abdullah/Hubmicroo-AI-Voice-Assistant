#!/usr/bin/env bash
# One-time local setup (no Docker). Run from the project root.
set -e

echo "==> 1/4  Python dependencies"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

echo "==> 2/4  Checking Ollama"
if ! command -v ollama >/dev/null 2>&1; then
  echo "    Ollama not found. Install it from https://ollama.com/download, then re-run."
  exit 1
fi

echo "==> 3/4  Pulling the self-hosted models (one-time download, no training)"
ollama pull qwen2.5:3b
ollama pull bge-m3

echo "==> 4/4  Building the first index from the demo catalogue + pages"
cd backend
python -c "from app import indexer; print(indexer.reindex_all())"

echo ""
echo "Done. Start the server with:"
echo "  source .venv/bin/activate && cd backend && uvicorn app.main:app --reload"
echo "Then open http://localhost:8000"
