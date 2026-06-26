# Running on Kaggle (Testing Only)

Kaggle gives you a **free T4 GPU (16 GB)** — enough to run Qdrant, Ollama + qwen3:4b, BGE-M3, and Whisper simultaneously.

## Quick start

1. Upload `run_on_kaggle.ipynb` to a new Kaggle notebook.
2. Enable **GPU T4 x2** accelerator in Settings → Accelerator.
3. Enable **Internet** access in Settings → Internet.
4. Run all cells — the last cell starts the backend and prints the public `ngrok` URL.
5. Open the URL + `/` to see the store widget, or `/admin` for the admin panel.

## What the notebook does

| Step | What happens |
|------|--------------|
| 1 | Install system deps (ffmpeg, curl) |
| 2 | Install Python packages from requirements.txt |
| 3 | Start Qdrant in background |
| 4 | Install Ollama, pull `qwen3:4b` |
| 5 | Download BGE-M3 embeddings |
| 6 | Clone/copy the repo code |
| 7 | Seed the vector index from `products.json` + `pages.json` |
| 8 | Start FastAPI with uvicorn |
| 9 | Expose via `pyngrok` (free tunnel) |
| 10 | Print public URL and run the eval harness |

## Notes

- **This is for testing only.** Kaggle sessions expire after 12 hours.
- The Kaggle free tier does not persist data between sessions.
- For production, run `docker compose up` on a VPS with 8–16 GB RAM.
- Piper TTS voice models are large — TTS is disabled by default on Kaggle (`TTS_ENABLED=false`).
