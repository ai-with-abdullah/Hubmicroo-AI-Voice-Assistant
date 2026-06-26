# Putting the Assistant on the real Hubmicroo website

The chat bubble is a **widget**. It runs on the **existing** hubmicroo.com site —
you do NOT replace their website, and the demo page (`index.html`) is only a test
page. You add 3 lines to their site, and the bubble appears on every page.

## How the pieces fit
```
Customer's browser on hubmicroo.com
        │  (the 3 widget lines load the chat bubble)
        ▼
YOUR server  ──►  backend (FastAPI)  ──►  Ollama (the AI brain)
                         │
                         └──►  live products + website knowledge (RAG)
```
- **Their website** = unchanged, just 3 extra lines.
- **Your server** = runs the backend + Ollama. This is where the AI lives.

## Step 1 — Put the backend on a server (one time)
Run the backend + Ollama on a server the public can reach (the client's server,
or a cloud box like Hetzner/DigitalOcean/AWS). Easiest way:
```bash
docker compose up --build -d
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec ollama ollama pull bge-m3
docker compose exec backend python -c "from app import indexer; print(indexer.reindex_all())"
```
Give it a domain + HTTPS (needed for the microphone), e.g. `https://ai.hubmicroo.com`.

## Step 2 — Add 3 lines to hubmicroo.com
Ask the client's web developer to paste this into the site template (so it shows
on every page), replacing `https://ai.hubmicroo.com` with your server address:
```html
<link rel="stylesheet" href="https://ai.hubmicroo.com/static/assistant.css">
<script>window.HUBMICROO_API_BASE = "https://ai.hubmicroo.com";</script>
<script src="https://ai.hubmicroo.com/static/assistant.js"></script>
```
✅ The 🎤 chat bubble now appears on their real website — text **and** voice,
English / Urdu / Arabic.

## Step 3 — Connect live products
In `.env`, set `STORE_CATALOGUE_URL` (or `STORE_SEARCH_URL`) to Hubmicroo's
product API so answers and cards use live data. Map field names if needed
(`FIELD_NAME`, `FIELD_PRICE`, …). Then open `/admin` → **Re-index**.

## Security before go-live
- Lock `CORS allow_origins` to `https://www.hubmicroo.com` (in `backend/app/main.py`).
- Put `/admin`, `/api/products`, `/api/reindex` behind a password.
- Serve over HTTPS.

## What to test on the live site
- Bubble appears bottom-right on hubmicroo.com.
- Ask a product question (EN/UR/AR) → spoken answer + product cards with buy links.
- Ask a policy question ("return policy?", "delivery to Lahore?") → grounded answer.
- Add a product in `/admin` → it's answerable within seconds (no retraining).
```
