"""Self-hosted LLM + embeddings via Ollama.

No third-party API keys. Everything talks to a local Ollama server that you run
on your own machine/server. Swap the model by changing LLM_MODEL in config.py.
"""
import requests

from . import config


class LLMError(RuntimeError):
    pass


def _post(path: str, payload: dict, timeout: float):
    url = f"{config.OLLAMA_URL}{path}"
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        raise LLMError(
            "Cannot reach Ollama at "
            f"{config.OLLAMA_URL}. Start it with `ollama serve` and pull the "
            f"models: `ollama pull {config.LLM_MODEL}` and "
            f"`ollama pull {config.EMBED_MODEL}`."
        )
    except requests.exceptions.HTTPError as e:
        raise LLMError(f"Ollama error: {e} — is the model pulled?")


def embed(texts):
    """Return a list of embedding vectors for a list of strings."""
    vectors = []
    for text in texts:
        data = _post(
            "/api/embeddings",
            {"model": config.EMBED_MODEL, "prompt": text,
             "keep_alive": config.KEEP_ALIVE},
            timeout=config.OLLAMA_TIMEOUT,
        )
        vectors.append(data["embedding"])
    return vectors


def embed_one(text):
    return embed([text])[0]


def chat(system_prompt: str, user_message: str) -> str:
    """Single-turn grounded answer from the self-hosted LLM."""
    data = _post(
        "/api/chat",
        {
            "model": config.LLM_MODEL,
            "stream": False,
            "keep_alive": config.KEEP_ALIVE,
            "options": {
                "temperature": config.LLM_TEMPERATURE,
                "num_predict": config.MAX_TOKENS,
            },
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        },
        timeout=config.OLLAMA_TIMEOUT,
    )
    return data["message"]["content"].strip()


def health() -> dict:
    """Report whether Ollama is up and the required models are present."""
    try:
        r = requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        names = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        names |= {m["name"] for m in r.json().get("models", [])}
        return {
            "ollama": True,
            "llm_model": config.LLM_MODEL,
            "llm_ready": config.LLM_MODEL in names
            or config.LLM_MODEL.split(":")[0] in names,
            "embed_model": config.EMBED_MODEL,
            "embed_ready": config.EMBED_MODEL in names
            or config.EMBED_MODEL.split(":")[0] in names,
        }
    except Exception:
        return {"ollama": False, "llm_model": config.LLM_MODEL,
                "embed_model": config.EMBED_MODEL}
