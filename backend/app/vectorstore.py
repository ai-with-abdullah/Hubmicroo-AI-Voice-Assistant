"""Tiny persistent vector store (numpy + cosine similarity).

Plenty fast for a store catalogue + website pages (hundreds–thousands of
chunks). For very large sites, swap the search loop for FAISS — the interface
stays the same. State is saved to data/vector/ so it survives restarts.

Each record has:
  text       : the chunk text
  source_id  : id of the thing it came from (product id or page url) so we can
               re-index just that item incrementally without rebuilding all.
  meta       : free-form dict (type, product payload, title, url ...)
"""
import json

import numpy as np

from . import config

_EMB_FILE = config.VECTOR_DIR / "embeddings.npy"
_META_FILE = config.VECTOR_DIR / "records.json"


class VectorStore:
    def __init__(self):
        self.vectors = np.zeros((0, 0), dtype="float32")
        self.records = []          # list of {text, source_id, meta}
        self.load()

    # ---- persistence ----------------------------------------------------
    def load(self):
        if _EMB_FILE.exists() and _META_FILE.exists():
            self.vectors = np.load(_EMB_FILE)
            self.records = json.loads(_META_FILE.read_text(encoding="utf-8"))

    def save(self):
        config.VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        np.save(_EMB_FILE, self.vectors)
        _META_FILE.write_text(
            json.dumps(self.records, ensure_ascii=False), encoding="utf-8"
        )

    # ---- mutation -------------------------------------------------------
    @staticmethod
    def _normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms

    def clear_source(self, source_id: str):
        """Drop every chunk that came from this product/page (for re-index)."""
        keep = [i for i, r in enumerate(self.records) if r["source_id"] != source_id]
        if len(keep) != len(self.records):
            self.records = [self.records[i] for i in keep]
            self.vectors = self.vectors[keep] if self.vectors.size else self.vectors

    def add(self, chunks, embeddings):
        """chunks: list of {text, source_id, meta}; embeddings: matching vectors."""
        if not chunks:
            return
        new = self._normalize(np.asarray(embeddings, dtype="float32"))
        if self.vectors.size == 0:
            self.vectors = new
        else:
            self.vectors = np.vstack([self.vectors, new])
        self.records.extend(chunks)

    def clear_all(self):
        self.vectors = np.zeros((0, 0), dtype="float32")
        self.records = []

    # ---- query ----------------------------------------------------------
    def search(self, query_vec, k: int, floor: float):
        if self.vectors.size == 0:
            return []
        q = np.asarray(query_vec, dtype="float32")
        q = q / (np.linalg.norm(q) or 1.0)
        scores = self.vectors @ q                      # cosine (all normalized)
        order = np.argsort(-scores)[:k]
        out = []
        for i in order:
            score = float(scores[i])
            if score < floor:
                continue
            rec = dict(self.records[i])
            rec["score"] = score
            out.append(rec)
        return out

    def __len__(self):
        return len(self.records)
