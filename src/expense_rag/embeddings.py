"""Turn text into a meaning fingerprint (embedding vector)."""

from __future__ import annotations

import os
from functools import lru_cache

EMBEDDING_DIM = 384
DEFAULT_MODEL = "all-MiniLM-L6-v2"


def embedding_model_name() -> str:
    return os.environ.get("EMBEDDING_MODEL", DEFAULT_MODEL)


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(embedding_model_name())


def embed_text(text: str) -> list[float]:
    vector = _model().encode(text, normalize_embeddings=True)
    values = [float(v) for v in vector.tolist()]
    if len(values) != EMBEDDING_DIM:
        raise ValueError(f"expected {EMBEDDING_DIM} dimensions, got {len(values)}")
    return values


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [embed_text(text) for text in texts]
