"""
Unified embedding backend for KEO.

Supports:
  - openai: text-embedding-3-small (requires OPENAI_API_KEY)
  - local: sentence-transformers model (air-gapped if weights are cached)
  - hash: deterministic bag-of-hashes fallback (no network; for smoke / CI)
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import List, Optional

import numpy as np

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore


class EmbeddingBackend:
    def __init__(
        self,
        backend: str = "openai",
        model: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        dim: int = 384,
    ):
        self.backend = (backend or "openai").lower()
        self.dim = dim
        self._st_model = None
        self._openai = None

        if self.backend == "openai":
            if OpenAI is None:
                raise ImportError("openai package required for embedding_backend=openai")
            key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OPENAI_API_KEY required for embedding_backend=openai")
            self._openai = OpenAI(api_key=key)
            self.model = model or "text-embedding-3-small"
        elif self.backend == "local":
            self.model = model or "sentence-transformers/all-MiniLM-L6-v2"
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(self.model)
                self.dim = int(self._st_model.get_sentence_embedding_dimension())
            except Exception as e:
                print(f"Warning: local ST model unavailable ({e}); falling back to hash embeddings")
                self.backend = "hash"
                self.model = "hash-v1"
        elif self.backend == "hash":
            self.model = model or "hash-v1"
        else:
            raise ValueError(f"Unknown embedding backend: {backend}")

    def embed(self, text: str) -> List[float]:
        text = (text or "").strip() or " "
        if self.backend == "openai":
            resp = self._openai.embeddings.create(
                model=self.model,
                input=text,
                encoding_format="float",
            )
            return list(resp.data[0].embedding)
        if self.backend == "local" and self._st_model is not None:
            vec = self._st_model.encode(text, normalize_embeddings=True)
            return vec.astype(float).tolist()
        return self._hash_embed(text)

    def _hash_embed(self, text: str) -> List[float]:
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        vec = np.zeros(self.dim, dtype=float)
        if not tokens:
            tokens = ["empty"]
        for tok in tokens:
            h = hashlib.md5(tok.encode("utf-8")).hexdigest()
            idx = int(h[:8], 16) % self.dim
            sign = 1.0 if int(h[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()
