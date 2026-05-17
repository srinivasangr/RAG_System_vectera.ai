"""Local embedding provider using sentence-transformers.

Runs entirely on CPU (or GPU if available). No API rate limits, no external
calls, no quota — just downloads the model once on first use and caches it
in `~/.cache/huggingface/`.

Default model: BAAI/bge-base-en-v1.5
  - 768 dims (matches our Snowflake VECTOR(FLOAT, 768) schema)
  - 440 MB
  - MTEB English score ~63.5
  - Public on HF (no login required)
"""

from __future__ import annotations

import logging
from typing import Sequence

from rag_system.config import settings
from rag_system.llm_providers.base import BaseEmbeddingProvider

log = logging.getLogger(__name__)


class LocalSentenceTransformersEmbedder(BaseEmbeddingProvider):
    name = "local"
    # Reasonable batch for CPU — keeps a single inference under ~2 GB of RAM
    BATCH_SIZE = 32

    def __init__(self, *, model: str | None = None, dim: int | None = None):
        self._model_name = model or settings.embedding_model or "BAAI/bge-base-en-v1.5"
        self.dim = dim or settings.embedding_dim or 768
        # Lazy-import so the rest of the app boots fast when local emb isn't used
        from sentence_transformers import SentenceTransformer
        log.info("loading sentence-transformer model: %s", self._model_name)
        self._model = SentenceTransformer(self._model_name)
        # Sanity-check dim
        try:
            actual = self._model.get_sentence_embedding_dimension()
            if actual != self.dim:
                log.warning(
                    "model dim %d != settings.embedding_dim %d — using model's dim",
                    actual, self.dim,
                )
                self.dim = actual
        except Exception:
            pass

    def embed(
        self,
        texts: Sequence[str],
        *,
        progress_cb=None,
    ) -> list[list[float]]:
        if not texts:
            return []
        texts = list(texts)
        total = len(texts)
        out: list[list[float]] = []

        for start in range(0, total, self.BATCH_SIZE):
            batch = texts[start:start + self.BATCH_SIZE]
            # normalize_embeddings=True so cosine similarity is just a dot product
            vecs = self._model.encode(
                batch,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
                batch_size=self.BATCH_SIZE,
            )
            out.extend(v.tolist() for v in vecs)
            if progress_cb:
                try:
                    progress_cb("embed_progress", {"done": len(out), "total": total})
                except Exception:
                    pass
        return out
