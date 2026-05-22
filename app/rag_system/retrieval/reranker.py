"""Cross-encoder reranker (stage 3).

A bi-encoder (our BGE embedder) scores query and chunk separately — fast but
imprecise, so RRF candidates are often noisy. A cross-encoder reads
`(query, chunk)` TOGETHER and outputs a single relevance score; it's the single
biggest precision lift in the pipeline. We only run it over the top-N RRF
candidates (it's ~100x slower than embedding, so not over the whole corpus).

Model: BAAI/bge-reranker-v2-m3 (local CPU, no API cost). Lazy singleton.
Feature-flagged: if the model can't load, we no-op and keep the RRF order.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

# Default to a FAST cross-encoder. bge-reranker-v2-m3 is higher quality but
# ~125s for 30 pairs on CPU (unusable interactively); MiniLM reranks 30 pairs
# in ~1-3s and is still a large precision lift over no rerank. Override with
# RERANKER_MODEL (e.g. BAAI/bge-reranker-base) if you have GPU / more latency budget.
_MODEL_NAME = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_MODEL = None
_DISABLED = False


def _get_model():
    global _MODEL, _DISABLED
    if _MODEL is None and not _DISABLED:
        try:
            from sentence_transformers import CrossEncoder
            log.info("loading reranker %s ...", _MODEL_NAME)
            _MODEL = CrossEncoder(_MODEL_NAME, max_length=512)
        except Exception as e:  # noqa: BLE001
            log.warning("reranker unavailable (%s) — keeping RRF order", e)
            _DISABLED = True
    return _MODEL


def rerank(query: str, chunks, *, top_k: int, text_attr: str = "text"):
    """Rerank `chunks` by cross-encoder relevance to `query`; return top_k.

    Sets `.rerank_score` on each chunk. If the model is unavailable, returns
    the first top_k unchanged (RRF order preserved).
    """
    if not chunks:
        return []
    model = _get_model()
    if model is None:
        return chunks[:top_k]

    pairs = [(query, (getattr(c, text_attr, "") or "")[:2000]) for c in chunks]
    try:
        scores = model.predict(pairs, show_progress_bar=False)
    except Exception as e:  # noqa: BLE001
        log.warning("rerank failed (%s) — keeping RRF order", e)
        return chunks[:top_k]

    for c, s in zip(chunks, scores):
        c.rerank_score = float(s)
        c.score = float(s)  # rerank score becomes the working score
    ranked = sorted(chunks, key=lambda c: (c.rerank_score if c.rerank_score is not None else -1e9),
                    reverse=True)
    return ranked[:top_k]
