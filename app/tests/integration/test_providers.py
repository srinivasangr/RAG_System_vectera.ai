"""Integration tests for the modular LLM / embedding / vision providers.

These need real API keys and live network. Auto-skipped when keys missing.
"""

import pytest

from rag_system.llm_providers import Message

pytestmark = pytest.mark.integration


@pytest.mark.gemini
def test_gemini_embedder_returns_768d_vectors():
    from rag_system.llm_providers._rate_limit import is_rate_limit_error
    from rag_system.llm_providers.gemini_provider import GeminiEmbedder

    emb = GeminiEmbedder()
    try:
        vecs = emb.embed(["hello world", "another sentence"])
    except Exception as e:
        if is_rate_limit_error(e) or "404" in str(e):
            pytest.skip(f"Gemini API unavailable / quota exhausted: {e}")
        raise

    assert len(vecs) == 2
    assert len(vecs[0]) == emb.dim == 768
    # Embedding is a unit-ish vector; not all zeros
    assert any(abs(x) > 0 for x in vecs[0])


@pytest.mark.cerebras
def test_cerebras_provider_responds():
    from rag_system.llm_providers.openai_compat import CerebrasProvider

    llm = CerebrasProvider()
    out = llm.generate(
        [Message(role="user", content="Reply with the single word: pong")],
        max_tokens=200,  # gpt-oss-* uses reasoning tokens — needs headroom
    )
    assert "pong" in out.lower()


def test_local_embedder_dim_matches_settings():
    """Pure local — no API. Should always work when sentence-transformers is installed."""
    pytest.importorskip("sentence_transformers")
    from rag_system.llm_providers.local_embedder import LocalSentenceTransformersEmbedder

    emb = LocalSentenceTransformersEmbedder()
    vecs = emb.embed(["test"])
    assert len(vecs) == 1
    assert len(vecs[0]) == emb.dim
