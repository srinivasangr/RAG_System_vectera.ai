"""Integration test for the full query path — embed → retrieve → generate.

Requires Snowflake (with ingested corpus) + Cerebras for generation.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.snowflake, pytest.mark.cerebras]


# gpt-oss-* models emit reasoning tokens before the answer; max_tokens must
# leave room for both. 1200 is the production default in the UI.
_MAX_TOKENS = 1200


def test_end_to_end_query_returns_grounded_answer():
    """A canonical question against the Digital Realty corpus must return citations."""
    from rag_system.generation import query

    ans = query(
        "What is Digital Realty's leverage ratio?",
        top_k=5,
        max_tokens=_MAX_TOKENS,
        write_log=False,
    )
    # Some text came back
    assert ans.answer and len(ans.answer) > 20
    # At least one citation
    assert len(ans.retrieved) > 0
    # Latency is reasonable (< 30s including potential cold start)
    assert ans.latency_ms < 30_000


def test_out_of_corpus_query_refuses_cleanly():
    """Insufficient-evidence path: model must refuse rather than hallucinate."""
    from rag_system.generation import query

    ans = query(
        "What is the share price of Apple as of today?",
        top_k=5,
        max_tokens=_MAX_TOKENS,
        write_log=False,
    )
    # The strict-citation prompt instructs the model to refuse with this phrase
    assert "don't have enough information" in ans.answer.lower()
