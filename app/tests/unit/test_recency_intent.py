"""Unit tests for the recency-intent auto-detection regex."""

import pytest

from rag_system.retrieval.hybrid import has_recency_intent


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "query, expected",
    [
        # Should trigger
        ("What is the current leverage ratio?",               True),
        ("Show me the latest financial figures",              True),
        ("What is the most recent strategy update?",          True),
        ("Most-recent guidance for FY26",                     True),
        ("What's new this quarter?",                          True),
        ("As of today, what is the FFO?",                     True),
        ("Recently announced acquisitions",                   True),
        ("Up-to-date capacity",                               True),
        ("Latest greatest",                                   True),

        # Should NOT trigger
        ("What was BXP's strategy in 2024?",                  False),
        ("Compare leverage between Dec 2025 and Mar 2026",    False),
        ("Tell me about sustainability",                      False),
        ("How many customers does the company have?",         False),

        # Word-boundary edge case: "recent" inside "recentralize" must not match
        ("recentralize their operations",                     False),
    ],
)
def test_has_recency_intent(query: str, expected: bool) -> None:
    assert has_recency_intent(query) is expected
