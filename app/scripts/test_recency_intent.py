"""Verify the recency-intent auto-detection regex."""

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.retrieval.hybrid import has_recency_intent


CASES = [
    # (query, expected)
    ("What is the current leverage ratio?",                  True),
    ("Show me the latest financial figures",                 True),
    ("What is the most recent strategy update?",             True),
    ("Most-recent guidance for FY26",                        True),
    ("What's new this quarter?",                             True),
    ("As of today, what is the FFO?",                        True),
    ("Recently announced acquisitions",                      True),
    ("Up-to-date capacity",                                  True),
    # Should NOT trigger
    ("What was BXP's strategy in 2024?",                     False),
    ("Compare leverage between Dec 2025 and Mar 2026",       False),
    ("Tell me about sustainability",                         False),
    ("How many customers does the company have?",            False),
    # Edge cases
    ("recentralize their operations",                        False),  # 'recent' as part of 'recentralize' — \b should block
    ("Latest greatest",                                      True),
]


def main() -> None:
    ok = 0
    fail = 0
    for q, expected in CASES:
        actual = has_recency_intent(q)
        tag = "OK " if actual == expected else "FAIL"
        if actual == expected: ok += 1
        else: fail += 1
        print(f"  [{tag}]  expected={expected!s:>5}  got={actual!s:>5}  '{q}'")
    print(f"\nPassed {ok}/{ok+fail}")


if __name__ == "__main__":
    main()
