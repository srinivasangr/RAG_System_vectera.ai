"""End-to-end smoke test of the query() entrypoint."""

import sys
from pathlib import Path

# Force UTF-8 stdout so model output with unicode (narrow nbsp, em dashes,
# bullets, etc.) doesn't crash on Windows cp1252 consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.generation import query


QUESTIONS = [
    "What is Digital Realty's leverage ratio (Net Debt to Adjusted EBITDA)?",
    "What's the breakdown of customer types by ARR?",
    "What is the company's stance on sustainability?",
    # Should refuse cleanly:
    "What is the share price of Apple as of today?",
]


def main() -> None:
    for q in QUESTIONS:
        print("\n" + "=" * 90)
        print(f"Q: {q}")
        a = query(q, top_k=6, max_tokens=1200)
        print(f"\n[answer] ({a.llm_provider}/{a.llm_model}, "
              f"retr {a.retrieval_ms}ms + gen {a.generation_ms}ms = {a.latency_ms}ms)")
        print(a.answer)
        if a.citations:
            print("\n[citations]")
            for c in a.citations:
                print(f"  [{c.n}] {c.company or '?'} {c.version_label or ''} "
                      f"p.{c.page_number} ({c.chunk_type})")


if __name__ == "__main__":
    main()
