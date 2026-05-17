"""End-to-end retrieval smoke test against whatever's in Snowflake right now."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.retrieval.filters import RetrievalFilters
from rag_system.retrieval.hybrid import retrieve


QUERIES = [
    ("What is Digital Realty's leverage ratio?", None),
    ("How is the data center business performing under AI demand?", None),
    ("Tell me about sustainability and renewable energy", None),
    ("How many customers does the company have?", None),
    ("What are the rental rate changes for renewals?", None),
    # Filtered: prefer recent
    ("What is the current strategy?", RetrievalFilters(prefer_recent=True)),
    # Filtered: by company
    ("data center demand", RetrievalFilters(companies=["Digital Realty"])),
]


def main() -> None:
    for q, f in QUERIES:
        print(f"\n{'='*80}")
        print(f"Q: {q}")
        if f and not f.is_empty():
            print(f"   filters: {f}")
        results = retrieve(q, filters=f or RetrievalFilters(), top_k=4)
        for i, r in enumerate(results, 1):
            tags = []
            if r.dense_rank: tags.append(f"d#{r.dense_rank}")
            if r.lexical_rank: tags.append(f"k#{r.lexical_rank}")
            print(f"  [{i}] score={r.score:.4f} [{','.join(tags)}] "
                  f"p.{r.page_number} ({r.chunk_type}) {r.company or '?'} {r.version_label or ''}")
            print(f"      {r.text[:160].strip()}...")


if __name__ == "__main__":
    main()
