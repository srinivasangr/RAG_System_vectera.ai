"""Verify what landed in Snowflake + run a quick retrieval test."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.llm_providers import get_embedder
from rag_system.storage.db import get_connection
from rag_system.storage.repository import corpus_stats


def main() -> None:
    print("=== Corpus stats ===")
    stats = corpus_stats()
    print(f"  documents: {stats['documents']}")
    print(f"  chunks:    {stats['chunks']}")
    for company, n_docs, n_chunks in stats["per_company"]:
        print(f"    {company}: {n_docs} doc(s), {n_chunks} chunks")

    print("\n=== Sample stored chunks ===")
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT chunk_id, page_number, chunk_type, token_count, LEFT(text, 120)
            FROM chunks
            WHERE chunk_type = 'chart_description'
            ORDER BY page_number, chunk_index
            LIMIT 3
        """)
        print("  chart_description samples:")
        for row in cur.fetchall():
            print(f"    [{row[0]}] p.{row[1]} ({row[2]}, {row[3]}tok)")
            print(f"      {row[4]}...")

        cur.close()

    print("\n=== Retrieval test (cosine similarity) ===")
    embedder = get_embedder()
    queries = [
        "What is Digital Realty's leverage ratio?",
        "How is the data center business performing?",
        "Tell me about sustainability and renewable energy",
    ]
    with get_connection() as conn:
        cur = conn.cursor()
        for q in queries:
            print(f"\nQ: {q}")
            qv = embedder.embed_one(q)
            vec_lit = "[" + ",".join(f"{x:.8f}" for x in qv) + "]"
            cur.execute(f"""
                SELECT page_number, chunk_type,
                       VECTOR_COSINE_SIMILARITY(embedding, {vec_lit}::VECTOR(FLOAT, 768)) AS score,
                       LEFT(text, 180)
                FROM chunks
                ORDER BY score DESC
                LIMIT 3
            """)
            for page, kind, score, text in cur.fetchall():
                print(f"  {score:.3f}  p.{page} ({kind}) {text}...")
        cur.close()


if __name__ == "__main__":
    main()
