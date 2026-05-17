"""Smoke-test Snowflake VECTOR storage + cosine similarity using a Gemini
embedding (768d). This proves the storage+retrieval plumbing works without
Cortex AI functions (which are blocked on trial)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from google.genai import types

from rag_system.config import settings
from rag_system.storage.db import get_connection


def embed(text: str) -> list[float]:
    client = genai.Client(api_key=settings.gemini_api_key)
    r = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    return list(r.embeddings[0].values)


def main() -> None:
    texts = {
        "smoke_a": "Digital Realty reported strong Q4 2025 data center demand.",
        "smoke_b": "BXP's office occupancy improved in the fourth quarter.",
        "smoke_c": "The cat sat on the mat.",
    }
    embeddings = {k: embed(v) for k, v in texts.items()}

    with get_connection() as conn:
        cur = conn.cursor()

        # Clean any previous smoke-test rows + a dummy parent doc
        cur.execute("DELETE FROM chunks WHERE chunk_id LIKE 'smoke_%'")
        cur.execute("DELETE FROM documents WHERE doc_id = 'smoke_doc'")

        cur.execute("""
            INSERT INTO documents (doc_id, source_path, company, doc_type)
            VALUES ('smoke_doc', 'scripts/test_snowflake_vector.py', 'TEST', 'smoke')
        """)

        # Insert via parameterized SQL with explicit VECTOR cast
        for cid, text in texts.items():
            vec_literal = "[" + ",".join(f"{x:.8f}" for x in embeddings[cid]) + "]"
            cur.execute(
                f"""
                INSERT INTO chunks
                  (chunk_id, doc_id, page_number, chunk_index, text, chunk_type, embedding)
                SELECT %s, %s, %s, %s, %s, %s, {vec_literal}::VECTOR(FLOAT, 768)
                """,
                (cid, "smoke_doc", 1, 0, text, "prose"),
            )
        conn.commit()
        print(f"[OK] Inserted {len(texts)} smoke chunks with 768d vectors.")

        # Cosine-similarity search using a query embedding
        q_vec = embed("How is the data center business performing?")
        q_literal = "[" + ",".join(f"{x:.8f}" for x in q_vec) + "]"
        cur.execute(f"""
            SELECT chunk_id, text,
                   VECTOR_COSINE_SIMILARITY(embedding, {q_literal}::VECTOR(FLOAT, 768)) AS score
            FROM chunks
            WHERE chunk_id LIKE 'smoke_%'
            ORDER BY score DESC
        """)
        print("\nQuery: 'How is the data center business performing?'")
        print("Ranked results:")
        for cid, text, score in cur.fetchall():
            print(f"  {score:.4f}  {cid}: {text}")

        # Cleanup
        cur.execute("DELETE FROM chunks WHERE chunk_id LIKE 'smoke_%'")
        cur.execute("DELETE FROM documents WHERE doc_id = 'smoke_doc'")
        conn.commit()
        cur.close()
        print("\n[OK] Cleaned up smoke rows. Vector storage + cosine search are working.")


if __name__ == "__main__":
    main()
