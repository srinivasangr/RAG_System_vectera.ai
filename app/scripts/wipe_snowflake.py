"""Wipe all documents/chunks/chunk_images from Snowflake.

Used when switching embedding models — existing vectors are in the old model's
space and won't match new-model query vectors. Safer to clear and re-ingest.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.storage.db import get_connection


def main() -> None:
    with get_connection() as conn:
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM documents")
        n_docs = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        n_chunks = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunk_images")
        n_images = cur.fetchone()[0]
        print(f"Before: {n_docs} docs / {n_chunks} chunks / {n_images} images")

        if n_docs == 0 and n_chunks == 0 and n_images == 0:
            print("Already empty.")
            return

        cur.execute("DELETE FROM chunk_images")
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM documents")
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM documents")
        a = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        b = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunk_images")
        c = cur.fetchone()[0]
        print(f"After:  {a} docs / {b} chunks / {c} images")
        print("[OK] wiped")


if __name__ == "__main__":
    main()
