"""Dedupe chunks table (in case the same chunk_id was inserted twice).

This happens when the same PDF (or a renamed copy) is ingested twice in a
single run — upsert_document sets status='unchanged' but insert_chunks runs
anyway, and Snowflake doesn't enforce PRIMARY KEY by default.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.storage.db import get_connection


def main() -> None:
    with get_connection() as conn:
        cur = conn.cursor()

        # Count duplicates
        cur.execute("""
            SELECT chunk_id, COUNT(*) AS cnt
            FROM chunks
            GROUP BY chunk_id
            HAVING COUNT(*) > 1
            ORDER BY cnt DESC
        """)
        dupes = cur.fetchall()
        if not dupes:
            print("No duplicates found.")
            return

        total_dup_rows = sum(c - 1 for _, c in dupes)
        print(f"Found {len(dupes)} chunk_ids with duplicates "
              f"({total_dup_rows} extra rows to delete)")
        for cid, c in dupes[:5]:
            print(f"  {cid}: {c} copies")
        if len(dupes) > 5:
            print(f"  ...and {len(dupes)-5} more")

        # Dedupe via temp-table swap
        print("\nDeduping...")
        cur.execute("""
            CREATE OR REPLACE TEMP TABLE _chunks_uniq AS
            SELECT * FROM chunks
            QUALIFY ROW_NUMBER() OVER (PARTITION BY chunk_id ORDER BY created_at) = 1
        """)
        cur.execute("SELECT COUNT(*) FROM _chunks_uniq")
        new_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM chunks")
        old_count = cur.fetchone()[0]
        print(f"  chunks before: {old_count}")
        print(f"  chunks after:  {new_count}")
        print(f"  will remove:   {old_count - new_count}")

        cur.execute("DELETE FROM chunks")
        cur.execute("INSERT INTO chunks SELECT * FROM _chunks_uniq")
        conn.commit()

        cur.execute("SELECT COUNT(*) FROM chunks")
        final_count = cur.fetchone()[0]
        print(f"\n[OK] chunks table now has {final_count} unique rows")
        cur.close()


if __name__ == "__main__":
    main()
