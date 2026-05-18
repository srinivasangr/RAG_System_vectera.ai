

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.storage.db import get_connection


def main() -> None:
    with get_connection() as conn:
        cur = conn.cursor()

        # --- Distribution ---
        print("=== Chunk counts by type ===")
        cur.execute("""
            SELECT chunk_type, COUNT(*) AS n,
                   AVG(token_count)::INT AS avg_tok,
                   MIN(token_count) AS min_tok,
                   MAX(token_count) AS max_tok
            FROM chunks
            GROUP BY chunk_type
            ORDER BY n DESC
        """)
        for kind, n, avg_t, min_t, max_t in cur.fetchall():
            print(f"  {kind:<22} {n:>4}  avg={avg_t}tok  range={min_t}-{max_t}")

        # --- Per-page coverage ---
        print("\n=== Per-page coverage (pages 1-37) ===")
        cur.execute("""
            SELECT page_number,
                   SUM(IFF(chunk_type='prose',             1, 0)) AS prose,
                   SUM(IFF(chunk_type='table',             1, 0)) AS tables,
                   SUM(IFF(chunk_type='chart_description', 1, 0)) AS charts
            FROM chunks
            GROUP BY page_number
            ORDER BY page_number
        """)
        print(f"  {'page':<6} {'prose':>6} {'tables':>7} {'charts':>7}")
        for p, prose, tables, charts in cur.fetchall():
            flag = ""
            if prose == 0 and tables == 0 and charts == 0:
                flag = "  ← empty page"
            print(f"  {p:<6} {prose:>6} {tables:>7} {charts:>7}{flag}")

        # --- Table samples ---
        print("\n=== 3 TABLE chunks (verify structure preserved) ===")
        cur.execute("""
            SELECT chunk_id, page_number, text
            FROM chunks
            WHERE chunk_type='table'
            ORDER BY token_count DESC
            LIMIT 3
        """)
        for cid, page, text in cur.fetchall():
            print(f"\n--- {cid} (p.{page}) ---")
            print(text[:1200])
            if len(text) > 1200:
                print(f"... [+{len(text)-1200} more chars]")

        # --- Chart description samples ---
        print("\n=== ALL chart_description chunks (Gemini vision output) ===")
        cur.execute("""
            SELECT chunk_id, page_number, token_count, text
            FROM chunks
            WHERE chunk_type='chart_description'
            ORDER BY page_number
        """)
        rows = cur.fetchall()
        if not rows:
            print("  (none — vision pass may have been skipped or all returned NOT_A_CHART)")
        for cid, page, ntok, text in rows:
            print(f"\n--- {cid} (p.{page}, {ntok}tok) ---")
            print(text)

        cur.close()


if __name__ == "__main__":
    main()
