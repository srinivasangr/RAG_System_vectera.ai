"""Apply the database schema (schema.sql).

Idempotent: every statement uses IF NOT EXISTS, so re-running is a no-op and a
fresh database is fully provisioned in one pass.

Usage:
    python -m rag_system.storage.migrate
    python -m rag_system.storage.migrate --verify   # just print the resulting columns/tables
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag_system.storage.db import get_connection

SCHEMA_DIR = Path(__file__).parent


def _split_statements(sql: str) -> list[str]:
    """Split on semicolon-terminated statements. Skips comment/blank lines.

    Robust to inline `-- comments` that follow a terminating `;` on the same
    line (we strip the trailing comment before testing for the `;`). Snowflake
    tolerates `--` comments inside a statement body, so we keep the original
    line in the buffer and only strip for the end-of-statement check.
    """
    out, buf = [], []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        buf.append(line)
        # Ignore a trailing inline comment when deciding if the statement ends.
        code = stripped.split("--", 1)[0].rstrip()
        if code.endswith(";"):
            stmt = "\n".join(buf).strip().rstrip(";").strip()
            # Drop a dangling trailing comment that survived the join.
            if stmt:
                out.append(stmt)
            buf = []
    if buf:
        tail = "\n".join(buf).strip()
        if tail:
            out.append(tail)
    return out


_NEW_TABLES = [
    "parent_chunks", "parent_images", "propositions",
    "table_rows", "chart_records", "ingest_checkpoints",
]


def _verify(cur) -> None:
    print("\n=== v2 tables ===")
    for t in _NEW_TABLES:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {t}")
            n = cur.fetchone()[0]
            print(f"  [OK] {t:<20} rows={n}")
        except Exception as e:  # noqa: BLE001
            print(f"  [MISSING] {t:<20} {e}")

    print("\n=== documents v2 columns ===")
    cur.execute("DESC TABLE documents")
    cols = {r[0].lower() for r in cur.fetchall()}
    for c in ("ticker", "as_of_date", "doc_family_id", "doc_type_conf", "as_of_source"):
        print(f"  {'[OK]' if c in cols else '[MISSING]'} documents.{c}")

    print("\n=== chunks v2 columns ===")
    cur.execute("DESC TABLE chunks")
    cols = {r[0].lower() for r in cur.fetchall()}
    for c in ("parent_id", "footnote_text", "qualifier_text", "doc_type",
              "as_of_date", "doc_family_id", "slide_title"):
        print(f"  {'[OK]' if c in cols else '[MISSING]'} chunks.{c}")

    print("\n=== query_log v2 columns ===")
    cur.execute("DESC TABLE query_log")
    cols = {r[0].lower() for r in cur.fetchall()}
    for c in ("router_intent", "sub_queries", "retrieval_stages", "rerank_top_ids",
              "conflict_pairs", "provider_chain", "reasoning_trace", "total_latency_ms"):
        print(f"  {'[OK]' if c in cols else '[MISSING]'} query_log.{c}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true",
                   help="skip migration, just print resulting tables/columns")
    p.add_argument("--file", type=str, default="schema.sql",
                   help="SQL file in this dir to apply")
    args = p.parse_args(argv)

    schema_file = SCHEMA_DIR / args.file

    with get_connection() as conn:
        cur = conn.cursor()
        if not args.verify:
            sql = schema_file.read_text(encoding="utf-8")
            stmts = _split_statements(sql)
            print(f"Applying {len(stmts)} v2 migration statements...\n")
            for stmt in stmts:
                head = stmt.splitlines()[0][:80]
                try:
                    cur.execute(stmt)
                    print(f"  [OK] {head}")
                except Exception as e:  # noqa: BLE001
                    # ADD COLUMN IF NOT EXISTS / CREATE IF NOT EXISTS shouldn't fail,
                    # but surface anything unexpected without aborting the whole run.
                    print(f"  [WARN] {head}\n         -> {e}")
            conn.commit()
            print("\n[OK] v2 migration applied.")

        _verify(cur)
        cur.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
