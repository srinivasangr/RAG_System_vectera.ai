"""Idempotent Snowflake bootstrap: warehouse + database + schema + tables.

Usage:
    python -m rag_system.storage.init_snowflake
"""

from pathlib import Path

import snowflake.connector

from rag_system.config import settings


SCHEMA_FILE = Path(__file__).parent / "schema.sql"


def _split_statements(sql: str) -> list[str]:
    """Split on semicolons that terminate statements. Naive but works for our DDL."""
    out, buf = [], []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") or not stripped:
            continue
        buf.append(line)
        if stripped.endswith(";"):
            stmt = "\n".join(buf).rstrip(";").strip()
            if stmt:
                out.append(stmt)
            buf = []
    if buf:
        tail = "\n".join(buf).strip()
        if tail:
            out.append(tail)
    return out


def main() -> None:
    print(f"Connecting to Snowflake account: {settings.snowflake_account}")

    # First connection: no DB/schema yet — we'll create them
    conn = snowflake.connector.connect(
        account=settings.snowflake_account,
        user=settings.snowflake_user,
        password=settings.snowflake_password,
        role=settings.snowflake_role,
    )
    try:
        cur = conn.cursor()
        sql = SCHEMA_FILE.read_text(encoding="utf-8")
        for stmt in _split_statements(sql):
            print(f">>> {stmt.splitlines()[0][:80]}...")
            cur.execute(stmt)
        cur.close()
        print("\n[OK] Snowflake schema ready.")
        print(f"  Database : {settings.snowflake_database}")
        print(f"  Schema   : {settings.snowflake_schema}")
        print(f"  Warehouse: {settings.snowflake_warehouse}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
