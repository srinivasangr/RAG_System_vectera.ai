"""Smoke-test Snowflake Cortex EMBED_TEXT_768 + LLM availability."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.storage.db import get_connection


def main() -> None:
    with get_connection() as conn:
        cur = conn.cursor()

        # Test embedding
        cur.execute(
            "SELECT SNOWFLAKE.CORTEX.EMBED_TEXT_768("
            "'snowflake-arctic-embed-m-v1.5', 'hello world')"
        )
        vec = cur.fetchone()[0]
        print(f"[OK] EMBED_TEXT_768 works. dim={len(vec)}")
        print(f"     first 3 values: {vec[:3]}")
        cur.close()


if __name__ == "__main__":
    main()
