"""Integration tests for Snowflake — connection, schema, vector ops.

All tests in this file are auto-skipped when Snowflake creds aren't set
(see tests/conftest.py).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.snowflake]


def test_connection_opens():
    """Smoke: we can open a connection and run a no-op query."""
    from rag_system.storage.db import get_connection

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT 1")
        assert cur.fetchone()[0] == 1
        cur.close()


def test_required_tables_exist():
    """Schema migrations have been applied — all four tables present."""
    from rag_system.storage.db import get_connection

    expected = {"DOCUMENTS", "CHUNKS", "CHUNK_IMAGES", "QUERY_LOG"}
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SHOW TABLES IN SCHEMA RAG_DB.RAG_SCHEMA")
        names = {row[1] for row in cur.fetchall()}
        cur.close()
    missing = expected - names
    assert not missing, f"missing tables: {missing}"


def test_vector_cosine_similarity_returns_floats():
    """VECTOR_COSINE_SIMILARITY works against a literal 768-dim vector."""
    from rag_system.storage.db import get_connection

    vec_lit = "[" + ",".join("0.1" for _ in range(768)) + "]"
    sql = (
        f"SELECT VECTOR_COSINE_SIMILARITY("
        f"{vec_lit}::VECTOR(FLOAT, 768), {vec_lit}::VECTOR(FLOAT, 768))"
    )
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(sql)
        sim = cur.fetchone()[0]
        cur.close()
    # Same vector vs itself → cosine similarity == 1.0 (within float tolerance)
    assert 0.999 <= sim <= 1.0001
