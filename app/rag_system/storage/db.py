"""Snowflake connection helper.

Yields a connection configured from `settings`. Keep it simple — one
connection per call; Snowflake handles pooling on its side.
"""

from contextlib import contextmanager
from typing import Iterator

import snowflake.connector

from rag_system.config import settings


@contextmanager
def get_connection() -> Iterator[snowflake.connector.SnowflakeConnection]:
    conn = snowflake.connector.connect(
        account=settings.snowflake_account,
        user=settings.snowflake_user,
        password=settings.snowflake_password,
        role=settings.snowflake_role,
        warehouse=settings.snowflake_warehouse,
        database=settings.snowflake_database,
        schema=settings.snowflake_schema,
        client_session_keep_alive=False,
    )
    try:
        yield conn
    finally:
        conn.close()
