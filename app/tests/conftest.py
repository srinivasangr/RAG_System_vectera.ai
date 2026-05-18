"""Shared pytest fixtures + skip helpers.

Integration tests register themselves with markers (snowflake / gemini /
cerebras / docling). Those markers also trigger an automatic skip when the
required environment / dependency is missing, so the suite runs cleanly on
a machine that only has some of the credentials configured.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the `rag_system` package importable from tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Auto-skip behavior driven by markers
# ---------------------------------------------------------------------------
def _has_env(name: str) -> bool:
    return bool(os.environ.get(name, "").strip())


def _has_provider_key(env_name: str, settings_attr: str) -> bool:
    """Look in os.environ first, then fall through to pydantic-settings (which
    loads .env). Lets us run integration tests locally without re-exporting
    everything that's already in .env."""
    if _has_env(env_name):
        return True
    try:
        from rag_system.config import settings
        return bool(getattr(settings, settings_attr, ""))
    except Exception:
        return False


def _has_snowflake_creds() -> bool:
    return (
        _has_provider_key("SNOWFLAKE_ACCOUNT",  "snowflake_account")
        and _has_provider_key("SNOWFLAKE_USER", "snowflake_user")
        and _has_provider_key("SNOWFLAKE_PASSWORD", "snowflake_password")
    )


def _has_docling() -> bool:
    try:
        import docling  # noqa: F401
        return True
    except Exception:
        return False


_SKIP_REASONS = {
    "snowflake": ("Snowflake credentials not configured", _has_snowflake_creds),
    "gemini":    ("GEMINI_API_KEY not set",
                  lambda: _has_provider_key("GEMINI_API_KEY", "gemini_api_key")),
    "cerebras":  ("CEREBRAS_API_KEY not set",
                  lambda: _has_provider_key("CEREBRAS_API_KEY", "cerebras_api_key")),
    "docling":   ("docling not installed",                _has_docling),
}


def pytest_collection_modifyitems(config, items):
    for item in items:
        for marker_name, (reason, has_it) in _SKIP_REASONS.items():
            if marker_name in item.keywords and not has_it():
                item.add_marker(pytest.mark.skip(reason=reason))


# ---------------------------------------------------------------------------
# Common fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def documents_dir() -> Path:
    """Path to the source PDFs (../Documents from the app root)."""
    from rag_system.config import settings
    return settings.documents_path
