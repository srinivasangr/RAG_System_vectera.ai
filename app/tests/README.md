# Tests

Two tiers, both runnable with `pytest`:

```
tests/
├── unit/          # pure logic, no external services, always runs
│   ├── test_metadata.py        # filename → company/date/version parsing
│   └── test_recency_intent.py  # "current/latest" query detector
└── integration/   # needs Snowflake / Gemini / Cerebras — auto-skips when keys missing
    ├── test_snowflake.py       # connection, schema, VECTOR ops
    ├── test_providers.py       # Gemini / Cerebras / local embedders
    └── test_retrieval.py       # full query → grounded answer
```

## Running

From `app/`:

```bash
# All tests (integration ones auto-skip without creds)
pytest

# Unit only — fast, no creds required
pytest tests/unit

# Integration only
pytest tests/integration

# A single test
pytest tests/unit/test_metadata.py::test_doc_id_is_stable_across_reads -v
```

## Auto-skip behavior

`tests/conftest.py` checks each marker and skips when the prereq is missing:

| Marker | Prerequisite |
|---|---|
| `@pytest.mark.snowflake` | `SNOWFLAKE_ACCOUNT/USER/PASSWORD` populated (env or `.env`) |
| `@pytest.mark.gemini` | `GEMINI_API_KEY` set |
| `@pytest.mark.cerebras` | `CEREBRAS_API_KEY` set |
| `@pytest.mark.docling` | `docling` importable |

So `pytest` on a fresh checkout with only `CEREBRAS_API_KEY` set runs every
unit test plus the Cerebras-only integration test; everything else skips
with a clear reason.

## What's NOT in here

Operational and dev helpers live in `app/scripts/` — bulk ingest,
Snowflake wipe, dedupe, status queries, benchmarks. They're scripts you
run on demand, not tests.
