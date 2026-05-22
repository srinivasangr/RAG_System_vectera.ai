# RAG System — Document Q&A

A retrieval-augmented question-answering system over complex business documents
(slide decks, reports, filings) containing dense text, tables, charts, and maps.
Ask natural-language questions and get grounded, cited answers with the exact
source pages.

The system is **domain-agnostic**: it carries no hardcoded entities, metrics, or
document types — everything is derived by reading the documents — so the same
code works on any PDF corpus.

---

## What it does

- **Content-driven ingestion** — identifies each document (issuer, type, as-of
  date) from its content; chunks structure-aware (footnotes kept with their
  numbers); page-level vision extracts tables, charts, and figures into
  structured records; everything is embedded into a multi-vector index.
- **Multi-stage retrieval** — query routing → dense + lexical + structured
  retrieval → reciprocal-rank fusion → cross-encoder rerank → diversification →
  version-pair expansion → small-to-big context → conflict detection.
- **Grounded generation** — every claim is cited; conflicting values across
  document versions are surfaced with attribution; stale sources are flagged;
  the system refuses honestly when the documents don't support an answer.
- **Observable** — a per-query trace (router plan, stage timings, token usage,
  engine chain) and a query history, both in the UI and persisted.

See **[docs/architecture.md](docs/architecture.md)** for the full design and
**[docs/comparison_v1_vs_v2.md](docs/comparison_v1_vs_v2.md)** for what changed
from the first iteration.

---

## Tech stack

| Layer | Choice |
|---|---|
| Parsing | Docling (layout-aware text + tables) + PyMuPDF (page rendering) |
| Vision | Gemini (page-level structured extraction) |
| Embeddings | BAAI/bge-base-en-v1.5 (local, 768-dim) |
| Reranker | Cross-encoder (MiniLM, local) |
| Vector store | Snowflake `VECTOR` + `VECTOR_COSINE_SIMILARITY` |
| Generation / routing | Gemini (provider-agnostic interface; OpenAI-compatible providers also supported) |
| API + UI | FastAPI serving a minimal streaming UI |

---

## Repository layout

```
app/
  api/                  FastAPI app + minimal UI (main.py, jobs.py, templates/, static/)
  rag_system/
    ingest/             parse · metadata · chunk · vision · propositions · pipeline
    retrieval/          router · retrieve · reranker · pipeline · filters
    generation/         generate · citations
    storage/            db · schema.sql · migrate · repository · init_snowflake
    llm_providers/      base · factory · gemini · openai_compat · local_embedder
    config.py
  eval/                 battery.yaml · run_battery · judge · ragas_metrics · baselines/
  tests/                unit + integration
  requirements.txt
docs/                   architecture.md · comparison_v1_vs_v2.md
Documents/              your source PDFs (not committed; see below)
Dockerfile · docker-compose.yml
```

---

## Setup (local)

**Prerequisites:** Python 3.11+, a Snowflake account, and a Gemini API key.

```bash
git clone <repo-url>
cd RAG_System/app

# 1. virtual environment + dependencies
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt

# 2. credentials
cp .env.example .env              # then fill in the values (see below)

# 3. create the Snowflake database + schema
python -m rag_system.storage.init_snowflake
python -m rag_system.storage.migrate

# 4. run
uvicorn api.main:app --port 8000
# open http://localhost:8000
```

### `.env`

```
SNOWFLAKE_ACCOUNT=...
SNOWFLAKE_USER=...
SNOWFLAKE_PASSWORD=...
SNOWFLAKE_WAREHOUSE=RAG_WH
SNOWFLAKE_DATABASE=RAG_DB
SNOWFLAKE_SCHEMA=RAG_SCHEMA

GEMINI_API_KEY=...                # generation, routing, and vision

# optional: raise on a paid Gemini tier to speed the vision pass
GEMINI_VISION_RPM=200
VISION_CONCURRENCY=5
```

### Ingest documents

Place your PDFs in the `Documents/` folder (default; override with
`DOCUMENTS_DIR`). Source PDFs are not committed to the repository (size and
copyright). Then either upload a PDF in the **Ingest & Corpus** tab (watch each
stage run live), or ingest from the command line:

```bash
python -m rag_system.ingest.pipeline        # ingests every PDF in the documents folder
```

Re-running is idempotent (unchanged files are skipped); a failed ingest resumes
at the next unfinished stage.

---

## Using the app

- **Ask** — type a question, optionally narrow the search to specific documents,
  and get a cited answer. Each source card links to the actual page image; a
  **Trace** panel shows the router plan, stage timings, token usage, and the
  prompt sent to the model.
- **Ingest & Corpus** — upload documents with a live per-stage status; browse
  the ingested corpus.
- **History** — every query is logged (intent, timings, engine, latency).

---

## Evaluation

Three complementary tracks (see **[app/eval/baselines/v2_RESULTS.md](app/eval/baselines/v2_RESULTS.md)**):

1. a 24-question diagnostic battery scored Pass / Partial / Fail,
2. an independent LLM-judge first pass (a different model from the generator),
3. automated RAGAS-style metrics (faithfulness, answer relevance, context
   precision/recall).

```bash
python -m eval.run_battery          # runs the battery + judge + RAGAS metrics
```

Headline metrics vs. the first iteration: context precision 0.09 → 0.49,
faithfulness 0.70 → 0.95.

---

## Deployment

The app runs as a single FastAPI process (API + UI). It is local-first; to host
it, deploy the same process to any container platform (e.g. Render, Railway,
Fly.io) with the same environment variables — Snowflake and the model providers
are reached over the network.

---

## Limitations and next steps

See **[docs/architecture.md](docs/architecture.md) §9**. In brief: the largest
open improvement is injecting matched table rows / chart records / footnotes
directly into the generation context (rather than the parent slide), and fuller
recovery of data encoded as positions on maps.
