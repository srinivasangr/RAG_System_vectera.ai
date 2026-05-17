# Vectera RAG System — System Design

> Source documents: ~10 REIT investor presentation PDFs in `Documents/`
> Assessment brief: `Vectera_RAG_System_Technical_Assessment.pdf`
> Target repo: https://github.com/srinivasangr/RAG_System_vectera.ai

---

## 1. Problem Framing

A user (investment analyst / PM) needs to ask natural-language questions over a corpus of REIT investor decks and receive **source-grounded answers with citations**. The corpus is small (~10 PDFs, ~50–150 pages each), but messy: heavy on charts, tables, and multiple versions of the same company's material.

**This is not a search engine.** It is a *reasoning-over-evidence* system. The retrieval layer must surface the right *snippets*, and the LLM layer must answer *only* from those snippets while preserving attribution and refusing to merge conflicting facts.

---

## 2. User Requirements

| # | User Need | What it implies for the system |
|---|-----------|-------------------------------|
| U1 | Ask free-text questions in English | Chat-style UI, no query DSL |
| U2 | Get an answer in seconds, not minutes | End-to-end p95 < 8s |
| U3 | Trust the answer — see *where it came from* | Inline citations + click-through to source page/snippet |
| U4 | Compare across versions ("How has FFO changed from Dec 2025 to Mar 2026?") | Document metadata must carry `company`, `doc_date`, `version` |
| U5 | Know when sources disagree | Prompt must instruct the LLM to surface conflicts, not paper over them |
| U6 | Know when the system *can't* answer (e.g. data lives in a chart image) | Explicit "insufficient evidence" path |
| U7 | (Optional) Scope queries to one company / one version | Filter chips in the UI; metadata filters at retrieval time |

---

## 3. Functional Requirements

**Must-have (from brief):**
- FR1. Ingest PDFs → text
- FR2. Chunk documents
- FR3. Embed chunks + store in a DB-backed vector index
- FR4. Retrieve top-k relevant chunks for a query
- FR5. Generate an LLM answer grounded in retrieved chunks
- FR6. Return inline citations referencing source doc + page/section
- FR7. Version awareness — do not blindly mix conflicting values across document versions
- FR8. Cross-document conflict awareness — preserve attribution
- FR9. Handle (or honestly disclaim) charts/tables
- FR10. Web UI (Streamlit), not CLI

**Should-have:**
- FR11. Metadata filters in UI (company, doc_date)
- FR12. Show retrieved chunks (transparency panel)
- FR13. Re-ranking step before LLM call

**Nice-to-have:**
- FR14. Multi-turn conversation
- FR15. Per-tenant access control hook (stubbed — see §13)
- FR16. Lightweight eval harness with a small Q&A set

---

## 4. Non-Functional Requirements

| Category | Target |
|---|---|
| Latency | p95 query < 8s end-to-end (retrieval < 1s, LLM < 6s) |
| Ingestion | One-shot batch; ~10 PDFs in < 5 min |
| Cost | Stay within free tiers — Snowflake free trial, OpenAI/Anthropic pay-per-call (~<$5 total for dev) |
| Reproducibility | `.env.example`, pinned `requirements.txt`, one `make ingest` + one `streamlit run` |
| Observability | Log every query → retrieved chunk IDs → final answer; enough to debug "why did it cite this?" |

---

## 5. Type of RAG — and Why

This is a **Naive RAG with version-aware metadata filtering + cite-everything prompting**, not an agentic / multi-hop RAG.

**Decision rationale:**

| Option | Verdict | Why |
|---|---|---|
| Naive RAG (single retrieval → single LLM call) | ✅ **Chosen** | Corpus is small (~10 PDFs). Most questions are factual single-hop. Cheap, debuggable, fits the 6–10h budget. |
| Hybrid retrieval (BM25 + dense) | ✅ **Add as enhancement** | Investor decks have lots of ticker symbols, named metrics (FFO, NOI, AFFO) — keyword recall matters. |
| Re-ranker (cross-encoder) | ✅ **Optional layer** | Cohere Rerank or `bge-reranker-base`. Cheap quality win. |
| Agentic / ReAct RAG | ❌ Skip | Adds latency + cost + failure modes for marginal gain on this corpus. |
| GraphRAG | ❌ Skip | Overkill; entities aren't dense enough to justify graph construction. |
| Multi-vector / ColBERT | ❌ Skip | Infra overhead; pgvector / Snowflake VECTOR is enough at this scale. |
| Long-context "stuff everything" | ❌ Skip | Corpus is ~thousands of pages — won't fit, defeats citation precision. |

**Differentiator: version-aware retrieval.** Each chunk carries `company` and `doc_date`. When the user asks "*current* strategy", we boost the most recent doc per company. When they ask "*how has X changed*", we deliberately retrieve from *both* versions and the prompt asks the LLM to compare with attribution.

---

## 6. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       Streamlit UI                              │
│   ┌──────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│   │ Query input  │  │ Answer + cites  │  │ Retrieved chunks │   │
│   └──────┬───────┘  └────────▲────────┘  └────────▲─────────┘   │
└──────────┼───────────────────┼────────────────────┼─────────────┘
           │ query             │ answer             │ debug panel
           ▼                   │                    │
┌─────────────────────────────────────────────────────────────────┐
│                    FastAPI service (or in-proc)                 │
│                                                                 │
│   ┌─────────┐  ┌────────────┐  ┌───────────┐  ┌──────────────┐  │
│   │ Query   │→ │ Retriever  │→ │ Re-ranker │→ │ Prompt +     │  │
│   │ rewrite │  │ (hybrid)   │  │ (optional)│  │ LLM call     │  │
│   └─────────┘  └─────┬──────┘  └───────────┘  └──────┬───────┘  │
│                      │                                │         │
└──────────────────────┼────────────────────────────────┼─────────┘
                       │                                │
                       ▼                                ▼
       ┌──────────────────────────┐         ┌────────────────────┐
       │   Database (Snowflake    │         │   LLM provider     │
       │   or Postgres+pgvector)  │         │  (Claude / OpenAI) │
       │                          │         └────────────────────┘
       │  ┌────────────────────┐  │
       │  │ documents          │  │   ▲
       │  │ chunks (+ vector)  │  │   │
       │  │ ingest_runs        │  │   │
       │  └────────────────────┘  │   │ batch upsert
       └──────────────────────────┘   │
                  ▲                   │
                  │                   │
       ┌──────────┴──────────────────┐│
       │   Ingestion pipeline (CLI)  ││
       │                             ││
       │  PDF → parse → chunk →      ││
       │  enrich metadata → embed →  ├┘
       │  upsert                     │
       └─────────────────────────────┘
```

**Two distinct planes:**
1. **Ingestion plane** (offline, batch): runs once per corpus change. Idempotent.
2. **Query plane** (online): stateless, fast, the user-facing path.

Keeping them separate means embedding model changes, chunking changes, or re-parses never block the query path.

---

## 7. Stage-by-Stage Input/Output Contracts

### Stage 1 — PDF Ingestion
**In:** A directory of PDFs.
**Out:** Structured per-page records.
```json
{
  "doc_id": "digital_realty_2026_03",
  "page_number": 14,
  "text": "...extracted text...",
  "tables": [ { "page": 14, "rows": [...] } ],
  "source_path": "Documents/Digital Realty_Investor Presentation March 2026.pdf"
}
```
**Tooling:** `pypdf` for text + `pdfplumber` for tables. `pymupdf` (fitz) as fallback for messy layouts. **No OCR** in v1 — flagged as a limitation.

### Stage 2 — Metadata Enrichment
**In:** Per-page records + filename.
**Out:** Adds `company`, `doc_date`, `doc_type`, `version_label`.

Parse filename with regex + a small lookup table:
```
"Digital Realty_Investor Presentation March 2026.pdf"
  → company: "Digital Realty"
  → doc_date: "2026-03-01"
  → doc_type: "investor_presentation"
  → version_label: "Mar 2026"
```
For ambiguous names (e.g. `Simon The Impact of Brick and Mortar Shopping.pdf`), default `doc_type: "third_party_report"` and leave `doc_date` null.

### Stage 3 — Chunking
**In:** Per-page text with metadata.
**Out:** Chunks of ~800 tokens with 100-token overlap.
```json
{
  "chunk_id": "digital_realty_2026_03::p14::c0",
  "doc_id": "...", "page_number": 14,
  "company": "Digital Realty", "doc_date": "2026-03-01",
  "text": "...", "token_count": 786
}
```
**Strategy:** Page-aware recursive splitter (LangChain `RecursiveCharacterTextSplitter`). Never split across pages — keeps citations precise to a page. Tables get their own chunks (one chunk per table, serialized as Markdown).

### Stage 4 — Embedding
**In:** Chunk text.
**Out:** 1536-dim (or 1024-dim) vector.
**Model:** `text-embedding-3-small` (OpenAI) for cost, or `bge-small-en-v1.5` for local. Batch size 64. Idempotent by `chunk_id`.

### Stage 5 — Storage
**In:** Chunk + vector + metadata.
**Out:** Rows in `chunks` table with VECTOR column. See §9 for schema.

### Stage 6 — Query
**In:** User question + optional filters (company, doc_date).
**Out:** Top-k chunks (k=8 after rerank, from ~30 initial).

Steps:
1. (Optional) Query rewrite — expand acronyms (FFO → "Funds From Operations"), one cheap LLM call. Skippable in v1.
2. Embed query.
3. Hybrid retrieve: dense ANN (cosine) ∪ BM25 keyword. Take top-30 union.
4. Apply metadata filters if provided.
5. Re-rank with cross-encoder → top-8.
6. Version-aware boost: if query contains "current"/"latest", boost most recent `doc_date` per company.

### Stage 7 — Answer Generation
**In:** Question + top-k chunks (each with `[Source N]` tag).
**Out:** Markdown answer with inline `[Source N]` citations + structured citations list.

The prompt enforces:
- Answer **only** from provided sources.
- Tag every claim with `[Source N]`.
- If two sources disagree, surface both with attribution (don't average).
- If insufficient, say "I don't have enough information in the provided documents."

---

## 8. UI Design (Streamlit)

```
┌────────────────────────────────────────────────────────┐
│ Vectera RAG — REIT Investor Docs                       │
├────────────────────────────────────────────────────────┤
│ Sidebar:                                               │
│   ▸ Company filter [dropdown, multi]                   │
│   ▸ Date range    [date picker]                        │
│   ▸ Top-k         [slider 3-15]                        │
│   ▸ Show retrieval debug [checkbox]                    │
├────────────────────────────────────────────────────────┤
│  Ask a question:                                       │
│  ┌──────────────────────────────────────────────────┐  │
│  │ How has Digital Realty's FFO changed?            │  │
│  └──────────────────────────────────────────────────┘  │
│                                  [Ask]                 │
│                                                        │
│  Answer:                                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │ According to the Dec 2025 deck [Source 1] ...    │  │
│  │ The Mar 2026 update reports ... [Source 2]       │  │
│  │ NOTE: these two values reflect different         │  │
│  │ reporting periods.                               │  │
│  └──────────────────────────────────────────────────┘  │
│                                                        │
│  Citations:                                            │
│   [1] Digital Realty Dec 2025 — p.14                   │
│   [2] Digital Realty Mar 2026 — p.11                   │
│                                                        │
│  ▾ Retrieved chunks (debug)                            │
│     ┌─ Source 1: ... excerpt ...                       │
│     └─ Source 2: ... excerpt ...                       │
└────────────────────────────────────────────────────────┘
```

**UI requirements:**
- Single page, conversation-optional
- Citations rendered as clickable expanders showing the raw chunk text
- "Show retrieval debug" toggle reveals all retrieved chunks + scores (huge for the demo video)
- Sidebar filters propagate to retrieval — proves version-awareness visually

---

## 9. Backend & Data Model (LLD)

### Database choice: **Postgres + pgvector** (recommended over Snowflake for this assessment)

| Option | Pros | Cons |
|---|---|---|
| Snowflake (preferred by brief) | Marks the brief, scales, has VECTOR type | Free tier expires, slower iteration, account setup overhead, vector search still maturing |
| **Postgres + pgvector (Supabase)** | Free tier, mature, fast iteration, hybrid (BM25 via `tsvector` + ANN via `pgvector`) | Smaller scale ceiling — fine for ~10 docs |
| ChromaDB / FAISS | Trivial to set up | Brief explicitly asks for a *database service* |

**Recommendation:** Build on **Supabase (Postgres + pgvector)** for dev speed, document Snowflake as an alternative path with a thin DAO interface so the storage layer can swap. Snowflake setup is doable but eats 2–3h of the budget.

### Schema

```sql
CREATE TABLE documents (
  doc_id          TEXT PRIMARY KEY,
  source_path     TEXT NOT NULL,
  company         TEXT,
  doc_date        DATE,
  doc_type        TEXT,
  version_label   TEXT,
  page_count      INT,
  ingested_at     TIMESTAMPTZ DEFAULT now(),
  checksum        TEXT  -- sha256 of file; dedupe key
);

CREATE TABLE chunks (
  chunk_id        TEXT PRIMARY KEY,
  doc_id          TEXT REFERENCES documents(doc_id) ON DELETE CASCADE,
  page_number     INT NOT NULL,
  chunk_index     INT NOT NULL,
  text            TEXT NOT NULL,
  token_count     INT,
  chunk_type      TEXT,   -- 'prose' | 'table' | 'caption'
  embedding       VECTOR(1536),
  text_tsv        TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX chunks_embedding_idx ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX chunks_tsv_idx       ON chunks USING gin  (text_tsv);
CREATE INDEX chunks_company_date  ON chunks ((doc_id));  -- joined with documents

CREATE TABLE query_log (
  query_id      UUID PRIMARY KEY,
  question      TEXT,
  filters       JSONB,
  retrieved_ids TEXT[],
  answer        TEXT,
  latency_ms    INT,
  created_at    TIMESTAMPTZ DEFAULT now()
);
```

### Module layout (Python)

```
rag_system/
├── ingest/
│   ├── parse.py          # PDF → page records (pypdf + pdfplumber)
│   ├── metadata.py       # filename → company/date/version
│   ├── chunk.py          # page records → chunks
│   ├── embed.py          # chunks → vectors (batched)
│   └── pipeline.py       # orchestrates the above; CLI entry
├── storage/
│   ├── db.py             # connection, migrations
│   ├── repository.py     # CRUD: documents, chunks, query_log
│   └── schema.sql
├── retrieval/
│   ├── hybrid.py         # dense + bm25 fusion (RRF)
│   ├── rerank.py         # cross-encoder rerank
│   └── filters.py        # company/date filter logic + recency boost
├── generation/
│   ├── prompt.py         # system + user prompt templates
│   ├── llm.py            # provider abstraction (Anthropic/OpenAI)
│   └── citations.py      # parse [Source N] back into structured cites
├── api/
│   └── service.py        # thin function: query(q, filters) -> Answer
├── ui/
│   └── streamlit_app.py
├── eval/
│   ├── questions.yaml    # ~15 hand-written Q&A
│   └── run_eval.py
├── tests/
├── .env.example
├── requirements.txt
└── README.md
```

---

## 10. Component Choices — and Why

| Concern | Choice | Reasoning |
|---|---|---|
| PDF text | `pypdf` | Fast, good enough for digital-native decks |
| PDF tables | `pdfplumber` | Best free table extractor; serialize to Markdown |
| Fallback parser | `pymupdf` (fitz) | Better with multi-column / styled layouts |
| OCR (skipped in v1) | `pytesseract` | Decks are digital, not scanned; document as limitation |
| Chunking | LangChain `RecursiveCharacterTextSplitter`, ~800/100 | Sentence-aware, language-agnostic, well-tested |
| Embedding | `text-embedding-3-small` (1536d) | $0.02/1M tokens; strong retrieval on financial text. Local alt: `bge-small-en-v1.5` |
| Vector DB | Postgres + pgvector | Same DB as metadata → cheap hybrid search; HNSW index is fast at this scale |
| Lexical | Postgres `tsvector` | No new infra; fuses with dense via RRF |
| Re-rank | Cohere Rerank v3 *or* `bge-reranker-base` | Big precision lift for ~30→8 narrowing |
| LLM | Claude Sonnet 4.6 (`claude-sonnet-4-6`) | Strong long-context grounding + low hallucination on cited tasks; OpenAI `gpt-4o-mini` is cheaper fallback |
| LLM client | `anthropic` SDK with **prompt caching** on system prompt | Same system prompt every call → cache hit cuts cost ~90% |
| Orchestration | Plain Python — **no LangChain at runtime** | LangChain useful for splitter, not for orchestration. Keep control of prompts and retries. |
| API | FastAPI (optional) or direct function call from Streamlit | At this scale, in-process is fine. Wrap as `query()` so it's trivially HTTP-able later. |
| UI | Streamlit | Brief recommends it. Fastest path to a demo-able UI. |
| Config | `pydantic-settings` + `.env` | Type-safe, one place for keys + models |
| Testing | `pytest` | Industry standard |
| Eval | Custom YAML + LLM-as-judge | See §11 |
| Logging | `structlog` + Postgres `query_log` | JSON logs to console + persistent audit table |

---

## 11. Evaluation & Metrics

### Retrieval metrics (offline, on `eval/questions.yaml`)
For each Q&A, mark the **gold chunk(s)** by `chunk_id` and measure:

| Metric | What it tells you |
|---|---|
| **Recall@k** (k=5, 10) | Did we surface the right evidence at all? |
| **MRR** | How high in the ranking is the right answer? |
| **nDCG@10** | Quality of ranking when multiple chunks are relevant |
| **Hit rate by query type** | Single-doc vs cross-doc vs version-compare — where do we fail? |

### Generation metrics
| Metric | Method |
|---|---|
| **Faithfulness / groundedness** | LLM-as-judge: "Is every claim supported by cited chunks?" (1–5) |
| **Citation accuracy** | Are `[Source N]` tags resolvable and correct? Automated parse + lookup |
| **Answer relevance** | LLM-as-judge: "Does it answer the question?" |
| **Conflict handling** | Hand-graded on conflict-specific eval questions |
| **Refusal correctness** | For "unanswerable" questions, did it correctly refuse? |

### Eval set composition (~15 questions)
- 5 single-fact lookups (e.g., "What is BXP's Q4 2025 occupancy?")
- 3 version-compare (e.g., "How has Digital Realty's revenue changed Dec→Mar?")
- 3 cross-document (e.g., "Compare data center strategies between Digital Realty and equinox-mentioning docs")
- 2 conflict cases (where two versions disagree)
- 2 unanswerable / chart-only ("What does the chart on PSA p.7 show?") — should refuse

### System metrics (live)
- p50 / p95 latency per stage
- Cost per query
- Retrieval cache hit ratio (if added)

---

## 12. Testing Strategy

| Layer | What to test | How |
|---|---|---|
| Unit | Filename → metadata parser; chunker boundary cases; citation parser | `pytest`, table-driven |
| Integration | Ingest → query roundtrip on 1 sample PDF | `pytest` + ephemeral Postgres (testcontainers) |
| Retrieval eval | Recall@k, MRR on `eval/questions.yaml` | `python -m eval.run_eval` (CI-runnable) |
| Generation eval | LLM-judge faithfulness/relevance | Same harness |
| Smoke | Streamlit boots, end-to-end query returns answer | Manual + a `make smoke` |
| Regression | Re-run eval set before every commit on retrieval/prompt files | Pre-commit or GH Action |

**No load testing.** Out of scope for the brief.

---

## 13. Infrastructure & Deployment

**For the assessment** (what's actually built):
- Local dev: Python 3.11, Postgres via Docker (`docker-compose.yml`)
- Or zero-setup: Supabase free tier (Postgres + pgvector hosted)
- Secrets in `.env` (gitignored); `.env.example` checked in
- Single `make` targets: `make install`, `make ingest`, `make run`, `make eval`
- Streamlit on `localhost:8501`

**What scaling would look like** (documented in README, not built):

```
┌─────────────┐       ┌─────────────────┐       ┌──────────────┐
│  Cloudflare │──────▶│  FastAPI on     │──────▶│  Snowflake / │
│   / Vercel  │       │  Cloud Run / ECS│       │   Postgres   │
│  (Streamlit │       │  (autoscale)    │       │   (managed)  │
│   or React) │       └────────┬────────┘       └──────────────┘
└─────────────┘                │
                               ▼
                      ┌────────────────┐
                      │  Anthropic /   │
                      │  OpenAI API    │
                      │  (with caching)│
                      └────────────────┘

   Ingestion: separate worker (Cloud Run Job / Lambda) triggered on
   S3/GCS upload event. Writes to same DB. Idempotent by checksum.
```

**Access control (optional ask in brief):**
Add `tenant_id` column to `documents` and `chunks`. Every query carries an authenticated `tenant_id`; retrieval SQL has `WHERE tenant_id = $current_tenant` baked in at the repository layer (not the caller — defense in depth). Snowflake row-access policies or Postgres RLS would enforce it at the DB level in prod.

---

## 14. Key Tradeoffs (the "design considerations" section)

| Tradeoff | Decision | Why |
|---|---|---|
| Snowflake vs Postgres | Postgres+pgvector primary, Snowflake documented | Iteration speed; brief allows equivalents |
| Naive RAG vs agentic | Naive | Corpus too small to justify; latency + failure-mode cost too high |
| Chunk size 800 vs 400 vs 1500 | 800 + 100 overlap | Balances retrieval precision (smaller better) with LLM context coherence (larger better); financial paragraphs are often 500–900 tokens |
| Page-aware chunking vs free flow | Page-aware | Citations need page numbers; small recall cost is worth it |
| OCR or not | Not in v1 | Decks are digital; OCR adds 4–6h and Tesseract quality on slides is poor |
| Charts: extract or disclaim | Disclaim, capture captions only | Brief explicitly says this is acceptable |
| Hybrid vs pure dense | Hybrid (RRF fusion) | Tickers + metric acronyms hurt pure dense recall |
| Re-rank or not | Yes, optional flag | Quality win > latency cost; toggleable for ablation |
| LangChain heavy vs light | Light — splitter only | LangChain abstractions obscure prompts; we want full control of citations |
| Per-doc vs global ingestion | Idempotent by `checksum` | Re-ingest is safe; supports doc updates |
| Streaming response | Not in v1 | Streamlit `st.write_stream` is one-liner; add if time permits |
| Memory / multi-turn | Not in v1 | Out of scope for brief; easy to add |

---

## 15. Known Limitations (state these in README + demo)

1. **Chart/image content is not extracted** — only text captions near charts. Questions whose answer lives only in a chart will (correctly) refuse.
2. **Table extraction is best-effort** — `pdfplumber` handles clean tables well, struggles with merged cells.
3. **No OCR** — assumes digital-native PDFs.
4. **No multi-turn memory** in v1.
5. **LLM hallucination risk is reduced, not zero** — even with strict prompting, ~5% of grounded-LLM answers introduce minor unsupported phrasing.
6. **Version detection is filename-based** — fragile to renames. Production would parse the cover page.
7. **No real-time ingestion** — batch only.
8. **Single-tenant** — multi-tenant access control sketched but not enforced.

---

## 16. What "more time" would unlock (for README)

- Snowflake as primary store with Cortex vector search
- Vision-LLM pass over chart images (Claude Sonnet w/ image input) → text descriptions stored as chunks
- Multi-turn chat with conversation memory
- Query rewriting + HyDE for hard questions
- Eval-driven prompt tuning loop in CI
- Tenant isolation with Postgres RLS
- Streaming responses
- Caching layer for repeated queries (Redis)

---

## 17. Build Order (within the 6–10h budget)

1. **Hour 0–1:** Scaffold repo, Postgres+pgvector via Docker, schema migration.
2. **Hour 1–3:** Ingestion pipeline: parse → metadata → chunk → embed → upsert. Run end-to-end on `Documents/`.
3. **Hour 3–4:** Hybrid retrieval + filters. Verify with hand queries via Python REPL.
4. **Hour 4–6:** Prompt + LLM call + citation parsing. Get groundedness right before UI.
5. **Hour 6–7:** Streamlit UI — input, answer, citations, debug panel, filters.
6. **Hour 7–8:** Eval set (15 Qs) + `run_eval.py`. Run it, iterate on chunking/prompts.
7. **Hour 8–9:** README + architecture diagram + limitations.
8. **Hour 9–10:** Record demo video.

---

## Open questions for you

Before we start building, three calls to make:

1. **DB choice — Postgres+pgvector (faster) or Snowflake (closer to brief)?** I'd pick Postgres but document Snowflake.
2. **LLM provider — Claude Sonnet 4.6 (better grounding) or OpenAI gpt-4o-mini (cheaper)?** I'd pick Claude.
3. **Re-ranker — Cohere Rerank (API, fast) or local `bge-reranker-base` (free, slower)?** I'd skip re-ranker in v1 and add only if eval recall is bad.
