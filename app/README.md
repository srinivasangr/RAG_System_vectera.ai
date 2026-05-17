# Vectera RAG — REIT Investor Documents

A Retrieval-Augmented Generation system over a corpus of REIT investor
presentation PDFs. Built for the [Vectera.ai technical
assessment](../Vectera_RAG_System_Technical_Assessment.pdf).

Ask natural-language questions; receive **source-grounded answers with
inline `[N]` citations** that map back to specific document pages.

---

## Architecture

> **Full diagram:** [`../docs/architecture.drawio`](../docs/architecture.drawio)
> — open in [https://app.diagrams.net](https://app.diagrams.net) (File → Open)
> or in the VS Code **Draw.io Integration** extension.
>
> Inline-renderable Mermaid version: [`../docs/architecture.md`](../docs/architecture.md).

Two distinct planes:

1. **Ingestion (offline, batch):** PDF → Docling → Markdown + images →
   Gemini Vision describes each chart → chunk → Gemini embeddings →
   Snowflake. Image bytes also persisted to `chunk_images`.
2. **Query (online, stateless):** question → embed → hybrid retrieve
   (dense + lexical, fused with RRF) → prompt-format with `[N]` source
   markers → Cerebras `gpt-oss-120b` → parse citations back → render
   (with the original chart PNG for `chart_description` citations).

---

## How the database (Snowflake) is used

Everything lives in three Snowflake tables in `RAG_DB.RAG_SCHEMA`:

| Table | What it stores |
|---|---|
| `documents` | one row per PDF; metadata (company, doc_date, version_label, checksum) |
| `chunks` | one row per chunk; text, `chunk_type`, `embedding VECTOR(FLOAT, 768)`, denormalized `company` / `doc_date` for filter pushdown |
| `query_log` | one row per user query; question, retrieved chunk ids, answer, llm/provider, latency |

**Vector search** uses Snowflake's built-in
`VECTOR_COSINE_SIMILARITY(embedding, query_vec)` against the `chunks`
table. No external vector DB.

**Lexical** uses `LOWER(text) LIKE` with a per-token score expression
(Snowflake `SEARCH()` is gated on higher tiers). For ~hundreds of chunks
per document, this is fast and avoids a second store.

> **Trial-account note.** Snowflake's Cortex AI functions
> (`EMBED_TEXT_768`, `COMPLETE`, etc.) are *not* available on free trial
> accounts (error `399258`). We therefore use Snowflake for **storage
> and vector search only** and run embeddings + LLM through external
> providers. If the account had Cortex access, the embedder layer would
> be a one-line swap (see `rag_system/llm_providers/factory.py`).

---

## Chunking strategy

- **Page-aware.** Chunks never cross a page boundary, so every citation
  is precise to a page number.
- **Markdown-first.** Docling outputs structured Markdown (headings,
  lists, GitHub-style tables). The chunker recognises and preserves
  tables intact (one chunk per table unless > 2× the size limit).
- **~800 tokens / 100-token overlap** via `RecursiveCharacterTextSplitter`
  with markdown-friendly separators (`\n## `, `\n### `, `\n\n`, ...).
- **Tagged by content type.** Each chunk carries
  `chunk_type ∈ {prose, table, chart_description}`. This drives both
  retrieval ranking and how citations are rendered in the UI.

Token counts are computed with `tiktoken cl100k_base` — a stable proxy
that avoids per-provider tokenizer dependencies.

---

## Retrieval approach

**Hybrid: dense (cosine) ∪ lexical, fused with Reciprocal Rank Fusion.**

Why hybrid? Investor decks are dense with tickers, defined acronyms
(FFO, AFFO, NOI, ARR), and named metrics. Pure dense embedding misses
exact-string matches; pure keyword misses paraphrases. RRF is
parameter-free, robust to score-scale differences, and trivial to
implement.

Pipeline per query:

1. Embed the question with `gemini-embedding-001` (768d, `RETRIEVAL_DOCUMENT`).
2. Run two retrievers in parallel against Snowflake:
   - **Dense** — top-30 by `VECTOR_COSINE_SIMILARITY`.
   - **Lexical** — top-30 by sum of per-token `LIKE` hits.
3. Apply metadata filters (`company`, `doc_date`, `doc_type`) at the SQL
   layer.
4. RRF-fuse the two candidate sets → top-8.
5. (Optional, on "current / latest" queries) recency boost: small
   constant added to chunks from the most-recent-dated doc per company.

The fused result is returned with `dense_rank` / `lexical_rank` metadata
so the UI can show *why* a chunk was retrieved.

---

## Generation + citations

**LLM:** Cerebras `gpt-oss-120b` by default (frontier model served fast);
swap on the fly in the UI sidebar (Gemini 2.5 / OpenAI / Anthropic /
OpenRouter all supported via the modular provider layer).

**Prompt design** (see `rag_system/generation/prompt.py`):

- System prompt instructs: answer **only** from sources, cite every
  factual claim with `[N]`, surface disagreement with attribution, refuse
  cleanly when evidence is insufficient.
- User payload lists numbered sources with `[N]` headers carrying
  *company + version + page + chunk_type* so the model can disambiguate
  versions.

**Citation parsing:** `[N]` markers in the answer are regex-extracted and
mapped back to the originating chunks. The UI renders each cited chunk
in an expandable panel so the user can verify the grounding.

---

## How we handle the hard parts

### Versioning
- Each document gets a `version_label` (`Mar 2026`) and a `doc_date` from
  filename parsing.
- Both fields are denormalized onto every chunk so filters and the
  recency-boost rank push down to SQL.
- The prompt explicitly tells the model to attribute by version when
  versions disagree (*"Per the Mar 2026 deck [3]... whereas the Dec 2025
  deck [5] reported..."*).

### Conflicting information
- We never collapse conflicts. The prompt forbids averaging or silent
  picking; the model is told to surface both sides with citations.
- The retriever returns multiple candidates per query (top-k = 8), giving
  the model the *opportunity* to see the disagreement in the first place.

### Charts / tables / structured content
- **Tables:** Docling's structured table extraction → GitHub-style
  Markdown → embedded as their own chunks with `chunk_type='table'`.
  Tables on a page are not merged into surrounding prose.
- **Charts / figures:** a vision-LLM pass (Gemini 2.5 Flash) describes
  each image as structured Markdown (figure type, axes, key data
  points, trend). Descriptions are embedded as
  `chunk_type='chart_description'` chunks. An aggressive pre-filter
  drops tiny images (< 200 px shortest side) and extreme aspect ratios
  before vision is called, conserving free-tier quota.
- **Hard charts:** complex visualisations whose meaning can't be put
  into a paragraph (e.g. dense geographic heatmaps) get a useful but
  approximate description. Pure-decoration images get `NOT_A_CHART` from
  the vision pass and are dropped.

---

## Setup

### 1. Prereqs

- Python **3.13** (works on 3.11+)
- A Snowflake account (free trial is fine — *but no Cortex AI*; see note above)
- At least one external LLM provider key (Cerebras / Gemini / OpenAI / Anthropic / OpenRouter)
- A Gemini API key from [aistudio.google.com](https://aistudio.google.com/apikey) — used for *embeddings* and *chart-image descriptions* (free tier sufficient for this corpus)

### 2. Install

```powershell
cd app
python -m venv .venv
.venv\Scripts\Activate.ps1                    # Windows
# source .venv/bin/activate                   # macOS/Linux
pip install -r requirements.txt
```

### 3. Configure

```powershell
copy .env.example .env
# Edit .env to fill in:
#   SNOWFLAKE_ACCOUNT / USER / PASSWORD
#   GEMINI_API_KEY     (required — for embeddings + vision)
#   CEREBRAS_API_KEY   (default LLM)
#   any other provider keys you want available in the UI
```

### 4. Initialize Snowflake schema

```powershell
python -m rag_system.storage.init_snowflake
```

This is idempotent — creates `RAG_WH` (XSMALL warehouse with 60s
auto-suspend), `RAG_DB.RAG_SCHEMA`, and the three tables. Re-running it
is a no-op.

### 5. Ingest documents

Put PDFs in `../Documents/` (relative to `app/`), then:

```powershell
# Full corpus (~50 min for 10 PDFs; respects Gemini free-tier vision quota):
python -m rag_system.ingest.pipeline --vision-budget 200

# Or one doc at a time:
python -m rag_system.ingest.pipeline --doc "Digital Realty_Investor Presentation December 2025.pdf"

# Or dry-run (parse + chunk only, no embedding / Snowflake writes):
python -m rag_system.ingest.pipeline --doc "<file.pdf>" --no-vision --dry-run
```

Ingestion is **idempotent**: re-running on an unchanged file (same
sha256 checksum) is a no-op; on a changed file it replaces the
document's chunks.

### 6. Run the UI

```powershell
streamlit run rag_system/ui/streamlit_app.py
```

Open http://localhost:8501.

### 7. Run evals

```powershell
python -m eval.run_eval --skip-requires    # questions answerable now
python -m eval.run_eval                    # full set (after full ingest)
```

---

## Project layout

```
app/
├── rag_system/
│   ├── config/            # pydantic-settings + .env loader
│   ├── ingest/            # PDF → parse → metadata → vision → chunk → upsert
│   ├── storage/           # Snowflake schema + DAO + bootstrap
│   ├── retrieval/         # hybrid (dense + lexical) + filters + RRF
│   ├── generation/        # prompt + LLM call + citation parsing + query()
│   ├── llm_providers/     # modular: OpenAI / Anthropic / Gemini / Cerebras / OpenRouter
│   └── ui/                # Streamlit app
├── eval/                  # questions.yaml + run_eval.py
├── tests/                 # pytest unit tests
├── scripts/               # standalone smoke-test scripts
├── .env.example
├── Makefile
└── requirements.txt
```

---

## Known limitations

- **Snowflake Cortex AI is blocked on trial accounts.** We use external
  Gemini for embeddings; on a paid account, Cortex is a one-line swap.
- **OCR is off.** Investor decks are digital-native; OCR adds 5–10×
  parse time and Tesseract quality on slide layouts is poor. Scanned
  PDFs would currently return no text.
- **Vision-LLM rate limits.** Gemini 2.5 Flash free tier is 250 RPD;
  we pre-filter images and cap calls per run. A large corpus may need
  to be ingested across multiple days.
- **Lexical retrieval is `LIKE`-based**, not full-text. Works well at
  this scale but won't scale to millions of chunks. Snowflake `SEARCH()`
  on a higher tier would replace it cleanly.
- **No reranker** in this build. Eval shows recall@8 ≈ 78% on the
  current sample; if recall drops on larger corpora, adding
  `bge-reranker-base` or Cohere Rerank between candidate-30 and top-8
  is a ~30-min addition.
- **Single tenant.** Multi-tenant access control is sketched in the
  schema (the chunks table has space for a `tenant_id` column) but
  not enforced. Production would use Postgres RLS / Snowflake row
  access policies.
- **Vision describes the visible figure faithfully**, but cannot infer
  numbers that aren't visually rendered (e.g. precise tick values on a
  small inline sparkline).

---

## What I'd improve with more time

- **Snowflake Cortex on a paid account** — embedding + LLM in one place
  + Cortex Search (managed BM25 + vector) would simplify retrieval.
- **Cross-encoder reranker** between candidate-30 and top-8.
- **Multi-turn chat** with conversation memory (out of scope for the
  brief).
- **Eval-driven prompt tuning loop in CI** — run `eval/run_eval.py` on
  every prompt change and fail the build if must-contain drops.
- **Streaming responses** in the UI (`st.write_stream` one-liner).
- **Tenant isolation** with Snowflake row-access policies + a query-time
  `tenant_id` injected by the auth layer.
- **HyDE / query rewriting** for harder questions (e.g. expand acronyms
  before retrieval).
- **Caching layer** (Redis) for repeated queries; trivial speedup since
  retrieval is the latency-dominant step.

---

## Eval results (current snapshot)

Run on Digital Realty Dec 2025 only (full corpus pending final ingestion):

| Metric | Value |
|---|---|
| Recall@8 | 78% |
| Mean MRR | 0.68 |
| Must-contain rate | 100% |
| Refusal correctness | 100% |
| Mean latency | ~5–6s (one cold-start outlier at 56s) |

The 22% retrieval miss does not translate to wrong answers — the same
facts appeared in multiple chunks; the model still answered correctly.

---

## Submission checklist (per brief)

- [x] Working application with UI (Streamlit, not CLI)
- [x] PDF ingestion → chunking → embedding → retrieval → LLM
- [x] Citations on every answer
- [x] Version awareness (filename-driven metadata, recency boost)
- [x] Conflict awareness (prompt explicitly surfaces disagreements)
- [x] Table extraction (Docling structured tables → Markdown)
- [x] Chart awareness (vision-LLM descriptions; limits disclosed)
- [x] README with setup, architecture, choices, limitations
- [ ] Demo video (5–10 min) — *recorded separately*
