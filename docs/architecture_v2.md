# Architecture v2 — Restructure Design

> Status: **Proposal / design contract**. No code changes yet. v1 (current main) stays the source of truth until each phase below ships and beats baseline on eval.

This document captures the target architecture for the next iteration of the RAG system. It covers chunking & indexing, retrieval orchestration, production-grade scaffolding, and evaluation — sequenced into shippable phases so we never break "working > new."

---

## 1. Motivation

Current system (v1) works end-to-end with reasonable numbers:

| Metric | v1 baseline |
|---|---|
| Recall@8 | 82% |
| Refusal rate (out-of-scope) | 100% |
| Answer relevance | 96% |
| Faithfulness | 70% |
| **Context precision** | **9%** ← weakest link |

The 9% context precision means most of what we retrieve isn't directly relevant — the generator compensates, but we're leaving headroom on the table. v2 targets:

1. **Higher precision** without losing recall — primarily via cross-encoder reranking.
2. **Better grounding** — propositions + small-to-big context expansion.
3. **Smarter queries** — routing, rewriting, HyDE for hard queries.
4. **Production-grade scaffolding** — async ingest, observability, API decoupling, CI regression gates.
5. **Scalable eval** — synthetic Q&A generation, tiered eval, regression CI.

---

## 2. Target Architecture

### 2.1 High-level shape

```
┌──────────────────────────────────────────────────────────────────────┐
│                          INGEST PIPELINE                              │
│  PDF → Docling parse → Structure-aware split → Semantic refine        │
│   → Proposition extraction (LLM) → Hierarchical index (parent/child)  │
│   → Multi-vector embed → Snowflake (chunks, propositions, parents)    │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       RETRIEVAL ORCHESTRATOR                          │
│                                                                       │
│   Query → Router (LLM) ──┬─► Metadata filter (company/date/type)     │
│                          │                                            │
│        ┌─────────────────┼─────────────────┐                         │
│        ▼                 ▼                 ▼                          │
│   Query Rewriter    Multi-Query Gen    HyDE Embedding                 │
│   (decompose)       (N variants)       (hypothetical answer)          │
│        │                 │                 │                          │
│        └────────┬────────┴────────┬────────┘                          │
│                 ▼                 ▼                                   │
│        Dense (proposition)  Lexical (chunk)                           │
│                 │                 │                                   │
│                 └────────┬────────┘                                   │
│                          ▼                                            │
│                    RRF Fusion (top 50)                                │
│                          ▼                                            │
│                Cross-Encoder Rerank (BGE-reranker-v2)                 │
│                          ▼                                            │
│              Small-to-Big: expand to parent chunks (top 8)            │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     GENERATION + GUARDRAILS                           │
│   Grounded prompt → LLM → Citation validator → Faithfulness check     │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│                  EVAL & OBSERVABILITY (continuous)                    │
│  Per-stage metrics │ Synthetic Q&A gen │ Golden set │ Regression CI   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chunking & Indexing

v2 layers three complementary techniques rather than picking one:

### 3.1 Layered index design

| Layer | What it stores | Used for | Typical size |
|---|---|---|---|
| **Propositions** | Atomic facts ("Digital Realty's Q4 2025 NOI was $X") | Dense retrieval target | ~30–80 tokens |
| **Child chunks** | Semantic chunks (topic-coherent) | Lexical retrieval target | ~300–500 tokens |
| **Parent chunks** | Page or section-level | Context expansion (small-to-big) | ~1500–2500 tokens |

**Why this works:** propositions are dense and noise-free → high embedding quality. But propositions alone strip context, so at generation time we expand back to the parent. Lexical search runs on child chunks (keywords / acronyms / tickers live there).

### 3.2 Pipeline stages

1. **Docling parse** — keep (it works).
2. **Structure-aware split** — tables stay whole, lists stay whole, prose flows free.
3. **Semantic refine** — within prose, split on embedding-similarity drops (SemanticSplitter, breakpoint percentile = 95).
4. **Proposition extraction** — for each prose chunk, prompt LLM: *"Decompose into standalone factual statements"* → store as separate rows linked to parent.
5. **Multi-vector embedding** — embed propositions individually + embed full parent for fallback.

### 3.3 Snowflake schema additions

```sql
-- new tables
CREATE TABLE propositions (
  prop_id VARCHAR PRIMARY KEY,
  chunk_id VARCHAR REFERENCES chunks(chunk_id),  -- the parent child chunk
  doc_id VARCHAR,
  text VARCHAR,
  embedding VECTOR(FLOAT, 768),
  -- propagate filters for fast pruning
  company VARCHAR, doc_date DATE, version_label VARCHAR
);

CREATE TABLE parent_chunks (
  parent_id VARCHAR PRIMARY KEY,
  doc_id VARCHAR,
  page_number INT,
  text VARCHAR,  -- ~2000 tokens
  child_chunk_ids ARRAY
);

ALTER TABLE chunks ADD COLUMN parent_id VARCHAR REFERENCES parent_chunks(parent_id);
```

---

## 4. Retrieval Orchestrator

All four upgrades wired as a **sequential pipeline**, not parallel paths.

```
Query
  │
  ▼
[Router LLM]  →  decides: needs-comparison? needs-time-filter? out-of-scope?
  │              outputs: {intent, filters, sub_queries[]}
  ▼
[Query Rewriter / HyDE]
  │  • Rewrite: "latest DLR guidance" → "Digital Realty 2026 forward guidance EPS FFO"
  │  • HyDE: LLM generates fake answer paragraph → embed it (better semantic match)
  │  • Multi-query: produce 3 variants
  ▼
[Parallel retrieval per variant]
  │  dense (over propositions) + lexical (over child chunks) + filters from router
  ▼
[RRF fusion]  →  top 50 child chunks
  ▼
[Cross-encoder rerank]  →  BGE-reranker-v2-m3, score query×chunk pairs
  │  this is THE biggest precision lift — context precision target 9% → 60%+
  ▼
[Small-to-big expansion]  →  for top 8 chunks, fetch their parent_chunk
  ▼
LLM
```

### 4.1 Tech choices

| Component | Pick | Why |
|---|---|---|
| Router | Cerebras gpt-oss-120b | Already wired, fast, free for our usage |
| Query rewriter | Same model | Single JSON output, one call |
| HyDE | Optional toggle | Adds 1 LLM call + 1 embed; worth it for semantic queries only |
| Reranker | `BAAI/bge-reranker-v2-m3` (local) | 568MB, ~50ms/query on CPU, no API cost |
| Vector store | Stay on Snowflake | `VECTOR_COSINE_SIMILARITY` scales fine to ~1M chunks |

### 4.2 Retrieval modes (latency control)

Expose a `RetrievalMode` enum + UI toggle:

| Mode | Stages | Target latency |
|---|---|---|
| `FAST` | dense + lexical + RRF | ~200ms |
| `BALANCED` (default) | + cross-encoder rerank + small-to-big | ~500ms |
| `THOROUGH` | + HyDE + multi-query | ~2s |

---

## 5. Production-grade scaffolding

Runs as a separate track from the model-quality work — don't gate one on the other.

| Concern | Add |
|---|---|
| **Async ingest** | Celery or arq worker pool, Redis queue. Submit returns immediately, status polled. |
| **Observability** | OpenTelemetry traces per-stage; Langfuse or Phoenix for LLM call tracing |
| **Caching** | Redis cache: query → results (15min TTL). Embed cache: text→vector (permanent). |
| **Rate limiting** | Token bucket per provider, shared across workers via Redis |
| **Config** | Pydantic Settings with env layering (base/dev/prod), not just `.env` |
| **API layer** | FastAPI in front of Streamlit — Streamlit calls REST endpoints. Decouples UI from logic, enables future clients. |
| **CI** | GitHub Actions: unit tests + eval-on-PR with regression gates |
| **Migrations** | Alembic-style versioned schema migrations, not raw `schema.sql` |

---

## 6. Evaluation overhaul

### 6.1 Three-tier eval

```
Tier 1: Unit-level eval (per stage)        ← runs in CI on every PR
  • Retriever recall@k per query class
  • Reranker NDCG@5
  • Citation extractor: precision/recall on bracket parse

Tier 2: End-to-end eval (RAGAS + TruLens) ← runs nightly
  • Faithfulness, answer_relevance, context_precision, context_recall
  • Custom: refusal_correctness, version_correctness
  • Dashboard: Langfuse or Phoenix

Tier 3: Synthetic eval (scale)            ← runs weekly
  • LLM reads each doc, generates Q + ground-truth A + source chunks
  • Filter via critic LLM (drop ambiguous Qs)
  • Grows golden set automatically as corpus grows
```

### 6.2 Golden set hygiene

- Version-controlled YAML, ~50–100 questions across categories: simple lookup, comparison, multi-hop, version-sensitive, refusal, chart, table, computation.
- Each question tagged with `expected_doc_ids` for retriever-only eval.

### 6.3 Regression gates

In CI: if `recall@8` drops > 5% or `faithfulness` drops > 10% vs main on the golden set, **block merge**.

---

## 7. Sequenced migration plan

Six phases, each shippable independently. Each phase has a measurable eval win — if a phase doesn't move the needle, stop.

| Phase | Scope | Expected eval lift | Effort |
|---|---|---|---|
| **0** | Freeze current as `v1` branch. Run baseline eval, save numbers. | — | 1h |
| **1** | Add cross-encoder reranker only. No other changes. | **+30–50% context precision** (biggest single win) | 1 day |
| **2** | Add `parent_chunks` table + small-to-big expansion | +5–10% faithfulness | 2 days |
| **3** | Add proposition extraction + multi-vector index | +5–10% recall@k | 3 days |
| **4** | Add query rewriter + router (skip HyDE/multi-query) | +10% recall on hard queries | 2 days |
| **5** | Add HyDE + multi-query as opt-in THOROUGH mode | +5% recall, +1.5s latency | 1 day |
| **6** | Synthetic eval generation + nightly CI regression | infra, no model lift | 2 days |

Production scaffolding (Celery, OTel, FastAPI, Redis) runs in parallel as a separate track.

---

## 8. Honest trade-offs

1. **Propositions are expensive.** One LLM call per chunk during ingest. For 10 decks ≈ 1000 chunks ≈ 1000 LLM calls. Plan for cost + ~30min ingest time. Make it opt-in via config flag.
2. **HyDE doesn't always help.** It hurts on factoid queries ("what is BXP's ticker?"). The router should decide when to apply it.
3. **Cross-encoder reranker is the single best investment.** If only one phase ships, it's this one. 9% → 60%+ context precision is realistic.
4. **Multi-query fusion has diminishing returns** if the reranker is already strong. Don't add unless eval shows specific failure modes.
5. **Snowflake at this scale is overkill but fine.** Don't switch to pgvector/Qdrant unless there's a real reason — the storage layer isn't the bottleneck.

---

## 9. Out of scope (intentionally)

- **Streaming responses** — UX nice-to-have, no eval impact. Defer.
- **Multi-turn conversation memory** — separate problem; don't conflate with retrieval quality.
- **Fine-tuning the embedder** — high cost, marginal lift given BGE-base is already strong on financial text.
- **GraphRAG / knowledge graph** — interesting but premature for a 10-document corpus.

---

## 10. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-19 | Approve v2 design doc | Capture target shape before code changes; phases keep risk bounded |
| 2026-05-19 | Reranker first (Phase 1) | Single biggest expected eval lift; cheapest to implement |
| 2026-05-19 | Stay on Snowflake | Storage isn't the bottleneck; switching costs > benefit |

---

## 11. Next action

Phase 0: freeze v1, re-run baseline eval, snapshot numbers into `eval/baselines/v1_baseline.json`. Then Phase 1 (reranker).
