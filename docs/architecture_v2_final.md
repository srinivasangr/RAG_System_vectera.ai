# Architecture v2 — Final Design (locked)

> **Status:** Design contract. No code changes until this is reviewed and approved.
> **Author:** Srinivasan
> **Date:** 2026-05-19
> **Replaces:** `docs/architecture_v2.md` (yesterday's generic draft)
> **Branch:** all v2 work lands on `v2` branch → merge to `main` on Thursday.

---

## 0. TL;DR

v2 transforms our linear RAG pipeline into a **multi-stage, doc-aware, multi-vector retrieval system** with **bbox-grounded vision** and **conflict-aware generation**, targeting the 24-question Vectera battery.

**Baseline:** v1 scores **6 Pass / 7 Partial / 11 Fail** (full breakdown in `app/eval/baselines/v1_battery_scored.md`).

**v2 goal:** ≥ **16 Pass / 0 confidently-wrong Fails** (Fails allowed only as honest refusals).

The single most important design principle: **every component is justified by which battery failure mode (F1–F8) it closes.** Nothing is built for complexity's sake.

---

## 1. Failure modes v2 must close (from v1 baseline)

| # | Gap | v1-failed Qs | v2 mechanism |
|---|---|---|---|
| **F1** | doc_type-aware ranking | Q2, Q11, Q12 | `doc_type` enum + router-driven boost weights |
| **F2** | version-pair surfacing | Q1, Q5 | `doc_family_id` + sibling expansion stage |
| **F3** | publication-date staleness | Q4 | LLM-extracted `as_of_date` + prompt staleness rule |
| **F4** | temporal-delta synthesis | Q3 | Delta-intent router → per-doc sub-queries → delta-aware prompt |
| **F5** | vision spatial mapping | Q15, Q16 | bbox-grounded vision → `chart_records` structured rows |
| **F6** | cross-page synthesis | Q17, Q19 | Two-hop retrieval + same-doc adjacency boost |
| **F7** | retrieval diversification | Q22, Q24 | MMR + per-doc quota + per-entity round-robin |
| **F8** | table column→entity preservation | Q14 | `table_rows` structured table with column labels |

> Concept-level explanations for each gap live in `docs/v2_concepts_primer.md`.

---

## 2. End-to-end pipeline (high-level shape)

```
                       ┌────────────────────────────────────────┐
                       │            OFFLINE — INGEST             │
                       └────────────────────────────────────────┘
PDF
 │
 ├─► [1] Docling parse (layout-aware: text + tables + image regions)
 │
 ├─► [2] Doc-level metadata enrichment
 │       • doc_type (LLM classify from filename + first page)
 │       • as_of_date (LLM extract from pages 1–2 cover content)
 │       • doc_family_id (company + doc series hash)
 │
 ├─► [3] Page classification
 │       • prose-heavy / table-heavy / chart-heavy / photo / cover
 │       • content-density filter drops empty-photo pages
 │
 ├─► [4] Structure-aware chunking
 │       • prose: semantic split (similarity-drop), 300–500 tok target
 │       • table: kept whole, then per-row decomposition with column labels
 │       • list: 2-col bullet slides kept as single chunk
 │       • footnote attachment: footnotes glued to their body chunk
 │
 ├─► [5] Parent-chunk generation
 │       • each slide becomes a parent (~1500–2500 tok)
 │       • child chunks reference parent_id
 │
 ├─► [6] Vision pass (chart/map/logo pages only)
 │       • Gemini 2.5 Flash primary, Pro fallback, bbox prompt
 │       • Output: structured chart_records (label↔value pairs with bboxes)
 │
 ├─► [7] Proposition extraction
 │       • LLM decomposes each prose chunk into atomic facts
 │       • each proposition inherits parent's doc metadata
 │
 ├─► [8] Multi-vector embedding (BGE-base-en-v1.5, 768d)
 │       • propositions  (highest-precision retrieval target)
 │       • child chunks  (lexical retrieval target)
 │       • parent chunks (context expansion target)
 │
 └─► [9] Snowflake upsert
         tables: documents, parent_chunks, chunks, propositions,
                 table_rows, chart_records, query_log


                       ┌────────────────────────────────────────┐
                       │            ONLINE — RETRIEVE            │
                       └────────────────────────────────────────┘
Query
 │
 ├─► [1] Router (LLM, 1 call, JSON out)
 │       intent: {lookup | compare | delta | recency | enumerate | refuse}
 │       entities: companies[], metrics[]
 │       doc_type_boost: {Q4_Update: 1.5, Investor_Day: 1.0, ...}
 │       sub_queries: [...] (1 or more)
 │       temporal: {as_of_date_filter, prefer_recent}
 │       use_vision: bool, use_tables: bool
 │
 ├─► [2] Per-sub-query parallel retrieval (entity-filtered)
 │       • dense (over propositions)
 │       • lexical (over child chunks, BM25-style LIKE+scoring)
 │       • structured (table_rows / chart_records when use_tables=true)
 │       → RRF fuse per sub-query → 50 candidates
 │
 ├─► [3] Cross-encoder rerank (BGE-reranker-v2-m3, local)
 │       → 20 candidates
 │
 ├─► [4] Diversification (MMR λ=0.5 + per-doc quota K_per_doc=3)
 │       → 12 candidates
 │
 ├─► [5] Version-pair expansion
 │       for each retained chunk, if doc has family siblings AND
 │       query has no explicit date filter → fetch sibling chunks
 │       covering the same proposition/slide title
 │       → 12 + sibling additions
 │
 ├─► [6] Small-to-big expansion
 │       replace each chunk with its parent_chunk; dedupe parents
 │       → ~6–10 parent chunks
 │
 ├─► [7] Conflict detection
 │       for each (entity, metric) pair, group chunks; if 2+ chunks
 │       claim different values, tag them as a conflict_pair
 │       → annotated chunks for prompt
 │
 ├─► [8] LLM generation (Cerebras gpt-oss-120b primary)
 │       conflict-aware prompt template; staleness rule;
 │       per-entity refusal for missing data
 │
 └─► [9] Logging
         query_log row with retrieval trace, stage timings, citations
```

---

## 3. Snowflake schema (exact SQL)

### 3.1 New tables

```sql
-- Parent chunks: slide-level context for small-to-big expansion
CREATE TABLE parent_chunks (
  parent_id     VARCHAR PRIMARY KEY,
  doc_id        VARCHAR REFERENCES documents(doc_id),
  page_number   INT,
  slide_title   VARCHAR,
  text          VARCHAR,           -- ~1500–2500 tokens, full slide content
  page_image_b64 VARCHAR           -- thumbnail for UI citation modal
);

-- Propositions: atomic facts as the dense retrieval target
CREATE TABLE propositions (
  prop_id       VARCHAR PRIMARY KEY,
  chunk_id      VARCHAR REFERENCES chunks(chunk_id),
  parent_id     VARCHAR REFERENCES parent_chunks(parent_id),
  doc_id        VARCHAR REFERENCES documents(doc_id),
  text          VARCHAR,           -- single atomic statement
  embedding     VECTOR(FLOAT, 768),
  -- propagated metadata for fast filtered retrieval
  company       VARCHAR,
  doc_type      VARCHAR,
  doc_date      DATE,
  as_of_date    DATE,
  doc_family_id VARCHAR,
  version_label VARCHAR
);

-- Table rows: structured table data with column labels preserved
CREATE TABLE table_rows (
  row_id        VARCHAR PRIMARY KEY,
  chunk_id      VARCHAR REFERENCES chunks(chunk_id),
  doc_id        VARCHAR REFERENCES documents(doc_id),
  page_number   INT,
  table_id      VARCHAR,           -- groups rows of the same table
  row_idx       INT,
  columns_json  VARCHAR,           -- {"col_label_1":"val_1","col_label_2":"val_2",...}
  -- normalized denormalized columns to enable LIKE search
  flat_text     VARCHAR,           -- "Metric: Same-Store Occupancy; PSA: 92.0%; NSA: 84.3%; ..."
  embedding     VECTOR(FLOAT, 768),
  company       VARCHAR,
  doc_type      VARCHAR,
  doc_date      DATE
);

-- Chart records: vision-extracted bar/label tuples for charts/maps
CREATE TABLE chart_records (
  record_id     VARCHAR PRIMARY KEY,
  chunk_id      VARCHAR REFERENCES chunks(chunk_id),
  doc_id        VARCHAR REFERENCES documents(doc_id),
  page_number   INT,
  chart_id      VARCHAR,           -- groups records from same subchart
  chart_kind    VARCHAR,           -- bar | pie | map | logo_table | other
  label         VARCHAR,           -- e.g. "PSA", "Caesars Palace"
  value         VARCHAR,           -- e.g. "78%", "39%", "" if just a name
  unit          VARCHAR,           -- "%", "M sq ft", "GW", etc.
  bbox          VARCHAR,           -- "[x,y,w,h]" for debugging only
  confidence    FLOAT,             -- 0–1; below 0.6 → flagged in prompt
  vision_model  VARCHAR,           -- gemini-2.5-flash / gemini-2.5-pro / ...
  company       VARCHAR
);
```

### 3.2 Modifications to existing tables

```sql
-- documents: add doc_type, as_of_date, doc_family_id
ALTER TABLE documents ADD COLUMN doc_type VARCHAR;
-- enum-style values (enforced at app layer, not Snowflake):
--   investor_day | q4_update | q3_update | company_update | merger_presentation |
--   roadshow | third_party_report | initiation_report | analyst_day
ALTER TABLE documents ADD COLUMN as_of_date DATE;
ALTER TABLE documents ADD COLUMN doc_family_id VARCHAR;
-- doc_family_id = hash(company_ticker + doc_series_tag)
-- e.g. "DLR_investor_presentation_quarterly" groups Dec25 + Mar26 decks

-- chunks: add parent linkage and qualifier preservation
ALTER TABLE chunks ADD COLUMN parent_id VARCHAR;
ALTER TABLE chunks ADD COLUMN footnote_text VARCHAR;   -- footnote glued during chunking
ALTER TABLE chunks ADD COLUMN qualifier_text VARCHAR;  -- "including development projects"
ALTER TABLE chunks ADD COLUMN doc_type VARCHAR;        -- propagated from documents
ALTER TABLE chunks ADD COLUMN as_of_date DATE;
ALTER TABLE chunks ADD COLUMN doc_family_id VARCHAR;

-- query_log: extend for observability
ALTER TABLE query_log ADD COLUMN router_intent VARCHAR;
ALTER TABLE query_log ADD COLUMN sub_queries VARCHAR;      -- JSON array
ALTER TABLE query_log ADD COLUMN retrieval_stages VARCHAR; -- JSON of stage->ms+count
ALTER TABLE query_log ADD COLUMN rerank_top_ids VARCHAR;   -- JSON list of chunk_ids
ALTER TABLE query_log ADD COLUMN conflict_pairs VARCHAR;   -- JSON list
ALTER TABLE query_log ADD COLUMN provider_chain VARCHAR;   -- JSON list of {provider, ok, latency_ms}
```

### 3.3 Re-ingest strategy

- Full re-ingest of all 11 PDFs on Wednesday morning after schema lands
- No in-place migration — schema differences are too significant
- Ingest is idempotent on `(doc_id, checksum)` → safe to re-run
- Estimated time: ~30 min total (parse 5 min + vision 3 min + propositions 15 min + embed/upsert 7 min)

---

## 4. Ingestion pipeline detail

### 4.1 Doc-level metadata enrichment (closes F1, F3)

```python
# rag_system/ingest/metadata_v2.py
def enrich_doc_metadata(pdf_path: Path, first_two_pages_text: str) -> DocMetadata:
    # 1. doc_type — single LLM call, classify from filename + cover text
    doc_type = classify_doc_type(pdf_path.name, first_two_pages_text)
    #   returns one of: investor_day | q4_update | q3_update | company_update |
    #                   merger_presentation | roadshow | third_party_report

    # 2. as_of_date — extract from cover content (not filename)
    as_of_date = extract_as_of_date(first_two_pages_text)
    #   LLM prompt: "What date does this presentation cover/report data as of?
    #               Return ISO date or NULL."

    # 3. doc_family_id — group versions of the same source
    series_tag = derive_series_tag(doc_type, pdf_path.name)
    doc_family_id = sha1(f"{company_ticker}_{series_tag}")[:16]
    return DocMetadata(doc_type=doc_type, as_of_date=as_of_date,
                       doc_family_id=doc_family_id, ...)
```

### 4.2 Structure-aware chunking (closes F8, partially F2)

- **Prose:** semantic split via embedding-similarity drops (percentile threshold 95)
- **Tables:** stay whole as one chunk; additionally explode into `table_rows`
- **Lists:** 2-column bullet layouts (detected via Docling layout boxes) kept as one chunk
- **Footnotes:** detected via superscript markers in Docling output; text attached to the body chunk via `footnote_text` column
- **Qualifiers:** sentences like "including development projects" that modify a number are kept in `qualifier_text`

### 4.3 Vision pass (closes F5)

```python
# rag_system/ingest/vision_v2.py
def extract_chart_records(page_image: bytes, page_num: int) -> list[ChartRecord]:
    # Bbox prompt — Gemini 2.5 Flash, structured JSON output
    prompt = """
    You are reading a chart, map, or logo table.
    Return JSON ONLY:
    {
      "chart_kind": "bar|pie|map|logo_table|other",
      "subcharts": [
        {
          "chart_id": "...",
          "labels": [{"text":"PSA","bbox":[x,y,w,h]}, ...],
          "values": [{"text":"7.8%","bbox":[x,y,w,h]}, ...]
        }
      ]
    }
    Do NOT describe the chart in prose. Do NOT guess label↔value mappings —
    just list what you see with coordinates. We'll do the mapping.
    """
    # Then for each subchart, compute label↔value by spatial proximity (Euclidean
    # distance on bbox centers within the same subchart group).
    # confidence = how close the nearest label is, normalized.
```

**Fallback chain:** Gemini 2.5 Flash (free) → Gemini 2.5 Pro (free, harder pages) → OpenRouter free vision (last resort) → log `vision_unavailable` and continue ingest.

### 4.4 Proposition extraction (closes F2 indirectly via cleaner embeddings)

```python
# rag_system/ingest/propositions.py
def extract_propositions(chunk_text: str) -> list[str]:
    prompt = f"""
    Decompose the text below into atomic factual statements.
    Each statement must:
    - Be self-contained (no pronouns referring to outside text)
    - Carry exactly one fact
    - Preserve the original units, dates, and qualifiers
    - Begin with the entity name (no anaphora)

    Return a JSON list of strings, one per statement.

    Text:
    {chunk_text}
    """
    # 1 LLM call per chunk during ingest, ~1000 chunks total
    # Use Cerebras gpt-oss-120b for speed
```

**Note:** propositions are the primary *dense retrieval* target. The chunk + parent are used for *generation context*.

---

## 5. Multi-stage retrieval detail

### 5.1 Stage 1 — Router (closes F1, F4, F7)

```python
ROUTER_SYSTEM = """
You are a query classifier for a REIT investor-presentation Q&A system.
Output STRICT JSON only.

For each input query, output:
{
  "intent": "lookup|compare|delta|recency|enumerate|refuse",
  "entities": {
    "companies": ["Digital Realty", "BXP", ...],  // canonical names
    "metrics":  ["FFO", "NOI margin", "occupancy", ...]
  },
  "doc_type_boost": {
    "q4_update": 1.5, "investor_day": 1.0, ...
  },
  "sub_queries": [
    "<rewritten or decomposed sub-query>", ...
  ],
  "temporal": {
    "as_of_date_filter": "YYYY-MM-DD or null",
    "prefer_recent": true/false
  },
  "use_vision": true/false,
  "use_tables": true/false
}

Examples:
- "What's BXP 2026 occupancy?" -> intent=lookup, doc_type_boost={q4_update:1.5}
- "Compare DLR and VICI strategy" -> intent=compare, sub_queries=[per-company]
- "What's changed between BXP Investor Day and Q4?" -> intent=delta, sub_queries=[per-doc]
- "Latest DLR guidance" -> intent=recency, temporal.prefer_recent=true
- "VICI's 10 trophy assets" -> intent=enumerate (triggers two-hop retrieval)
"""
```

Single Cerebras call, ~500ms. Failure mode: bad JSON → fallback to intent=lookup, no sub-queries, no boosts.

### 5.2 Stage 2 — Per-sub-query parallel retrieval

For each sub-query in router output:

| Retriever | Target table | Filter | k |
|---|---|---|---|
| Dense | `propositions` | entity + temporal | 30 |
| Lexical | `chunks` (text + footnote_text) | entity | 30 |
| Structured | `table_rows` (flat_text) | entity + use_tables=true | 15 |
| Vision | `chart_records` (label, value) | entity + use_vision=true | 15 |

RRF fuse per sub-query → top 50.

### 5.3 Stage 3 — Cross-encoder rerank

- Model: `BAAI/bge-reranker-v2-m3`
- Local CPU, ~50ms for 50 candidates
- Outputs raw logit score; we keep top 20

### 5.4 Stage 4 — Diversification (closes F7)

```python
def diversify(candidates, lambda_=0.5, k_per_doc=3, top_k=12):
    selected = []
    doc_counts = {}
    while len(selected) < top_k and candidates:
        best, best_score = None, -inf
        for cand in candidates:
            if doc_counts.get(cand.doc_id, 0) >= k_per_doc:
                continue
            relevance = cand.rerank_score
            redundancy = max(
                (cosine(cand.embedding, s.embedding) for s in selected),
                default=0,
            )
            mmr_score = lambda_ * relevance - (1 - lambda_) * redundancy
            if mmr_score > best_score:
                best, best_score = cand, mmr_score
        selected.append(best)
        candidates.remove(best)
        doc_counts[best.doc_id] = doc_counts.get(best.doc_id, 0) + 1
    return selected
```

### 5.5 Stage 5 — Version-pair expansion (closes F2)

```python
def expand_version_pairs(chunks, router_output):
    if router_output.temporal.as_of_date_filter:
        return chunks  # user explicitly scoped a date — don't expand
    out = list(chunks)
    seen_props = {c.prop_id for c in chunks if c.prop_id}
    for c in chunks:
        siblings = fetch_family_siblings(
            doc_family_id=c.doc_family_id,
            slide_title=c.parent.slide_title,
            exclude_doc_id=c.doc_id,
        )
        for s in siblings:
            if s.prop_id not in seen_props:
                out.append(s)
                seen_props.add(s.prop_id)
    return out
```

### 5.6 Stage 6 — Small-to-big expansion

Replace each chunk with its `parent_chunk`. Dedupe parents (one slide may have multiple matching child chunks). Final context = parent chunks, not child chunks.

### 5.7 Stage 7 — Conflict detection

```python
def detect_conflicts(parent_chunks):
    # Heuristic: extract (entity, metric, value) tuples via regex + LLM helper
    # Group by (entity, metric); if 2+ distinct values, flag as conflict_pair
    pairs = []
    by_key = defaultdict(list)
    for p in parent_chunks:
        for triple in extract_triples(p.text):  # ("BXP", "dividend yield", "3.9%")
            by_key[(triple.entity, triple.metric)].append((p, triple))
    for key, items in by_key.items():
        values = {t.value for _, t in items}
        if len(values) > 1:
            pairs.append({"key": key, "members": items})
    return pairs
```

---

## 6. Generation prompt (closes F1, F2, F3, F4)

### 6.1 Conflict-aware system prompt

```
You are a senior research analyst answering questions about REIT investor
presentations. You are given numbered source excerpts. Follow these rules
in this exact order of precedence:

[RULE 1 — Grounding]
Every numeric or quoted claim MUST be cited with [N] markers tied to
a source. If a claim cannot be cited, do not make it.

[RULE 2 — Refusal]
If the sources do not contain enough information to answer, say:
"I don't have enough information in the provided documents to answer that."
Then briefly state what IS in the sources, so the user can refine.

[RULE 3 — Conflict-pairs]
You will sometimes be given a CONFLICT_PAIRS block listing chunks
that disagree on the same (entity, metric). When this happens:
  - Present BOTH values with full attribution
  - Format: "{value_a} per {doc_type_a} ({as_of_date_a}) [{n_a}],
             {value_b} per {doc_type_b} ({as_of_date_b}) [{n_b}]"
  - Do NOT pick one silently
  - Do NOT average them
  - If the differing values are due to a methodology change, explain it

[RULE 4 — Staleness]
For any cited source whose as_of_date is more than 2 years older than the
other cited sources OR more than 18 months before today, prepend the
answer with: "As of {as_of_date_year}, this source reports: ..." OR
"This data is from {year} and may no longer reflect current conditions."

[RULE 5 — Per-entity refusal]
For multi-entity questions ("for each X in {list}"), produce one bullet
per entity. If a particular entity has no data in the provided sources,
explicitly state: "No 2026 FFO guidance disclosed in the {company}
materials provided." Do NOT silently omit.

[RULE 6 — doc_type attribution]
When citing, include the doc_type and as_of_date in the citation context:
  "[3] BXP Q4 2025 Update, p.7 (Mar 2026)"
not just "[3] BXP".

[RULE 7 — Delta synthesis]
If the question asks what changed between two documents, structure the
answer as: (a) stable elements, (b) changed elements with old+new values,
(c) net-new disclosures.
```

### 6.2 Source format passed to LLM

```
[1] Boston Properties — Q4 2025 Update — p.131 — as_of 2026-03-10
SLIDE: 2026 Guidance
─────────────────────────────────────
{parent chunk text}
{any footnote_text}
{any qualifier_text}
─────────────────────────────────────

[2] Boston Properties — Investor Day 2025 — p.58 — as_of 2025-06-30
SLIDE: 2026 Occupancy Outlook
─────────────────────────────────────
{parent chunk text}
─────────────────────────────────────

CONFLICT_PAIRS:
- ("BXP", "2026 occupancy"): sources [1], [2]
```

---

## 7. Provider routing & failover

| Role | Primary | Fallback 1 | Fallback 2 | Failure handling |
|---|---|---|---|---|
| Router LLM | Cerebras gpt-oss-120b | Gemini 2.5 Flash | OpenRouter free | If all fail → default intent=lookup, no sub-queries |
| Generation LLM | Cerebras gpt-oss-120b | Gemini 2.5 Flash | OpenRouter free | If all fail → return retrieved context with apologetic message |
| Vision | Gemini 2.5 Flash | Gemini 2.5 Pro | OpenRouter Llama-Vision | If all fail → store `vision_unavailable`, continue ingest |
| Embedding | BGE-base-en-v1.5 local | — | — | No remote call → no failover needed |
| Reranker | BGE-reranker-v2-m3 local | (disable rerank stage) | — | No remote call → no failover needed |
| Proposition extraction | Cerebras gpt-oss-120b | Gemini 2.5 Flash | — | If all fail → skip proposition, use chunk only |

**Implementation:** `rag_system/llm_providers/router.py` — a `ProviderRouter` class that wraps the call site, attempts primary, catches rate-limit and 5xx errors, walks the chain. Logs each attempt to `query_log.provider_chain`.

---

## 8. Observability (extends `query_log`)

Every query writes a row with:

```json
{
  "query_id": "uuid",
  "ts": "2026-05-21T14:23:12Z",
  "question": "...",
  "router_intent": "compare",
  "sub_queries": ["...", "..."],
  "retrieval_stages": {
    "dense": {"ms": 80, "n_candidates": 50},
    "lexical": {"ms": 120, "n_candidates": 50},
    "rrf": {"ms": 5, "n_candidates": 50},
    "rerank": {"ms": 60, "n_candidates": 20},
    "diversify": {"ms": 10, "n_candidates": 12},
    "version_pair": {"ms": 40, "n_candidates": 14},
    "small_to_big": {"ms": 20, "n_parents": 8},
    "conflict_detect": {"ms": 80, "n_pairs": 2}
  },
  "rerank_top_ids": ["chunk_x", "chunk_y", ...],
  "conflict_pairs": [{"key":["BXP","dividend yield"], "n_members": 2}],
  "provider_chain": [
    {"provider":"cerebras","model":"gpt-oss-120b","ok":true,"latency_ms":2400}
  ],
  "answer": "...",
  "citations": [...],
  "total_latency_ms": 3200
}
```

Streamlit UI gets a new **debug panel** that renders this JSON for each query — invaluable for the demo (showing live multi-stage breakdown).

---

## 9. Component choices — locked

| Layer | Choice | Why this, not that |
|---|---|---|
| **Parser** | Docling (keep) | Layout-aware, preserves table structure, free, already wired. Alternatives: Unstructured.io (more cloud-leaning), LlamaParse (paid). |
| **Vision** | Gemini 2.5 Flash → Pro → OpenRouter | Free tier sufficient for 30 chart pages. GPT-4o/Claude considered for hardest pages — defer to Day 3 if time. |
| **Embedder** | BGE-base-en-v1.5 (keep) | Local, no rate limit, 768d works fine on financial text. OpenAI text-embedding-3-large considered — adds cost + rate limit for no measurable lift here. |
| **Reranker** | BGE-reranker-v2-m3 (local) | Cohere Rerank API considered — adds dependency, $/call, no quality lift over BGE-v2-m3 on this corpus size. |
| **Vector store** | Snowflake (keep) | `VECTOR_COSINE_SIMILARITY` is native, multi-tenancy via row-access policies, no separate store to operate. pgvector/Qdrant: switching cost > benefit at this scale. |
| **Primary LLM** | Cerebras gpt-oss-120b (keep) | Free, fast (~1s), already wired. Has reasoning channel. Claude/GPT-4o considered for higher quality — held in reserve for hardest cases. |
| **Orchestration** | Plain Python (sync) | LangGraph / LlamaIndex agents considered. Rejected: harder to debug, harder to explain in interview, no quality benefit for our stage count. |
| **API layer** | Streamlit (direct, no FastAPI split for v2) | FastAPI split is sketched in design only — adds 4hr work and zero eval impact. Documented as prod path. |
| **Cache** | In-process LRU (queries + embeddings) | Redis sketched in design only. Not needed at single-instance demo scale. |
| **Async ingest** | Sync (works for 11 docs) | Celery/arq sketched in design only. Demo-scale doesn't need it. |
| **Observability** | Extend `query_log` table + Streamlit debug panel | Langfuse/Phoenix/OTel sketched in design only. |

---

## 10. Production scaffolding (design doc only — NOT in v2 code)

These are the "scale this to a real production multi-tenant system" answers Prashant is testing for. Sketched here, explicitly out of v2's coding scope:

### 10.1 Multi-tenant architecture

```sql
ALTER TABLE documents      ADD COLUMN tenant_id VARCHAR NOT NULL;
ALTER TABLE chunks         ADD COLUMN tenant_id VARCHAR NOT NULL;
ALTER TABLE propositions   ADD COLUMN tenant_id VARCHAR NOT NULL;
ALTER TABLE parent_chunks  ADD COLUMN tenant_id VARCHAR NOT NULL;
ALTER TABLE chart_records  ADD COLUMN tenant_id VARCHAR NOT NULL;
ALTER TABLE table_rows     ADD COLUMN tenant_id VARCHAR NOT NULL;

-- Row-access policy
CREATE ROW ACCESS POLICY tenant_isolation AS (tenant_id VARCHAR)
RETURNS BOOLEAN ->
  CURRENT_ROLE() = 'SYSADMIN'
  OR tenant_id = CURRENT_SESSION_VARIABLES():tenant_id::VARCHAR;
```

Streamlit session sets `tenant_id` via auth; every query carries it; Snowflake enforces.

### 10.2 FastAPI split

- Streamlit becomes thin UI calling REST endpoints
- FastAPI exposes `/query`, `/ingest`, `/docs`, `/health`, `/metrics`
- Enables alternate clients (API, Slackbot, CLI)
- OpenAPI schema for downstream consumers

### 10.3 Async ingest pipeline

- Celery worker pool consuming Redis queue
- Ingest jobs become long-running tasks with progress polling
- Decouples large-document ingest from UI thread
- Per-tenant fair-share scheduling

### 10.4 Observability stack

- OpenTelemetry traces per request → Tempo/Jaeger
- Langfuse for LLM call traces + cost tracking
- Prometheus metrics for retrieval-stage latencies + provider success rates
- Grafana dashboards

### 10.5 CI eval regression gate

- GitHub Action runs full 24-Q battery on every PR
- Compares to baseline scores in `app/eval/baselines/`
- Blocks merge if pass-rate drops > 5% or any new "confidently wrong" fail

### 10.6 Cache layer

- Redis: query → answer cache, 15min TTL, keyed on (query, doc_set_hash)
- Persistent embedding cache (text→vector hash table) — embeddings are deterministic

---

## 11. Out of scope (explicitly deferred)

Even with infinite time, these don't help the battery. They are mentioned in design only:

- **Streaming responses** — UX nice-to-have, no eval impact
- **Multi-turn conversation memory** — separate problem
- **GraphRAG / knowledge graph** — premature at this corpus size
- **Fine-tuning embedder** — marginal lift, large effort
- **Synthetic eval generation** — the 24-Q battery IS the eval target
- **HyDE** — reranker captures most of the lift; HyDE made opt-in only

---

## 12. Two-day execution plan

> Effective working time: ~25 hours (Tue eve + Wed full + Thu until evening).

### Wednesday (12 hrs) — Schema + ingestion + retrieval skeleton

| Hours | Block | Output |
|---|---|---|
| 0–1 | Schema migration scripts | `app/scripts/migrate_v2.sql`, `app/rag_system/storage/schema_v2.sql` |
| 1–3 | Doc-level metadata enrichment | `metadata_v2.py` — doc_type classifier, as_of_date extractor, doc_family_id |
| 3–5 | Structure-aware chunker | `chunk_v2.py` — footnote attachment, qualifier preservation, table-row decomposition |
| 5–6 | Parent-chunk generator | `chunk_v2.py` cont'd — slide-level parents with image_b64 |
| 6–8 | Vision pass v2 | `vision_v2.py` — bbox prompt, chart_records output, provider failover |
| 8–10 | Proposition extractor | `propositions.py` — Cerebras LLM, JSON list output |
| 10–11 | Full re-ingest | All 11 PDFs through new pipeline |
| 11–12 | Audit + fix | Snowflake row counts, sanity-check `chart_records`, fix bugs |

### Thursday (10 hrs) — Retrieval + generation + eval

| Hours | Block | Output |
|---|---|---|
| 0–1.5 | Router LLM call | `retrieval/router.py` — JSON schema, prompt, fallback |
| 1.5–3 | Multi-source retrieval | `retrieval/v2_hybrid.py` — dense (props) + lexical (chunks) + structured (table_rows, chart_records) |
| 3–4 | Cross-encoder reranker | `retrieval/reranker.py` — BGE-reranker-v2-m3 |
| 4–5 | Diversification + version-pair + small-to-big | `retrieval/v2_pipeline.py` — orchestrator wiring all stages |
| 5–6 | Conflict detection + new prompt | `generation/prompt_v2.py` |
| 6–6.5 | Provider router wiring | `llm_providers/router.py` — chain fallback |
| 6.5–7.5 | Streamlit debug panel | Show retrieval trace per query |
| 7.5–8.5 | Run full 24-Q battery against v2 | `app/eval/baselines/v2_battery_results.{json,md}` |
| 8.5–9.5 | Manual scoring + failure analysis doc | `app/eval/baselines/v2_battery_scored.md` with v1→v2 delta table |
| 9.5–10 | Update README + record demo + submit | `README_v2.md`, 5-min demo video |

### Thursday evening — Ship

- Merge `v2` → `main`
- Push to remote
- Update Streamlit Cloud env vars (no app code change needed)
- Send revised submission email with: GitHub link, hosted link, demo video, scored battery

---

## 13. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Re-ingest takes longer than 30 min | Medium | Parallel doc processing; if too slow, ingest 5 most-critical docs first (BXP×2, DLR×2, PSA×2) |
| Proposition extraction hits Cerebras rate limits | Low | Already wired with rate limiter; fall back to per-chunk skipping (no propositions for that chunk) |
| Vision bbox prompts return malformed JSON | Medium | Retry with stricter prompt; if 2 retries fail, store `vision_unavailable` and continue |
| Cross-encoder reranker too slow on CPU | Low | Limit to top 30 candidates instead of 50; warm model on app start |
| BGE-reranker model download fails on demo machine | Low | Pre-download as Wed morning task; check into setup script |
| Conflict-detection regex misses real conflicts | High | Acceptable — partial implementation OK if obvious conflicts (e.g. Q5, Q9) work |
| v2 scores worse than v1 on a question | Medium | Re-ingest is reversible; v1 stays on `v1` branch as fallback |
| Streamlit Cloud doesn't have BGE-reranker model | Medium | Test on Wed evening; if fails, run reranker only locally and host without it (flag in UI) |

---

## 14. Decision log

| Date | Decision | Why |
|---|---|---|
| 2026-05-19 | Approve v2 design doc (this file) | Eval-driven; gap-by-gap mapping to F1–F8; clear scope |
| 2026-05-19 | Full re-ingest, no in-place migration | Schema additions are too significant for ALTER-and-backfill |
| 2026-05-19 | Stay on Snowflake | Switching costs > benefits; native VECTOR fits multi-tenant |
| 2026-05-19 | Cerebras primary, no streaming for v2 | Streaming has zero eval impact; keep code simple |
| 2026-05-19 | Sync ingest in v2; Celery in design only | Demo scale = 11 docs; async adds risk without quality lift |
| 2026-05-19 | Plain Python orchestration, no LangGraph | Easier to debug + explain; no quality benefit at our depth |
| 2026-05-19 | BGE-reranker-v2-m3 (local) | Cohere Rerank rejected — no quality lift, adds API dep |
| 2026-05-19 | Two-day target with Thu evening submission | User commitment |

---

## 15. Sign-off checklist

Before any code on this branch:

- [ ] User reviews this doc end-to-end
- [ ] Schema additions are acceptable (especially: doc_type enum values, doc_family_id approach)
- [ ] Provider routing chain is acceptable
- [ ] Component picks are acceptable (no overrides on parser/vision/embedder/reranker/LLM)
- [ ] Two-day execution plan is realistic
- [ ] Out-of-scope list is acceptable (FastAPI, Celery, OTel, Langfuse, multi-tenant remain design-only)
- [ ] Risks are acceptable (especially: vision JSON malformation, conflict-detection partial coverage)

Once these are checked, no further architecture changes until Thursday submission.

---

## 16. References

- v1 baseline: `app/eval/baselines/v1_battery_scored.md`
- Concepts primer: `docs/v2_concepts_primer.md`
- Eval source: `app/eval/battery_v1.yaml`, sourced from `Vectera_RAG_Self_Evaluation_Battery.docx`
- v1 architecture (for diff): `docs/architecture.md`, `docs/architecture.drawio`
