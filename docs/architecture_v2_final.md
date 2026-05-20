# Architecture v2 — Final Design (locked)

> **Status:** APPROVED & LOCKED (2026-05-19). No further architecture changes until Thursday submission.
> **Author:** Srinivasan
> **Date:** 2026-05-19
> **Replaces:** `docs/architecture_v2.md` (yesterday's generic draft)
> **Branch:** all v2 work lands on `v2` branch → merge to `main` on Thursday.

---

## 0. TL;DR

v2 transforms our linear RAG pipeline into a **multi-stage, doc-aware, multi-vector retrieval system** with **bbox-grounded vision** and **conflict-aware generation**, served by a **FastAPI backend + minimal streaming UI**, with **optional Langfuse observability** and **checkpointed ingestion**. It targets the 24-question Vectera battery.

**Baseline:** v1 scores **6 Pass / 7 Partial / 11 Fail** (full breakdown in `app/eval/baselines/v1_battery_scored.md`).

**v2 goal:** ≥ **16 Pass / 0 confidently-wrong Fails** (Fails allowed only as honest refusals).

**Two design principles govern everything:**
1. **Eval-driven** — every component is justified by which battery failure mode (F1–F8) it closes. Nothing built for complexity's sake.
2. **Domain-agnostic** — the system carries NO hardcoded domain knowledge (no "REIT", "FFO", company lists). Prompts read the corpus's metadata at runtime (§5.1, §6.1), so the same code works on medical, legal, or any other corpus. This directly resolves the overfitting risk in the prompts.

**Scope (locked):** Core F1–F8 + FastAPI/streaming UI + optional Langfuse + checkpointed ingest, by Thursday, with a fixed cut order (§12) for graceful degradation. Agentic = planned decomposition (§17). Multi-tenant / Celery / OTel / cache / CI = design-only (§10).

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

**Design principle: domain-agnostic.** The router carries NO hardcoded entities,
metrics, or document types. It is told what the corpus contains *at runtime* by
querying the system's own metadata. The same prompt works on REIT decks, medical
PDFs, legal contracts, or any other corpus — because the only domain knowledge is
data read from the database, never baked into the prompt.

```python
# rag_system/retrieval/router.py

def build_corpus_profile(conn) -> dict:
    """Read what the corpus actually contains — injected into the prompt.
    This is the ONLY source of domain knowledge. Nothing is hardcoded."""
    return {
        "doc_types":  query("SELECT DISTINCT doc_type FROM documents"),
        "entities":   query("SELECT DISTINCT company  FROM documents"),  # generic 'entity' column
        "date_range": query("SELECT MIN(as_of_date), MAX(as_of_date) FROM documents"),
        "modalities": query("SELECT DISTINCT chunk_type FROM chunks"),   # prose/table/chart...
    }

ROUTER_SYSTEM = """
You are a query analyzer for a document question-answering system.
You will be given a CORPUS PROFILE describing what the current corpus contains.
Use ONLY that profile — do not assume any domain.

Output STRICT JSON only:
{
  "intent": "lookup | compare | delta | recency | enumerate | refuse",
  "entities":  [ ... ],   // named entities mentioned IN THIS QUERY (extract freely; do not use a fixed list)
  "attributes":[ ... ],   // measured quantities/metrics mentioned IN THIS QUERY (extract freely)
  "sub_queries":[ ... ],  // decompose if multi-entity or multi-hop; else echo the query
  "temporal": {
    "as_of_date_filter": "YYYY-MM-DD or null",
    "prefer_recent": true/false
  },
  "doc_type_preference": [ ... ],  // ordered subset of the profile's doc_types relevant to this intent, or []
  "needs_tables": true/false,      // inferred from query language (asks for numbers/comparison/rows)
  "needs_charts": true/false       // inferred from query language (asks about a figure/visual/map)
}

INTENT DEFINITIONS (domain-agnostic):
- lookup    : a single fact about one entity
- compare   : contrast 2+ entities or 2+ attributes  -> decompose into per-entity sub_queries
- delta     : what changed between two documents/versions -> decompose into per-document sub_queries
- recency   : user wants the most current value ("latest", "current", "now") -> prefer_recent=true
- enumerate : user wants a list that may be split across pages -> enables two-hop retrieval
- refuse    : query is unanswerable from any document corpus

RANKING GUIDANCE (generic — no hardcoded hierarchy):
- For recency/forward-looking intents, prefer documents with a more recent as_of_date
  within the same document family.
- doc_type is only a TIE-BREAKER when dates are close. Reason about it generically:
  e.g. a periodic 'update' is typically more current than a one-off 'overview'.
- Never invent a document type that is not in the supplied CORPUS PROFILE.
"""

# At call time:
#   messages = [system, {"role":"user","content": f"CORPUS PROFILE:\n{profile}\n\nQUERY:\n{query}"}]
```

Single Cerebras call (reasoning model), ~500ms. **Failure mode:** bad JSON → fall
back to `intent=lookup`, `sub_queries=[query]`, no boosts. The reasoning trace is
captured and logged (see §18).

> Why this is not overfit: swap the corpus to clinical-trial PDFs and the profile
> becomes `doc_types=[trial_protocol, drug_label, ...]`, `entities` are extracted
> from each query, and the same prompt routes correctly with zero edits.

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
    # Domain-agnostic: extract (entity, attribute, value) triples via regex +
    # a small LLM helper. Group by (entity, attribute); if 2+ distinct values,
    # flag as a conflict_pair. Example triple shape: (entity, attribute, value).
    pairs = []
    by_key = defaultdict(list)
    for p in parent_chunks:
        for triple in extract_triples(p.text):
            by_key[(triple.entity, triple.attribute)].append((p, triple))
    for key, items in by_key.items():
        values = {t.value for _, t in items}
        if len(values) > 1:
            pairs.append({"key": key, "members": items})
    return pairs
```

---

## 6. Generation prompt (closes F1, F2, F3, F4)

### 6.1 Conflict-aware system prompt

**Design principle: domain-agnostic.** No domain words ("REIT", "FFO"), no
hardcoded examples. The prompt reasons about generic *entities*, *attributes*,
*values*, *dates*, and *document types*. The concrete entity/attribute/doc_type
strings arrive only in the SOURCE block at runtime — they are data, not prompt.

```
You are a careful research analyst answering questions strictly from the
numbered source excerpts provided. Apply these rules in this order of
precedence:

[RULE 1 — Grounding]
Every factual, numeric, or quoted claim MUST carry a [N] citation tied to a
source. If a claim cannot be cited from the sources, do not make it.

[RULE 2 — Refusal]
If the sources lack enough information to answer, say:
"I don't have enough information in the provided documents to answer that."
Then briefly state what the sources DO contain so the user can refine.

[RULE 3 — Conflict-pairs]
You may be given a CONFLICT_PAIRS block listing sources that disagree on the
same (entity, attribute). When present:
  - Present EVERY conflicting value with full attribution
  - Format: "{value} per {doc_type} ({as_of_date}) [{n}]" for each
  - Do NOT pick one silently. Do NOT average them.
  - If a footnote/qualifier explains the difference (e.g. a methodology or
    basis change), state that explicitly.

[RULE 4 — Staleness]
If a cited source's as_of_date is much older than the others (more than ~2
years), or substantially before today, flag it:
"As of {year}, {source} reports …" or
"This figure is from {year} and may no longer reflect current conditions."

[RULE 5 — Per-entity completeness]
For "for each X" / multi-entity questions, produce one item per entity. If an
entity has no supporting data in the sources, state explicitly:
"Not disclosed in the provided {entity} materials." Never silently omit an
entity the user asked about.

[RULE 6 — Attribution detail]
Each citation should expose the document type and date so the reader can judge
authority, e.g. "[3] {source_title}, p.{page} ({as_of_date})" — not just "[3]".

[RULE 7 — Delta synthesis]
If asked what changed between two documents/versions, structure the answer as:
(a) stable elements, (b) changed elements with both old and new values,
(c) net-new disclosures.

[RULE 8 — Qualifier preservation]
When a number carries a scope qualifier in the source (e.g. "including X",
"as of date Y", "top-N basis"), keep the qualifier attached to the number.
Two numbers that look equal but have different qualifiers are NOT duplicates.
```

> The rules encode *behaviors* (cite, refuse, surface conflicts, flag staleness,
> preserve qualifiers) that are correct for any document corpus. The only
> corpus-specific content — entity names, attribute names, document types — is
> injected via the SOURCE block (§6.2), never written into the prompt.

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

The minimal UI (§20) gets a **debug panel** that renders this JSON for each query — invaluable for the demo (showing live multi-stage breakdown). The same trace is mirrored to Langfuse (§18) when keys are configured.

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
| **Orchestration** | Plain Python — **planned decomposition** (router → sub-queries → deterministic execution) | Full agentic loop considered (§17). Rejected for v2: latency + reliability risk under deadline. Bounded loop is a documented stretch. LangGraph rejected: harder to debug/explain. |
| **API layer** | **FastAPI** — serves REST API **and** a minimal streaming UI (one process) | **Now built (was design-only).** Replaces Streamlit. Faster evaluator UX (SSE streaming, no rerun), independently hostable. See §20. |
| **UI** | **Minimal HTML + vanilla JS + SSE** served by FastAPI | **Now built.** No Streamlit, no React build step. Streamlit retired (preserved on `v1` branch). See §20. |
| **Cache** | In-process LRU (queries + embeddings) | Redis sketched in design only. Not needed at single-instance demo scale. |
| **Async ingest** | Sync **with per-stage checkpointing** (resume on crash) | **Checkpointing now built (§19).** Celery/Redis distributed workers remain design-only. |
| **Observability** | **Langfuse (optional) + extended `query_log`** + UI debug panel | **Now built (§18).** Langfuse is optional (no keys → still runs). OTel/Phoenix/Prometheus remain design-only. |
| **Reasoning** | Capture gpt-oss `reasoning` channel; surface in debug panel | **Now built.** Already emitted by the model; v1 discarded it. |

---

## 10. Production scaffolding

> **Update (decisions locked 2026-05-19):** FastAPI, the minimal UI, optional
> Langfuse observability, and checkpointed ingestion **moved from design-only to
> BUILD** — see §18–§21. The items below (multi-tenant, Celery distributed
> workers, full OTel/Prometheus/Grafana, cache layer, CI gate) remain
> **design-only** for v2: they're the "scale to production multi-client" answers
> Prashant is probing for, articulated here but not built under the 2-day clock.

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

The authenticated session sets `tenant_id`; every query carries it; Snowflake's
row-access policy enforces isolation server-side. The FastAPI layer (§20, built)
already gives us the request context to attach `tenant_id` — so multi-tenant is a
*data + auth* addition on top of the v2 API, not a re-architecture.

### 10.2 FastAPI split — **built in v2 (§20)**

The FastAPI backend + minimal streaming UI is **built** (§20), not design-only.
What remains design-only here is the *multi-client hardening* on top of it:
- per-tenant auth + API keys, rate-limit quotas per tenant
- `/metrics` Prometheus endpoint, structured access logs
- alternate clients (CLI, Slackbot, MCP server §17) consuming the same API
- OpenAPI contract versioning for downstream consumers

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

- **Multi-turn conversation memory** — separate problem
- **GraphRAG / knowledge graph** — premature at this corpus size
- **Fine-tuning embedder** — marginal lift, large effort
- **Synthetic eval generation** — the 24-Q battery IS the eval target
- **HyDE** — reranker captures most of the lift; HyDE made opt-in only
- **Full agentic loop** — planned decomposition chosen instead (§17); bounded loop is a documented stretch goal
- **MCP server** — retrieval-as-a-tool for external agents; design-only extension point (§17)

> Note: **streaming responses moved INTO scope** — SSE streaming is part of the
> FastAPI minimal UI (§20), since it's the main driver of the faster evaluator UX.

---

## 12. Two-day execution plan

> Effective working time: ~25 hours (Tue eve + Wed full + Thu until evening).
> Scope is aggressive — the **cut order** below is fixed up front so a time
> crunch degrades gracefully instead of leaving the core half-built.

### Cut order (if behind schedule, drop from the TOP of this list)

```
Cut 1st: Streamlit retirement   → if FastAPI UI slips, Streamlit stays as fallback
Cut 2nd: per-stage checkpointing → ingest works without it (just less resilient)
Cut 3rd: Langfuse                → query_log already gives traceability → Langfuse becomes design-only
Cut 4th: FastAPI UI polish       → fall back to functional-but-plain page
NEVER cut: F1–F8 core + domain-agnostic prompts (the assessment criteria)
```

### Wednesday (12 hrs) — Schema + ingestion + retrieval core

| Hours | Block | Output |
|---|---|---|
| 0–1 | Schema migration + checkpoint table | `app/rag_system/storage/schema_v2.sql`, `ingest_checkpoints` table (§19) |
| 1–3 | Doc-level metadata enrichment (**domain-agnostic**) | `metadata_v2.py` — doc_type classifier, as_of_date extractor, doc_family_id |
| 3–5 | Structure-aware chunker | `chunk_v2.py` — footnote attachment, qualifier preservation, table-row decomposition |
| 5–6 | Parent-chunk generator | slide-level parents with image_b64 |
| 6–8 | Vision pass v2 (bbox) + provider failover | `vision_v2.py` — chart_records output |
| 8–9.5 | Proposition extractor | `propositions.py` |
| 9.5–10 | Wire checkpointing into pipeline | resume-on-crash per stage (§19) |
| 10–11 | Full re-ingest (checkpointed) | all 11 PDFs through new pipeline |
| 11–12 | Audit + fix | row counts, sanity-check `chart_records` |

### Thursday (13 hrs) — Retrieval + generation + API/UI + eval

| Hours | Block | Output |
|---|---|---|
| 0–1.5 | Router (**domain-agnostic**, corpus-profile injected) + planned decomposition | `retrieval/router.py` |
| 1.5–3 | Multi-source retrieval | `retrieval/v2_hybrid.py` — dense(props)+lexical(chunks)+structured(tables/charts) |
| 3–3.75 | Cross-encoder reranker | `retrieval/reranker.py` — BGE-reranker-v2-m3 |
| 3.75–5 | Diversification + version-pair + small-to-big | `retrieval/v2_pipeline.py` orchestrator |
| 5–6 | Conflict detection + domain-agnostic prompt + reasoning capture | `generation/prompt_v2.py` |
| 6–6.5 | Provider router (LLM failover chain) | `llm_providers/router.py` |
| 6.5–8.5 | **FastAPI backend + minimal streaming UI** (§20) | `app/api/main.py`, `app/api/templates/`, SSE `/api/query` |
| 8.5–9.25 | **Langfuse wiring (optional)** + UI debug panel (§18) | `@observe` decorators; trace panel |
| 9.25–10.5 | Run full 24-Q battery against v2 | `app/eval/baselines/v2_battery_results.{json,md}` |
| 10.5–11.5 | Manual scoring + failure analysis | `app/eval/baselines/v2_battery_scored.md` (v1→v2 delta) |
| 11.5–13 | README + sample_docs + record demo + submit | `README.md`, `sample_docs/`, 5–7 min demo |

### Thursday evening — Ship

- Include the 10 PDFs in `sample_docs/` so the evaluator can ingest locally
- Merge `v2` → `main`, push
- Deploy FastAPI app (API + UI, one process) to host; set secrets as env vars
- Send revised submission: GitHub link, hosted link, demo video, scored battery,
  read-only Snowflake creds for local validation

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
| Reranker model (568MB) too big for free host RAM | Medium | Host on a tier with ≥1GB RAM (Render/Railway); if not, run reranker locally only + flag in UI. Reranker stage is feature-flagged. |
| FastAPI + minimal UI slips past Thursday | Medium | Cut order: Streamlit (preserved on `v1`) is the fallback UI; ship API-only if needed |
| Time crunch on the full scope | High | Fixed cut order (§12); F1–F8 core is never cut |
| Sharing read-only Snowflake creds | Low | Scoped read-only role; rotate after the process; creds go only in the private email, never the repo |

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
| 2026-05-19 | **Domain-agnostic prompts** (router + generation) | Corpus metadata discovered at runtime; prompts carry no REIT/domain knowledge → works on any corpus (§5.1, §6.1) |
| 2026-05-19 | **Planned decomposition**, not full agentic | Router decomposes → deterministic execution. Reliable + debuggable under deadline. Bounded loop + MCP = design-only (§17) |
| 2026-05-19 | **Capture gpt-oss reasoning channel** | Already emitted; v1 discarded it. Surface in debug panel for transparency |
| 2026-05-19 | **Langfuse (optional) + extended query_log** | Two-layer observability; Langfuse optional so system still runs without keys (§18) |
| 2026-05-19 | **Checkpointed ingestion** (per-stage resume) | Solves expensive-restart pain; Celery distributed remains design-only (§19) |
| 2026-05-19 | **FastAPI + minimal HTML/JS/SSE UI**, retire Streamlit | Faster evaluator UX, independently hostable, one deployable. React build rejected (risk). (§20) |
| 2026-05-19 | **Local-first deploy** + optional hosted | Evaluator runs locally with own creds, or uses hosted demo with my secrets. Read-only creds for local validation against populated data (§21) |
| 2026-05-19 | **Ship the 10 PDFs in `sample_docs/`** | So the evaluator can run ingestion end-to-end locally |

---

## 15. Sign-off checklist

- [x] User reviewed the doc and raised the overfitting concern → resolved via domain-agnostic prompts (§5.1, §6.1)
- [x] Schema additions accepted (doc_type, doc_family_id, as_of_date, new tables) — full re-ingest OK
- [x] Provider routing chain accepted (§7)
- [x] Component picks accepted — **with updates**: FastAPI+UI, Langfuse, checkpointing moved to BUILD (§9, §14)
- [x] Scope confirmed: Core (F1–F8) + FastAPI/UI + Langfuse + checkpointing, by Thursday, with fixed cut order (§12)
- [x] Agentic stance: planned decomposition; bounded loop + MCP design-only (§17)
- [x] Build-vs-design split: multi-tenant, Celery, OTel/Prometheus/Grafana, cache, CI gate remain design-only (§10)
- [x] Deployment: local-first + optional hosted; ship `sample_docs/`; read-only creds for local validation (§21)
- [x] Risks accepted incl. cut order for graceful degradation (§13)

**Status: locked.** No further architecture changes until Thursday submission.

---

## 16. References

- v1 baseline: `app/eval/baselines/v1_battery_scored.md`
- Concepts primer: `docs/v2_concepts_primer.md`
- Eval source: `app/eval/battery_v1.yaml`, sourced from `Vectera_RAG_Self_Evaluation_Battery.docx`
- v1 architecture (for diff): `docs/architecture.md`, `docs/architecture.drawio`

---

## 17. Agentic stance — planned decomposition (build) vs full agent (design-only)

**Decision: planned decomposition for v2. Bounded agentic loop and MCP are design-only.**

### What we build: planned decomposition

The router (§5.1) does the *planning* in one LLM call; the pipeline executes
*deterministically*. This captures most of the agentic benefit (multi-hop,
decomposition, per-entity coverage) without an open-ended control loop.

```
Query ──► Router (1 LLM call)
            └─ emits sub_queries[]  (decomposition / two-hop plan)
         ──► deterministic parallel retrieval over sub_queries
         ──► fuse → rerank → diversify → expand → generate
```

- **Multi-hop (Q17, Q19):** for `enumerate` intent, the router emits a framing
  sub-query AND an entity sub-query derived from the framing chunk's text.
- **Multi-entity (Q22, Q24):** for `compare`, the router emits one sub-query per
  entity → per-entity round-robin merge.
- **Reliable + debuggable + fast:** no loop that can fail to converge mid-demo.

### What we DON'T build (design-only): bounded agentic loop

A true agent gives the LLM `search()` + `finish()` tools and lets it iterate.
Documented as a **stretch**: a hard cap of 3 iterations, gated to fire only on
`enumerate`/`delta` intents, with fallback to planned decomposition on failure.

```python
# DESIGN SKETCH — not built in v2
TOOLS = [search_tool, finish_tool]
def agentic_answer(query, max_hops=3):
    ctx = []
    for hop in range(max_hops):
        action = llm.decide(query, ctx, tools=TOOLS)   # function-calling
        if action.name == "finish":
            return action.answer
        ctx += retrieve(action.args.query, action.args.filters)
    return generate(query, ctx)  # forced finish after cap
```

**Why not now:** 2–5× latency, convergence/looping risk, needs verified
function-calling reliability. The planned-decomposition path covers the battery.

### What we DON'T build (design-only): MCP server

MCP is a protocol for sharing tools *across processes/vendors*. For a single
internal app, plain function-calling is simpler. The clean extension is to
**expose `/retrieve` as an MCP server** so any MCP client (Claude Desktop, other
agents) can query this corpus as a reusable tool. Articulated; not built.

---

## 18. Observability — Langfuse (optional) + extended query_log

**Two layers, both giving full traceability. Langfuse is OPTIONAL — absent keys,
the system still runs and `query_log` still captures everything.**

| Layer | Captures | Where | Required? |
|---|---|---|---|
| **`query_log` (Snowflake)** | router intent, sub-queries, per-stage timings + candidate counts, rerank top IDs, conflict pairs, provider chain, citations, total latency | our DB (§8) | always on |
| **Langfuse** | per-LLM-call traces: prompt, completion, **token counts**, **cost**, latency, the **reasoning trace**, nested spans across router→generation | Langfuse cloud (free tier) or self-host | optional |

```python
# rag_system/observability/trace.py
from functools import wraps

def observe(name):
    """Wrap an LLM call. If LANGFUSE_* keys exist, send a span; always no-op safe."""
    def deco(fn):
        @wraps(fn)
        def wrap(*a, **k):
            if not _langfuse_enabled():        # no keys → pure pass-through
                return fn(*a, **k)
            with _langfuse.span(name=name) as span:
                out = fn(*a, **k)
                span.update(input=..., output=..., usage=out.usage)
                return out
        return wrap
    return deco
```

- **Demo value:** "here's our observability dashboard" answers Prashant's
  *operational thinking* criterion directly — token usage, cost per query,
  latency breakdown, full request traces.
- **Local-hostable:** Langfuse can self-host via Docker for the data-control story;
  for v2 we default to the free cloud tier and keep it optional.
- **Design-only extensions:** OpenTelemetry → Tempo/Jaeger, Prometheus metrics,
  Grafana dashboards (§10.4).

---

## 19. Checkpointed ingestion (resume-on-crash)

**Solves the real pain: a crash at page 30 should not redo from page 1.**
Each document moves through ordered stages; each completed stage writes a
checkpoint. Re-running resumes from the last good stage.

```sql
CREATE TABLE ingest_checkpoints (
  doc_id      VARCHAR,
  checksum    VARCHAR,         -- file sha256 — idempotency key
  stage       VARCHAR,         -- parse|chunk|parent|vision|propositions|embed|upsert
  status      VARCHAR,         -- pending|done|failed
  payload_ref VARCHAR,         -- where the stage's intermediate output is cached
  updated_at  TIMESTAMP,
  PRIMARY KEY (doc_id, stage)
);
```

```
Ingest(doc):
  for stage in [parse, chunk, parent, vision, propositions, embed, upsert]:
      if checkpoint(doc, stage) == done:   # idempotent: skip completed work
          continue
      run(stage); write_checkpoint(doc, stage, done)
```

- **Stage isolation:** vision failing (rate limit / bad JSON) doesn't lose parse,
  chunk, or parent work — they're already checkpointed.
- **Idempotent on checksum:** re-ingesting an unchanged file is a no-op.
- **Resumable re-runs:** the expensive stages (vision, propositions, embed) are
  never repeated unnecessarily.
- **Design-only extension:** promote stages to a Celery/Redis worker pool with
  progress events for distributed, multi-tenant ingest (§10.3).

---

## 20. FastAPI backend + minimal streaming UI (replaces Streamlit)

**One process serves the REST API and a minimal chat UI. No Streamlit, no React
build step. Faster evaluator UX via SSE token streaming.**

### Layered architecture (the API analog of MVC)

```
app/api/
  main.py            # FastAPI app, route wiring, static + template mounts
  routes/
    query.py         # POST /api/query   (SSE stream), GET /api/query/{id}/trace
    ingest.py        # POST /api/ingest  -> job_id, GET /api/ingest/{id} (progress)
    documents.py     # GET  /api/documents, DELETE /api/documents/{id}
    health.py        # GET  /health, GET /api/corpus-profile
  services/          # business logic (calls retrieval + generation + ingest)
  templates/
    index.html       # single chat page (server-rendered shell)
  static/
    app.js           # vanilla JS: EventSource(SSE), source cards, debug panel
    styles.css
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Minimal chat UI (HTML shell) |
| `/api/query` | POST | **SSE stream** — tokens stream as generated; final event carries citations + trace |
| `/api/query/{id}/trace` | GET | Full retrieval trace JSON for the debug panel |
| `/api/ingest` | POST | Trigger ingest → returns `job_id` (async, checkpointed §19) |
| `/api/ingest/{id}` | GET | Poll ingest progress |
| `/api/documents` | GET/DELETE | List / delete corpus docs |
| `/api/corpus-profile` | GET | The runtime corpus profile fed to the router (§5.1) |
| `/health` | GET | Liveness + dependency checks (Snowflake, model load) |

- **Controller → Service → Repository** separation: routes are thin; services hold
  logic; the existing `repository.py` is the data layer. Clean, testable, the
  layered analog of MVC the user asked about.
- **Streaming:** `/api/query` returns `text/event-stream`; the browser uses
  `EventSource`. Answer appears token-by-token — no Streamlit rerun lag.
- **Independently hostable:** the API can run headless (UI optional). Enables
  future clients (CLI, Slackbot, MCP server §17).
- **OpenAPI:** FastAPI auto-generates `/docs` — free API documentation for the
  evaluator.

---

## 21. Deployment — local-first + optional hosted

### Local (primary — what the evaluator does)

```
git clone <repo>
cp .env.example .env          # fill: SNOWFLAKE_*, CEREBRAS_API_KEY, GEMINI_API_KEY,
                              #       (optional) LANGFUSE_*
./setup.sh   |  .\setup.ps1   # venv + deps + model downloads + snowflake check
uvicorn app.api.main:app --reload      # OR: docker compose up
# open http://localhost:8000
```

- The evaluator supplies **their own** Snowflake + API keys, ingests the bundled
  `sample_docs/` (the 10 PDFs), and queries — full end-to-end locally.
- For zero-ingest validation, the submission email includes **read-only Snowflake
  creds** pointing at the already-populated instance: paste into `.env`, run, query.

### Hosted (secondary — the demo link)

- One FastAPI process (API + UI) deployed to Render / Railway / Fly.io free tier.
- **Secrets as platform env vars** — my Snowflake creds + my API keys. The
  evaluator just opens the URL; the system uses my configured secrets.
- Replaces the previous Streamlit Cloud deployment.

### Secrets matrix

| Secret | Local (evaluator) | Hosted (me) |
|---|---|---|
| `SNOWFLAKE_*` | their account, or my read-only creds | my account |
| `CEREBRAS_API_KEY` | their key | my key |
| `GEMINI_API_KEY` | their key (vision/optional) | my key |
| `LANGFUSE_*` | optional | optional |

- Secrets live only in `.env` (gitignored) or platform env vars — **never in the repo**.
- `.env.example` documents every variable with comments.
- Storage abstraction (Snowflake ↔ local DuckDB/FAISS) is **design-only future
  work** — noted so the system *could* run fully offline without cloud creds.

---

## Sign-off status

**APPROVED & LOCKED — 2026-05-19.** All §15 checklist items confirmed via the
scope/agentic/observability/ingestion/UI/deployment decisions recorded in §14.
Domain-agnostic prompts (§5.1, §6.1) resolve the overfitting concern. No further
architecture changes until Thursday submission; deviations get logged in §14.
