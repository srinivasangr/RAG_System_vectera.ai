# Architecture

A retrieval-augmented question-answering system over complex business documents
(slide decks, reports, filings) that contain dense text, tables, charts, and
maps. The system is domain-agnostic: it carries no hardcoded entities, metrics,
or document types — everything is derived by reading the documents.

---

## 1. Overview

The system has two halves:

- **Offline (ingestion):** documents are parsed, identified, chunked, enriched
  with vision-extracted structured data, embedded, and indexed.
- **Online (query):** a query is analyzed, routed to multiple retrieval sources,
  fused, reranked, diversified, expanded, and answered with cited, grounded
  generation.

```
                      ┌─────────────────────── OFFLINE ───────────────────────┐
  PDF ─▶ parse ─▶ identify ─▶ chunk ─▶ page-vision ─▶ propositions ─▶ embed ─▶ store
                  (text)     (LLM)    (structure)   (LLM, images)    (LLM)    (local)
                      └────────────────────────────────────────────────────────┘
                                              │  (Snowflake: documents, chunks,
                                              ▼   propositions, table_rows,
                      ┌─────────────────────── ONLINE ──── chart_records, page_images)
  query ─▶ route ─▶ multi-source retrieve ─▶ RRF ─▶ rerank ─▶ diversify
           (LLM)    (dense + lexical +              (cross-     (MMR / per-doc)
                     structured)                     encoder)        │
                                                                     ▼
            answer ◀─ generate ◀─ conflict-detect ◀─ small-to-big ◀─ version-pair
            (cited)   (LLM)        (entity/date)      (chunk→slide)   (sibling docs)
```

---

## 2. Offline ingestion

| Stage | Engine | Purpose |
|---|---|---|
| **Parse** | Docling (local) | Layout-aware text + table extraction + per-page reading order; detects which pages are visual |
| **Identify** | LLM (1 call/doc) | Reads the cover to infer issuer, document type, as-of date, and a document-family id — no hardcoded lists |
| **Chunk** | local | Slide-level parent chunks + child chunks; footnotes attached to their body text; scope qualifiers preserved |
| **Page-vision** | LLM (image, per visual page) | Classifies each page element (table / chart / figure / map) and extracts structured data + a description |
| **Propositions** | LLM (per prose chunk) | Decomposes prose into atomic, self-contained facts — the high-precision dense-retrieval target |
| **Embed** | BGE-base-en-v1.5 (local) | 768-dim vectors for chunks, propositions, and table rows |
| **Store** | Snowflake | Upsert all artifacts; per-stage checkpoints enable resume-on-failure |

Key properties:

- **Document identification is content-driven.** Issuer, type, and dates come
  from the document text (cover + footnotes), not the filename, so version and
  staleness can be reasoned about later.
- **Page-level vision** sends a whole rendered page to the vision model rather
  than pre-cropped images, so legends, multi-panel charts, maps, and footnotes
  are interpreted in their spatial context. Each extracted element carries a
  confidence score; low-confidence extractions are flagged rather than asserted.
- **Idempotent + resumable.** A file is keyed by content checksum; an unchanged
  file is skipped. Each stage checkpoints, so a crash resumes at the next
  unfinished stage instead of redoing the document.

---

## 3. Online retrieval

A multi-stage pipeline, each stage addressing a specific failure mode of naive
single-shot retrieval.

1. **Route** — one LLM call analyzes the query against the live corpus profile
   (the distinct document types / entities / date range read from the database).
   It returns intent (lookup / compare / delta / recency / enumerate / refuse),
   the entities and attributes mentioned, decomposed sub-queries, and temporal
   hints. The corpus profile is the only source of domain knowledge, so the same
   prompt works on any corpus.
2. **Multi-source retrieve** — for each sub-query, three sources run, all keyed
   back to a chunk:
   - **dense** — cosine similarity over propositions *and* over all chunk
     embeddings (so chart/table/figure chunks are semantically findable, not
     only prose);
   - **lexical** — keyword match over chunk text (catches tickers, acronyms,
     exact numbers);
   - **structured** — keyword match over table rows and chart records.
3. **Fuse** — Reciprocal Rank Fusion combines the source rankings (rank-based,
   parameter-free, robust to score-scale differences).
4. **Rerank** — a cross-encoder scores each (query, chunk) pair together for a
   precise relevance ordering.
5. **Diversify** — a per-document quota (and query decomposition for comparison
   queries) prevents one document from dominating the top-k.
6. **Version-pair expansion** — when a result's document has sibling versions in
   the same family, the matching slide from each sibling is surfaced so multiple
   versions can be compared rather than silently picking one.
7. **Small-to-big** — each matched chunk is expanded to its parent slide so the
   model sees full context (surrounding bullets, footnotes, qualifiers).
8. **Conflict detection** — when the same entity appears with multiple as-of
   dates in the result set, the sources are tagged as a conflict group so the
   answer presents each value with attribution instead of blending them.

---

## 4. Generation

The retrieved parent slides are formatted into a numbered source block (each
labelled with document type and as-of date, conflict groups noted) and passed to
the generation model with a grounding-first prompt. Behaviours enforced:

- every factual/numeric claim must carry a citation, or it is not stated;
- when sources conflict, present every value with its date and document type;
- flag sources materially older than the others (staleness);
- for "for each X" questions, address every entity, stating "not disclosed" where
  a source is absent rather than silently omitting it;
- refuse honestly when the sources do not support an answer.

Off-topic or trivial input is short-circuited before retrieval: greetings are
answered directly, and queries the router classifies as out-of-scope are declined
after a single classification call (no retrieval or generation).

A provider fallback chain retries generation on a secondary model if the primary
returns a transient error, so a single query does not fail on a provider hiccup.

---

## 5. Storage schema (Snowflake)

| Table | Holds | Embedded |
|---|---|---|
| `documents` | one row per file: identity (issuer, type, as-of date, family), file metadata | — |
| `document_files` | the original PDF bytes (provenance / re-processing) | — |
| `parent_chunks` | full slide/page text (small-to-big context target) | — |
| `page_images` | a compact image of each rendered page (citation thumbnails) | — |
| `chunks` | child chunks: prose / table / chart / figure, each with type and confidence | yes |
| `propositions` | atomic facts decomposed from prose | yes |
| `table_rows` | one row per table row, column labels preserved | yes |
| `chart_records` | vision-extracted (label, value, unit, confidence) from figures | — |
| `ingest_checkpoints` | per-stage ingestion progress (resume-on-failure) | — |
| `query_log` | per-query trace: intent, sub-queries, stage timings, conflicts, engine, latency | — |

Relationships (logical join keys; Snowflake does not enforce foreign keys):

```
documents(doc_id)
  ├─ document_files(doc_id)
  ├─ parent_chunks(parent_id) ─ page_images(parent_id)
  │     └─ chunks(chunk_id) ─┬─ propositions(chunk_id)
  │                          ├─ table_rows(chunk_id)
  │                          └─ chart_records(chunk_id)
```

---

## 6. Components and rationale

| Layer | Choice | Why |
|---|---|---|
| Parser | Docling | Layout-aware text + native table structure; local, free |
| Vision | Gemini (page-level, structured output) | Interprets charts/maps/logos in spatial context; structured + confidence |
| Embeddings | BAAI/bge-base-en-v1.5 (local) | No rate limits or per-call cost; strong on business text |
| Reranker | Cross-encoder (MiniLM) | Largest precision lift; runs on CPU at interactive latency |
| Vector store | Snowflake `VECTOR` + `VECTOR_COSINE_SIMILARITY` | Native vector search alongside relational metadata; one store |
| Generation / routing | Gemini (configurable) | Fast, structured output; provider-agnostic interface allows swapping |
| API + UI | FastAPI serving a minimal streaming UI | Single deployable; observable; no separate frontend build |

The LLM, embedding, and vision providers sit behind a common interface, so any
of them can be swapped without touching the pipeline.

---

## 7. Observability

Every query is logged to `query_log` with its full trace: router intent and
sub-queries, per-stage timings and candidate counts, the engine chain with
per-call token usage, conflict groups, and the final citations. The UI exposes
this per query (a trace panel) and as a query-history view. Token usage and the
exact prompt sent to the model are captured for inspection.

---

## 8. Deployment

The application runs as a single FastAPI process serving both the REST API and
the UI. It is local-first: clone, configure credentials in `.env`, run, and open
the local URL. It can also be deployed to any container host; the storage layer
(Snowflake) and model providers are reached over the network via configured
credentials.

---

## 9. Known limitations and next steps

- **Generation context assembly.** When a relevant slide is retrieved but the
  needed value lives in a specific table cell or footnote, the model is given
  the parent slide text rather than the matched `table_rows` / `chart_records` /
  footnote directly. Injecting the matched structured detail into the prompt is
  the highest-leverage next improvement.
- **Map / spatial figures.** Data encoded purely as positions on a map is only
  partially recovered; multi-hop cross-page synthesis would improve these.
- **Scale-out (documented, not built):** asynchronous distributed ingestion,
  multi-tenant row isolation, a managed tracing backend, and a query/result
  cache are the production-scaling path.
