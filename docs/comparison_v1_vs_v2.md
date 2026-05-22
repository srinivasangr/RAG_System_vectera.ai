# v1 → v2: what changed

v1 was a working single-pass RAG demo. v2 is a multi-stage, document-aware
system designed around the failure modes that distinguish naive RAG from a
robust one. The two versions live on separate branches (`v1`, `v2`).

---

## Architecture changes

| Area | v1 | v2 |
|---|---|---|
| **Document identification** | Hardcoded issuer alias table + keyword document-type map | LLM reads the cover; issuer, type, as-of date, and document family inferred from content — no hardcoded lists |
| **Chunking** | Page-aware splitter; prose / table / free-text "chart description" | Slide-level **parent** chunks + child chunks; footnotes attached to body text; scope qualifiers preserved |
| **Tables** | Serialized to text inside a chunk | **Reconstructed into structured rows** with column labels preserved (`table_rows`) |
| **Charts / figures** | Optional vision producing a free-text caption | **Page-level vision** classifies each element and extracts structured `(label, value, unit, confidence)` records; figures get a description; page images stored |
| **Index** | Single `chunks` table (dense + lexical) | **Multi-vector**: `chunks`, `propositions` (atomic facts), `table_rows`, `chart_records`, plus `parent_chunks`, `page_images`, `document_files` |
| **Retrieval** | Dense ∪ lexical → RRF → recency boost | **Router → multi-source (dense over props + all chunks, lexical, structured) → RRF → cross-encoder rerank → diversify → version-pair expansion → small-to-big → conflict detection** |
| **Generation** | Strict citation prompt + refusal | Conflict-aware (present all values with dates), staleness flagging, per-entity completeness, qualifier preservation, off-topic guardrails, provider fallback chain |
| **Interface** | Streamlit app | FastAPI serving a minimal streaming UI: live stage status, document filter, citation page images, history, per-query trace |
| **Ingestion robustness** | Single run | Content-checksum dedup + per-stage checkpoints (resume-on-failure) |
| **Observability** | Basic query log | Full per-query trace (intent, sub-queries, stage timings, token usage, engine chain, conflicts) surfaced in the UI |
| **Evaluation** | Hand-written questions + RAGAS-style metrics | 24-question diagnostic battery + independent LLM-judge + RAGAS metrics + evidence-based failure analysis |

---

## Functionality comparison

| Capability | v1 | v2 |
|---|---|---|
| Find a single fact and cite it | ✓ | ✓ |
| Keyword / acronym / exact-number match | ✓ | ✓ |
| Surface **both versions** of a recurring document | ✗ | ✓ (document families + version-pair expansion) |
| Detect and present **conflicting values** with attribution | ✗ | ✓ (conflict detection + conflict-aware prompt) |
| **Doc-type / scope** awareness (standalone vs combined, NOI vs ARO) | ✗ | ✓ (router + metadata) |
| Keep **footnote / scope qualifiers** with their numbers | partial | ✓ (footnote-attached chunking) |
| **Table cell → entity** preservation | ✗ | ✓ (`table_rows` with column labels) |
| **Chart / figure** data as structured records | ✗ (free-text only) | ✓ (`chart_records` with confidence) |
| **Cross-document diversification** (no single doc dominates) | ✗ | ✓ (per-doc quota + query decomposition) |
| **Flag stale sources** | ✗ | ✓ |
| Semantic search over **charts/tables**, not just prose | ✗ | ✓ (dense over all chunk types) |
| Reranking for precision | ✗ | ✓ (cross-encoder) |
| Off-topic / greeting short-circuit | ✗ | ✓ |
| Live query trace + token/cost observability | ✗ | ✓ |

---

## Measured improvement (RAGAS-style, same judge)

| Metric | v1 | v2 |
|---|---|---|
| Context precision (retrieved chunks that are relevant) | 0.09 | **0.49** |
| Faithfulness (answer grounded in sources) | 0.70 | **0.95** |
| Answer relevance | 0.96 | 0.95 |

On the 24-question diagnostic battery, the largest remaining gap is **generation
context assembly** — when a relevant slide is retrieved but the needed value sits
in a specific table cell or footnote, the model currently receives the parent
slide text rather than the matched structured row. This is the documented next
improvement (see `architecture.md` §9). Retrieval quality itself improved
substantially (context precision 5×; near-zero hallucination at 0.95
faithfulness).
