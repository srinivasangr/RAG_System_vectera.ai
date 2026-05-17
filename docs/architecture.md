# System Architecture

> Source-of-truth diagram: [`architecture.drawio`](architecture.drawio) — open in
> [https://app.diagrams.net](https://app.diagrams.net) (File → Open) or in the
> VS Code **Draw.io Integration** extension.
> The Mermaid version below renders inline on GitHub.

---

## Diagram (Mermaid)

```mermaid
flowchart TB
    %% ============== USER ==============
    user["👤 Analyst (Browser)<br/>natural-language questions"]:::user

    %% ============== UI ==============
    subgraph UI["Streamlit UI — http://localhost:8501"]
        direction LR
        ui_sb["Sidebar<br/>• company / date filters<br/>• provider + model<br/>• top-K · recency boost"]:::ui
        ui_q["Query box<br/>[Ask]"]:::ui
        ui_a["Answer panel<br/>• markdown + [N] cites<br/>• image + text per citation<br/>• latency metrics"]:::ui
        ui_d["Debug panel<br/>retrieved chunks<br/>+ dense/keyword ranks<br/>+ RRF score"]:::ui
    end

    %% ============== QUERY SERVICE ==============
    subgraph QSVC["Query Service — rag_system.generation.query()"]
        direction LR
        s1["1. Embed query<br/>Gemini embedding-001<br/>→ 768d vector"]:::svc
        s2["2. Hybrid Retrieve<br/>• Dense cosine top-30<br/>• Lexical LIKE top-30<br/>• RRF fusion → top-K<br/>• Filters · recency boost"]:::svc
        s3["3. Format prompt<br/>• System: cite, surface<br/>  conflicts, refuse cleanly<br/>• User: numbered sources"]:::svc
        s4["4. LLM + parse [N]<br/>Default Cerebras<br/>gpt-oss-120b"]:::svc
        s1 --> s2 --> s3 --> s4
    end

    %% ============== INGEST ==============
    subgraph ING["Ingestion Pipeline — offline / batch"]
        direction LR
        i1["Docling parse<br/>page batches of 5<br/>Markdown + images"]:::ing
        i2["Gemini 2.5 Flash<br/>vision describes<br/>each chart"]:::ing
        i3["Chunk<br/>page-aware<br/>~800 tok / 100 ovl<br/>tag: prose / table /<br/>chart_description"]:::ing
        i4["Embed (Gemini 768d)<br/>+ Upsert<br/>idempotent by checksum"]:::ing
        i1 --> i2 --> i3 --> i4
    end

    %% ============== PROVIDERS ==============
    subgraph PROV["LLM Provider Abstraction"]
        direction TB
        base["Abstract bases:<br/>BaseLLMProvider<br/>BaseEmbeddingProvider<br/>BaseVisionProvider<br/><br/>Factory: get_llm() /<br/>get_embedder() / get_vision()"]:::prov
        p_oa["OpenAI<br/>gpt-4o-mini"]:::prov
        p_an["Anthropic<br/>claude-sonnet-4-6"]:::prov
        p_ge["⭐ Gemini<br/>LLM + embed + vision"]:::provActive
        p_ce["⭐ Cerebras<br/>gpt-oss-120b"]:::provActive
        p_or["OpenRouter<br/>any open model"]:::prov
    end

    %% ============== SNOWFLAKE ==============
    subgraph SF["Snowflake — RAG_DB.RAG_SCHEMA"]
        direction LR
        t_docs[("documents<br/>doc_id, company,<br/>doc_date, version_label,<br/>checksum")]:::sf
        t_ch[("chunks<br/>chunk_id, page_number,<br/>chunk_type, text,<br/><b>embedding VECTOR(FLOAT,768)</b><br/>company, doc_date")]:::sf
        t_im[("chunk_images<br/>chunk_id PK,<br/>width, height,<br/>image_b64 (PNG)")]:::sf
        t_log[("query_log<br/>question, cites,<br/>answer, llm,<br/>latency_ms")]:::sf
    end

    %% ============== SOURCE ==============
    pdfs[/"📑 Source PDFs<br/>Documents/ — 10 REIT decks<br/>BXP · DLR · EGP · PSA · O · SPG · VICI"/]:::pdf

    %% ============== FLOWS — QUERY (solid) ==============
    user --> UI
    UI --> QSVC
    QSVC -- "Answer + Citations" --> UI
    s2 <-->|cosine + LIKE| t_ch
    s4 -.->|log| t_log
    s1 -.uses.-> p_ge
    s4 ==>|default| p_ce
    t_im -- "PNG bytes for<br/>chart citations" --> ui_a

    %% ============== FLOWS — INGEST (dashed) ==============
    pdfs -. read .-> i1
    i2 -. uses .-> p_ge
    i4 -. uses .-> p_ge
    i4 -. "upsert<br/>(idempotent)" .-> t_ch
    i4 -. write .-> t_docs
    i2 -. PNG .-> t_im

    %% ============== STYLES ==============
    classDef user fill:#dae8fc,stroke:#6c8ebf,color:#000
    classDef ui fill:#fff2cc,stroke:#d6b656,color:#000
    classDef svc fill:#d5e8d4,stroke:#82b366,color:#000
    classDef ing fill:#fad7ac,stroke:#b46504,color:#000
    classDef prov fill:#e1d5e7,stroke:#9673a6,color:#000
    classDef provActive fill:#d5e8d4,stroke:#82b366,color:#000,stroke-width:3px
    classDef sf fill:#cce5ff,stroke:#1c6ea4,color:#000
    classDef pdf fill:#f8cecc,stroke:#b85450,color:#000
```

---

## Two distinct planes

### 1. Query plane (online, stateless, runs on every user click)

```
Browser  →  Streamlit  →  query()  ──┬─→ Gemini (embed query, 768d)
                                     │
                                     ├─→ Snowflake (dense cosine + LIKE)
                                     │      ↓
                                     │   top-K chunks (RRF-fused)
                                     │
                                     └─→ Cerebras gpt-oss-120b
                                            ↓
                                       answer + [N] citations
                                            ↓
                                       chunk_images.image_b64 for chart cites
                                            ↓
                                       rendered in UI
```

**Latency budget** (observed on Digital Realty Dec 2025, top-k 8):

| Stage | p50 |
|---|---|
| Query embedding (Gemini API) | 300–500 ms |
| Hybrid retrieval (Snowflake) | 2 000–3 500 ms |
| Prompt format + token count | < 5 ms |
| LLM generation (Cerebras gpt-oss-120b) | 500–1 500 ms |
| **End-to-end** | **~3–6 s** |

### 2. Ingest plane (offline, batch, one-shot or on file change)

```
Documents/*.pdf  ⇢  Docling parse (page batches of 5)
                  ⇢  per page: Gemini Vision describes each chart image
                                (skips logos, tiny, extreme-aspect)
                  ⇢  chunk_page() — tag prose / table / chart_description
                  ⇢  Gemini embeddings (768d, batched)
                  ⇢  Snowflake upsert
                        - documents (one row per PDF)
                        - chunks (text + embedding + metadata)
                        - chunk_images (PNG bytes for chart_description chunks)
```

**Idempotency:** every PDF is hashed (sha256). Re-running the pipeline on an
unchanged file is a no-op; on a changed file it replaces the document's rows
inside a single transaction.

---

## Key components, in code

| Concern | Module | What it does |
|---|---|---|
| Settings | `rag_system/config/settings.py` | Typed Pydantic settings loaded from `.env` |
| PDF parsing | `rag_system/ingest/parse.py` | Docling, batched by page, returns per-page Markdown + image bytes |
| Filename → metadata | `rag_system/ingest/metadata.py` | Regex-driven company/date/version extraction |
| Chart description | `rag_system/ingest/vision_extract.py` | Gemini 2.5 Flash; aggressive pre-filter on size/aspect |
| Chunking | `rag_system/ingest/chunk.py` | Page-aware, table-isolating, tagged by content type |
| Pipeline | `rag_system/ingest/pipeline.py` | CLI orchestrator with `--limit / --doc / --no-vision / --dry-run` |
| Snowflake schema | `rag_system/storage/schema.sql` | `documents`, `chunks` (with `VECTOR(FLOAT,768)`), `chunk_images`, `query_log` |
| DAO | `rag_system/storage/repository.py` | Idempotent upserts, image fetch, query log |
| Hybrid retrieval | `rag_system/retrieval/hybrid.py` | Dense (cosine) ∪ Lexical (LIKE) → RRF → recency boost |
| Generation | `rag_system/generation/prompt.py` + `service.py` | Strict citation prompt, refusal path, citation parsing |
| LLM providers | `rag_system/llm_providers/` | Base interfaces + 5 swappable adapters |
| UI | `rag_system/ui/streamlit_app.py` | Question / answer / cite expanders / debug / image render |
| Eval | `eval/run_eval.py` + `eval/questions.yaml` | Recall@k, MRR, must-contain, refusal correctness |

---

## Hard-problem handling, at a glance

| Problem | Where it's solved |
|---|---|
| **Multiple versions of same company** | Filename → `doc_date` + `version_label`; chunks carry both; recency boost in `_apply_recency_boost`; prompt told to attribute by version |
| **Conflicting facts across docs** | Retriever returns top-K from *all* docs; prompt forbids averaging/silent picking; model surfaces disagreement with attribution |
| **Charts / figures** | Gemini Vision pass → structured Markdown description embedded as `chart_description` chunks; original PNG stored in `chunk_images` and rendered in UI |
| **Tables** | Docling extracts structured tables → Markdown; chunker isolates them as `table` chunks |
| **Citations not invented** | LLM prompt enforces `[N]`-only; `resolve_citations` regex-validates each `[N]` maps to an actual source |
| **Insufficient evidence** | Explicit refusal sentence in system prompt; eval validates refusal on out-of-corpus questions |
| **Cortex not on trial** | Embedder + LLM are external providers behind a swappable interface; Snowflake used purely as DB + vector search |
