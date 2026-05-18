# Design notes & interview talking points

Concise reference for design decisions, the tradeoffs they involve, and the
vocabulary to use when describing them.

---

## 1. The system in one paragraph

A retrieval-augmented generation pipeline over a small corpus of PDF
investor decks. PDFs are parsed with a layout-aware extractor, chunked
with page boundaries preserved, embedded with a local sentence-transformer,
and stored in Snowflake using its native `VECTOR(FLOAT, 768)` column.
Queries run a hybrid retrieval (dense cosine + lexical `LIKE`, fused with
Reciprocal Rank Fusion), then a strict-citation prompt sends the top-K
chunks to a hosted LLM (Cerebras `gpt-oss-120b`). The UI is Streamlit;
the same backend powers the eval harness and a future REST API.

---

## 2. The big decisions, and the alternatives I rejected

| Decision | Picked | Considered (and rejected) | Reason |
|---|---|---|---|
| Storage + vector search | **Snowflake `VECTOR(FLOAT, 768)`** | Pinecone, Qdrant, Chroma, Weaviate | The brief required Snowflake. Beyond compliance, keeping metadata + vectors in one DB means filters push down to SQL and there's no second piece of infra to operate. |
| PDF parser | **Docling** (IBM, OSS) | `pypdf` + `Camelot`, Unstructured.io, LlamaParse, Mistral OCR | Layout-aware, handles slide-heavy decks, outputs Markdown including tables, runs offline. No API rate limits, no per-page cost. |
| Embedding model | **`BAAI/bge-base-en-v1.5`** (local, 768d) | Gemini `gemini-embedding-001`, OpenAI `text-embedding-3-small`, Snowflake Cortex `EMBED_TEXT_768` | Cortex is blocked on Snowflake trial; Gemini's 100 RPM free-tier limit was a constant source of failed ingests. Local removes the rate-limit variable, costs nothing, and 768d matches the schema. |
| LLM (answer gen) | **Cerebras `gpt-oss-120b`** | OpenAI `gpt-4o-mini`, Anthropic `claude-sonnet-4`, Gemini `gemini-2.5-flash` | Frontier-class OSS model served at very low latency (~500 ms for short answers), generous free tier. The LLM is a swappable layer — any of the others is one dropdown change in the UI. |
| Retrieval | **Hybrid (dense ∪ lexical, RRF-fused)** | Dense-only (vector), Lexical-only (BM25/`LIKE`), Reranker as the only step | Investor decks are dense with tickers and metric acronyms (FFO, NOI, AFFO, ARR). Pure dense misses exact-string matches; pure lexical misses paraphrases. RRF combines both without parameter tuning. |
| Chunking | **Page-aware, ~800 tokens / 100 overlap, table-isolating** | Fixed-size sliding window across pages, semantic chunking | Citations need precise page numbers. Tables get their own `chunk_type='table'` chunk so the structured rows stay intact rather than being chopped mid-row. |
| Charts/figures | **Optional Gemini-vision-as-extractor → text** | True multimodal embeddings (ColPali, Nomic, Voyage Multimodal-3) | Multimodal embeddings need a GPU and a different vector space. The vision-LLM approach reuses the same text embedder + citation infra and works on free-tier APIs (it's just rate-limited at 5 RPM). |
| UI | **Streamlit (synchronous upload)** | FastAPI + React, Flask + HTMX, Streamlit + background worker | The brief recommends Streamlit. Synchronous upload (blocks the UI for the parse-embed-store cycle) is rock-solid on Windows; the background-worker version had Docling-on-thread init races. |

---

## 3. Tradeoffs I knowingly accepted

1. **UI blocks during upload.** Each upload runs `ingest_one()` synchronously
   in the main Streamlit thread (3–5 min for a typical deck). Reliability
   over responsiveness — the background-worker version had race conditions.
2. **No cross-encoder reranker.** Top-K is returned straight from RRF. The
   eval shows context precision at ~9% (the LLM judges 1 of 8 retrieved
   chunks as truly relevant). A reranker between candidate-30 and final
   top-8 is the obvious next lever.
3. **Single-turn LLM context.** The chat UI shows multi-message threads,
   but each query is sent to the LLM in isolation — no prior turns passed.
   Follow-ups like "tell me more about that" lose context.
4. **No vector ANN index.** We do exact cosine over the chunks table
   (~700 rows). Fine at this scale; would need an HNSW-equivalent at
   millions of chunks.
5. **Chat persistence is in-memory only.** `chats` and `messages` live in
   `st.session_state`. Refresh = lost. Documents and `query_log` are
   persistent in Snowflake.
6. **Vision is off by default.** Gemini Vision's 5-RPM / 250-RPD free-tier
   limit makes it unreliable for bulk chart description. Toggle on in
   the UI when there's quota.

---

## 4. Vocabulary cheat sheet for the conversation

| Term | One-line definition |
|---|---|
| **Embedding** | Dense float vector representation of text in a model's semantic space (e.g. 768 dims for BGE-base) |
| **Cosine similarity** | Angle-based vector similarity; scale-invariant; the standard for normalized text embeddings |
| **Dense retrieval** | Find chunks whose vectors are closest to the query vector |
| **Lexical retrieval** | Find chunks containing the query's literal tokens (BM25, TF-IDF, or in our case `LIKE`) |
| **Hybrid retrieval** | Run both, fuse the results |
| **RRF (Reciprocal Rank Fusion)** | Parameter-free fusion: combined score = Σ 1/(k + rank_i); robust to different score scales |
| **Top-K** | The number of chunks the retriever returns to the LLM (we use K=8) |
| **Candidate-K** | The larger set retrieved per method before fusion (we use 30) |
| **ANN index** | Approximate Nearest Neighbor index (HNSW, IVF-PQ); we don't need one at 700 chunks |
| **Bi-encoder vs cross-encoder** | Bi-encoders embed query and doc independently (fast, used at retrieval); cross-encoders score (query, doc) together (slow, used at rerank) |
| **Recall@k** | Did at least one relevant chunk appear in top-k? |
| **MRR** | Mean Reciprocal Rank: 1/rank of the first relevant chunk, averaged |
| **nDCG** | Normalized Discounted Cumulative Gain; ranks-aware variant; we don't use it for n=14 |
| **Faithfulness** | Are the answer's atomic claims grounded in the retrieved chunks? (RAGAS metric) |
| **Answer relevance** | Does the answer address the question? (RAGAS) |
| **Context precision / recall** | Of the retrieved chunks, what fraction is relevant? / Of the gold facts, what fraction is in the retrieved chunks? |
| **LLM-as-judge** | Using an LLM to score quality metrics rather than human annotators |
| **Grounding** | The constraint that the answer must be supported by the supplied context |
| **Hallucination** | LLM output that isn't supported by the context; the failure mode RAG is designed to prevent |
| **Chunking** | Splitting documents into retrievable units; tradeoff between cohesion (large chunks) and precision (small chunks) |
| **Page-aware chunking** | Chunks never cross page boundaries; preserves citation precision |
| **Reranker** | Second-stage scorer that re-orders the candidate set using a cross-encoder for higher precision |
| **HyDE (Hypothetical Document Embeddings)** | Generate a fake answer first, embed *that*, retrieve against it; helps with abstract queries |
| **Multi-turn conversation memory** | Including prior turns in the LLM prompt so follow-ups have context |
| **Idempotency (in ingest)** | Re-running on the same file does nothing; we key by sha256 checksum |

---

## 5. How the brief's "key capabilities" map to specific code

| Capability | Mechanism |
|---|---|
| **Version awareness** | Filenames parsed to `(company, doc_date, version_label)` and denormalized onto every chunk. Optional **recency boost** auto-activates when the query contains `current`/`latest`/`recent`/`now`/`most recent`. The prompt also instructs the LLM to attribute by version. |
| **Cross-document conflicts** | Top-K is pulled from all docs without per-doc quotas. Prompt rule #3: *"If the SOURCES disagree, surface the disagreement explicitly with attribution. Do not average, blend, or silently pick one side."* |
| **Tables** | Docling identifies table regions; the chunker pulls each table out as its own `chunk_type='table'` chunk so rows stay intact. |
| **Charts/figures** | Optional Gemini-vision pass turns each chart image into a structured Markdown description (figure type, axes, key data points, takeaway) and stores both the description and the source PNG (in `chunk_images`). Hidden behind a UI toggle because of the free-tier daily quota. |
| **Cite everything; refuse cleanly when unknown** | System prompt enforces `[N]` markers on every factual claim; `resolve_citations()` regex-parses markers and drops any that point outside the supplied source list. Out-of-corpus questions trigger the verbatim refusal sentence. |

---

## 6. The eval, in one paragraph

14 hand-written Q&A in `eval/questions.yaml`, tagged with `gold_pages`
(for Recall@k and MRR), `must_contain` substrings (for Recall-style
answer checks), and `refuse: true` for questions the model should
refuse. RAGAS-style **faithfulness**, **answer relevance**, and
**context precision** use the same Cerebras model as the judge; **context
recall** is a deterministic substring check against `must_contain`.
Run with `python -m eval.run_eval --ragas`. Latest scores in
[`app/eval/RESULTS.md`](../app/eval/RESULTS.md).

---

## 7. Things to NOT say in the conversation

- "I used LangChain" — we use exactly one helper from LangChain
  (`RecursiveCharacterTextSplitter`) and nothing else. The orchestration,
  retrieval, prompt formatting, and provider routing are all custom.
- "RAGAS is a library we installed" — we wrote the LLM-judge prompts
  ourselves in `eval/ragas_metrics.py`. Same methodology, no extra dep.
- "We need a vector DB like Pinecone" — we use Snowflake's native VECTOR
  column. Same primitives (cosine similarity), one fewer thing to operate.
- "Multi-turn is implemented" — the UI tracks multiple messages per
  chat but the LLM sees each query in isolation today.

---

## 8. Common interview probes and short answers

| Q | A |
|---|---|
| Why not just use BM25? | BM25 misses paraphrases. The retrieval-only quality on questions like *"how is data center demand growing?"* (where the deck phrases it as *"cloud transformation fundamentals"*) is markedly worse without dense embeddings. |
| Why local embeddings if Gemini is free? | The 100 RPM free-tier limit kept halting bulk ingest. Local embedding (BGE on CPU) takes ~17 ms per chunk and has no quota. |
| Why 768 dimensions specifically? | BGE-base is 768d, the Snowflake schema is `VECTOR(FLOAT, 768)`. The alternative 1024d (BGE-M3, multilingual) doesn't add value for an English-only financial corpus. |
| How would you scale this to 1M chunks? | Two changes: (a) add an HNSW vector index in Snowflake's table (or move to a dedicated ANN store), (b) lexical search via Snowflake `SEARCH()` (full-text, gated on higher tiers) instead of `LIKE`. The retrieval API doesn't change. |
| What if a question spans multiple docs that disagree? | The retriever doesn't filter; top-K pulls from any doc. The prompt forbids averaging or silent picking — the model must surface the disagreement with attribution. |
| Why no reranker? | Eval shows it would be the biggest single quality lever (context precision currently ~9%). It was deferred for time; `bge-reranker-base` is a 30-min add. |
| Why Streamlit and not React + FastAPI? | The brief recommends it, and the backend is decoupled enough (`generation/service.py::query()`) that a FastAPI route is a 15-line addition with no business-logic change. |
| How do you prevent hallucination? | The prompt enforces `[N]` citations on every claim and explicitly refuses when evidence is insufficient. Citation parsing validates that every `[N]` resolves to a real chunk; mismatches are dropped. The RAGAS faithfulness metric measures the residual hallucination rate. |
