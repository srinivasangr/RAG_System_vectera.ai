# v2 Concepts Primer — RAG techniques explained for the interview

> Purpose: every concept referenced in the v2 architecture, explained from scratch with examples from our REIT corpus. Read this once; you'll be able to answer any interview question on "why this technique, not that one."

---

## Part 1 — Foundation concepts (terminology used throughout)

### 1.1 Embedding (dense vector)

**What it is.** A numerical fingerprint of a piece of text. The embedding model reads "Digital Realty's Q4 2025 NOI was $5.2B" and outputs a 768-number vector. Similar meanings produce vectors that are close together in space.

**Why we use it.** It captures *semantic* similarity — "data center capacity" matches "IT power" even though they share no words.

**How we measure closeness.** Cosine similarity — the angle between two vectors. Range: 0 (unrelated) to 1 (identical meaning).

**Our model.** BGE-base-en-v1.5 (`BAAI/bge-base-en-v1.5`). 768 dimensions. Runs locally on CPU. No API rate limit.

**Further reading**
- https://huggingface.co/BAAI/bge-base-en-v1.5
- https://www.pinecone.io/learn/vector-embeddings/ (best intro)

---

### 1.2 Dense retrieval vs lexical retrieval

**Dense retrieval.** Embed the query → find chunks whose embedding is closest. Catches paraphrases, synonyms, and conceptual matches. **Misses** exact-string matches like tickers ("DLR", "BXP"), specific numbers ("$6.96"), and acronyms ("AFFO", "NOI") when the embedding doesn't know they're important.

**Lexical retrieval.** Find chunks that literally contain the query's keywords. Misses paraphrases ("data center" vs "compute facility") but is unbeatable on specifics.

**Why we need both.** In our REIT corpus, both kinds of question exist:
- "What's BXP's strategy?" → dense wins (paraphrased content)
- "What is BXP's 2026 FFO/share?" → lexical wins (literal `$6.96` in a table)

**Our setup.** Hybrid: run both, fuse the results.

**Further reading**
- https://www.elastic.co/blog/improving-information-retrieval-elastic-stack-hybrid (the classic intro)

---

### 1.3 Reciprocal Rank Fusion (RRF)

**The problem it solves.** Dense and lexical retrievers produce scores on totally different scales. Dense cosine is in [0,1]; lexical TF-IDF or BM25 can be 0–50+. You can't just add them.

**How RRF works.** Throw away the scores. Use only the *rank* (1st, 2nd, 3rd...). For each chunk:

```
RRF_score(chunk) = sum over retrievers of  1 / (k + rank)
```

where `k` is a constant (60 is standard). A chunk ranked 1st by dense and 5th by lexical gets `1/61 + 1/65 = 0.0317`. A chunk ranked 1st by both gets `1/61 + 1/61 = 0.0328`.

**Why it's good.** Parameter-free, robust to score-scale differences, no tuning needed.

**v1 uses this already.** It's in `app/rag_system/retrieval/hybrid.py`.

**Further reading**
- Cormack et al., "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (2009) — http://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf

---

### 1.4 Cross-encoder reranker

**The problem it solves.** Embedding similarity is a *bi-encoder* — query and chunk are embedded separately, then compared. Fast but imprecise. Two unrelated chunks can have high cosine just because they share vocabulary.

**How a cross-encoder works.** It takes `(query, chunk)` together as one input and outputs a single relevance score. The model can attend across both — it knows which words in the query match which words in the chunk.

```
bi-encoder:    embed(query) · embed(chunk) → score
cross-encoder: model([query, SEP, chunk]) → score
```

**Trade-off.** Cross-encoders are 100× slower. You can't score 1M chunks. So the standard pattern is:
1. **Retrieve** ~50 candidates with bi-encoder (fast)
2. **Rerank** them with cross-encoder (precise)
3. Keep top 8

**Our pick.** `BAAI/bge-reranker-v2-m3`. 568MB. Runs locally on CPU at ~50ms/query for 50 candidates. No API cost.

**Why this is the single biggest precision lift.** Real example from our v1 baseline:
- Q11 (PSA NOI margin): v1 retrieved 8 chunks all from Company Update because BGE-base sees both Merger Presentation chunks and Company Update chunks as semantically similar. A cross-encoder reading `(query, chunk)` together sees that the Merger Presentation chunk talks about a *combined entity* and ranks it lower for "PSA NOI margin" — surfacing the standalone chunk.

**Further reading**
- https://www.sbert.net/examples/applications/cross-encoder/README.html (definitive)
- https://huggingface.co/BAAI/bge-reranker-v2-m3

---

### 1.5 Maximal Marginal Relevance (MMR)

**The problem it solves.** Top-K retrieval often returns 8 near-duplicate chunks — all from the same page, or all from the same document. The LLM has redundant context and misses other relevant docs.

**How MMR works.** Greedy diversification. Pick the top chunk. Then for each subsequent pick, score chunks by:

```
score = λ · relevance(query, chunk)  −  (1-λ) · max_similarity(chunk, already_picked_chunks)
```

`λ` controls the trade-off. λ=1 = pure relevance (no diversification). λ=0 = pure diversity (ignore relevance). Typical: λ=0.5.

**Real example from our v1 baseline.** Q22 ("How is AI affecting demand across the different real estate sectors?"). v1 returned 6 BXP chunks out of 8 because BGE clustered all BXP AI mentions in the top-K. With MMR + per-doc quota, after picking the first BXP chunk, the next chunks compete against it on similarity — so DLR, PSA, Realty Income chunks rise.

**Per-doc quota.** A simpler heuristic that works well: enforce "no more than 2 chunks per document in top-K." Forces cross-doc diversity directly.

**Further reading**
- Carbonell & Goldstein, "The Use of MMR, Diversity-Based Reranking" (1998) — https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
- https://docs.llamaindex.ai/en/stable/examples/vector_stores/MMR.html

---

### 1.6 Propositions (atomic facts)

**The problem it solves.** A chunk like "EastGroup operates ~65M sq ft including development projects, value-add acquisitions in lease-up, and properties under construction. Florida is the largest state at 25% ABR..." mixes many facts. Embedding the whole chunk averages all of them — the resulting vector matches none of them precisely.

**How propositions help.** Use an LLM to decompose the chunk into atomic statements:
- "EastGroup operates ~65 million square feet."
- "EastGroup's 65M sq ft figure includes development projects."
- "EastGroup's 65M sq ft figure includes value-add acquisitions in lease-up."
- "Florida represents 25% of EastGroup's ABR."

Each proposition is embedded separately. Now "What's EastGroup's Florida exposure?" matches the Florida proposition cleanly, not a noisy mixed-topic chunk.

**Caveat.** Propositions strip context. You retrieve the proposition but pass the parent chunk to the LLM (see "small-to-big" below) so it sees the qualifiers.

**Cost.** One LLM call per chunk during ingest. For our 1000 chunks: ~30 minutes + small cost.

**Further reading**
- Chen et al., "Dense X Retrieval: What Retrieval Granularity Should We Use?" (2024) — https://arxiv.org/abs/2312.06648 (the paper that popularized this)

---

### 1.7 Small-to-big (parent-child chunking)

**The problem it solves.** You want chunks small (precise embedding) but the LLM needs context (surrounding bullets, footnotes, slide title).

**How it works.** Index small chunks (300–500 tokens) for retrieval. After retrieval, replace each chunk with its larger *parent* (the full slide or section, ~1500–2500 tokens) before passing to the LLM.

**Real example from our v1 baseline.** Q7 ("BXP yield as of Aug 29 2025") passed because the 5.47% footnote happened to be in the retrieved chunk. Q6 partially passed because the chart body 3.9% number was in a different chunk than its footnote. If the chunk-with-footnote retrieved on its own, the LLM saw 5.47% in isolation. With parent expansion, the LLM would have seen 3.9% body + 5.47% footnote together and could attribute both.

**Further reading**
- https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/ (LlamaIndex calls this "auto-merging retriever")

---

### 1.8 HyDE (Hypothetical Document Embedding)

**The problem it solves.** Queries are short and use question-style language. Documents are long and use declarative style. Their embeddings can be far apart even when the document contains the answer.

**How HyDE works.**
1. Ask the LLM: "Pretend you know the answer. Write a paragraph that would answer this question."
2. Embed *that fake answer* (not the original query).
3. Search with the fake answer's embedding.

**Why this works.** The fake answer's writing style matches the corpus's writing style. Even if it's factually wrong, it lands in the right neighborhood of vector space.

**When NOT to use it.** Factoid queries ("What is BXP's ticker?"). HyDE adds latency and can hurt — the LLM might write a plausible-sounding but misleading paragraph.

**Our plan.** Make it opt-in via the router. Use only for semantic/comparison queries.

**Further reading**
- Gao et al., "Precise Zero-Shot Dense Retrieval without Relevance Labels" (2022) — https://arxiv.org/abs/2212.10496

---

### 1.9 Query decomposition / multi-query

**The problem it solves.** "Compare AI demand across REIT sectors" is a multi-entity question. Embedding it as one query retrieves chunks that mention "AI" + "REIT" + "demand" — most likely all BXP. The query needs to be broken up.

**How decomposition works.** LLM reads the query and produces sub-queries:
- "Digital Realty AI demand"
- "Boston Properties AI demand"
- "Public Storage AI"
- "Realty Income AI"
- ...

Retrieve for each sub-query, then fuse. Now each company gets at least one bite at the apple.

**v2 detail.** The "router" LLM call does decomposition as part of its output JSON: `{intent: "compare", sub_queries: [...]}`.

**Further reading**
- https://blog.langchain.dev/query-transformations/ (LangChain's blog covers all transforms)

---

## Part 2 — The 8 failure modes (F1–F8)

For each failure mode: what it is, how we verify v1 broke on it, what v2 does to fix it, and a real-world analogy.

---

### F1 — doc_type-aware ranking

**The concept in plain English.** Different document types convey different "truth weights" for different question types. A Q4 earnings deck released March 2026 supersedes a mid-2025 Investor Day for *forward-looking guidance* — but the Investor Day is still authoritative for *strategic intent*.

**Analogy.** If your friend told you their salary in January and again in October, you'd believe October. But if you asked their *career goal* and they told you both times, you'd give equal weight.

**How v1 fails this.** v1 has no `doc_type` column. All chunks compete by cosine similarity alone. For Q2 ("BXP 2026 occupancy"), v1 returned an Investor Day range (87.25–88%) and never surfaced the Q4 deck's tighter 88% point estimate — even though Q4 was more recent and the question was about forward guidance.

**How v2 fixes it.**
1. **At ingest:** classify each doc into an enum (`Investor_Day`, `Q4_Update`, `Company_Update`, `Merger_Presentation`, `Roadshow`, `Third_Party`) via filename + first-page LLM classification.
2. **At retrieval:** the router LLM tags the query intent. If intent is "forward guidance," boost `Q4_Update` and `Company_Update`. If "strategic direction," boost `Investor_Day`.
3. **At generation:** if both doc types in retrieved context, present each with attribution.

**Verifies on.** Q2, Q11, Q12 — all need doc_type to disambiguate scope.

**Further reading**
- "Metadata Filtering" in retrieval: https://docs.llamaindex.ai/en/stable/module_guides/indexing/metadata_extraction/

---

### F2 — Version-pair surfacing

**The concept in plain English.** Two slides with nearly identical titles ("Offering a Global Data Center Platform" in DLR Dec 2025 and DLR Mar 2026) have different numbers (2.9 GW vs 3 GW + 5 GW future). A naive retriever sees them as near-duplicates and may return just one. A reranker may collapse them as redundant. Both must surface, and the answer must attribute each to its source.

**Analogy.** Two photographs of the same room six months apart — different furniture, same caption. You don't want a slideshow that shows just one.

**How v1 fails this.** Q1 ("DLR total IT capacity") — v1 returned 8 chunks, most from one of the two decks, and refused. No mechanism flagged "these are version siblings."

**How v2 fixes it.**
1. **At ingest:** add `doc_family_id` — DLR Dec25 and DLR Mar26 get the same family ID (derived from company + doc series).
2. **At retrieval:** after the main retrieval pass, for the top-K chunks, fetch sibling chunks from other family members covering the same topic (slide title or proposition).
3. **At rerank:** disable the "deduplicate near-identical chunks" behavior when chunks are version-siblings.
4. **At generation:** prompt enforces explicit attribution to (doc_type, as_of_date) when multiple family members are present.

**Verifies on.** Q1, Q5.

---

### F3 — Content-extracted publication date + staleness flag

**The concept in plain English.** The Simon report says "November 2018" on its page 2 cover — not in its filename. A naive ingest reads the filename and gets nothing useful. The Simon data is ~7 years stale compared to the rest of the corpus. The generator should flag this rather than presenting 2017 numbers as current fact.

**Analogy.** Citing a 2017 statistics report in a 2026 essay without saying "as of 2017" is academically dishonest.

**How v1 fails this.** Q4 — v1 reported `$695M property tax, 311,870 retail jobs` as if current. No staleness annotation. Filename had no date, content-date extraction wasn't attempted.

**How v2 fixes it.**
1. **At ingest:** run a per-doc LLM call on pages 1–2 to extract `as_of_date` from cover content. Fall back to filename parsing only if nothing found.
2. **At retrieval:** carry `as_of_date` through every chunk.
3. **At generation:** prompt has a rule: if cited source's `as_of_date` is >2 years older than other cited sources OR >18 months from `today`, prepend "**As of {date}:**" or "**This source is from {year} — context has likely changed.**"

**Verifies on.** Q4.

**Further reading**
- This is a real production concern; "temporal relevance" papers are an active research area. Start with: https://arxiv.org/abs/2310.06756

---

### F4 — Temporal-delta synthesis

**The concept in plain English.** "What changed in BXP's strategy between the 2025 Investor Day and the Q4 2025 update?" — the system must read both documents, identify which parts are stable (strategic framework) and which have evolved (dividend reset announced → reported as completed, occupancy guidance tightened, 2026 FFO disclosed).

**Analogy.** "What's different about this draft compared to last month's?" requires diff-reading, not just reading one of them.

**How v1 fails this.** Q3 — v1 retrieved chunks from both docs but refused to synthesize, saying "no quoted content from source [5]." The LLM didn't have a frame for "produce a delta."

**How v2 fixes it.**
1. **At retrieval:** router detects "delta" intent (`changed between X and Y`) → produces two sub-queries, one per doc, with `doc_id` filters → returns interleaved chunks tagged with their source doc.
2. **At generation:** delta-aware prompt template kicks in: "Below are chunks from [doc A] and [doc B]. Identify (1) stable elements, (2) changed elements with both old and new values, (3) net-new disclosures."

**Verifies on.** Q3.

---

### F5 — Vision spatial mapping (bbox-grounded)

**The concept in plain English.** PSA's page 7 has six bar charts arranged 2×3. Each chart has a fixed legend (PSA/CubeSmart/Extra Space/NSA) at the bottom, but each chart re-orders the bars by performance. The numbers are visible, but text-only extraction just yields a row like `9.0 6.8 6.2 5.5` with no idea which bar belongs to which company. The reader needs the *spatial position* (x-coordinate or bbox) of each label and each bar to make the mapping.

**Analogy.** "What's everyone's favorite color?" from a photo of name-tag-wearing people pointing at colored squares. You need bounding boxes — who is pointing at what — not just a list of names and colors.

**How v1 fails this.** Q15 and Q16. v1 used Gemini vision with free-text-description prompts ("describe this chart"). The model produced plausible captions like "PSA leads with 78%" but made up the rankings. Q15: confidently wrong (gave growth = margin for every company). Q16: wrong tenant (Cherokee instead of Caesars at 39%).

**How v2 fixes it.**
1. **At ingest:** detect chart/map/logo pages via Docling layout (large image regions, low text density).
2. **Vision pass with bbox prompt:** Gemini 2.5 Flash, structured JSON output:
   ```json
   {"labels": [{"text":"PSA","bbox":[x,y,w,h]},...],
    "bars":   [{"value":7.8,"bbox":[x,y,w,h]},...]}
   ```
3. **Post-process:** for each bar, find the nearest label (by x-distance within the same subchart group). Store as `chart_records(page, chart_id, label, value)` rows.
4. **At retrieval:** numeric/comparison queries fetch from `chart_records`, not from prose chunks.
5. **At generation:** if vision data is missing or low-confidence, the prompt instructs honest refusal ("I can see the chart shows values for these companies but cannot reliably map them") instead of fabrication.

**Fallback:** Gemini Pro on hard pages → GPT-4o or Claude Sonnet on the hardest few if needed.

**Verifies on.** Q15, Q16. Also helps Q18 (geo-positioned map data).

**Further reading**
- Google docs on bounding-box prompting: https://ai.google.dev/gemini-api/docs/vision (has a 2D detection example)

---

### F6 — Cross-page synthesis within a document

**The concept in plain English.** VICI's page 14 says "10 trophy assets on the Las Vegas Strip" but doesn't list them (they're baked into a map graphic). Pages 13 and 16 name several of those assets in body text (Caesars Palace, Venetian, MGM Grand, Mandalay Bay, etc.). To answer "what are VICI's 10 trophy assets?", the system must combine page-14 framing with page-13/16 specifics.

**Analogy.** A book's table of contents says "10 chapters" but doesn't list titles. The titles appear inside the chapters. Answering "what are the chapter titles?" needs both.

**How v1 fails this.** Q17 — refused. Q19 — refused even though page 8 had the data. v1's single-shot retrieval returned ~5 chunks all on the topic page; the supplementary pages didn't make top-K.

**How v2 fixes it.**
1. **At retrieval:** for "list" or "enumerate" intents, the router produces two sub-queries: (a) the framing query ("10 trophy assets Las Vegas Strip"), (b) the entity query ("Caesars Venetian MGM Mandalay") — the second is generated by extracting candidate entities from the first retrieval's chunk text.
2. **Two-hop retrieval:** first hop finds the framing chunk; second hop uses that chunk's text as a query to find related body-text chunks.
3. **Same-document boost:** when one chunk from a doc retrieves, its sibling chunks within ±5 pages get a score boost.

**Verifies on.** Q17, Q19.

**Further reading**
- "Multi-hop retrieval" papers — e.g., Khattab et al., DSPy / Multi-hop QA: https://arxiv.org/abs/2305.14283

---

### F7 — Retrieval diversification (MMR + per-doc quota)

**The concept in plain English.** Without diversification, top-K can be 8 chunks from one document for a question that asks about multiple documents. MMR ([§1.5](#15-maximal-marginal-relevance-mmr)) and per-doc quotas force cross-document spread.

**How v1 fails this.** Q22 ("AI across REIT sectors") — 6/8 retrieved chunks were Boston Properties. Q24 ("2026 FFO for each REIT") — all 8 chunks were BXP+PSA only. The other 5 companies in the corpus weren't even queried.

**How v2 fixes it.**
1. **Per-doc quota:** in the candidate set after RRF, no more than `K_per_doc=3` chunks per document.
2. **MMR rerank:** after the cross-encoder, apply MMR with λ=0.5 to spread topically.
3. **Round-robin per-entity:** for explicitly cross-company queries (router output: `intent=compare`, `companies=[...]`), do separate retrievals per company and round-robin the merge.
4. **Honest negatives:** for "for each X in {list}" queries, if a particular company yields zero chunks above relevance threshold, the prompt requires "no 2026 FFO disclosed in the provided {company} materials" — not silent omission.

**Verifies on.** Q22, Q24.

---

### F8 — Table column → entity preservation

**The concept in plain English.** PSA Merger Presentation page 4 has a 3-column table: PSA 92.0% | NSA 84.3% | Pro Forma 90.3%. If the chunker turns this into a flat string `"92.0% 84.3% 90.3%"`, the column labels are gone — the LLM can't tell which number is PSA's and which is NSA's.

**Analogy.** A spreadsheet where you accidentally delete the header row. The numbers still exist but mean nothing.

**How v1 fails this.** Q14 — refused with "no NSA same-store occupancy figure" even though it was right there in the table, because the chunker flattened the column header.

**How v2 fixes it.**
1. **At parse:** Docling already outputs structured tables (rows × columns with headers). v1 then flattens them into prose during chunking — v2 keeps the structure.
2. **At index:** add a `table_rows` table — one row per table-row, with column labels attached to each value:
   ```
   chunk_id | row_idx | columns_json
   ───────────────────────────────────
   t_42_r1  | 1       | {"Metric":"Same-Store Occupancy","PSA":"92.0%","NSA":"84.3%","Pro Forma":"90.3%"}
   ```
3. **At retrieval:** numeric/comparison queries hit `table_rows` directly. The retrieved row is rendered back as a small markdown table for the LLM so it sees the column labels.
4. **At generation:** the prompt template for "compare X and Y" pulls from `table_rows` when both X and Y appear as columns in the same row.

**Verifies on.** Q14. Also assists Q11 (PSA NOI margin in 3-column table).

**Further reading**
- TableRAG / table-aware retrieval is its own subfield: https://arxiv.org/abs/2410.04739
- Docling's table extraction: https://github.com/DS4SD/docling

---

## Part 3 — How to verify each gap is fixed

For each of F1–F8 we have a *target question* in the battery. The v2-vs-v1 delta is the primary KPI.

| Gap | v1 status on target Q | v2 success criterion |
|---|---|---|
| F1 | Q2, Q11, Q12 — Partial/Fail | All three Pass (system surfaces both doc types with attribution) |
| F2 | Q1, Q5 — Fail | Both Pass (both versions surfaced; differences explicit) |
| F3 | Q4 — Fail | Pass (answer flags 2017 vintage) |
| F4 | Q3 — Fail | Pass (produces explicit delta) |
| F5 | Q15, Q16 — Fail (confidently wrong) | At least Partial (honest refusal acceptable; no fabrication) |
| F6 | Q17, Q19 — Fail | At least Partial (Q19 should Pass with page-8 data) |
| F7 | Q22, Q24 — Fail/Partial | Both Pass (≥3 distinct companies in top-K) |
| F8 | Q14 — Fail | Pass (system computes 7.7pp gap from table) |

**Aggregate goal.** Move from 6/24 Pass to ≥16/24 Pass. Even 14/24 with no Fail-fabrications (all Fail's at least become Partial honest-refusal) would be a strong submission.

---

## Part 4 — Glossary (one-line definitions)

| Term | One-liner |
|---|---|
| **Bi-encoder** | Embeds query and doc separately; cosine similarity. Fast, less precise. |
| **Cross-encoder** | Reads `(query, doc)` together; outputs single score. Slow, more precise. |
| **BM25** | Classical lexical retrieval algorithm; TF-IDF on steroids. |
| **HyDE** | Generate hypothetical answer → embed it → search. |
| **MMR** | Maximal Marginal Relevance — greedy diversity-aware reranking. |
| **MRR@k** | Mean reciprocal rank of the first correct result, capped at depth k. |
| **NDCG** | Normalized Discounted Cumulative Gain — graded relevance metric. |
| **Recall@k** | Fraction of relevant items returned in top-k. |
| **Precision@k** | Fraction of top-k items that are relevant. |
| **Faithfulness** | (RAGAS) Does every claim in the answer trace to retrieved context? |
| **Context precision** | (RAGAS) Of the retrieved chunks, what fraction is actually relevant? |
| **Context recall** | (RAGAS) Of the gold facts, what fraction is covered by retrieved chunks? |
| **Answer relevance** | (RAGAS) Does the answer address what was asked? |
| **Proposition / atomic fact** | LLM-decomposed single-claim sentence. |
| **Parent-child / small-to-big** | Index small chunks, generate with large parents. |
| **RRF** | Reciprocal Rank Fusion — rank-based ensemble of retrievers. |
| **Reranker** | Second-stage scorer over top-K retrieval candidates. |
| **Router** | LLM that classifies query intent and routes to the right retrieval path. |
| **Decomposition** | LLM breaks a complex query into sub-queries. |
| **Multi-hop** | Two or more retrieval rounds where round N uses round N-1's output. |
| **Hybrid retrieval** | Dense + lexical (often + structured) fused with RRF. |
| **Doc family** | A group of documents that are versions/snapshots of the same source. |
| **Chunk** | A retrieved unit of text. ~300–500 tokens here. |
| **Embedding** | A vector representation of text. 768-dim here. |
| **Vector DB** | A database that indexes embeddings for fast similarity search. |

---

## Part 5 — Recommended reading order (if you have 2 hours total)

1. (20 min) **Pinecone vector embeddings intro** — https://www.pinecone.io/learn/vector-embeddings/
2. (15 min) **Elastic hybrid retrieval** — https://www.elastic.co/blog/improving-information-retrieval-elastic-stack-hybrid
3. (20 min) **LangChain query transformations** — https://blog.langchain.dev/query-transformations/
4. (20 min) **Sentence-Transformers cross-encoder docs** — https://www.sbert.net/examples/applications/cross-encoder/README.html
5. (20 min) **LlamaIndex auto-merging retriever** (small-to-big) — https://docs.llamaindex.ai/en/stable/examples/retrievers/auto_merging_retriever/
6. (25 min) **Dense X Retrieval paper** (propositions) — https://arxiv.org/abs/2312.06648 — read intro + section 3

If you only read one, read **the LangChain query transformations** blog — it covers HyDE, decomposition, multi-query, and routing in one place.

---

## Part 6 — Interview talking points (use this if asked "why these choices")

**Why hybrid retrieval and not just dense?** Financial documents have tickers, acronyms, exact numbers. Embeddings miss exact-token matches; lexical catches them. RRF fuses without tuning.

**Why a cross-encoder reranker?** Single biggest precision lift available. 568MB local, no API cost. Industry standard pattern: retrieve broad with bi-encoder, rerank narrow with cross-encoder.

**Why propositions?** Atomic statements make embeddings cleaner — less topic-mixing per vector. Cost is one LLM call per chunk at ingest; benefit is significantly higher retrieval precision on factual queries.

**Why MMR plus per-doc quota?** MMR diversifies by similarity; per-doc quota diversifies by source. They protect against different failure modes — MMR against near-duplicate chunks, quota against one-document-dominance.

**Why Snowflake and not pgvector/Qdrant?** Storage isn't the bottleneck at our scale. Snowflake has native `VECTOR_COSINE_SIMILARITY`, fits multi-tenant via row policies, and consolidates storage + analytics in one place. Switching costs > benefits for our use case.

**Why a router LLM?** Different query intents need different pipelines. A lookup query doesn't need HyDE. A comparison query needs decomposition. A delta query needs version-pair retrieval. The router classifies once so downstream stages can specialize.

**Why generate-then-verify (citations)?** Citations are the only honest interface between retrieval and generation. Without them, "the model said it so it must be true." With them, every claim has an audit trail.

**What would you build next given another week?** Multi-turn conversation memory, streaming generation, full RAGAS-tracked CI regression gates, GPT-4o on the hardest vision pages, and a synthetic eval generator that grows the golden set as the corpus grows.
