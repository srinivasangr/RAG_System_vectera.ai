# Evaluation results

Run on the 11-doc Snowflake corpus, `top_k=8`,
LLM = Cerebras `gpt-oss-120b`, embedder = local `BAAI/bge-base-en-v1.5`,
LLM-as-judge = same Cerebras model.

Reproduce:

```bash
python -m eval.run_eval --ragas --skip-requires
```

## Aggregate scores

| Tier | Metric | Score | Notes |
|---|---|---|---|
| Core | **Recall@8** | **81.8%** | At least one gold-page chunk in top-8 for 9/11 questions |
| Core | **Mean MRR** | 0.524 | First gold chunk averages rank ~2 |
| Core | **Must-contain** | 81.8% | Required keywords in the answer for 9/11 questions |
| Core | **Refusal correctness** | **100%** | Out-of-corpus questions (n=2) refuse cleanly |
| Core | Mean latency | 14.5 s | Two outliers ≈55 s — Cerebras retry on transient errors |
| RAGAS | **Faithfulness** | 70.4% | Atomic-claim grounding (n=8) |
| RAGAS | **Answer relevance** | **95.9%** | Direct answer-to-question scoring (n=11) |
| RAGAS | Context precision | 8.9% | Fraction of top-8 chunks judged relevant (n=7) |
| RAGAS | **Context recall** | 55.6% | Must-have facts found in retrieved chunks (n=9) |

## What each metric means + how it's computed

| Metric | Definition | Computation |
|---|---|---|
| **Recall@k** | Did at least one gold-labeled chunk appear in the top-k retrieved? | Walk retrieved chunks; first match against `gold_pages` in the YAML scores 1. |
| **MRR** | Reciprocal rank of the first gold-labeled chunk | `1 / rank` of the first match, 0 if none. |
| **Must-contain** | Fraction of required substrings present in the final answer | Lowercase substring check against `must_contain` list in the YAML. |
| **Refusal correctness** | For questions tagged `refuse: true`, did the model say the no-info sentence? | Substring check for `"don't have enough information"` in the answer. |
| **Faithfulness** | Of the answer's atomic claims, what fraction are directly supported by a retrieved chunk? | LLM judge splits the answer into claims, scores each against the sources, averages. |
| **Answer relevance** | How directly does the answer address the question? | LLM judge scores on a five-point scale (0.0 / 0.3 / 0.5 / 0.7 / 1.0). |
| **Context precision** | Of the retrieved chunks, what fraction is judged relevant to the question? | LLM judge marks each chunk relevant / not, averages. |
| **Context recall** | Of the must-have facts (`must_contain` phrases), what fraction appears in any retrieved chunk? | Substring check across all retrieved chunk texts. |

## Reading the numbers

The signal is consistent with how the system was built:

- **Generation is solid.** Answer relevance 95.9% and refusal correctness 100% say the LLM follows the prompt: it answers what's asked and refuses cleanly when evidence is missing.
- **Retrieval finds the right region but is noisy.** Recall@8 of 82% and MRR 0.52 mean we usually surface a gold-labeled chunk in the top few results — good enough that the LLM has the material to answer. But context precision of 8.9% says the *other* 7 chunks in the top-8 are largely irrelevant by the judge's strict criterion. There is real headroom for a cross-encoder reranker between candidate-30 and final top-8.
- **Faithfulness 70%** means roughly 3 in 10 atomic claims fail the judge's "directly supported by a source" test. Some of this is the judge being strict about hedging language; some is genuine over-claiming. Tightening the prompt to forbid inference beyond the snippets would help.
- **Context recall 55.6%** says about half the must-have facts make it into the retrieved chunks. Combined with high Recall@8 (82%), this tells us that when retrieval succeeds it finds *the right pages*, but a single page sometimes doesn't carry every fact named in `must_contain`. Expanding chunks or lowering chunk overlap would move this number.

## Caveats

- **Eval set is hand-written, n=14 (11 active).** Statistically thin; use as a regression baseline, not a benchmark for absolute quality.
- **3 questions skipped** by `--skip-requires` because the eval YAML names companies by ticker (`BXP`, `PSA`, `VICI`) while the ingested rows carry full names (`Boston Properties`, `Public Storage`, `VICI Properties`). The questions are answerable — the skip heuristic is just conservative.
- **2 latency outliers (~55s)** are SDK retries after transient Cerebras errors, not steady-state behaviour. Median latency is ~5s.
- **LLM-as-judge variance.** Same metric re-run on the same data fluctuates ±5-10% because the judge model isn't deterministic at non-trivial reasoning. The numbers above are one run.

## What would move the numbers

| Lever | Expected effect |
|---|---|
| Add a cross-encoder reranker (`bge-reranker-base` between candidate-30 and top-8) | **+context precision** (large) — re-orders for true relevance |
| Lower top-K from 8 → 4 | **+context precision**, **−context recall** (tradeoff) |
| Increase chunk size / merge adjacent | **+context recall** — more facts per chunk |
| Tighten the answering prompt (forbid all inference) | **+faithfulness**, **−answer completeness** |
| Add multi-turn context | **+answer relevance** on follow-up Q&A (not currently tested) |
| Use a stronger judge (Claude Sonnet 4.6) | More reliable RAGAS scores, less JSON parse failure |
