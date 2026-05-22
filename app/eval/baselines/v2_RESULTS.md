# v2 — Evaluation Results & Failure Analysis

_Generation: `gemini-3.1-flash-lite` · Retrieval: multi-stage (router → dense(props+chunks) + lexical + structured → RRF → cross-encoder rerank → diversify → version-pair → small-to-big → conflict-detect) · Judge: `gemini-2.5-flash` (separate model)._

This evaluation uses **three complementary tracks** (per the recommended hybrid
methodology): (A) the 24-question Vectera battery scored Pass/Partial/Fail,
(B) an independent **LLM-judge** first pass, and (C) automated **RAGAS-style
metrics**. The judge grades against each question's diagnostic criteria + the
retrieved context — never its own memory — and a different model from the one
that generated the answer (no self-grading).

---

## Headline

| | Pass | Partial | Fail |
|---|---|---|---|
| **Candidate self-assessment** (human review of the answers) | 14 | 5 | 5 |
| **Independent LLM-judge** (stricter, automated) | 7 | 9 | 8 |
| **v1 baseline** (manual) | 6 | 7 | 11 |

The human and judge scores diverge because the judge is strict on
"missing-a-nuance" Partials and on cases where the right slide was retrieved but
a specific value wasn't extracted (see Failure Analysis). Both are reported
transparently; the battery asks the candidate to self-assess, and the judge is
supporting rigor.

## RAGAS-style metrics (mean over 24 Q, judge = gemini-2.5-flash)

| Metric | v1 | v2 | Δ |
|---|---|---|---|
| **Faithfulness** — answer grounded in retrieved sources | 0.70 | **0.95** | **+0.25** |
| **Answer relevance** — answer addresses the question | 0.96 | 0.95 | ≈ |
| **Context precision** — fraction of retrieved chunks that are relevant | **0.09** | **0.49** | **5×** |
| **Context recall** — expected facts present in retrieval | — | 0.57 | — |

**The two headline wins:** context precision **9% → 49%** (the reranker +
structured retrieval), and faithfulness **0.70 → 0.95** (when v2 answers, it's
grounded — minimal hallucination).

---

## Where the failures actually are (the diagnostic)

Judge `likely_failure_stage` distribution: **generation 10 · retrieval 6 · citation 1 · pass 7**.

But a deeper, evidence-based look (checking whether the needed content exists in
the index AND was retrieved) shows the labels understate retrieval's progress:

| Failing Q | Content in index? | Retrieved into top sources? | True root cause |
|---|---|---|---|
| Q14 (PSA 92% vs NSA 84.3%) | ✅ | ✅ (merger comparison slide) | **Context assembly** — `table_rows` not injected into the prompt; model can't pull `84.3%` from the parent-slide text |
| Q8 (DLR top-10 → top-100 basis) | ✅ | ✅ (both customer-base slides) | **Context assembly** — methodology footnote not surfaced to the model |
| Q1 (DLR total IT capacity) | ✅ | ✅ after the dense-over-chunks fix | Partial — surfaces one version; version-pair didn't pull the sibling's capacity slide |
| Q17 (VICI 10 trophy assets) | partial (names in a map image) | partial | **Vision/cross-page** — names baked into a map graphic; needs multi-hop synthesis from pp.13/16 |
| Q18 (Realty Income region ABR) | partial (geo-positioned on a map) | weak | **Vision spatial** — map data not fully extracted |
| Q24 (per-REIT 2026 FFO) | mixed | partial | Per-company coverage + the model hallucinated PSA FFO instead of refusing |

**A/B confirmation:** re-running the worst 5 on the stronger `gemini-2.5-flash`
generation model **did not fix them** — proving the bottleneck is **not** model
strength. When the right slide is retrieved but the answer still fails, it's
because the **precise structured detail (table cells / footnote qualifiers) is
not in the generation context** — we pass the parent slide text, not the matched
`table_rows`/`chart_records`/`footnote_text`.

### Highest-leverage next improvement
**Inject the matched `table_rows`, `chart_records`, and `footnote_text` into the
generation prompt** (alongside the parent slide). Evidence says this would
directly recover Q14, Q8, Q11, and likely Q2/Q6 — the value is *in the index and
retrieved*, just not in the prompt. This is a context-assembly change, not a
retrieval or model change.

---

## What clearly works in v2 (Pass)

- **Q7** — footnote-as-truth: returns the 5.47% market yield from the footnote, not the 3.9% chart-body construct.
- **Q12 / Q13** — scope & metric-definition separation (standalone vs pro-forma; NOI vs ARO).
- **Q22 / Q23** — cross-document diversification: balanced AI-across-sectors and VICI-vs-Realty-Income gaming (per-doc quota + query decomposition).
- **Q9** — intra-doc scope ambiguity (10 vs 9 countries) surfaced with breakdown.
- **Q20** — content-density: substantive EastGroup strategy, not photo-caption noise.

These exercise exactly the capabilities v1 lacked (version-pair surfacing,
conflict detection, doc-type/scope awareness, footnote integrity, diversification).

---

## Methodology notes
- **No self-grading:** generation = `gemini-3.1-flash-lite`; judge = `gemini-2.5-flash`; the judge grades the answer as an external artifact against the battery criteria + retrieved context.
- **Human-in-the-loop:** every Fail/Partial was reviewed; the candidate self-assessment is the authoritative battery score.
- **RAGAS-style** metrics are implemented in-house (`eval/ragas_metrics.py`) with the same judge — the same metrics are interchangeable with the RAGAS / TruLens / DeepEval libraries.
- Reproduce: `python -m eval.run_battery_v2` (writes `v2_battery_results.json` + `v2_battery_scored.md`); A/B in `v2_battery_ab25_*`.
