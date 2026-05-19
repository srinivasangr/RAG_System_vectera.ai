# v1 Baseline — Vectera Self-Evaluation Battery (Scored)

**System:** v1 (current `main` branch, frozen as `v1` branch)
**Date:** 2026-05-19
**Corpus:** 11 ingested PDFs (10 REIT decks + 1 third-party report)
**Top-k:** 8
**Total elapsed:** 300.6s (avg ~12.5s/question, p95 ~60s)

---

## Headline numbers

| Outcome | Count | % |
|---|---|---|
| **Pass** | 6 / 24 | **25%** |
| **Partial** | 7 / 24 | **29%** |
| **Fail** | 11 / 24 | **46%** |

Honest read: ~half the battery questions break the system in some way. The passes are concentrated in cases where retrieval coincidentally surfaced the right chunk; the partials show the LLM has the capacity but the pipeline isn't feeding it the right context; the failures are systematic — they all map to **8 missing capabilities** that v2 must address.

---

## Per-question scores

### Group A — Version & temporal awareness

| ID | Q | Score | Note |
|---|---|---|---|
| Q1 | DLR total IT capacity | **Fail** | Refused. Retrieved DLR pages but missed both capacity slides. No version-pair surfacing of Dec25 (2.9 GW) vs Mar26 (3 GW + 5 GW future). |
| Q2 | BXP 2026 occupancy | **Partial** | Gave Investor Day range "87.25–88%, year-end ~89%". Missed the Q4 deck's tighter 88% point estimate. No doc-type-aware ranking. |
| Q3 | BXP strategy delta (ID vs Q4) | **Fail** | Refused. Retrieved both docs but couldn't synthesize delta. No temporal-delta reasoning. |
| Q4 | Simon economic impact | **Fail** | Presented 2017 numbers as current fact ($695M property tax, etc.) with **no staleness flag**. No content-extracted publication date. |

### Group B — Intra-document inconsistency & footnote awareness

| ID | Q | Score | Note |
|---|---|---|---|
| Q5 | DLR customer count | **Fail** | Picked "more than 5,000" silently. Did not surface the page 3 (5,500+) vs page 23 (5,000+) inconsistency. |
| Q6 | BXP dividend yield | **Partial** | Got 5.47% (footnote) and 5.2% (Mar 2026). Missed the 3.9% chart body number — so the **footnote-vs-body integrity test wasn't triggered**, but coincidentally got two of three figures. |
| Q7 | BXP yield as of Aug 29, 2025 | **Pass** | Correctly returned 5.47% from the footnote, not the 3.9% chart body. Lucky that retrieval surfaced the footnote chunk. |
| Q8 | DLR credit-quality methodology change | **Pass** | Excellent — surfaced both 51% (top-100, $6.1B) and 50% (top-10, $4.5B) with explicit methodology change. The diagnostic question itself worked. |
| Q9 | Realty Income countries | **Pass** | Surfaced the 10 vs 9 vs 8 conflict across pages and explicitly said "sources disagree." Hybrid retrieval got enough variety here. |
| Q10 | EastGroup portfolio size | **Pass** | Preserved the qualifier: "65M SF — including existing properties, development projects, and value-add acquisitions in lease-up." Only got one of two definitions but kept the scope intact. |

### Group C — Document-type & scope distinctions

| ID | Q | Score | Note |
|---|---|---|---|
| Q11 | PSA NOI margin | **Fail** | Got 78% from Company Update only. **Missed entire Merger Presentation** (69% NSA, 77% pro-forma). No doc_type metadata, no retrieval pulled from Merger doc. |
| Q12 | PSA 2026 outlook | **Partial** | Got Company Update standalone numbers cleanly. Missed Merger Presentation FFO-neutral/accretive timeline. Same root cause as Q11. |
| Q13 | BXP CBD market % | **Partial** | Got 90.8% NOI + ~90% ARO. Missed the 90.5% NOI from Investor Day and the 70.9% historical comparison. No metric-definition awareness in attribution. |
| Q14 | PSA vs NSA same-store occupancy diff | **Fail** | "Don't have enough information... PSA 92.0% but no NSA figure." Retrieved 8 PSA pages — all from Company Update, none from the Merger Presentation 3-column table. |

### Group D — Chart, table, and map parsing (vision-required)

| ID | Q | Score | Note |
|---|---|---|---|
| Q15 | Self-storage REIT NOI growth + margin | **Fail** | **Confidently wrong.** Said PSA 78% growth / 78% margin, Extra Space 71% / 71%, etc. — fabricated mappings where growth = margin (clearly impossible). Vision pass missed the multi-subchart re-sorting. |
| Q16 | VICI largest tenant | **Fail** | **Confidently wrong.** Said "Eastern Band of Cherokee Indians, 39%." It's Caesars at 39%. Logo-rendered names + chunk text near them led to bad alignment. |
| Q17 | VICI 10 trophy Strip assets | **Fail** | Refused. Did not cross-reference pages 13/16 where Caesars, Venetian, MGM Grand, Mandalay Bay etc. appear in body text. No cross-page synthesis. |
| Q18 | Realty Income U.S. regions vs Europe ABR | **Partial** | Got Europe 19%. Honestly refused on U.S. regional breakdown ("not in provided sources"). Spatial map parsing not attempted. |
| Q19 | EGP markets ranked | **Fail** | Refused — "lists states but no ranking." Page 8 explicitly has TX 35% / FL 25% / CA 15% / AZ 8% / NC 5% but retrieval missed it. |

### Group E — Ingestion quality

| ID | Q | Score | Note |
|---|---|---|---|
| Q20 | EGP property selection strategy | **Pass** | Substantive strategy answer. Content-density filtering issue didn't bite here because retrieval ranked strategy pages above photo caption chunks. |
| Q21 | BXP key strategy | **Partial** | Covers the spirit (FFO growth, top 15% markets, moderate leverage). Doesn't surface the structured 6-point SUMMARY slide. Slide-level structured list not preserved as a chunk. |

### Group F — Cross-document synthesis

| ID | Q | Score | Note |
|---|---|---|---|
| Q22 | AI impact across REIT sectors | **Fail** | 6 of 8 retrieved chunks were BXP. Answer is BXP-dominated. Missed DLR data-center angle, PSA's internal-efficiency framing, Realty Income's data-center diversification. Classic non-diversification failure. |
| Q23 | VICI vs Realty Income gaming | **Pass** | Balanced the comparison: VICI 54 gaming + 39 experiential, Realty Income's single Encore Boston Harbor property. |
| Q24 | 2026 FFO outlook per REIT | **Partial** | Got BXP ($6.96 midpoint) and PSA ($16.35–$17.00 range). **Did not honestly report** that VICI, DLR, Realty Income, EGP, Simon don't disclose 2026 FFO. Just stopped after 2 companies. No per-company round-robin retrieval. |

---

## Failure-mode inventory (the systematic gaps v2 must close)

The 11 fails + 7 partials collapse to **8 root-cause capabilities** v1 lacks:

| # | Missing capability | Hurts | Q's affected |
|---|---|---|---|
| **F1** | **doc_type-aware ranking** (Investor Day vs Q4 update vs Merger Presentation) | Forward-guidance questions get the wrong horizon; scope conflation | Q2, Q11, Q12 (3 partial/fail) |
| **F2** | **version-pair surfacing** (same slide, two versions) | Refuses or picks one silently | Q1, Q5 (2 fail) |
| **F3** | **content-extracted publication date + staleness flag** | Presents 2017 figures as current | Q4 (1 fail) |
| **F4** | **temporal-delta synthesis** (what changed between v1 and v2 of same doc?) | Refuses when both docs are retrieved | Q3 (1 fail) |
| **F5** | **vision spatial mapping** (bbox-grounded labels, not free-text descriptions) | Confidently-wrong chart/logo answers | Q15, Q16 (2 fail, both wrong) |
| **F6** | **cross-page synthesis within a document** | Refuses when names live on different pages | Q17, Q19 (2 fail) |
| **F7** | **retrieval diversification** (MMR + per-doc quota) | Top-K dominated by one doc → biased answers | Q22, Q24 (1 fail + 1 partial) |
| **F8** | **table column→entity preservation** (each cell carries its column label) | Misses NSA column in 3-column comparison | Q14 (1 fail) |

Other contributing factors (lower priority but present):
- **Footnote-body integrity** at chunk boundary (Q6, Q7) — got lucky here, will be flaky
- **Structured-list preservation** (Q21) — slide-level 6-point lists fragment
- **Honest negative reporting** — fine when retrieval truly empty (Q14 refused honestly); breaks when retrieval is one-sided (Q24)
- **Doc-density filtering** during ingest (Q20 passed by luck; will bite on photo-heavy decks)

---

## Patterns I want to call out

### What works
- **Hybrid retrieval + LIKE lexical** finds the right page about 65% of the time on factual lookups
- **The LLM (gpt-oss-120b) handles conflicts well WHEN they're in the retrieved context** — Q8 and Q9 both came back excellent because the conflicting chunks both made it into top-K
- **Honest refusal** works for fully out-of-corpus cases (no hallucination of made-up numbers)

### What breaks
- **Vision pass = generic descriptions**, not spatial/structured records → Q15 and Q16 produce confidently-wrong outputs
- **No diversification means one doc can dominate top-K** → Q22 had 6 BXP chunks out of 8 for a cross-sector question
- **Retrieval is doc-blind** — it doesn't know that "BXP Q4 deck" and "BXP Investor Day" need to be treated as different document types for ranking
- **No version-family grouping** — DLR Dec25 and Mar26 compete as separate docs, not as v1/v2 of the same source
- **Generation prompt has no conflict-pair concept** — when retrieval returns conflicting chunks, the LLM picks one rather than presenting both with attribution (Q5)

---

## Implications for v2

This baseline gives us the exact priority order for v2 work:

1. **Highest leverage** (touches Q2, 11, 12, 13, 22, 24 — 6 questions): doc_type as first-class metadata + retrieval diversification (MMR + per-doc quota)
2. **High leverage** (Q1, Q5, Q14): version-family grouping + table-row chunking with column labels
3. **Vision rebuild** (Q15, Q16): bbox-grounded vision → structured `chart_records`, not free-text descriptions
4. **Cross-page synthesis** (Q17, Q19): query decomposition that allows multi-hop retrieval within a document
5. **Generation upgrades** (Q3, Q4, Q5): conflict-aware prompt, staleness flagging, temporal-delta synthesis
6. **Ingestion polish** (Q6, Q21): footnote-attached chunking, structured-list preservation

Detailed v2 architecture lives in `docs/architecture_v2_final.md` (to be written next).

---

## Reproducing this baseline

```bash
cd app
.venv/Scripts/python.exe -m eval.run_battery
# Outputs:
#   eval/baselines/v1_battery_results.json   (full retrieval logs)
#   eval/baselines/v1_battery_results.md     (unscored skeleton)
#   eval/baselines/v1_battery_scored.md      (this file — scored manually)
```

System config at time of run:
- Snowflake VECTOR(FLOAT, 768) + `VECTOR_COSINE_SIMILARITY`
- BGE-base-en-v1.5 local embeddings
- Cerebras gpt-oss-120b for generation
- Hybrid retrieval (dense ∪ lexical) + RRF, no reranker
- 11 documents, ~1000 chunks total
