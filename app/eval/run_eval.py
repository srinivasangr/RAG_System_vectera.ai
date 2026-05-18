"""Run the eval set and report per-question + aggregate metrics.

Core metrics (always run, no LLM calls beyond the answer itself):
  - retrieval_hit:   did at least one gold page appear in the top-k? (recall@k)
  - mrr_at_k:        reciprocal rank of the first gold-page chunk
  - must_contain:    fraction of required substrings present in the answer
  - refuse_ok:       for refuse=true questions, did the model say so?
  - citation_count:  number of unique citations the model used

RAGAS metrics (opt-in via --ragas; +4 LLM calls per question, ~5s each):
  - faithfulness        are the answer's claims grounded in the sources?
  - answer_relevance    does the answer actually address the question?
  - context_precision   of the retrieved chunks, what fraction is relevant?
  - context_recall      of the gold facts, what fraction is in the retrieved chunks?

Usage:
  python -m eval.run_eval                       # core metrics only
  python -m eval.run_eval --ragas               # core + RAGAS
  python -m eval.run_eval --ragas --limit 5     # quick sample
  python -m eval.run_eval --skip-requires       # skip Qs needing missing docs
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

# Force UTF-8 stdout for model output with unicode chars
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.generation import query
from rag_system.storage.repository import corpus_stats


@dataclass
class CaseResult:
    id: str
    category: str
    retrieval_hit: bool
    mrr: float
    must_contain_frac: float
    refuse_ok: bool | None
    citation_count: int
    latency_ms: int
    answer_snippet: str
    # RAGAS metrics (None when --ragas isn't passed or metric isn't measurable)
    faithfulness: float | None = None
    answer_relevance: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    skipped: bool = False
    skip_reason: str = ""


REFUSE_SENTINEL = "don't have enough information"


def _gold_match(retrieved_chunks, gold_pages) -> tuple[bool, float]:
    """Returns (hit, MRR). gold_pages may specify company alone or company+page."""
    if not gold_pages:
        return (True, 1.0)  # nothing to verify
    for rank, ch in enumerate(retrieved_chunks, start=1):
        for gp in gold_pages:
            if gp.get("company") and ch.company != gp["company"]:
                continue
            if gp.get("page") is not None and ch.page_number != gp["page"]:
                continue
            return (True, 1.0 / rank)
    return (False, 0.0)


def _check_must_contain(answer: str, must: list[str]) -> float:
    if not must:
        return 1.0
    lower = answer.lower()
    hit = sum(1 for s in must if s.lower() in lower)
    return hit / len(must)


def _check_refuse(answer: str) -> bool:
    return REFUSE_SENTINEL.lower() in answer.lower()


def _present_companies() -> set[str]:
    # Filter out None — a doc with an unparseable filename can land in the DB
    # with company=None; we don't want to crash the `requires:` check on it.
    return {c for c, _, _ in corpus_stats()["per_company"] if c}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-requires", action="store_true",
                   help="skip questions whose `requires` doc isn't in the corpus yet")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--ragas", action="store_true",
                   help="also run RAGAS-style LLM-judge metrics "
                        "(faithfulness, answer_relevance, context_precision/recall). "
                        "~4 extra LLM calls per question.")
    args = p.parse_args(argv)

    eval_file = Path(__file__).parent / "questions.yaml"
    cases = yaml.safe_load(eval_file.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    present_companies = _present_companies()

    results: list[CaseResult] = []
    for case in cases:
        # Determine if we should skip (`requires` not met)
        requires = case.get("requires") or []
        should_skip = False
        skip_reason = ""
        if args.skip_requires and requires:
            for r in requires:
                # `requires:` entries can be specific (e.g. "Digital Realty Mar 2026"
                # or just "BXP") — skip only if no ingested company name appears
                # anywhere in the requirement string.
                if not any(c.lower() in r.lower() for c in present_companies):
                    should_skip = True
                    skip_reason = f"missing: {r}"
                    break

        if should_skip:
            results.append(CaseResult(
                id=case["id"], category=case["category"],
                retrieval_hit=False, mrr=0.0, must_contain_frac=0.0,
                refuse_ok=None, citation_count=0, latency_ms=0,
                answer_snippet="", skipped=True, skip_reason=skip_reason,
            ))
            continue

        ans = query(case["question"], top_k=args.top_k, write_log=False)

        hit, mrr = _gold_match(ans.retrieved, case.get("gold_pages") or [])
        must_frac = _check_must_contain(ans.answer, case.get("must_contain") or [])
        refuse_ok = _check_refuse(ans.answer) if case.get("refuse") else None

        # RAGAS metrics (opt-in)
        faith = relevance = precision = recall = None
        if args.ragas:
            from eval.ragas_metrics import evaluate_one
            scores = evaluate_one(
                question=case["question"],
                answer=ans.answer,
                retrieved_chunks=ans.retrieved,
                must_contain=case.get("must_contain") or [],
            )
            faith = scores.faithfulness
            relevance = scores.answer_relevance
            precision = scores.context_precision
            recall = scores.context_recall

        results.append(CaseResult(
            id=case["id"], category=case["category"],
            retrieval_hit=hit, mrr=mrr,
            must_contain_frac=must_frac,
            refuse_ok=refuse_ok,
            citation_count=len(ans.citations),
            latency_ms=ans.latency_ms,
            answer_snippet=ans.answer[:200].replace("\n", " "),
            faithfulness=faith,
            answer_relevance=relevance,
            context_precision=precision,
            context_recall=recall,
        ))

    # --- Per-case ---
    print(f"\n{'id':<28} {'cat':<16} {'hit':>4} {'mrr':>6} {'mc':>5} {'ref':>4} {'cites':>6} {'ms':>6}")
    print("-" * 96)
    for r in results:
        if r.skipped:
            print(f"{r.id:<28} {r.category:<16}  SKIP  ({r.skip_reason})")
            continue
        print(
            f"{r.id:<28} {r.category:<16} "
            f"{'Y' if r.retrieval_hit else 'N':>4} "
            f"{r.mrr:>6.2f} "
            f"{r.must_contain_frac:>5.2f} "
            f"{('Y' if r.refuse_ok else 'N') if r.refuse_ok is not None else '-':>4} "
            f"{r.citation_count:>6d} "
            f"{r.latency_ms:>6d}"
        )

    active = [r for r in results if not r.skipped]
    if active:
        recall_hit_rate = sum(1 for r in active if r.retrieval_hit) / len(active)
        mean_mrr = sum(r.mrr for r in active) / len(active)
        mean_mc = sum(r.must_contain_frac for r in active) / len(active)
        refuse_cases = [r for r in active if r.refuse_ok is not None]
        refuse_rate = (
            sum(1 for r in refuse_cases if r.refuse_ok) / len(refuse_cases)
            if refuse_cases else None
        )
        mean_lat = sum(r.latency_ms for r in active) / len(active)
        print("\n--- Aggregate ---")
        print(f"  active cases:        {len(active)} / {len(results)}")
        print(f"  Recall@{args.top_k}:           {recall_hit_rate:.2%}")
        print(f"  Mean MRR:            {mean_mrr:.3f}")
        print(f"  Mean must-contain:   {mean_mc:.2%}")
        if refuse_rate is not None:
            print(f"  Refusal correctness: {refuse_rate:.2%}  ({len(refuse_cases)} cases)")
        print(f"  Mean latency:        {mean_lat:.0f} ms")

        if args.ragas:
            def _mean(field: str) -> float | None:
                vals = [getattr(r, field) for r in active if getattr(r, field) is not None]
                return sum(vals) / len(vals) if vals else None

            print("\n--- RAGAS-style (LLM-as-judge) ---")
            for label, key in [
                ("Faithfulness",      "faithfulness"),
                ("Answer relevance",  "answer_relevance"),
                ("Context precision", "context_precision"),
                ("Context recall",    "context_recall"),
            ]:
                m = _mean(key)
                if m is None:
                    print(f"  {label:<22} (no measurable cases)")
                else:
                    n_measured = sum(1 for r in active if getattr(r, key) is not None)
                    print(f"  {label:<22} {m:.2%}  (n={n_measured})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
