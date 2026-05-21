"""Run the 24-question Vectera self-eval battery against the current system.

Unlike `run_eval.py` (which scores against a hand-built gold set), this runner
just executes every battery question and captures:
  - the verbatim answer
  - the retrieved chunks (id, doc_id, company, page, score, text snippet)
  - latency

Output:
  - app/eval/baselines/v1_battery_results.json   (machine-readable, full retrieval)
  - app/eval/baselines/v1_battery_results.md     (human-readable, for self-eval table)

Pass/Partial/Fail scoring is done MANUALLY against the diagnostic notes after
this runs — these 24 questions require judgment that's hard to automate.

Usage:
  python -m eval.run_battery                  # all 24 questions
  python -m eval.run_battery --limit 5        # first 5 (sanity check)
  python -m eval.run_battery --only Q15,Q18   # specific question IDs
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

# Force UTF-8 stdout
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.generation import query


BASELINE_DIR = Path(__file__).parent / "baselines"
BATTERY_FILE = Path(__file__).parent / "battery_v1.yaml"


def _chunk_record(ch) -> dict:
    """Compact representation of a retrieved chunk for the JSON log."""
    return {
        "chunk_id": ch.chunk_id,
        "doc_id": ch.doc_id,
        "company": ch.company,
        "page": ch.page_number,
        "type": ch.chunk_type,
        "doc_date": ch.doc_date.isoformat() if ch.doc_date else None,
        "version_label": ch.version_label,
        "score": round(float(ch.score), 4),
        "dense_rank": ch.dense_rank,
        "lexical_rank": ch.lexical_rank,
        "text_snippet": (ch.text or "")[:300].replace("\n", " "),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None,
                   help="run only the first N questions")
    p.add_argument("--only", type=str, default=None,
                   help="comma-separated IDs e.g. Q15,Q18")
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--out-prefix", type=str, default="v1_battery_results",
                   help="output filename prefix in app/eval/baselines/")
    args = p.parse_args(argv)

    cases = yaml.safe_load(BATTERY_FILE.read_text(encoding="utf-8"))
    if args.only:
        wanted = {x.strip().upper() for x in args.only.split(",")}
        cases = [c for c in cases if c["id"].upper() in wanted]
    if args.limit:
        cases = cases[: args.limit]

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(cases)} battery questions (top_k={args.top_k})...\n")
    started = datetime.utcnow()
    results = []
    for i, case in enumerate(cases, start=1):
        qid = case["id"]
        question = case["question"]
        print(f"[{i:>2}/{len(cases)}] {qid}: {question[:80]}")
        try:
            ans = query(question, top_k=args.top_k, write_log=False)
            results.append({
                "id": qid,
                "group": case["group"],
                "group_name": case["group_name"],
                "question": question,
                "diagnoses": case.get("diagnoses", "").strip(),
                "answer": ans.answer,
                "citations": [
                    {"n": c.n, "chunk_id": c.chunk_id,
                     "company": c.company, "page": c.page_number}
                    for c in ans.citations
                ],
                "retrieved": [_chunk_record(ch) for ch in ans.retrieved],
                "latency_ms": ans.latency_ms,
                "llm_provider": ans.llm_provider,
                "llm_model": ans.llm_model,
                "error": None,
            })
            print(f"    -> {len(ans.retrieved)} chunks, "
                  f"{len(ans.citations)} cites, {ans.latency_ms} ms\n")
        except Exception as e:
            print(f"    !! ERROR: {e}\n")
            results.append({
                "id": qid,
                "group": case["group"],
                "group_name": case["group_name"],
                "question": question,
                "diagnoses": case.get("diagnoses", "").strip(),
                "answer": "",
                "citations": [],
                "retrieved": [],
                "latency_ms": 0,
                "error": str(e),
            })

    finished = datetime.utcnow()
    elapsed = (finished - started).total_seconds()

    # --- Write JSON ---
    json_path = BASELINE_DIR / f"{args.out_prefix}.json"
    json_path.write_text(
        json.dumps({
            "system": "v1",
            "started_utc": started.isoformat() + "Z",
            "finished_utc": finished.isoformat() + "Z",
            "elapsed_seconds": round(elapsed, 2),
            "top_k": args.top_k,
            "n_questions": len(results),
            "results": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # --- Write Markdown self-eval skeleton ---
    md_path = BASELINE_DIR / f"{args.out_prefix}.md"
    lines = [
        "# v1 Baseline — Vectera Self-Evaluation Battery",
        "",
        f"_System: v1 (current main). Run at {started.isoformat()}Z. "
        f"Total elapsed: {elapsed:.1f}s. top_k={args.top_k}._",
        "",
        "Each row below is **unscored** — manual Pass / Partial / Fail review needed.",
        "",
        "---",
        "",
    ]

    current_group = None
    for r in results:
        if r["group"] != current_group:
            lines += [
                f"## Group {r['group']} — {r['group_name']}",
                "",
            ]
            current_group = r["group"]

        snippet = (r["answer"] or "").strip().replace("\n", " ")
        if len(snippet) > 800:
            snippet = snippet[:800] + "…"
        cites = ", ".join(
            f"[{c['n']}] {c['company']} p.{c['page']}" for c in r["citations"]
        ) or "_(no citations)_"
        retrieved_summary = ", ".join(
            f"{ch['company']} p.{ch['page']} ({ch['type']})"
            for ch in r["retrieved"][:5]
        )

        lines += [
            f"### {r['id']} — {r['question']}",
            "",
            f"**Diagnoses:** {r['diagnoses']}",
            "",
            f"**Top-{args.top_k} retrieved (first 5):** {retrieved_summary}",
            "",
            f"**Citations:** {cites}",
            "",
            "**Answer:**",
            "",
            f"> {snippet}" if snippet else "> _(empty answer)_",
            "",
            "**Score:** ☐ Pass · ☐ Partial · ☐ Fail",
            "",
            "**Note:** _(fill in after manual review)_",
            "",
            "---",
            "",
        ]

    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n✓ JSON saved: {json_path}")
    print(f"✓ Markdown saved: {md_path}")
    print(f"  Elapsed: {elapsed:.1f}s | {len(results)} questions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
