"""Run the 24-Q Vectera battery against the v2 system + auto-grade with a judge LLM.

Per question:
  v2 answer_query  -> answer + citations + retrieved Sources (filenames/pages)
  judge LLM        -> Pass/Partial/Fail + score + failure_stage + one-line note

Outputs (app/eval/baselines/):
  v2_battery_results.json   full machine-readable (answers + sources + judgments)
  v2_battery_scored.md      human-readable scored sheet (review Fails/Partials)

Generation model and judge model are DIFFERENT (no self-grading). The judge
grades against the battery's diagnostic criteria, not its own memory.

Usage:
  python -m eval.run_battery_v2                       # all 24
  python -m eval.run_battery_v2 --only Q1,Q7,Q15      # subset
  python -m eval.run_battery_v2 --gen-model gemini-3.1-flash-lite
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import yaml

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.judge import judge_answer
from rag_system.generation.generate_v2 import answer_query
from rag_system.llm_providers import get_llm

BASELINE_DIR = Path(__file__).parent / "baselines"
BATTERY_FILE = Path(__file__).parent / "battery_v1.yaml"

# v1 baseline (from app/eval/baselines/v1_battery_scored.md) for the delta.
V1_BASELINE = {"Pass": 6, "Partial": 7, "Fail": 11}


def _sources_summary(sources) -> str:
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(
            f"[{i}] {s.get('company')} — {s.get('doc_type')} — p.{s.get('page_number')} "
            f"— as of {s.get('as_of_date')} — {s.get('filename')}\n"
            f"    {s.get('snippet','')[:240]}"
        )
    return "\n".join(lines) or "(no sources retrieved)"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--only", type=str, default=None, help="comma-separated IDs e.g. Q1,Q7")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--gen-provider", default="gemini")
    p.add_argument("--gen-model", default="gemini-3.1-flash-lite")
    p.add_argument("--judge-provider", default="gemini")
    p.add_argument("--judge-model", default="gemini-2.5-flash")
    args = p.parse_args(argv)

    cases = yaml.safe_load(BATTERY_FILE.read_text(encoding="utf-8"))
    if args.only:
        want = {x.strip().upper() for x in args.only.split(",")}
        cases = [c for c in cases if c["id"].upper() in want]
    if args.limit:
        cases = cases[: args.limit]

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    judge_llm = get_llm(args.judge_provider, args.judge_model)

    print(f"Running {len(cases)} Qs | gen={args.gen_model} judge={args.judge_model}\n")
    started = datetime.utcnow()
    results = []
    for i, case in enumerate(cases, start=1):
        qid, q = case["id"], case["question"]
        print(f"[{i:>2}/{len(cases)}] {qid}: {q[:70]}")
        try:
            a = answer_query(q, provider=args.gen_provider, model=args.gen_model,
                             write_log=False)
            sources = [{
                "company": s.company, "doc_type": s.doc_type, "page_number": s.page_number,
                "as_of_date": str(s.as_of_date) if s.as_of_date else None,
                "filename": s.filename, "snippet": (s.text or "")[:240],
                "cited": (idx + 1) in set(a.cited_numbers),
            } for idx, s in enumerate(a.sources)]
            j = judge_answer(
                question=q, diagnoses=case.get("diagnoses", ""),
                answer=a.answer, citations=a.cited_numbers,
                retrieved_summary=_sources_summary(sources), llm=judge_llm,
            )
            results.append({
                "id": qid, "group": case["group"], "group_name": case["group_name"],
                "question": q, "diagnoses": case.get("diagnoses", "").strip(),
                "answer": a.answer, "intent": getattr(a.plan, "intent", None),
                "cited_numbers": a.cited_numbers, "conflicts": a.conflicts,
                "n_sources": len(a.sources), "sources": sources,
                "timings": {k: v for k, v in a.timings.items() if k != "provider_chain"},
                "judge": j,
            })
            print(f"     -> {j['result']} (score {j.get('score')}) "
                  f"[{j.get('likely_failure_stage')}] {j.get('one_line_note','')[:80]}\n")
        except Exception as e:  # noqa: BLE001
            print(f"     !! ERROR: {e}\n")
            results.append({"id": qid, "group": case["group"], "question": q,
                            "answer": "", "judge": {"result": "Fail",
                            "one_line_note": f"error: {e}", "likely_failure_stage": "error"}})

    finished = datetime.utcnow()
    counts = Counter(r["judge"]["result"] for r in results)
    n = len(results)

    # --- JSON ---
    (BASELINE_DIR / "v2_battery_results.json").write_text(json.dumps({
        "system": "v2", "gen_model": args.gen_model, "judge_model": args.judge_model,
        "started_utc": started.isoformat() + "Z", "n": n,
        "counts": dict(counts), "results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- Scored markdown ---
    lines = [
        "# v2 — Vectera Self-Evaluation Battery (LLM-judge graded)", "",
        f"_Generation: {args.gen_model} · Judge: {args.judge_model} (separate model). "
        f"Run {started.isoformat()}Z._", "",
        "Judge grades against each question's diagnostic criteria + retrieved context "
        "(not its own memory). **Fails/Partials should be human-reviewed.**", "",
        "## Headline", "",
        f"| Outcome | v2 | v1 |", "|---|---|---|",
        f"| **Pass** | {counts.get('Pass',0)}/{n} | {V1_BASELINE['Pass']}/24 |",
        f"| **Partial** | {counts.get('Partial',0)}/{n} | {V1_BASELINE['Partial']}/24 |",
        f"| **Fail** | {counts.get('Fail',0)}/{n} | {V1_BASELINE['Fail']}/24 |", "",
        f"**v1 → v2 delta: Pass {V1_BASELINE['Pass']} → {counts.get('Pass',0)}**", "",
        "---", "",
    ]
    cur_group = None
    for r in results:
        if r.get("group") != cur_group:
            cur_group = r.get("group")
            lines += [f"## Group {cur_group} — {r.get('group_name','')}", ""]
        j = r["judge"]
        badge = {"Pass": "✅", "Partial": "🟡", "Fail": "❌"}.get(j["result"], "❓")
        ans = (r.get("answer") or "").strip().replace("\n", " ")
        if len(ans) > 600:
            ans = ans[:600] + "…"
        cites = ", ".join(
            f"[{i+1}] {s['company']} p.{s['page_number']}"
            for i, s in enumerate(r.get("sources", [])) if s.get("cited"))
        lines += [
            f"### {badge} {r['id']} — {r['question']}",
            f"**Result:** {j['result']} (score {j.get('score')}) · "
            f"**failure stage:** {j.get('likely_failure_stage')} · "
            f"**citations:** {j.get('citation_quality')}",
            f"**Note:** {j.get('one_line_note','')}", "",
            f"**Answer:** {ans}", "",
            f"**Cited sources:** {cites or '—'}", "",
        ]
        if j.get("critical_facts_missing"):
            lines.append(f"**Missing:** {'; '.join(map(str, j['critical_facts_missing']))}")
        lines += ["", "---", ""]
    (BASELINE_DIR / "v2_battery_scored.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n=== v2 RESULT ===")
    print(f"  Pass {counts.get('Pass',0)} / Partial {counts.get('Partial',0)} / Fail {counts.get('Fail',0)}  (n={n})")
    print(f"  v1 was: Pass {V1_BASELINE['Pass']} / Partial {V1_BASELINE['Partial']} / Fail {V1_BASELINE['Fail']}")
    print(f"  saved: baselines/v2_battery_results.json + v2_battery_scored.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
