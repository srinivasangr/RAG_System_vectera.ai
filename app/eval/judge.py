"""LLM-as-judge for the Vectera battery.

Grades a RAG answer against the battery's diagnostic criteria + the retrieved
context — NOT the judge model's own memory. Independence safeguards:
  - the judge is a DIFFERENT model from the one that generated the answer
    (generation = gemini-3.1-flash-lite; judge = gemini-2.5-flash by default);
  - it grades the answer as an external artifact (never its own output);
  - ground truth is the battery's "what it diagnoses" text + sources;
  - every Fail/Partial is meant for human review (first-pass automation only).
"""

from __future__ import annotations

import json
import logging
import re

from rag_system.llm_providers import get_llm
from rag_system.llm_providers.base import Message

log = logging.getLogger(__name__)

JUDGE_MODEL = ("gemini", "gemini-2.5-flash")  # different from the generation model

_SYSTEM = """You are evaluating a RAG system's answer. Do NOT answer the question
yourself. Your only job is to grade whether the CANDIDATE ANSWER satisfies the
EXPECTED BEHAVIOR, using the retrieved sources as the grounding context.

Grade strictly:
- Pass:    all critical facts correct, correctly scoped/dated, well cited, no
           hallucination, and it handled the specific challenge the question tests.
- Partial: directionally correct but misses an important qualifier, version,
           citation, source, footnote, or limitation.
- Fail:    wrong, unsupported, hallucinated, blends sources, misses a required
           document/contradiction, or fabricates when it should refuse.

A truthful "I don't have enough information" is better than a confident wrong
answer: grade an honest refusal as Partial (not Fail) when the data genuinely
isn't in the sources, but Fail if the data WAS available and it refused anyway.

Return STRICT JSON only:
{
  "result": "Pass | Partial | Fail",
  "score": 0,
  "critical_facts_found": [],
  "critical_facts_missing": [],
  "incorrect_or_unsupported_claims": [],
  "citation_quality": "good | weak | missing | wrong",
  "likely_failure_stage": "retrieval | chunking | metadata | table_parsing | vision_ocr | reranking | generation | citation | none",
  "one_line_note": ""
}
score is 0-5 (5=perfect Pass, 3=Partial, 0-2=Fail). likely_failure_stage is
"none" for a Pass."""


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    s = text.find("{")
    if s == -1:
        return None
    depth = 0
    for i in range(s, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[s:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def judge_answer(*, question, diagnoses, answer, citations, retrieved_summary,
                 llm=None) -> dict:
    """Grade one answer. retrieved_summary = compact text of the sources used."""
    llm = llm or get_llm(*JUDGE_MODEL)
    user = (
        f"QUESTION:\n{question}\n\n"
        f"EXPECTED BEHAVIOR / WHAT THIS QUESTION DIAGNOSES:\n{diagnoses}\n\n"
        f"CANDIDATE ANSWER:\n{answer}\n\n"
        f"CITATIONS USED: {citations}\n\n"
        f"RETRIEVED SOURCES (grounding context the answer had access to):\n{retrieved_summary}"
    )
    try:
        # thinking_budget=0 disables Gemini 2.5 'thinking' so the JSON isn't
        # truncated. Falls back gracefully for providers without the kwarg.
        try:
            reply = llm.generate(
                [Message(role="system", content=_SYSTEM), Message(role="user", content=user)],
                temperature=0.0, max_tokens=1024, thinking_budget=0,
            )
        except TypeError:
            reply = llm.generate(
                [Message(role="system", content=_SYSTEM), Message(role="user", content=user)],
                temperature=0.0, max_tokens=2048,
            )
        data = _extract_json(reply) or {}
    except Exception as e:  # noqa: BLE001
        log.warning("judge failed: %s", e)
        data = {}

    result = (data.get("result") or "Fail").strip().title()
    if result not in ("Pass", "Partial", "Fail"):
        result = "Fail"
    return {
        "result": result,
        "score": data.get("score"),
        "critical_facts_found": data.get("critical_facts_found") or [],
        "critical_facts_missing": data.get("critical_facts_missing") or [],
        "incorrect_or_unsupported_claims": data.get("incorrect_or_unsupported_claims") or [],
        "citation_quality": data.get("citation_quality") or "unknown",
        "likely_failure_stage": data.get("likely_failure_stage") or "unknown",
        "one_line_note": data.get("one_line_note") or "",
    }
