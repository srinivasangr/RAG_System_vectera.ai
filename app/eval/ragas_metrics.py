"""RAGAS-style RAG evaluation metrics using LLM-as-judge.

Implements four standard metrics from the RAGAS framework
(https://github.com/explodinggradients/ragas), simplified for our use case:

    1. faithfulness        — are the answer's claims grounded in the sources?
    2. answer_relevance    — is the answer actually answering the question?
    3. context_precision   — of the retrieved chunks, what fraction is relevant?
    4. context_recall      — of the must-have facts, what fraction is covered
                              by the retrieved chunks?

All metrics return a float in [0.0, 1.0] (higher is better). They share a
single LLM judge — by default the same Cerebras model used for answer
generation, since it's free and fast. Each metric call costs 1 LLM round trip.

Why LLM-as-judge over RAGAS's official library:
    * No extra dependency. RAGAS pulls openai + langchain + a chunky stack.
    * Single judge model (configurable). Easy to swap.
    * Transparent prompts — what's measured is visible right here.

Prompts are intentionally terse to minimise reasoning-token spend on
gpt-oss-style models.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable

from rag_system.llm_providers import Message, get_llm

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge helpers
# ---------------------------------------------------------------------------
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _ask_judge(prompt: str, *, max_tokens: int = 600) -> str:
    """Run the configured judge LLM with no system prompt."""
    llm = get_llm()
    return llm.generate(
        [Message(role="user", content=prompt)],
        temperature=0.0,
        max_tokens=max_tokens,
    )


def _parse_json(raw: str) -> dict | None:
    """Pull the first JSON object out of an LLM reply. Returns None on failure."""
    if not raw:
        return None
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _format_chunks(chunks, *, max_chars: int = 600) -> str:
    """Render chunks as [N] Source ... blocks the judge can reason over."""
    out = []
    for i, c in enumerate(chunks, start=1):
        text = (c.text or "").strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        out.append(f"[{i}] {text}")
    return "\n\n".join(out)


# ---------------------------------------------------------------------------
# 1. Faithfulness
# ---------------------------------------------------------------------------
_FAITHFULNESS_PROMPT = """\
You are evaluating whether an answer is supported by the source documents.

QUESTION:
{question}

ANSWER:
{answer}

SOURCES:
{sources}

Step 1: Break the answer into atomic factual claims. Skip filler / hedging.
Step 2: For each claim, decide if AT LEAST ONE source directly supports it.
        - score 1: supported
        - score 0: not supported (hallucinated, or only partial / inferred)

Return ONLY this JSON, no prose:
{{
  "claims": [{{"claim": "<text>", "score": 0 | 1}}],
  "faithfulness": <float between 0.0 and 1.0 = mean of scores>
}}
"""


def judge_faithfulness(question: str, answer: str, retrieved_chunks) -> float | None:
    """Fraction of the answer's atomic claims that are grounded in sources."""
    if not answer.strip() or not retrieved_chunks:
        return None
    prompt = _FAITHFULNESS_PROMPT.format(
        question=question.strip(),
        answer=answer.strip(),
        sources=_format_chunks(retrieved_chunks),
    )
    raw = _ask_judge(prompt, max_tokens=1500)
    parsed = _parse_json(raw)
    if not parsed:
        log.warning("faithfulness: judge returned unparseable JSON")
        return None
    val = parsed.get("faithfulness")
    if isinstance(val, (int, float)):
        return max(0.0, min(1.0, float(val)))
    # Fall back to averaging claim scores if the aggregate field is missing
    claims = parsed.get("claims") or []
    if claims:
        scores = [c.get("score") for c in claims if isinstance(c.get("score"), (int, float))]
        if scores:
            return sum(scores) / len(scores)
    return None


# ---------------------------------------------------------------------------
# 2. Answer relevance
# ---------------------------------------------------------------------------
_RELEVANCE_PROMPT = """\
Rate how directly the ANSWER addresses the QUESTION.

QUESTION:
{question}

ANSWER:
{answer}

Scoring:
  1.0 — directly and completely answers the question
  0.7 — substantially answers but misses something asked
  0.5 — partially relevant; covers the topic but not the specific question
  0.3 — tangential
  0.0 — does not answer the question (or is a refusal when one was unwarranted)

A correct refusal ("I don't have enough information...") to an unanswerable
question scores 1.0 (it is the right response).

Return ONLY this JSON:
{{"relevance": <float between 0.0 and 1.0>, "reason": "<one short sentence>"}}
"""


def judge_answer_relevance(question: str, answer: str) -> float | None:
    if not answer.strip():
        return None
    prompt = _RELEVANCE_PROMPT.format(question=question.strip(), answer=answer.strip())
    raw = _ask_judge(prompt, max_tokens=400)
    parsed = _parse_json(raw)
    if not parsed:
        log.warning("answer_relevance: judge returned unparseable JSON")
        return None
    val = parsed.get("relevance")
    if isinstance(val, (int, float)):
        return max(0.0, min(1.0, float(val)))
    return None


# ---------------------------------------------------------------------------
# 3. Context precision
# ---------------------------------------------------------------------------
_PRECISION_PROMPT = """\
For each retrieved CHUNK below, decide whether it is RELEVANT to answering the QUESTION.

QUESTION:
{question}

CHUNKS:
{chunks}

A chunk is RELEVANT (score 1) if it contains information that helps answer the
question, even partially. Otherwise score 0.

Return ONLY this JSON:
{{"scores": [<0|1 for chunk 1>, <0|1 for chunk 2>, ...]}}

The array length must equal the number of chunks.
"""


def judge_context_precision(question: str, retrieved_chunks) -> float | None:
    if not retrieved_chunks:
        return None
    prompt = _PRECISION_PROMPT.format(
        question=question.strip(),
        chunks=_format_chunks(retrieved_chunks),
    )
    raw = _ask_judge(prompt, max_tokens=400)
    parsed = _parse_json(raw)
    if not parsed:
        log.warning("context_precision: judge returned unparseable JSON")
        return None
    scores = parsed.get("scores") or []
    if not isinstance(scores, list) or not scores:
        return None
    nums = [s for s in scores if isinstance(s, (int, float))]
    if not nums:
        return None
    return sum(nums) / len(nums)


# ---------------------------------------------------------------------------
# 4. Context recall (heuristic + LLM judge for absent must_contain)
# ---------------------------------------------------------------------------
def judge_context_recall(
    must_contain: Iterable[str],
    retrieved_chunks,
) -> float | None:
    """Fraction of `must_contain` phrases that appear somewhere in the
    retrieved chunks. Pure substring check — no LLM call needed. Returns None
    if no must_contain phrases are defined (i.e. unmeasurable)."""
    phrases = [p for p in (must_contain or []) if p and p.strip()]
    if not phrases:
        return None
    if not retrieved_chunks:
        return 0.0
    joined = "\n".join((c.text or "").lower() for c in retrieved_chunks)
    hits = sum(1 for p in phrases if p.lower() in joined)
    return hits / len(phrases)


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------
@dataclass
class RagasScores:
    faithfulness: float | None
    answer_relevance: float | None
    context_precision: float | None
    context_recall: float | None

    def as_dict(self) -> dict:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevance": self.answer_relevance,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
        }


def evaluate_one(
    *,
    question: str,
    answer: str,
    retrieved_chunks,
    must_contain: Iterable[str] = (),
) -> RagasScores:
    """Run all four RAGAS-style metrics for a single question. Each metric is
    independently None if it can't be computed (e.g. empty answer)."""
    return RagasScores(
        faithfulness=judge_faithfulness(question, answer, retrieved_chunks),
        answer_relevance=judge_answer_relevance(question, answer),
        context_precision=judge_context_precision(question, retrieved_chunks),
        context_recall=judge_context_recall(must_contain, retrieved_chunks),
    )
