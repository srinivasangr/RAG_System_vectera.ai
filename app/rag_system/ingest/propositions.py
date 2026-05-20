"""Proposition extraction — decompose prose chunks into atomic facts.

Each prose chunk is decomposed by an LLM into self-contained single-fact
statements. These are the primary DENSE retrieval target: a clean, one-fact
embedding matches a question far better than a noisy mixed-topic chunk.

Domain-agnostic: the prompt asks for atomic facts with units/dates/qualifiers
preserved and entity names spelled out — no assumption about the subject matter.

Robustness: if the LLM call fails or returns nothing usable, we fall back to a
single proposition equal to the chunk text, so dense retrieval still has a
target for that chunk.
"""

from __future__ import annotations

import json
import logging
import re

from rag_system.llm_providers import get_llm
from rag_system.llm_providers.base import Message

log = logging.getLogger(__name__)

_SYSTEM = """You decompose a passage into atomic factual statements for retrieval.

Return STRICT JSON only: {"propositions": ["...", "..."]}

Each statement must:
- carry exactly ONE fact
- be self-contained: spell out the entity/subject (no "it"/"they"/"the company")
- preserve original numbers, units, dates, and any scope qualifier
  (e.g. "including assets under construction", "as of Dec 31, 2025")
- copy facts faithfully — do not infer, compute, or add anything not stated

If the passage has no factual content (e.g. a heading or a photo caption),
return {"propositions": []}.
"""


def _extract_json(text: str) -> dict | None:
    if not text:
        return None
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_propositions(chunk_text: str, *, llm=None, max_props: int = 15) -> list[str]:
    """Return a list of atomic statements for one chunk (LLM, with fallback)."""
    text = (chunk_text or "").strip()
    if not text:
        return []
    llm = llm or get_llm()
    try:
        reply = llm.generate(
            [Message(role="system", content=_SYSTEM),
             Message(role="user", content=text[:4000])],
            temperature=0.0,
            max_tokens=800,
        )
        data = _extract_json(reply) or {}
        props = data.get("propositions") or []
        props = [str(p).strip() for p in props if str(p).strip()]
        if props:
            return props[:max_props]
    except Exception as e:  # noqa: BLE001
        log.warning("proposition extraction failed: %s", e)
    # Fallback: the chunk itself becomes one proposition so dense retrieval works.
    return [text[:1000]]
