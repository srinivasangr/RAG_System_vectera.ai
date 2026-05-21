"""Query router — the first stage of multi-stage retrieval.

One LLM call analyzes the query and returns a plan: intent, the entities/
attributes mentioned, decomposed sub-queries, temporal hints, and which
sources to hit. It is DOMAIN-AGNOSTIC: it carries no hardcoded entities or
document types — instead it is handed the live CORPUS PROFILE (distinct
doc_types / entities / date range read from the DB) so the same prompt works
on any corpus.

Failure-safe: bad/empty JSON → a sensible default plan (lookup, single
sub-query = the original query, no boosts).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime

from rag_system.llm_providers import get_llm
from rag_system.llm_providers.base import Message
from rag_system.storage import repository_v2 as repo

log = logging.getLogger(__name__)

INTENTS = {"lookup", "compare", "delta", "recency", "enumerate", "refuse"}


@dataclass
class RoutePlan:
    intent: str = "lookup"
    entities: list[str] = field(default_factory=list)
    attributes: list[str] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    as_of_date_filter: date | None = None
    prefer_recent: bool = False
    doc_type_preference: list[str] = field(default_factory=list)
    needs_tables: bool = True
    needs_charts: bool = True
    raw: dict = field(default_factory=dict)


_SYSTEM = """You are a query analyzer for a document question-answering system.
You are given a CORPUS PROFILE (what the current corpus contains) and a user
QUERY. Use ONLY the profile — assume no particular industry or domain.

Return STRICT JSON only:
{
  "intent": "lookup | compare | delta | recency | enumerate | refuse",
  "entities":  ["named entities mentioned IN THE QUERY; extract freely, no fixed list"],
  "attributes":["metrics/quantities mentioned IN THE QUERY, e.g. revenue, occupancy"],
  "sub_queries":["decompose multi-entity/multi-hop queries into focused sub-queries; else echo the query"],
  "as_of_date_filter": "YYYY-MM-DD or null (only if the query names a specific date)",
  "prefer_recent": true/false,
  "doc_type_preference": ["ordered subset of the profile's doc_types relevant to this query, or []"],
  "needs_tables": true/false,
  "needs_charts": true/false
}

INTENT DEFINITIONS (domain-agnostic):
- lookup    : one fact about one entity
- compare   : contrast 2+ entities or attributes -> one sub_query per entity
- delta     : what changed between two documents/versions -> one sub_query per side
- recency   : wants the most current value ("latest","current","now") -> prefer_recent=true
- enumerate : wants a list that may be split across pages -> sub_queries broaden the search
- refuse    : not answerable from any document corpus

GUIDANCE:
- entities/attributes are EXTRACTED from the query; never use a built-in list.
- needs_tables=true if the query asks for numbers/comparisons/rows.
- needs_charts=true if it asks about a figure/visual/share/breakdown/map.
- Keep sub_queries short and specific. 1 for simple lookups; up to ~6.
"""


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


def _parse_date(s) -> date | None:
    if not s or not isinstance(s, str):
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def build_corpus_profile() -> dict:
    """Live corpus metadata fed to the router. The ONLY source of domain info."""
    try:
        return repo.corpus_profile()
    except Exception as e:  # noqa: BLE001
        log.warning("corpus_profile failed: %s", e)
        return {"n_documents": 0, "doc_types": [], "entities": [], "date_range": [None, None]}


def route(query: str, *, llm=None, profile: dict | None = None) -> RoutePlan:
    """Analyze a query into a retrieval plan (1 LLM call, with safe fallback)."""
    profile = profile or build_corpus_profile()
    llm = llm or get_llm()
    user = (
        f"CORPUS PROFILE:\n{json.dumps(profile, default=str)}\n\n"
        f"QUERY:\n{query}"
    )
    data = {}
    try:
        reply = llm.generate(
            [Message(role="system", content=_SYSTEM), Message(role="user", content=user)],
            temperature=0.0, max_tokens=500,
        )
        data = _extract_json(reply) or {}
    except Exception as e:  # noqa: BLE001
        log.warning("router LLM failed: %s", e)

    intent = (data.get("intent") or "lookup").strip().lower()
    if intent not in INTENTS:
        intent = "lookup"
    subs = [str(s).strip() for s in (data.get("sub_queries") or []) if str(s).strip()]
    if not subs:
        subs = [query]

    plan = RoutePlan(
        intent=intent,
        entities=[str(e).strip() for e in (data.get("entities") or []) if str(e).strip()],
        attributes=[str(a).strip() for a in (data.get("attributes") or []) if str(a).strip()],
        sub_queries=subs[:6],
        as_of_date_filter=_parse_date(data.get("as_of_date_filter")),
        prefer_recent=bool(data.get("prefer_recent", False)) or intent == "recency",
        doc_type_preference=[str(d).strip() for d in (data.get("doc_type_preference") or [])],
        needs_tables=bool(data.get("needs_tables", True)),
        needs_charts=bool(data.get("needs_charts", True)),
        raw=data,
    )
    log.info("route(%r) -> intent=%s entities=%s subs=%d recent=%s",
             query[:60], plan.intent, plan.entities, len(plan.sub_queries), plan.prefer_recent)
    return plan
