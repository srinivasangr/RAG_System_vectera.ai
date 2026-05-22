"""Conflict-aware, cited generation (consumes the v2 retrieval result).

Domain-agnostic: the prompt encodes behaviors (cite, refuse, surface conflicts,
flag staleness, preserve qualifiers) — all corpus-specific content (entities,
attributes, doc types, dates) arrives only in the SOURCE block at runtime.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import date

from rag_system.config import settings
from rag_system.generation.citations import parse_citation_ns
from rag_system.llm_providers import get_llm
from rag_system.llm_providers.base import Message
from rag_system.retrieval import router as router_mod
from rag_system.retrieval.filters import RetrievalFilters
from rag_system.retrieval.pipeline import Source, retrieve

log = logging.getLogger(__name__)

REFUSAL = "I don't have enough information in the provided documents to answer that."

# ---------------------------------------------------------------------------
# Input guardrails — cheap gating so off-topic input doesn't run the whole
# retrieval+generation loop.
# ---------------------------------------------------------------------------
_GREETING_RE = re.compile(
    r"^\s*(hi+|hey+|hello+|yo|good\s*(morning|afternoon|evening|night)|"
    r"thanks?|thank\s*you|thx|ty|ok|okay|cool|nice|great|bye|"
    r"how\s*are\s*you|who\s*are\s*you|what\s*can\s*you\s*do)\b[\s.!,?]*$",
    re.IGNORECASE,
)
_GREETING_MSG = (
    "Hi! I'm a document Q&A assistant for the documents currently loaded in this "
    "corpus. Ask me anything about them — e.g. financial metrics, guidance, "
    "strategy, occupancy, or comparisons across the companies."
)
_OUT_OF_SCOPE_MSG = (
    "I can only answer questions grounded in the loaded documents, and that "
    "doesn't appear to be answerable from them. Try asking about the companies, "
    "metrics, or topics covered in the corpus."
)


def _quick_guard(query: str) -> str | None:
    """Return a canned reply for trivial/non-question input (no LLM call)."""
    q = (query or "").strip()
    if len(q) < 2:
        return "Please type a question about the documents."
    if _GREETING_RE.match(q):
        return _GREETING_MSG
    return None

SYSTEM_PROMPT = f"""You are a careful research analyst. Answer the QUESTION using
ONLY the numbered SOURCES provided. Apply these rules in order of precedence:

[1 GROUNDING] Every factual, numeric, or quoted claim MUST carry a [N] citation
to the source it came from. If you can't cite it from the sources, don't say it.

[2 REFUSAL] If the sources lack the answer, say exactly:
"{REFUSAL}" then briefly note what the sources DO contain.

[3 CONFLICTS] You may be given a CONFLICTS note (same entity, multiple as-of
dates). When sources disagree on the same fact, present EVERY value with full
attribution — "{{value}} per {{doc_type}} as of {{date}} [N]" — do NOT pick one
silently and do NOT average. If a footnote/qualifier explains the difference
(methodology/basis change), say so.

[4 STALENESS] If a cited source is much older than the others (e.g. >2 years),
flag it: "As of {{year}}, {{source}} reports ... (this may be outdated)".

[5 COMPLETENESS] For "for each X" / comparison questions, address every entity
asked about. If an entity has no supporting source, state:
"Not disclosed in the provided {{entity}} materials." Never silently omit one.

[6 ATTRIBUTION] Make citations specific — tie each to the right source number;
prefer citing the document type + date so the reader can judge recency.

[7 DELTA] If asked what changed between two documents/versions, structure the
answer as: (a) what's stable, (b) what changed with old AND new values, (c)
net-new disclosures.

[8 QUALIFIERS] Keep scope qualifiers attached to numbers ("including X",
"as of date Y", "top-N basis"). Two numbers that look equal but have different
qualifiers are NOT the same — say so.

Be concise and factual. Use the citation markers like [1], [2,3]."""


@dataclass
class Answer:
    question: str
    answer: str
    plan: object                       # router.RoutePlan
    sources: list[Source]              # all retrieved (numbered 1..N)
    cited_numbers: list[int]
    conflicts: list[dict]
    llm_provider: str
    llm_model: str
    timings: dict = field(default_factory=dict)
    reasoning: str | None = None
    trace: dict = field(default_factory=dict)


# Generation provider fallback chain (architecture §7). On a transient failure
# (503 high-demand / rate limit) we retry the next engine so a single query
# doesn't fail. Both engines are paid Gemini (reliable); Cerebras free-tier was
# dropped — it rate-limited and never won the race in practice.
_FALLBACK_CHAIN = [("gemini", "gemini-3.1-flash-lite"), ("gemini", "gemini-2.5-flash")]


def _generate_with_fallback(messages, primary_llm, *, max_tokens):
    """Try primary, then the fallback chain. Returns (text, provider, model, chain)."""
    chain_log = []
    attempts = [("primary", primary_llm)]
    for prov, mdl in _FALLBACK_CHAIN:
        attempts.append((f"{prov}/{mdl or 'default'}", (prov, mdl)))
    last_err = None
    for label, spec in attempts:
        try:
            llm = spec if not isinstance(spec, tuple) else get_llm(spec[0], spec[1])
        except Exception as e:  # noqa: BLE001 — provider not configured; skip
            chain_log.append({"engine": label, "ok": False, "error": f"init: {e}"})
            continue
        try:
            t = time.perf_counter()
            text = llm.generate(messages, temperature=0.0, max_tokens=max_tokens)
            usage = getattr(llm, "last_usage", {}) or {}
            chain_log.append({"engine": label, "ok": True,
                              "ms": int((time.perf_counter() - t) * 1000),
                              "tokens": usage})
            return text, getattr(llm, "name", label), getattr(llm, "_default_model", ""), chain_log
        except Exception as e:  # noqa: BLE001
            last_err = e
            chain_log.append({"engine": label, "ok": False, "error": str(e)[:120]})
            log.warning("generation engine %s failed: %s", label, str(e)[:120])
    raise last_err or RuntimeError("all generation engines failed")


def _fmt_date(d) -> str:
    return d.isoformat() if isinstance(d, date) else "undated"


def format_sources(sources: list[Source], conflicts: list[dict]) -> str:
    """Render the numbered SOURCE block (+ CONFLICTS note) for the LLM."""
    lines: list[str] = []
    for i, s in enumerate(sources, start=1):
        head = (f"[{i}] {s.company or 'Unknown'} — {s.doc_type or 'document'} — "
                f"p.{s.page_number} — as of {_fmt_date(s.as_of_date)}")
        if s.conflict_group:
            head += "  (⚠ conflicting versions present)"
        lines.append(head)
        if s.slide_title:
            lines.append(f"SLIDE: {s.slide_title}")
        lines.append((s.text or "").strip()[:1800])
        lines.append("---")
    block = "\n".join(lines)

    if conflicts:
        notes = "; ".join(
            f"{c['company']} appears with multiple as-of dates ({', '.join(c['as_of_dates'])})"
            for c in conflicts
        )
        block += f"\n\nCONFLICTS: {notes}\nPresent each value with its date; do not blend."
    return block


def answer_query(
    query: str,
    *,
    llm=None,
    provider: str | None = None,
    model: str | None = None,
    top_k: int | None = None,
    filters: RetrievalFilters | None = None,
    progress_cb=None,
    write_log: bool = True,
) -> Answer:
    """End-to-end: retrieve → conflict-aware generate → parse citations."""
    t0 = time.perf_counter()
    llm = llm or get_llm(provider, model)

    def _emit(stage):
        if progress_cb:
            try:
                progress_cb(stage)
            except Exception:
                pass

    def _short(answer_text, plan, intent):
        """Build a short-circuit answer (no retrieval/generation)."""
        return Answer(
            question=query, answer=answer_text,
            plan=plan or router_mod.RoutePlan(intent=intent),
            sources=[], cited_numbers=[], conflicts=[],
            llm_provider=getattr(llm, "name", provider or "?"),
            llm_model=model or settings.llm_model,
            timings={"total_ms": int((time.perf_counter() - t0) * 1000)},
            trace={"short_circuit": intent},
        )

    # GUARDRAIL 1 — trivial/greeting input: instant canned reply, 0 LLM calls.
    canned = _quick_guard(query)
    if canned is not None:
        _emit("done")
        return _short(canned, None, "chitchat")

    # GUARDRAIL 2 — classify once; if not answerable from the corpus, decline
    # WITHOUT running the heavy retrieve+rerank+generate loop.
    _emit("routing")
    _t_route = time.perf_counter()
    plan = router_mod.route(query, llm=llm)
    route_ms = int((time.perf_counter() - _t_route) * 1000)
    router_usage = dict(getattr(llm, "last_usage", {}) or {})  # router call's tokens
    if plan.intent == "refuse":
        return _short(_OUT_OF_SCOPE_MSG, plan, "refuse")

    # 1) Retrieve (reuse the plan we just computed — no second router call)
    rr = retrieve(query, filters=filters, top_k=top_k, llm=llm,
                  plan=plan, progress_cb=progress_cb)

    # 2) No sources → honest refusal
    if not rr.sources:
        return Answer(
            question=query, answer=REFUSAL + " (Nothing relevant was retrieved.)",
            plan=rr.plan, sources=[], cited_numbers=[], conflicts=[],
            llm_provider=getattr(llm, "name", provider or settings.llm_provider),
            llm_model=model or settings.llm_model,
            timings={**rr.timings, "generate_ms": 0},
        )

    # 3) Generate
    src_block = format_sources(rr.sources, rr.conflicts)
    user = f"QUESTION:\n{query}\n\nSOURCES:\n{src_block}"
    messages = [Message(role="system", content=SYSTEM_PROMPT),
                Message(role="user", content=user)]
    _emit("generating")
    tg = time.perf_counter()
    gen_provider, gen_model, chain = getattr(llm, "name", "?"), model or "", []
    try:
        text, gen_provider, gen_model, chain = _generate_with_fallback(
            messages, llm, max_tokens=1500)
    except Exception as e:  # noqa: BLE001
        log.warning("all generation engines failed: %s", e)
        text = REFUSAL + f" (Generation error: {e})"
    gen_ms = int((time.perf_counter() - tg) * 1000)

    cited = parse_citation_ns(text)
    timings = {**rr.timings, "generate_ms": gen_ms,
               "route_ms": route_ms,  # actual router time (retrieve reused the plan)
               "total_ms": int((time.perf_counter() - t0) * 1000)}
    log.info("answer(%r): %d sources, cited=%s, engine=%s, %dms",
             query[:50], len(rr.sources), cited, gen_provider, timings["total_ms"])
    ans = Answer(
        question=query, answer=text.strip(), plan=rr.plan, sources=rr.sources,
        cited_numbers=cited, conflicts=rr.conflicts,
        llm_provider=gen_provider, llm_model=gen_model or (model or settings.llm_model),
        timings=timings,
    )
    ans.timings["provider_chain"] = chain

    # ---- Build the observability trace (spans + tokens + texts) ----
    gen_usage = {}
    for entry in chain:
        if entry.get("ok") and entry.get("tokens"):
            gen_usage = entry["tokens"]
            break

    def _tot(u):
        return (u or {}).get("total_tokens") or 0
    total_tokens = _tot(router_usage) + _tot(gen_usage)

    ans.trace = {
        "plan": {
            "intent": getattr(rr.plan, "intent", None),
            "entities": getattr(rr.plan, "entities", []),
            "attributes": getattr(rr.plan, "attributes", []),
            "sub_queries": getattr(rr.plan, "sub_queries", []),
            "needs_tables": getattr(rr.plan, "needs_tables", None),
            "needs_charts": getattr(rr.plan, "needs_charts", None),
            "prefer_recent": getattr(rr.plan, "prefer_recent", None),
        },
        "spans": [
            {"name": "route (router LLM)", "ms": timings.get("route_ms"), "tokens": router_usage},
            {"name": "retrieve (dense+lexical+structured → RRF)",
             "ms": timings.get("retrieve_ms"), "n_candidates": timings.get("n_candidates")},
            {"name": "rerank (cross-encoder)", "ms": timings.get("rerank_ms")},
            {"name": "generate (answer LLM)", "ms": timings.get("generate_ms"), "tokens": gen_usage},
        ],
        "provider_chain": chain,
        "tokens": {"router": router_usage, "generation": gen_usage, "total": total_tokens},
        "prompt_preview": user[:6000],
        "system_prompt_chars": len(SYSTEM_PROMPT),
        "n_sources": len(rr.sources),
    }

    # Best-effort query log (never break the answer on a logging error).
    if write_log:
        try:
            from rag_system.storage import repository as repo3
            stage_t = {k: v for k, v in timings.items() if k != "provider_chain"}
            repo3.log_query(
                question=query, answer=ans.answer, intent=getattr(rr.plan, "intent", None),
                sub_queries=getattr(rr.plan, "sub_queries", []),
                retrieved_ids=[s.parent_id for s in rr.sources],
                retrieval_stages=stage_t, conflict_pairs=rr.conflicts,
                provider_chain=chain, llm_provider=ans.llm_provider,
                llm_model=ans.llm_model, total_latency_ms=timings.get("total_ms"),
                doc_ids=(filters.doc_ids if filters else []),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("query_log write failed: %s", e)
    return ans
