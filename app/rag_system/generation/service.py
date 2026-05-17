"""High-level query() entrypoint used by Streamlit and the eval harness.

  query(question, filters, provider, model) -> Answer

The Answer carries the generated text, parsed citations, retrieval debug
info, and timing. We also log every query to Snowflake `query_log`.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from rag_system.config import settings
from rag_system.generation.citations import Citation, resolve_citations
from rag_system.generation.prompt import (
    SYSTEM_PROMPT,
    build_user_prompt,
    format_sources,
)
from rag_system.llm_providers import Message, get_llm
from rag_system.retrieval.filters import RetrievalFilters
from rag_system.retrieval.hybrid import RetrievedChunk, retrieve
from rag_system.storage.repository import log_query


@dataclass
class Answer:
    question: str
    answer: str
    citations: list[Citation]
    retrieved: list[RetrievedChunk]
    llm_provider: str
    llm_model: str
    latency_ms: int
    retrieval_ms: int
    generation_ms: int


def query(
    question: str,
    *,
    filters: RetrievalFilters | None = None,
    provider: str | None = None,
    model: str | None = None,
    top_k: int | None = None,
    max_tokens: int = 1200,
    write_log: bool = True,
) -> Answer:
    filters = filters or RetrievalFilters()
    t0 = time.perf_counter()

    # 1) Retrieve
    t_r0 = time.perf_counter()
    retrieved = retrieve(question, filters=filters, top_k=top_k)
    retrieval_ms = int((time.perf_counter() - t_r0) * 1000)

    if not retrieved:
        ans_text = (
            "I don't have enough information in the provided documents to "
            "answer that. (Nothing was retrieved for this query.)"
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        return Answer(
            question=question, answer=ans_text, citations=[],
            retrieved=[], llm_provider=provider or settings.llm_provider,
            llm_model=model or settings.llm_model,
            latency_ms=latency_ms, retrieval_ms=retrieval_ms, generation_ms=0,
        )

    # 2) Build prompt
    sources = format_sources(retrieved)
    user_prompt = build_user_prompt(question, sources)

    # 3) Generate
    llm = get_llm(provider=provider, model=model)
    t_g0 = time.perf_counter()
    ans_text = llm.generate(
        [
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    generation_ms = int((time.perf_counter() - t_g0) * 1000)

    # 4) Parse citations
    citations = resolve_citations(ans_text, sources)

    latency_ms = int((time.perf_counter() - t0) * 1000)
    ans = Answer(
        question=question, answer=ans_text, citations=citations,
        retrieved=retrieved,
        llm_provider=llm.name, llm_model=getattr(llm, "_model", "?"),
        latency_ms=latency_ms, retrieval_ms=retrieval_ms,
        generation_ms=generation_ms,
    )

    # 5) Log — non-fatal but no longer silently swallowed
    if write_log:
        try:
            log_query(
                question=question,
                filters={k: (v.isoformat() if hasattr(v, "isoformat") else v)
                         for k, v in asdict(filters).items()},
                retrieved_ids=[c.chunk_id for c in retrieved],
                answer=ans_text,
                llm_provider=ans.llm_provider,
                llm_model=ans.llm_model,
                latency_ms=latency_ms,
            )
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "query_log write failed: %s: %s", type(e).__name__, e,
            )
    return ans
