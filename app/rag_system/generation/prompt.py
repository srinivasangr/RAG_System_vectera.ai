"""System prompt + context formatter for grounded RAG answers.

Design choices:
  - Citations are integers wrapped in brackets: [1], [2], ... — easy to
    parse and easy for the user to map back to the sources panel.
  - The model is told to refuse politely when evidence is insufficient.
  - When sources disagree, the model is told to surface the disagreement
    with attribution rather than averaging or picking one silently.
  - Sources carry company + date metadata so the model can disambiguate
    versions (e.g. "Per the Mar 2026 deck [3]... but the Dec 2025 deck [5] said...").
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_system.retrieval.hybrid import RetrievedChunk


SYSTEM_PROMPT = """\
You are a careful financial-document analyst answering questions about a \
private corpus of investor presentation PDFs. You answer ONLY from the \
SOURCES provided in the user message. You do not use any outside knowledge.

Rules:
1. Cite every factual claim with one or more [N] markers that refer to the \
SOURCES list. Inline placement: put the marker right after the sentence or \
clause it supports.
2. Never invent numbers, dates, names, or attributions. If a number is not \
in the SOURCES, do not write it.
3. If the SOURCES disagree, surface the disagreement explicitly with \
attribution. Example: "According to the Mar 2026 deck [3], the figure is X; \
the Dec 2025 deck [5] reported Y." Do not average, blend, or silently pick \
one side.
4. If the SOURCES do not contain enough information to answer, reply \
exactly with: "I don't have enough information in the provided documents to \
answer that." (and then briefly say what is missing).
5. Be concise. Prefer 3-6 sentences unless a list or table is genuinely \
clearer. Do not pad.
6. Respect document versions. When the question is about "current" / \
"latest" / "most recent", prefer the source with the most recent date. \
When the question asks how something changed across versions, compare \
explicitly with attribution.
7. Do not echo the SOURCES verbatim in your answer; quote sparingly only \
when the exact wording matters (e.g. a defined term or a forward-looking \
statement).
"""


@dataclass
class FormattedSource:
    n: int                       # 1-based index used in the citation marker
    chunk: RetrievedChunk
    header: str
    body: str


def format_sources(chunks: list[RetrievedChunk], *, max_chars_per_chunk: int = 1800) -> list[FormattedSource]:
    """Number each chunk and produce a header + truncated body for the prompt."""
    out: list[FormattedSource] = []
    for i, c in enumerate(chunks, start=1):
        version = c.version_label or (c.doc_date.isoformat() if c.doc_date else "undated")
        company = c.company or "Unknown company"
        header = (
            f"[{i}] {company} — {version} — p.{c.page_number} "
            f"(chunk_type: {c.chunk_type})"
        )
        body = c.text.strip()
        if len(body) > max_chars_per_chunk:
            body = body[:max_chars_per_chunk].rstrip() + "..."
        out.append(FormattedSource(n=i, chunk=c, header=header, body=body))
    return out


def build_user_prompt(question: str, sources: list[FormattedSource]) -> str:
    """Assemble the user-turn payload sent to the LLM."""
    if not sources:
        return (
            f"QUESTION: {question}\n\n"
            "SOURCES: (none)\n\n"
            "Reply exactly with the insufficient-information sentence."
        )
    lines = ["SOURCES:", ""]
    for s in sources:
        lines.append(s.header)
        lines.append(s.body)
        lines.append("")
    lines.append(f"QUESTION: {question}")
    lines.append("")
    lines.append(
        "Answer using only these SOURCES. Cite with [N] markers. "
        "If the SOURCES disagree, surface the disagreement with attribution. "
        "If they do not contain the answer, say so."
    )
    return "\n".join(lines)
