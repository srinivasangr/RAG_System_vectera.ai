"""Parse [N]-style citation markers in a generated answer back to the chunks
that produced them, and surface a clean Citations list for the UI."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_system.generation.prompt import FormattedSource


# Match [N], [N, M], (N), 【N】 (CJK fullwidth — gpt-oss-* sometimes emits these),
# and "[Source N]". The captured group is always the comma-separated number list.
_CITE_RE = re.compile(
    r"[\[【(]\s*(?:Source\s*)?(\d+(?:\s*,\s*\d+)*)\s*[\]】)]",
    re.IGNORECASE,
)


@dataclass
class Citation:
    n: int
    company: str | None
    version_label: str | None
    page_number: int
    chunk_id: str
    chunk_type: str
    source_path: str | None
    text: str


def parse_citation_ns(answer: str) -> list[int]:
    """Return the ordered, de-duplicated list of citation numbers used."""
    used: list[int] = []
    seen: set[int] = set()
    for m in _CITE_RE.finditer(answer):
        for piece in m.group(1).split(","):
            try:
                n = int(piece.strip())
            except ValueError:
                continue
            if n not in seen:
                seen.add(n)
                used.append(n)
    return used


def resolve_citations(
    answer: str, sources: list[FormattedSource]
) -> list[Citation]:
    """Map each [N] in the answer to a Citation dataclass, in order of appearance.

    Citations that point to a number outside the provided sources are dropped
    (the UI will also flag this as a model error).
    """
    by_n = {s.n: s for s in sources}
    used = parse_citation_ns(answer)
    out: list[Citation] = []
    for n in used:
        s = by_n.get(n)
        if not s:
            continue
        c = s.chunk
        out.append(
            Citation(
                n=n,
                company=c.company,
                version_label=c.version_label,
                page_number=c.page_number,
                chunk_id=c.chunk_id,
                chunk_type=c.chunk_type,
                source_path=c.source_path,
                text=c.text,
            )
        )
    return out
