"""Parse [N]-style citation markers out of a generated answer."""

from __future__ import annotations

import re

# Match [N], [N, M], (N), 【N】 (some models emit CJK fullwidth brackets), and
# "[Source N]". The captured group is always the comma-separated number list.
_CITE_RE = re.compile(
    r"[\[【(]\s*(?:Source\s*)?(\d+(?:\s*,\s*\d+)*)\s*[\]】)]",
    re.IGNORECASE,
)


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
