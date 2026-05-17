"""Filter spec used by the retrieval and UI layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass
class RetrievalFilters:
    """Optional metadata filters applied at retrieval time."""
    doc_ids: list[str] = field(default_factory=list)    # restrict to these doc_ids
    companies: list[str] = field(default_factory=list)  # canonical company names
    doc_types: list[str] = field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None
    # When True, give a small score boost to chunks from the most recent doc
    # per company. Useful for queries like "current strategy" / "latest results".
    prefer_recent: bool = False

    def is_empty(self) -> bool:
        return not (
            self.doc_ids or self.companies or self.doc_types
            or self.date_from or self.date_to or self.prefer_recent
        )
