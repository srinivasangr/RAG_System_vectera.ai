"""Print a sample of chunks for eyeball QA — does the parser output something
sensible per page, including tables?
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.config import settings
from rag_system.ingest.chunk import chunk_page
from rag_system.ingest.metadata import extract_metadata
from rag_system.ingest.parse import parse_pdf


DEFAULT_PDF = "Digital Realty_Investor Presentation December 2025.pdf"


def main() -> None:
    pdf_path = settings.documents_path / (sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF)
    meta = extract_metadata(pdf_path)
    parsed = parse_pdf(pdf_path)

    all_chunks = []
    for page in parsed.pages:
        all_chunks.extend(
            chunk_page(
                doc_id=meta.doc_id,
                page_number=page.page_number,
                page_markdown=page.markdown,
                company=meta.company,
                doc_date=meta.doc_date,
                version_label=meta.version_label,
            )
        )

    print(f"Total chunks: {len(all_chunks)}\n")

    # Group by type
    by_type: dict[str, list] = {}
    for c in all_chunks:
        by_type.setdefault(c.chunk_type, []).append(c)
    for kind, lst in by_type.items():
        print(f"  {kind}: {len(lst)}")

    print("\n=== 3 prose chunks (varied pages) ===")
    prose = by_type.get("prose", [])
    for c in [prose[0], prose[len(prose)//2], prose[-1]] if len(prose) >= 3 else prose:
        print(f"\n--- {c.chunk_id} (p.{c.page_number}, {c.token_count} tok) ---")
        print(c.text[:600])
        if len(c.text) > 600:
            print(f"... [+{len(c.text)-600} chars]")

    print("\n=== 3 table chunks ===")
    tables = by_type.get("table", [])
    for c in [tables[0], tables[len(tables)//2], tables[-1]] if len(tables) >= 3 else tables:
        print(f"\n--- {c.chunk_id} (p.{c.page_number}, {c.token_count} tok) ---")
        print(c.text[:800])
        if len(c.text) > 800:
            print(f"... [+{len(c.text)-800} chars]")


if __name__ == "__main__":
    main()
