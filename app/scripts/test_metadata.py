"""Verify metadata extraction across all PDFs in Documents/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.config import settings
from rag_system.ingest.metadata import extract_metadata


def main() -> None:
    docs_dir = settings.documents_path
    print(f"Scanning: {docs_dir}\n")
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {docs_dir}")

    print(f"{'company':<25} {'date':<12} {'type':<22} {'version':<10}  source")
    print("-" * 120)
    for pdf in pdfs:
        m = extract_metadata(pdf)
        print(
            f"{(m.company or '?'):<25} "
            f"{(str(m.doc_date) if m.doc_date else '?'):<12} "
            f"{(m.doc_type or '?'):<22} "
            f"{(m.version_label or '?'):<10}  "
            f"{Path(m.source_path).name}"
        )


if __name__ == "__main__":
    main()
