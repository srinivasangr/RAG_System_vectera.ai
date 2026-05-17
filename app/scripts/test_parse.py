"""Eyeball parse of a single PDF.

Usage:
  python scripts/test_parse.py                  # parses default file
  python scripts/test_parse.py <pdf-name.pdf>   # parses given file
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.config import settings
from rag_system.ingest.metadata import extract_metadata
from rag_system.ingest.parse import parse_pdf


DEFAULT_PDF = "Digital Realty_Investor Presentation December 2025.pdf"


def main() -> None:
    docs_dir = settings.documents_path
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PDF
    pdf_path = docs_dir / target
    if not pdf_path.exists():
        raise SystemExit(f"Not found: {pdf_path}")

    print(f"=== Metadata ===")
    meta = extract_metadata(pdf_path)
    for k, v in vars(meta).items():
        print(f"  {k}: {v}")

    print(f"\n=== Parsing (first run downloads ~300MB of models) ===")
    parsed = parse_pdf(pdf_path)
    print(f"  page_count: {parsed.page_count}")
    print(f"  pages with markdown: {sum(1 for p in parsed.pages if p.markdown)}")
    total_images = sum(len(p.images) for p in parsed.pages)
    print(f"  total images extracted: {total_images}")

    print(f"\n=== Per-page summary ===")
    for p in parsed.pages[:10]:
        md_len = len(p.markdown)
        img_count = len(p.images)
        img_sizes = ", ".join(f"{im.width}x{im.height}" for im in p.images[:3])
        snippet = p.markdown[:120].replace("\n", " ") if p.markdown else "<empty>"
        print(f"  p.{p.page_number}: md={md_len} chars, imgs={img_count} [{img_sizes}]")
        print(f"        '{snippet}...'")

    print(f"\n=== Markdown sample (first 1500 chars of full doc) ===")
    print(parsed.full_markdown[:1500])

    # Save outputs for inspection
    out_dir = Path(__file__).resolve().parents[1] / ".cache" / pdf_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "full.md").write_text(parsed.full_markdown, encoding="utf-8")
    for p in parsed.pages:
        for im in p.images:
            fname = f"p{p.page_number:03d}_img{im.image_index:02d}.png"
            (out_dir / fname).write_bytes(im.png_bytes)
    print(f"\nArtifacts written to: {out_dir}")


if __name__ == "__main__":
    main()
