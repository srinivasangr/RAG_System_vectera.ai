"""Ingest every PDF in Documents/ that isn't already in Snowflake.

Skips files whose checksum already matches a stored document (idempotent).
Processes them one at a time. Safe to interrupt + resume.

Usage:
  python scripts/ingest_all_pending.py
  python scripts/ingest_all_pending.py --vision-budget 30
  python scripts/ingest_all_pending.py --skip-large 100   # skip PDFs > 100 pages
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
except Exception:
    pass
# Disable buffering so print() output shows up in tee'd logs immediately
import os as _os
_os.environ.setdefault("PYTHONUNBUFFERED", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from rag_system.config import settings
from rag_system.ingest.metadata import extract_metadata
from rag_system.ingest.pipeline import ingest_one
from rag_system.storage.db import get_connection


def _existing_checksums() -> set[str]:
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT checksum FROM documents WHERE checksum IS NOT NULL")
        out = {r[0] for r in cur.fetchall()}
        cur.close()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vision-budget", type=int, default=20)
    p.add_argument("--no-vision", action="store_true",
                   help="Skip chart-image vision pass entirely (faster, no chart_description chunks)")
    p.add_argument("--skip-large", type=int, default=None,
                   help="Skip PDFs with more than N pages (use pypdf to count)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    docs_dir = settings.documents_path
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {docs_dir}")
        return 1

    have = _existing_checksums()
    todo: list[Path] = []
    skipped_dup = []
    skipped_large = []

    for pdf in pdfs:
        meta = extract_metadata(pdf)
        if meta.checksum in have:
            skipped_dup.append(pdf.name)
            continue
        if args.skip_large:
            from pypdf import PdfReader
            try:
                npages = len(PdfReader(str(pdf)).pages)
            except Exception:
                npages = 0
            if npages > args.skip_large:
                skipped_large.append((pdf.name, npages))
                continue
        todo.append(pdf)

    print(f"\n=== Plan ===")
    print(f"  total in folder:  {len(pdfs)}")
    print(f"  already ingested: {len(skipped_dup)}")
    print(f"  skipped (large):  {len(skipped_large)}")
    print(f"  to ingest:        {len(todo)}\n")
    for f in todo:
        print(f"    - {f.name}")
    if skipped_large:
        print(f"\n  too large (>{args.skip_large}p):")
        for f, n in skipped_large:
            print(f"    - {f}  ({n}p)")

    if args.dry_run:
        return 0
    if not todo:
        print("\nNothing to do.")
        return 0

    print(f"\n=== Ingesting {len(todo)} PDF(s) sequentially ===\n")
    t_total = time.perf_counter()
    results = []
    for i, pdf in enumerate(todo, 1):
        print(f"\n[{i}/{len(todo)}] {pdf.name}", flush=True)
        t0 = time.perf_counter()
        try:
            r = ingest_one(
                pdf,
                with_vision=not args.no_vision,
                vision_call_budget=args.vision_budget,
            )
            print(f"  -> stored {r.get('chunks',0)} chunks "
                  f"({r.get('doc_status')}) in {time.perf_counter()-t0:.0f}s",
                  flush=True)
            results.append((pdf.name, "ok", r))
        except Exception as e:
            import traceback as _tb
            tb_short = _tb.format_exc().splitlines()[-4:]
            print(f"  -> FAILED: {type(e).__name__}: {e}", flush=True)
            for line in tb_short:
                print(f"     {line}", flush=True)
            results.append((pdf.name, "error", str(e)))

    print(f"\n=== Summary ({time.perf_counter()-t_total:.0f}s total) ===")
    for name, status, payload in results:
        if status == "ok":
            print(f"  ✓ {name}: {payload.get('chunks',0)} chunks, {payload.get('elapsed_s',0)}s")
        else:
            print(f"  ✗ {name}: {payload}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
