"""Run ingest_one() from CLI to verify the pipeline works outside Streamlit.

Usage:
  python scripts/cli_ingest_one.py "BXP Morning Session Deck web.pdf"
"""

import logging
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from rag_system.config import settings
from rag_system.ingest.pipeline import ingest_one


def _cb(event: str, payload: dict) -> None:
    print(f"  [event] {event:<18} {payload}", flush=True)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: cli_ingest_one.py <filename in Documents/>")
    pdf_path = settings.documents_path / sys.argv[1]
    if not pdf_path.exists():
        raise SystemExit(f"Not found: {pdf_path}")

    print(f"Running ingest_one on: {pdf_path}")
    result = ingest_one(
        pdf_path, with_vision=True, vision_call_budget=20, progress_cb=_cb,
    )
    print(f"\nResult: {result}")


if __name__ == "__main__":
    main()
