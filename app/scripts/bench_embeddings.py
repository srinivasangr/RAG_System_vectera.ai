"""Time how long it takes to embed N chunks via Gemini, batched vs per-item."""

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.llm_providers import get_embedder


def main() -> None:
    embedder = get_embedder()
    texts = [f"This is sample text number {i} about REIT financial metrics." for i in range(64)]

    t0 = time.perf_counter()
    vecs = embedder.embed(texts)
    elapsed = time.perf_counter() - t0

    assert len(vecs) == len(texts)
    assert all(len(v) == embedder.dim for v in vecs)
    print(f"[OK] embedded {len(texts)} chunks in {elapsed:.2f}s "
          f"({elapsed/len(texts)*1000:.0f} ms/chunk)")
    print(f"     dim={embedder.dim}, sample first[:3]={vecs[0][:3]}")


if __name__ == "__main__":
    main()
