"""End-to-end smoke test for the modular provider layer.

Tests whichever providers have keys present in .env.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_system.llm_providers import (
    Message,
    available_llm_providers,
    get_embedder,
    get_llm,
    get_vision,
)


def main() -> None:
    print("--- Available LLM providers (keys present) ---")
    providers = available_llm_providers()
    for p in providers:
        print(f"  - {p}")

    print("\n--- Generation smoke test ---")
    for p in providers:
        try:
            llm = get_llm(provider=p)
            out = llm.generate(
                [
                    Message(role="system", content="Reply with one word."),
                    Message(role="user", content="Say pong."),
                ],
                max_tokens=200,  # reasoning models (gpt-oss-*) need headroom
            )
            print(f"  [OK]   {p}: {out!r}")
        except Exception as e:
            print(f"  [FAIL] {p}: {type(e).__name__}: {str(e)[:160]}")

    print("\n--- Embedding smoke test ---")
    try:
        emb = get_embedder()
        v = emb.embed_one("hello world")
        print(f"  [OK] {emb.name}: dim={len(v)}, sample={v[:3]}")
    except Exception as e:
        print(f"  [FAIL] embedder: {type(e).__name__}: {e}")

    print("\n--- Vision provider (lazy init only; no image call) ---")
    try:
        v = get_vision()
        print(f"  [OK] {v.name} vision ready (no image sent in this test)")
    except Exception as e:
        print(f"  [FAIL] vision: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
