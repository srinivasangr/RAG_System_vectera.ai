"""Smoke-test Gemini embedding + find a working free-tier generation model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google import genai
from google.genai import types

from rag_system.config import settings


def main() -> None:
    client = genai.Client(api_key=settings.gemini_api_key)

    # Embedding with output_dimensionality=768 to match our Snowflake VECTOR(768)
    print("--- Embedding (truncated to 768) ---")
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents="hello world",
        config=types.EmbedContentConfig(output_dimensionality=768),
    )
    emb = result.embeddings[0].values
    print(f"[OK] dim={len(emb)} sample={emb[:3]}")

    # Try generation models in priority order
    print("\n--- Generation models ---")
    candidates = [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-flash-8b",
    ]
    working = []
    for model in candidates:
        try:
            resp = client.models.generate_content(
                model=model,
                contents="Reply with the single word: pong",
            )
            txt = (resp.text or "").strip()
            print(f"  [OK]   {model}: {txt[:50]}")
            working.append(model)
        except Exception as e:
            msg = str(e)
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                print(f"  [QUOTA] {model}: free-tier blocked")
            elif "NOT_FOUND" in msg or "404" in msg:
                print(f"  [404]  {model}: not available on key")
            else:
                print(f"  [ERR]  {model}: {type(e).__name__}: {msg[:120]}")

    print(f"\n>>> Working generation models: {working}")


if __name__ == "__main__":
    main()
