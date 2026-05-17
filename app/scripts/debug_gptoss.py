"""Inspect raw gpt-oss-120b response to see if reasoning tokens are eating the budget."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from rag_system.config import settings


def main() -> None:
    c = OpenAI(api_key=settings.cerebras_api_key, base_url="https://api.cerebras.ai/v1")

    for max_tok in (50, 500, 2000):
        print(f"\n=== max_tokens={max_tok} ===")
        resp = c.chat.completions.create(
            model="gpt-oss-120b",
            messages=[{"role": "user", "content": "Reply with the single word: pong"}],
            max_tokens=max_tok,
            temperature=0.0,
        )
        msg = resp.choices[0].message
        print(f"  finish_reason: {resp.choices[0].finish_reason}")
        print(f"  content:       {msg.content!r}")
        # gpt-oss may expose reasoning in a separate field
        for attr in ("reasoning", "reasoning_content"):
            if hasattr(msg, attr):
                val = getattr(msg, attr)
                if val:
                    print(f"  {attr}: {str(val)[:200]!r}")
        if resp.usage:
            print(f"  usage: prompt={resp.usage.prompt_tokens} "
                  f"completion={resp.usage.completion_tokens}")


if __name__ == "__main__":
    main()
