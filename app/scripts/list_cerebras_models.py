"""List models available on the Cerebras API."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openai import OpenAI

from rag_system.config import settings


def main() -> None:
    c = OpenAI(api_key=settings.cerebras_api_key, base_url="https://api.cerebras.ai/v1")
    for m in c.models.list().data:
        print(f"  {m.id}")


if __name__ == "__main__":
    main()
