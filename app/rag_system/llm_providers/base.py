"""Abstract base classes for LLM, embedding, and vision providers.

Every concrete provider implements the same interface so the rest of the
codebase can call `get_llm().generate(...)` without knowing which vendor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


# --- Message format (OpenAI-style, used as our universal format) -----------
@dataclass(frozen=True)
class Message:
    role: str  # "system" | "user" | "assistant"
    content: str


# ---------------------------------------------------------------------------
# LLM (text-in, text-out)
# ---------------------------------------------------------------------------
class BaseLLMProvider(ABC):
    """Generates text given a list of messages."""

    name: str = "base"

    @abstractmethod
    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        """Return the assistant's reply as a string."""
        ...


# ---------------------------------------------------------------------------
# Embeddings (text-in, vector-out)
# ---------------------------------------------------------------------------
class BaseEmbeddingProvider(ABC):
    """Produces fixed-dim embeddings for text."""

    name: str = "base"
    dim: int = 0

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one vector per input text. Always 2D, even for a single text."""
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


# ---------------------------------------------------------------------------
# Vision (image-in, text-out) — used for chart/figure description in ingest
# ---------------------------------------------------------------------------
class BaseVisionProvider(ABC):
    """Describes an image as text."""

    name: str = "base"

    @abstractmethod
    def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        mime_type: str = "image/png",
        model: str | None = None,
    ) -> str:
        """Return a text description of the image."""
        ...
