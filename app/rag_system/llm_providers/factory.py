"""Factory: pick a provider at runtime from a name string.

Used by both ingestion (embedder, vision) and query path (LLM).
The UI dropdown also calls these.
"""

from rag_system.config import settings
from rag_system.llm_providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseVisionProvider,
)


def get_llm(provider: str | None = None, model: str | None = None) -> BaseLLMProvider:
    provider = (provider or settings.llm_provider).lower()
    if provider == "openai":
        from rag_system.llm_providers.openai_compat import OpenAIProvider
        return OpenAIProvider(model=model)
    if provider == "cerebras":
        from rag_system.llm_providers.openai_compat import CerebrasProvider
        return CerebrasProvider(model=model)
    if provider == "openrouter":
        from rag_system.llm_providers.openai_compat import OpenRouterProvider
        return OpenRouterProvider(model=model)
    if provider == "anthropic":
        from rag_system.llm_providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(model=model)
    if provider == "gemini":
        from rag_system.llm_providers.gemini_provider import GeminiProvider
        return GeminiProvider(model=model)
    raise ValueError(f"Unknown LLM provider: {provider}")


def get_embedder(provider: str | None = None) -> BaseEmbeddingProvider:
    provider = (provider or settings.embedding_provider).lower()
    if provider == "local":
        # sentence-transformers, runs on CPU — no rate limits, no API quotas
        from rag_system.llm_providers.local_embedder import LocalSentenceTransformersEmbedder
        return LocalSentenceTransformersEmbedder()
    if provider == "gemini":
        from rag_system.llm_providers.gemini_provider import GeminiEmbedder
        return GeminiEmbedder()
    # snowflake_cortex would go here if not blocked on trial
    raise ValueError(
        f"Unsupported embedding provider for this build: {provider}. "
        "Use 'local' (recommended) or 'gemini' (rate-limited on free tier)."
    )


def get_vision(provider: str | None = None) -> BaseVisionProvider:
    provider = (provider or "gemini").lower()
    if provider == "gemini":
        from rag_system.llm_providers.gemini_provider import GeminiVisionProvider
        return GeminiVisionProvider()
    raise ValueError(f"Unknown vision provider: {provider}")


# Convenience: list of selectable LLM providers for the UI dropdown.
# Only those with a non-empty API key are returned.
def available_llm_providers() -> list[str]:
    options: list[tuple[str, str]] = [
        ("openai", settings.openai_api_key),
        ("anthropic", settings.anthropic_api_key),
        ("gemini", settings.gemini_api_key),
        ("cerebras", settings.cerebras_api_key),
        ("openrouter", settings.openrouter_api_key),
    ]
    return [name for name, key in options if key]
