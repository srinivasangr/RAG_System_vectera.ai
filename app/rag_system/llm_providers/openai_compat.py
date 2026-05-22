"""OpenAI-compatible providers: OpenAI, Cerebras, OpenRouter.

All three speak the OpenAI Chat Completions protocol — the only difference
is `base_url` and `api_key`. We use one shared class with per-vendor configs.
"""

from typing import Sequence

from openai import OpenAI

from rag_system.config import settings
from rag_system.llm_providers.base import BaseLLMProvider, Message


class _OpenAICompatLLM(BaseLLMProvider):
    """Shared logic for OpenAI Chat-Completions-compatible APIs."""

    PROVIDER_KEY: str = "base"
    DEFAULT_MODEL: str = ""
    BASE_URL: str | None = None

    def __init__(self, *, api_key: str, model: str | None = None):
        if not api_key:
            raise ValueError(f"{self.PROVIDER_KEY}: api_key is empty")
        self._client = (
            OpenAI(api_key=api_key, base_url=self.BASE_URL)
            if self.BASE_URL
            else OpenAI(api_key=api_key)
        )
        # Use settings.llm_model only when settings selects THIS provider
        active_default = (
            settings.llm_model
            if settings.llm_provider == self.PROVIDER_KEY and settings.llm_model
            else None
        )
        self._model = model or active_default or self.DEFAULT_MODEL
        self.name = self.PROVIDER_KEY

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        resp = self._client.chat.completions.create(
            model=model or self._model,
            messages=[{"role": m.role, "content": m.content} for m in messages],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # Capture token usage for observability (best-effort).
        try:
            u = resp.usage
            self.last_usage = {
                "prompt_tokens": getattr(u, "prompt_tokens", None),
                "completion_tokens": getattr(u, "completion_tokens", None),
                "total_tokens": getattr(u, "total_tokens", None),
            }
        except Exception:  # noqa: BLE001
            self.last_usage = {}
        return resp.choices[0].message.content or ""


class OpenAIProvider(_OpenAICompatLLM):
    PROVIDER_KEY = "openai"
    DEFAULT_MODEL = "gpt-4o-mini"
    BASE_URL = None

    def __init__(self, *, model: str | None = None):
        super().__init__(api_key=settings.openai_api_key, model=model)


class CerebrasProvider(_OpenAICompatLLM):
    PROVIDER_KEY = "cerebras"
    DEFAULT_MODEL = "qwen-3-235b-a22b-instruct-2507"
    BASE_URL = "https://api.cerebras.ai/v1"

    def __init__(self, *, model: str | None = None):
        super().__init__(api_key=settings.cerebras_api_key, model=model)


class OpenRouterProvider(_OpenAICompatLLM):
    PROVIDER_KEY = "openrouter"
    DEFAULT_MODEL = "meta-llama/llama-3.3-70b-instruct"
    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, *, model: str | None = None):
        super().__init__(api_key=settings.openrouter_api_key, model=model)
