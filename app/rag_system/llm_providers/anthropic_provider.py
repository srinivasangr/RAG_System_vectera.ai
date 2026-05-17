"""Anthropic Claude provider."""

from typing import Sequence

from anthropic import Anthropic

from rag_system.config import settings
from rag_system.llm_providers.base import BaseLLMProvider, Message


class AnthropicProvider(BaseLLMProvider):
    name = "anthropic"
    DEFAULT_MODEL = "claude-sonnet-4-6"

    def __init__(self, *, model: str | None = None):
        if not settings.anthropic_api_key:
            raise ValueError("anthropic: api_key is empty")
        self._client = Anthropic(api_key=settings.anthropic_api_key)
        active_default = (
            settings.llm_model
            if settings.llm_provider == "anthropic" and settings.llm_model
            else None
        )
        self._default_model = model or active_default or self.DEFAULT_MODEL

    def generate(
        self,
        messages: Sequence[Message],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> str:
        # Anthropic keeps system separate from messages
        system_parts = [m.content for m in messages if m.role == "system"]
        chat = [
            {"role": m.role, "content": m.content}
            for m in messages
            if m.role in ("user", "assistant")
        ]
        resp = self._client.messages.create(
            model=model or self._default_model,
            system="\n\n".join(system_parts) if system_parts else None,
            messages=chat,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # response.content is a list of blocks; concatenate the text parts
        return "".join(block.text for block in resp.content if getattr(block, "text", None))
