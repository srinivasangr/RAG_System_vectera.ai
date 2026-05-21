"""Gemini provider: LLM + embeddings + vision (one SDK)."""

import logging
import os
import time
from typing import Sequence

from google import genai
from google.genai import types

from rag_system.config import settings
from rag_system.llm_providers._rate_limit import (
    RateLimiter,
    is_rate_limit_error,
    parse_retry_after_seconds,
)
from rag_system.llm_providers.base import (
    BaseEmbeddingProvider,
    BaseLLMProvider,
    BaseVisionProvider,
    Message,
)

log = logging.getLogger(__name__)

# Client-side throttle caps. Defaults are FREE-TIER-SAFE (so a fresh checkout
# never trips 429s), but are overridable via env for a paid key:
#   - GEMINI_EMBED_RPM   (default 80)  — gemini-embedding free tier ~100/min
#   - GEMINI_VISION_RPM  (default 4)   — gemini-2.5-flash vision free tier ~5/min
# Paid Tier 1 allows ~1000+ RPM, so set GEMINI_VISION_RPM=100+ to stop the
# vision pass being throttled to free-tier speed.
_EMBED_RPM  = int(os.environ.get("GEMINI_EMBED_RPM", "80"))
_VISION_RPM = int(os.environ.get("GEMINI_VISION_RPM", "4"))
_EMBED_LIMITER  = RateLimiter(max_calls=_EMBED_RPM,  window_s=60.0, name="gemini-embed")
_VISION_LIMITER = RateLimiter(max_calls=_VISION_RPM, window_s=60.0, name="gemini-vision")


def _client() -> genai.Client:
    if not settings.gemini_api_key:
        raise ValueError("gemini: api_key is empty")
    return genai.Client(api_key=settings.gemini_api_key)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
class GeminiProvider(BaseLLMProvider):
    name = "gemini"
    # 2.0 series is rate-limited to 0 on free tier; 2.5 Flash Lite is the default
    DEFAULT_MODEL = "gemini-2.5-flash-lite"

    def __init__(self, *, model: str | None = None):
        self._client = _client()
        active_default = (
            settings.llm_model
            if settings.llm_provider == "gemini" and settings.llm_model
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
        # Gemini takes system_instruction separately from contents
        system_parts = [m.content for m in messages if m.role == "system"]
        contents = []
        for m in messages:
            if m.role == "user":
                contents.append({"role": "user", "parts": [{"text": m.content}]})
            elif m.role == "assistant":
                contents.append({"role": "model", "parts": [{"text": m.content}]})

        cfg = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction="\n\n".join(system_parts) if system_parts else None,
        )
        resp = self._client.models.generate_content(
            model=model or self._default_model,
            contents=contents,
            config=cfg,
        )
        # Capture token usage for observability (best-effort).
        try:
            u = resp.usage_metadata
            self.last_usage = {
                "prompt_tokens": getattr(u, "prompt_token_count", None),
                "completion_tokens": getattr(u, "candidates_token_count", None),
                "total_tokens": getattr(u, "total_token_count", None),
            }
        except Exception:  # noqa: BLE001
            self.last_usage = {}
        return (resp.text or "").strip()


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
class GeminiEmbedder(BaseEmbeddingProvider):
    name = "gemini"
    # Keep BATCH_SIZE <= _EMBED_LIMITER.max_calls so a single batch can fit
    # in one rate-limit window. The Gemini batchEmbedContents API hard limit
    # is 100, but our free-tier-aware limiter caps at 80/min.
    BATCH_SIZE = 50

    def __init__(self, *, model: str | None = None, dim: int | None = None):
        self._client = _client()
        self._model = model or settings.embedding_model or "gemini-embedding-001"
        self.dim = dim or settings.embedding_dim or 768

    def embed(
        self,
        texts: Sequence[str],
        *,
        progress_cb=None,
    ) -> list[list[float]]:
        """Batch-embed texts via Gemini's `contents=[...]` batch form.

        Honors the gemini-embedding free-tier limit (100 items/min) via a
        process-global rate limiter and retries 429s with the API's suggested
        delay.
        """
        if not texts:
            return []
        texts = list(texts)
        total = len(texts)
        out: list[list[float]] = []

        for start in range(0, total, self.BATCH_SIZE):
            batch = texts[start:start + self.BATCH_SIZE]
            batch_embeds = self._embed_batch_with_retry(batch)
            out.extend(batch_embeds)
            if progress_cb:
                try:
                    progress_cb("embed_progress", {"done": len(out), "total": total})
                except Exception:
                    pass

        return out

    def _embed_batch_with_retry(self, batch: list[str], *, max_retries: int = 3) -> list[list[float]]:
        """Embed one batch, throttled + retrying on 429."""
        attempt = 0
        while True:
            # Reserve quota for this many items before issuing the request
            _EMBED_LIMITER.acquire(n=len(batch))
            try:
                r = self._client.models.embed_content(
                    model=self._model,
                    contents=batch,
                    config=types.EmbedContentConfig(
                        output_dimensionality=self.dim,
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                return [list(e.values) for e in r.embeddings]
            except Exception as e:
                if is_rate_limit_error(e) and attempt < max_retries:
                    wait_s = parse_retry_after_seconds(str(e), default=30)
                    log.warning(
                        "gemini-embed 429 — sleeping %.0fs (retry %d/%d)",
                        wait_s, attempt + 1, max_retries,
                    )
                    time.sleep(wait_s + 1)
                    attempt += 1
                    continue
                raise


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
class GeminiVisionProvider(BaseVisionProvider):
    name = "gemini"

    def __init__(self, *, model: str | None = None):
        self._client = _client()
        # 2.5 Flash has the best image-understanding quality at this tier
        self._default_model = model or "gemini-2.5-flash"

    def describe_image(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        mime_type: str = "image/png",
        model: str | None = None,
    ) -> str:
        # Rate-limit vision strictly to stay under the 5 RPM free-tier limit
        _VISION_LIMITER.acquire(n=1)
        try:
            resp = self._client.models.generate_content(
                model=model or self._default_model,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    prompt,
                ],
            )
            return (resp.text or "").strip()
        except Exception as e:
            if is_rate_limit_error(e):
                # Caller (vision_extract.describe_images) treats RuntimeError
                # ending in "rate_limited" as non-fatal: image is skipped.
                wait_s = parse_retry_after_seconds(str(e), default=20)
                log.warning("gemini-vision 429 — would need to wait %.0fs; skipping image", wait_s)
                raise RuntimeError(f"vision rate_limited (retry-after ~{wait_s:.0f}s)") from e
            raise
