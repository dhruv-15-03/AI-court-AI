"""LLM client — supports GitHub Models (gpt-4o) and Ollama (local Llama3/Mistral).

Provider selected via ``LLM_PROVIDER`` env var:
    * ``github`` (default) — GitHub Models at models.inference.ai.azure.com (needs GITHUB_TOKEN)
    * ``ollama`` — Local Ollama server at $OLLAMA_BASE_URL (default http://localhost:11434/v1)

Both endpoints are OpenAI-compatible, so the same ``openai`` client handles them.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterator, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


_PROVIDER_DEFAULTS = {
    "github": {
        "base_url": "https://models.inference.ai.azure.com",
        "model": "gpt-4o",
        "api_key_env": "GITHUB_TOKEN",
    },
    "ollama": {
        # Ollama exposes an OpenAI-compatible shim at /v1.
        "base_url": "http://localhost:11434/v1",
        "model": "llama3",
        "api_key_env": "OLLAMA_API_KEY",
    },
}


class LLMClient:
    """Thin wrapper around an OpenAI-compatible endpoint for legal reasoning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        timeout: float = 120.0,
    ):
        provider = (provider or os.getenv("LLM_PROVIDER", "github")).lower().strip()
        if provider not in _PROVIDER_DEFAULTS:
            logger.warning("Unknown LLM_PROVIDER=%s, defaulting to 'github'", provider)
            provider = "github"
        defaults = _PROVIDER_DEFAULTS[provider]

        self.provider = provider
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or defaults["base_url"]
        self.model = model or os.getenv("LLM_MODEL") or defaults["model"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

        # Resolve API key: explicit > provider env > generic LLM_API_KEY > ollama placeholder
        self.api_key = (
            api_key
            or os.getenv(defaults["api_key_env"])
            or os.getenv("LLM_API_KEY")
            or ("ollama" if provider == "ollama" else "")
        )
        if not self.api_key:
            raise ValueError(
                f"No API key provided for provider={provider!r}. "
                f"Set {defaults['api_key_env']} or pass api_key."
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )
        logger.info(
            "[llm] provider=%s model=%s base_url=%s",
            self.provider, self.model, self.base_url,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str | Iterator[str]:
        """Send a chat completion request.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            stream: If True, returns an iterator of content chunks.

        Returns:
            Complete response text, or iterator of chunks if stream=True.
        """
        if stream:
            return self._stream(messages, temperature, max_tokens)
        return self._complete(messages, temperature, max_tokens)

    def _complete(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        t0 = time.perf_counter()
        retries = 0
        max_retries = 3

        while retries <= max_retries:
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature or self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                elapsed = time.perf_counter() - t0
                logger.info(
                    "LLM completion: provider=%s model=%s tokens=%s latency=%.1fs",
                    self.provider, self.model,
                    response.usage.total_tokens if response.usage else "?",
                    elapsed,
                )
                return response.choices[0].message.content or ""

            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    retries += 1
                    wait = 2 ** retries
                    logger.warning("Rate limited, retrying in %ds (attempt %d)", wait, retries)
                    time.sleep(wait)
                    continue
                if "5" in err_str[:3] and retries < max_retries:
                    retries += 1
                    wait = 2 ** retries
                    logger.warning("Server error, retrying in %ds: %s", wait, err_str[:100])
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError(f"LLM request failed after {max_retries} retries")

    def _stream(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def health_check(self) -> dict:
        """Quick check that the LLM endpoint is reachable."""
        try:
            result = self._complete(
                [{"role": "user", "content": "Reply with OK"}],
                temperature=0,
                max_tokens=5,
            )
            return {
                "status": "ok",
                "provider": self.provider,
                "model": self.model,
                "response": result.strip(),
            }
        except Exception as e:
            return {
                "status": "error",
                "provider": self.provider,
                "model": self.model,
                "error": str(e),
            }
