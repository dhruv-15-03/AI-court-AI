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
from typing import Any, Iterator, Literal, Optional, cast, overload

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

# Substrings that mark a provider error as worth retrying (transient): rate limits
# (HTTP 429), upstream 5xx, and connection/timeout blips. Kept conservative so we
# never retry 4xx client errors (bad request, auth) that will fail deterministically.
_RETRYABLE_MARKERS = (
    "429", "rate limit", "rate_limit", "too many requests",
    "500", "502", "503", "504", "overloaded", "server error",
    "timeout", "timed out", "connection", "temporarily unavailable",
)


def _is_retryable(err_str: str) -> bool:
    low = err_str.lower()
    return any(marker in low for marker in _RETRYABLE_MARKERS)


# Markers identifying a deterministic provider rejection of the `response_format`
# parameter (e.g. older Ollama builds, or models without JSON mode). Used by
# chat_json() to decide whether to transparently retry without structured-output
# mode rather than surfacing the error.
_RESPONSE_FORMAT_REJECTION_MARKERS = (
    "response_format", "response format", "json_object", "json schema",
    "unsupported", "not supported", "does not support", "unexpected keyword",
    "unknown parameter", "unrecognized", "unknown field", "extra inputs",
    "invalid_request_error",
)


def _is_unsupported_response_format(err_str: str) -> bool:
    low = err_str.lower()
    return any(marker in low for marker in _RESPONSE_FORMAT_REJECTION_MARKERS)


def parse_json_object(text: Optional[str]) -> dict[str, Any]:
    """Parse a JSON object from possibly-decorated model output.

    Tolerates ```json code fences, leading/trailing prose, and assistant chatter
    around the object by falling back to the outermost ``{...}`` span. This makes
    structured-output parsing robust even when a model ignores instructions to
    emit bare JSON.

    Raises:
        ValueError: if no JSON object can be recovered.
    """
    import json

    if not text or not text.strip():
        raise ValueError("empty LLM response")
    s = text.strip()

    # Strip a leading ```/```json fence and any trailing fence.
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(s[start : end + 1])
        except Exception as exc:
            raise ValueError(f"could not parse JSON object: {exc}") from exc
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in LLM response")


def _resolve_float(explicit: Optional[float], env_var: str, default: float) -> float:
    """Pick an explicit value, else env var, else default. Ignores invalid/<=0 env."""
    if explicit is not None:
        return explicit
    raw = (os.getenv(env_var) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def _resolve_int(explicit: Optional[int], env_var: str, default: int) -> int:
    """Pick an explicit value, else env var, else default. Ignores invalid/<0 env."""
    if explicit is not None:
        return explicit
    raw = (os.getenv(env_var) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value >= 0 else default


def route_timeout(route: str, default: float) -> float:
    """Resolve a per-route LLM wall-clock budget (seconds) from the environment.

    Reads ``LLM_TIMEOUT_<ROUTE>`` (e.g. ``LLM_TIMEOUT_CHAT``). Falls back to
    ``default`` when the variable is unset, blank, or not a positive number.
    Operators tune these so a single request's total LLM time (including retries)
    stays under the gunicorn worker ``timeout`` (default 120s), preventing the
    worker from being killed mid-call (which surfaces to clients as a 502).
    """
    raw = (os.getenv(f"LLM_TIMEOUT_{route.upper()}") or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        return default
    return value if value > 0 else default


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
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        max_backoff: Optional[float] = None,
    ):
        provider = (provider or os.getenv("LLM_PROVIDER") or "github").lower().strip()
        if provider not in _PROVIDER_DEFAULTS:
            logger.warning("Unknown LLM_PROVIDER=%s, defaulting to 'github'", provider)
            provider = "github"
        defaults = _PROVIDER_DEFAULTS[provider]

        self.provider = provider
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or defaults["base_url"]
        self.model = model or os.getenv("LLM_MODEL") or defaults["model"]
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Default wall-clock budget for a single chat() call (incl. retries). Kept
        # below the gunicorn worker timeout (120s) so a slow upstream never causes
        # a worker kill -> 502. Env-overridable; per-route callers pass tighter values.
        self.timeout = _resolve_float(timeout, "LLM_TIMEOUT", 90.0)
        # App-level retry budget for transient errors. The OpenAI SDK's own retries
        # are disabled below (max_retries=0) so this loop is the single source of
        # truth and total latency stays bounded by self.timeout.
        self.max_retries = _resolve_int(max_retries, "LLM_MAX_RETRIES", 1)
        self.max_backoff = _resolve_float(max_backoff, "LLM_RETRY_MAX_BACKOFF", 5.0)

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
            # Disable SDK-level retries: the app retry loop in _complete() owns
            # retries so cumulative latency can't silently exceed the worker budget.
            max_retries=0,
        )
        logger.info(
            "[llm] provider=%s model=%s base_url=%s timeout=%.0fs max_retries=%d",
            self.provider, self.model, self.base_url, self.timeout, self.max_retries,
        )

    @overload
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = ...,
        max_tokens: Optional[int] = ...,
        stream: Literal[False] = ...,
        timeout: Optional[float] = ...,
    ) -> str: ...

    @overload
    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = ...,
        max_tokens: Optional[int] = ...,
        stream: Literal[True] = ...,
        timeout: Optional[float] = ...,
    ) -> Iterator[str]: ...

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        timeout: Optional[float] = None,
    ) -> str | Iterator[str]:
        """Send a chat completion request.

        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": "..."}
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            stream: If True, returns an iterator of content chunks.
            timeout: Per-call wall-clock budget (seconds) for the whole request,
                including retries. Defaults to ``self.timeout``. Callers pass a
                per-route value (see ``route_timeout``) to keep total time under
                the gunicorn worker timeout.

        Returns:
            Complete response text, or iterator of chunks if stream=True.
        """
        if stream:
            return self._stream(messages, temperature, max_tokens, timeout)
        return self._complete(messages, temperature, max_tokens, timeout)

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        """Chat completion that returns a parsed JSON object.

        Requests a JSON-object response via ``response_format`` when the provider
        supports it (GitHub Models / OpenAI ``gpt-4o`` do; some Ollama models do
        not), transparently retrying once as a plain completion on a
        parameter-rejection error. The text is then parsed with a tolerant
        extractor (:func:`parse_json_object`) that strips code fences and
        surrounding prose, so a stray ```json fence or trailing sentence no longer
        derails downstream ``json.loads``.

        The prompt must still instruct the model to return JSON (OpenAI's
        json_object mode requires the word "json" to appear in the messages).

        Args:
            messages: Same shape as :meth:`chat`.
            temperature/max_tokens/timeout: As in :meth:`chat`.

        Returns:
            The parsed JSON object as a dict.

        Raises:
            ValueError: if the model output cannot be parsed into a JSON object.
        """
        try:
            raw = self._complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                response_format={"type": "json_object"},
            )
        except Exception as exc:
            # A provider/model that rejects `response_format` fails deterministically
            # (HTTP 400) -> retry once without it. Transient errors are re-raised so
            # the caller's existing fallback handles them and we don't silently
            # double the latency budget on an upstream outage.
            if _is_unsupported_response_format(str(exc)):
                logger.info("[llm] response_format unsupported, retrying without it")
                raw = self._complete(
                    messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            else:
                raise
        return parse_json_object(raw)

    def _complete(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
        response_format: Optional[dict[str, Any]] = None,
    ) -> str:
        budget = timeout if timeout is not None else self.timeout
        extra: dict[str, Any] = {}
        if response_format is not None:
            extra["response_format"] = response_format
        t0 = time.perf_counter()
        deadline = t0 + budget
        retries = 0

        while True:
            # First attempt gets the full budget; retries get only the time left so
            # the call can never run past `budget` wall-clock seconds in total.
            attempt_timeout = budget if retries == 0 else deadline - time.perf_counter()
            if attempt_timeout <= 0:
                raise RuntimeError(
                    f"LLM request exceeded its {budget:.0f}s budget after {retries} retr"
                    f"{'y' if retries == 1 else 'ies'}"
                )
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=cast(Any, messages),
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                    timeout=attempt_timeout,
                    **extra,
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
                if retries < self.max_retries and _is_retryable(err_str):
                    retries += 1
                    wait = min(2 ** retries, self.max_backoff)
                    # Don't sleep past the deadline — fail fast instead of overrunning.
                    if time.perf_counter() + wait >= deadline:
                        logger.warning(
                            "LLM transient error with no budget left to retry: %s",
                            err_str[:100],
                        )
                        raise
                    logger.warning(
                        "LLM transient error, retrying in %.0fs (attempt %d/%d): %s",
                        wait, retries, self.max_retries, err_str[:100],
                    )
                    time.sleep(wait)
                    continue
                raise

    def _stream(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> Iterator[str]:
        budget = timeout if timeout is not None else self.timeout
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages),
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            stream=True,
            timeout=budget,
        )
        for chunk in cast(Any, stream):
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def health_check(self) -> dict:
        """Quick check that the LLM endpoint is reachable."""
        try:
            result = self._complete(
                [{"role": "user", "content": "Reply with OK"}],
                temperature=0,
                max_tokens=5,
                timeout=route_timeout("health", 10.0),
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
