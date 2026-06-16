"""Response caching for deterministic AI predictions.

The classifier endpoints (``/api/analyze`` and ``/api/analyze/quick``) run a
TF-IDF + RandomForest pipeline (plus optional explainability / RAG lookups) whose
output is a pure function of the cleaned request payload and the loaded model.
Repeating the same request therefore repeats identical, expensive work. This
module memoises those responses.

Design notes:
  * Two tiers — an optional shared Redis backend (so every gunicorn worker and a
    fresh deploy can reuse entries) in front of a per-process TTL + LRU memory
    cache. Redis is entirely optional: if it is unset or unreachable the in-process
    tier still works and any Redis error degrades gracefully to memory-only
    without ever failing the request.
  * Keys are a SHA-256 over the model version and a canonicalized copy of the
    payload. Identity-only fields (user id, request id, timestamps) are stripped so
    unrelated callers still share a hit, while anything that changes the output
    (e.g. the subscription ``_plan``) stays in the key. The model version (run id)
    is part of the key, so a new model deploy never serves a stale prediction.
  * Only deterministic classifier output is cached. Non-deterministic LLM/agent
    responses must never be stored here; callers decide what to cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Fields that never change the prediction output and must be excluded from the
# cache key so requests from different users/sessions can still share a result.
_DEFAULT_IGNORED_KEYS = frozenset(
    {
        "_user_id",
        "user_id",
        "_request_id",
        "request_id",
        "_session_id",
        "session_id",
        "_timestamp",
        "timestamp",
    }
)


def normalize_text(text: str) -> str:
    """Collapse runs of whitespace so trivial spacing differences share an entry.

    Case is intentionally preserved — lowercasing could merge semantically
    distinct inputs and the downstream preprocessor handles casing itself.
    """
    return " ".join(text.split())


def _canonicalize(value: Any, ignored_keys: frozenset[str]) -> Any:
    """Return a JSON-stable copy: dict keys sorted, ignored keys dropped."""
    if isinstance(value, dict):
        return {
            k: _canonicalize(v, ignored_keys)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if k not in ignored_keys
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(v, ignored_keys) for v in value]
    return value


def make_key(
    namespace: str,
    payload: Any,
    *,
    model_version: str = "",
    ignored_keys: frozenset[str] = _DEFAULT_IGNORED_KEYS,
) -> str:
    """Build a stable cache key for a request payload.

    The key is ``"{namespace}:{sha256}"`` where the digest covers the namespace,
    model version and a canonicalized payload, making it deterministic across
    processes and insensitive to dict ordering or volatile identity fields.
    """
    canonical = _canonicalize(payload, ignored_keys)
    blob = json.dumps(
        {"ns": namespace, "mv": model_version, "p": canonical},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    return f"{namespace}:{digest}"


class ResponseCache:
    """Thread-safe TTL + LRU cache with an optional shared Redis tier."""

    def __init__(
        self,
        *,
        enabled: bool = True,
        ttl_seconds: int = 3600,
        max_size: int = 512,
        redis_url: str = "",
        key_prefix: str = "aicache",
    ) -> None:
        self.enabled = enabled
        self.ttl_seconds = max(0, ttl_seconds)
        self.max_size = max(1, max_size)
        self.key_prefix = key_prefix
        self._lock = threading.Lock()
        self._store: "OrderedDict[str, tuple[float, dict[str, Any]]]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._redis: Optional[Any] = None
        if enabled and redis_url:
            self._redis = self._connect_redis(redis_url)

    def _connect_redis(self, redis_url: str) -> Optional[Any]:
        try:
            import redis  # type: ignore[import-not-found]

            client = redis.Redis.from_url(
                redis_url,
                socket_connect_timeout=2,
                socket_timeout=2,
                decode_responses=True,
            )
            client.ping()
            logger.info("ResponseCache: connected to Redis backend")
            return client
        except Exception as exc:  # pragma: no cover - depends on runtime env
            logger.warning(
                "ResponseCache: Redis unavailable (%s); using in-process cache only",
                exc,
            )
            return None

    @property
    def redis_available(self) -> bool:
        return self._redis is not None

    def _redis_key(self, key: str) -> str:
        return f"{self.key_prefix}:{key}"

    def get(self, key: str) -> Optional[dict[str, Any]]:
        """Return a cached value for ``key`` or ``None`` on a miss/expiry."""
        if not self.enabled:
            return None
        now = time.time()
        # Memory tier first (cheapest, no network).
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                expires_at, value = entry
                if expires_at > now:
                    self._store.move_to_end(key)
                    self._hits += 1
                    return dict(value)
                del self._store[key]
        # Redis tier — promotes the entry back into the memory tier on a hit.
        if self._redis is not None:
            raw = None
            try:
                raw = self._redis.get(self._redis_key(key))
            except Exception as exc:  # pragma: no cover - depends on runtime env
                logger.warning("ResponseCache: Redis get failed (%s)", exc)
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    self._set_memory(key, parsed, now)
                    with self._lock:
                        self._hits += 1
                    return dict(parsed)
        with self._lock:
            self._misses += 1
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store ``value`` under ``key`` in both tiers (best effort)."""
        if not self.enabled:
            return
        now = time.time()
        self._set_memory(key, value, now)
        if self._redis is not None and self.ttl_seconds > 0:
            try:
                self._redis.setex(
                    self._redis_key(key),
                    self.ttl_seconds,
                    json.dumps(value, ensure_ascii=False, default=str),
                )
            except Exception as exc:  # pragma: no cover - depends on runtime env
                logger.warning("ResponseCache: Redis set failed (%s)", exc)

    def _set_memory(self, key: str, value: dict[str, Any], now: float) -> None:
        expires_at = now + self.ttl_seconds if self.ttl_seconds > 0 else float("inf")
        with self._lock:
            self._store[key] = (expires_at, dict(value))
            self._store.move_to_end(key)
            while len(self._store) > self.max_size:
                self._store.popitem(last=False)

    def clear(self) -> None:
        """Drop all entries from the memory tier (and our Redis namespace)."""
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
        if self._redis is not None:
            try:
                for redis_key in self._redis.scan_iter(f"{self.key_prefix}:*"):
                    self._redis.delete(redis_key)
            except Exception as exc:  # pragma: no cover - depends on runtime env
                logger.warning("ResponseCache: Redis clear failed (%s)", exc)

    def stats(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of cache counters/config."""
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": self.enabled,
                "backend": "redis+memory" if self.redis_available else "memory",
                "entries": len(self._store),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 3) if total else 0.0,
            }
