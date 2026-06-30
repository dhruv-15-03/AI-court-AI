"""Unit tests for LLMClient latency-bounding and per-route timeouts.

These exercise the retry/timeout plumbing with a fake OpenAI client so no
network or API key is required. They lock in the production guarantees added in
the AI-Flask-hardening change:

* a per-call ``timeout`` is forwarded to the OpenAI ``create()`` call;
* ``timeout`` acts as a TOTAL wall-clock budget — retries can never push a call
  past it (so a slow upstream can't outlive the gunicorn worker timeout -> 502);
* non-retryable (4xx) errors are not retried;
* ``route_timeout`` resolves per-route budgets from the environment.
"""
import time

import pytest

from ai_court.llm.client import LLMClient, route_timeout


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    total_tokens = 7


class _FakeResponse:
    def __init__(self, content="ok"):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _FakeCompletions:
    """Records the kwargs of the last create() call; scriptable behaviour."""

    def __init__(self, error=None):
        self.calls = []
        self.error = error

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return _FakeResponse()


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeOpenAI:
    def __init__(self, completions):
        self.chat = _FakeChat(completions)


def _client_with(completions, **kwargs):
    c = LLMClient(api_key="test-key", provider="github", **kwargs)
    c._client = _FakeOpenAI(completions)
    return c


def test_route_timeout_env(monkeypatch):
    monkeypatch.delenv("LLM_TIMEOUT_ANALYZE", raising=False)
    assert route_timeout("analyze", 60.0) == 60.0

    monkeypatch.setenv("LLM_TIMEOUT_ANALYZE", "25")
    assert route_timeout("analyze", 60.0) == 25.0

    # Invalid / non-positive values fall back to the default.
    monkeypatch.setenv("LLM_TIMEOUT_ANALYZE", "not-a-number")
    assert route_timeout("analyze", 60.0) == 60.0
    monkeypatch.setenv("LLM_TIMEOUT_ANALYZE", "0")
    assert route_timeout("analyze", 60.0) == 60.0


def test_chat_forwards_per_call_timeout():
    completions = _FakeCompletions()
    c = _client_with(completions, timeout=120.0)
    out = c.chat([{"role": "user", "content": "hi"}], timeout=12.5)
    assert out == "ok"
    # First attempt receives the full per-call budget.
    assert completions.calls[-1]["timeout"] == 12.5


def test_chat_defaults_to_client_timeout():
    completions = _FakeCompletions()
    c = _client_with(completions, timeout=33.0)
    c.chat([{"role": "user", "content": "hi"}])
    assert completions.calls[-1]["timeout"] == 33.0


def test_temperature_zero_is_respected():
    # Regression: `temperature or self.temperature` discarded an explicit 0.0.
    completions = _FakeCompletions()
    c = _client_with(completions, temperature=0.3)
    c.chat([{"role": "user", "content": "hi"}], temperature=0.0)
    assert completions.calls[-1]["temperature"] == 0.0


def test_retries_are_bounded_and_fast():
    # A persistently-failing retryable upstream must not exceed the budget.
    completions = _FakeCompletions(error=RuntimeError("503 server error: upstream"))
    c = _client_with(completions, timeout=0.5, max_retries=3, max_backoff=0.01)
    start = time.perf_counter()
    with pytest.raises(Exception):
        c.chat([{"role": "user", "content": "hi"}])
    elapsed = time.perf_counter() - start
    # Bounded by the budget, nowhere near the old 3x exponential-backoff worst case.
    assert elapsed < 2.0
    # 1 initial attempt + at most max_retries retries.
    assert 1 <= len(completions.calls) <= 4


def test_non_retryable_error_not_retried():
    completions = _FakeCompletions(error=RuntimeError("400 Bad Request: invalid prompt"))
    c = _client_with(completions, timeout=5.0, max_retries=3, max_backoff=0.01)
    with pytest.raises(Exception):
        c.chat([{"role": "user", "content": "hi"}])
    # 4xx is deterministic — exactly one attempt, no retries.
    assert len(completions.calls) == 1


def test_sdk_retries_disabled():
    # The app loop is the single retry authority; the SDK client must not add its own.
    c = LLMClient(api_key="test-key", provider="github")
    assert c._client.max_retries == 0
