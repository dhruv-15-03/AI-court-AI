"""Regression + performance test for LegalAgentPipeline.analyze() concurrency.

``analyze()`` overlaps the LLM "understand query" round-trip with the local
similar-case search (see ``ai_court/agent/pipeline.py``) because the search
only needs the raw query text, not the LLM's structured understanding. This
test asserts both the overlap actually happens (wall time ~= max(llm, search)
rather than their sum) and that the returned analysis is unaffected by the
change (same fields, same values sourced from each step).
"""
from __future__ import annotations

import time
from typing import Any

import pytest

from ai_court.agent.pipeline import LegalAgentPipeline


class _SlowLLMClient:
    """Fake LLM client whose chat() calls take a fixed, measurable amount of time."""

    model = "fake-model"

    def __init__(self, delay: float = 0.2):
        self.delay = delay
        self.calls: list[str] = []

    def chat(self, messages, **kwargs):
        self.calls.append("chat")
        time.sleep(self.delay)
        # The first chat() call is always understand_query() (it is the only
        # LLM call that runs concurrently with the search thread; analysis/
        # strategy always follow it), so return parseable JSON only for call
        # #1 -- later calls just need any string.
        if len(self.calls) == 1:
            return (
                '{"case_type": "Civil", "legal_issues": ["breach of contract"], '
                '"relevant_acts": ["Indian Contract Act"], "party_role": "plaintiff", '
                '"relief_sought": "damages", "relevant_sections": []}'
            )
        return "Fake LLM analysis output."


class _SlowSearchIndex(dict):
    """Fake search index whose lookups take a fixed, measurable amount of time."""

    def __init__(self, delay: float = 0.2):
        super().__init__()
        self.delay = delay
        self.lookups = 0

    def get(self, key, default=None):
        if key in ("vectorizer", "matrix", "meta"):
            self.lookups += 1
            time.sleep(self.delay / 3)  # split across the 3 .get() calls
        return super().get(key, default)


@pytest.fixture
def slow_pipeline():
    llm = _SlowLLMClient(delay=0.2)

    class _FakeClassifier:
        def predict(self, query: str, case_type: str) -> dict[str, Any]:
            return {"judgment": "Favorable", "confidence": 0.8, "all_probabilities": {}}

    pipeline = LegalAgentPipeline(
        llm_client=llm,  # type: ignore[arg-type]
        classifier=_FakeClassifier(),
        search_index=None,  # find_similar_cases short-circuits to [] cheaply
        statute_corpus=None,
    )
    return pipeline, llm


def test_analyze_overlaps_understanding_and_search(monkeypatch, slow_pipeline):
    pipeline, llm = slow_pipeline

    search_delay = 0.2
    call_order: list[str] = []

    def fake_find_similar_cases(query, k=5):
        call_order.append("search_start")
        time.sleep(search_delay)
        call_order.append("search_end")
        return [{"title": "Fake v. Case", "outcome": "Favorable", "score": 0.9}]

    monkeypatch.setattr(pipeline, "find_similar_cases", fake_find_similar_cases)

    t0 = time.perf_counter()
    result = pipeline.analyze("Test query about a contract dispute", include_strategy=False)
    elapsed = time.perf_counter() - t0

    # Sequential execution would take >= llm.delay (0.2s, x2 for the two chat()
    # calls: understand + analysis) + search_delay (0.2s) = ~0.6s. Overlapped,
    # the understand-query LLM call and the search run concurrently, so total
    # wall time should be close to 2 chat calls (0.4s) plus a small margin —
    # well under the fully-sequential 0.6s+ bound.
    sequential_lower_bound = llm.delay * 2 + search_delay
    assert elapsed < sequential_lower_bound - 0.05, (
        f"analyze() took {elapsed:.3f}s, expected concurrency to beat the "
        f"sequential bound of {sequential_lower_bound:.3f}s"
    )

    # Correctness: result still reflects both the LLM understanding and the
    # (mocked) search results, unaffected by running them concurrently.
    assert result["understanding"]["case_type"] == "Civil"
    assert result["prediction"]["judgment"] == "Favorable"
    assert result["similar_cases"][0]["title"] == "Fake v. Case"
    assert result["analysis"] == "Fake LLM analysis output."
    assert result["strategy"] is None  # include_strategy=False


def test_analyze_still_correct_when_search_returns_empty(slow_pipeline):
    pipeline, _llm = slow_pipeline
    result = pipeline.analyze("Another query", include_strategy=False)
    assert result["similar_cases"] == []
    assert result["understanding"]["case_type"] == "Civil"
