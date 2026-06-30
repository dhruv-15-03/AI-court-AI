"""Tests for the AI answer-quality wave.

Covers three production guarantees added together:

* **Structured outputs** — ``parse_json_object`` tolerates fenced/prose-wrapped
  model output, and ``LLMClient.chat_json`` requests ``response_format`` but
  degrades gracefully when a provider rejects it.
* **Citation faithfulness** — ``verify_citations`` flags case citations that do
  not map to any retrieved document (the core legal-AI hallucination guard).
* **Hybrid retrieval** — ``rag.pipeline.retrieve`` fuses dense + TF-IDF results
  via reciprocal-rank fusion, and falls back cleanly to a single retriever.

All tests are dependency-light: the LLM tests drive a fake OpenAI client (no
network/key), and the retrieval tests ``importorskip`` numpy.
"""
import pytest

from ai_court.llm.client import (
    LLMClient,
    parse_json_object,
    _is_unsupported_response_format,
)
from ai_court.llm.faithfulness import extract_case_citations, verify_citations


# --------------------------------------------------------------------------- #
# parse_json_object
# --------------------------------------------------------------------------- #

def test_parse_plain_json():
    assert parse_json_object('{"a": 1, "b": "x"}') == {"a": 1, "b": "x"}


def test_parse_fenced_json():
    text = '```json\n{"outcome": "win", "confidence": 0.8}\n```'
    assert parse_json_object(text) == {"outcome": "win", "confidence": 0.8}


def test_parse_bare_fence():
    text = '```\n{"k": "v"}\n```'
    assert parse_json_object(text) == {"k": "v"}


def test_parse_prose_wrapped_json():
    # Model adds a polite preamble/suffix around the object.
    text = 'Sure, here is the result:\n{"verdict": "guilty"}\nLet me know if you need more.'
    assert parse_json_object(text) == {"verdict": "guilty"}


def test_parse_empty_raises():
    with pytest.raises(ValueError):
        parse_json_object("")
    with pytest.raises(ValueError):
        parse_json_object("   ")
    with pytest.raises(ValueError):
        parse_json_object(None)


def test_parse_garbage_raises():
    with pytest.raises(ValueError):
        parse_json_object("not json at all, no braces here")


def test_parse_non_object_json_raises():
    # A bare JSON array is valid JSON but not the object we require.
    with pytest.raises(ValueError):
        parse_json_object("[1, 2, 3]")


# --------------------------------------------------------------------------- #
# _is_unsupported_response_format
# --------------------------------------------------------------------------- #

def test_unsupported_response_format_detection():
    assert _is_unsupported_response_format(
        "Error: response_format is not supported by this model"
    )
    assert _is_unsupported_response_format("unknown parameter: response_format")
    assert not _is_unsupported_response_format("503 server error: upstream overloaded")
    assert not _is_unsupported_response_format("connection timeout")


# --------------------------------------------------------------------------- #
# chat_json (fake OpenAI client)
# --------------------------------------------------------------------------- #

class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeUsage:
    total_tokens = 5


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


class _ScriptedCompletions:
    """Returns queued responses/errors in order, recording each call's kwargs."""

    def __init__(self, script):
        self.calls = []
        self._script = list(script)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResponse(item)


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


def test_chat_json_requests_response_format_and_parses():
    completions = _ScriptedCompletions(['{"outcome": "win"}'])
    c = _client_with(completions, timeout=30.0)
    out = c.chat_json([{"role": "user", "content": "return json"}])
    assert out == {"outcome": "win"}
    assert completions.calls[-1]["response_format"] == {"type": "json_object"}


def test_chat_json_retries_without_response_format_on_rejection():
    completions = _ScriptedCompletions([
        ValueError("400 unknown parameter: response_format"),
        '{"ok": true}',
    ])
    c = _client_with(completions, timeout=30.0)
    out = c.chat_json([{"role": "user", "content": "return json"}])
    assert out == {"ok": True}
    # First attempt carried response_format; retry dropped it.
    assert completions.calls[0].get("response_format") == {"type": "json_object"}
    assert "response_format" not in completions.calls[1]


def test_chat_json_reraises_transient_error():
    # A transient upstream error must NOT be swallowed as a format rejection —
    # otherwise we'd silently double the latency budget on an outage.
    completions = _ScriptedCompletions([
        RuntimeError("503 server error: upstream overloaded"),
    ])
    c = _client_with(completions, timeout=2.0, max_retries=0)
    with pytest.raises(Exception):
        c.chat_json([{"role": "user", "content": "return json"}])


# --------------------------------------------------------------------------- #
# verify_citations
# --------------------------------------------------------------------------- #

def _docs(*titles):
    return [{"title": t, "url": f"http://x/{i}"} for i, t in enumerate(titles)]


def test_extract_case_citations_variants():
    text = (
        "See Kesavananda Bharati v. State of Kerala and "
        "Maneka Gandhi vs Union of India and also ADM Jabalpur versus Shivkant Shukla."
    )
    cites = extract_case_citations(text)
    joined = " | ".join(cites).lower()
    assert "kesavananda" in joined
    assert "maneka gandhi" in joined
    assert "adm jabalpur" in joined


def test_verify_citations_grounded():
    answer = "As held in Maneka Gandhi v. Union of India, the right is not absolute."
    docs = _docs("Maneka Gandhi v. Union of India", "Some Other Case v. Anr")
    res = verify_citations(answer, docs)
    assert res["grounded"] is True
    assert res["unverified_citations"] == []
    assert any("Maneka" in c for c in res["verified_citations"])


def test_verify_citations_flags_hallucination():
    answer = "Per Fictional Imaginary Petitioner v. Nonexistent Respondent, claims fail."
    docs = _docs("Maneka Gandhi v. Union of India")
    res = verify_citations(answer, docs)
    assert res["grounded"] is False
    assert len(res["unverified_citations"]) == 1


def test_verify_citations_no_citation_is_grounded():
    res = verify_citations("This is a general statement with no case cited.", _docs("X v. Y"))
    assert res["grounded"] is True
    assert res["cited_cases"] == []


# --------------------------------------------------------------------------- #
# Hybrid retrieval (rag.pipeline.retrieve)
# --------------------------------------------------------------------------- #

def test_retrieve_semantic_only_path():
    pytest.importorskip("numpy")
    import numpy as np
    from ai_court.rag import pipeline

    semantic_index = {
        "embeddings": np.array([[1.0, 0.0], [0.0, 1.0]]),
        "meta": [
            {"title": "Doc A v. State", "url": "u/a"},
            {"title": "Doc B v. Union", "url": "u/b"},
        ],
    }

    def embed(_query):
        return np.array([[1.0, 0.0]])

    docs = pipeline.retrieve(
        "anything",
        semantic_index=semantic_index,
        query_embed_fn=embed,
        k=2,
    )
    assert docs, "semantic-only retrieval should return results"
    assert docs[0]["title"] == "Doc A v. State"
    assert docs[0]["rank"] == 1


def test_retrieve_fuses_both_legs(monkeypatch):
    pytest.importorskip("numpy")
    import numpy as np
    from ai_court.rag import pipeline

    lexical_docs = [
        {"title": "Shared Case v. State", "url": "u/shared", "score": 0.9, "rank": 1},
        {"title": "Lexical Only v. X", "url": "u/lex", "score": 0.5, "rank": 2},
    ]

    monkeypatch.setattr(pipeline, "_lexical_retrieve", lambda *a, **k: lexical_docs)

    semantic_index = {
        "embeddings": np.array([[1.0, 0.0], [0.9, 0.1]]),
        "meta": [
            {"title": "Shared Case v. State", "url": "u/shared"},
            {"title": "Semantic Only v. Y", "url": "u/sem"},
        ],
    }

    def embed(_query):
        return np.array([[1.0, 0.0]])

    docs = pipeline.retrieve(
        "q",
        search_index={"vectorizer": object(), "matrix": object(), "meta": []},
        semantic_index=semantic_index,
        query_embed_fn=embed,
        k=5,
    )

    # Fusion must surface the doc present in BOTH legs first and add fusion_score.
    assert docs[0]["title"] == "Shared Case v. State"
    assert "fusion_score" in docs[0]
    assert docs[0]["rank"] == 1
    titles = {d["title"] for d in docs}
    assert "Lexical Only v. X" in titles
    assert "Semantic Only v. Y" in titles
