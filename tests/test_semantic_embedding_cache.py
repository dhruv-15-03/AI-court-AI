"""Regression tests for the SentenceTransformer caching fix in dependencies.py.

Before the fix, semantic_query_embeddings() constructed a brand-new
SentenceTransformer (re-loading model weights + tokenizer from disk) on
every single call. That turned every semantic /api/search request into a
model-load operation. These tests assert the encoder is instantiated at
most once per model_name and reused across repeated calls / model_name.
"""
import sys
import types

import pytest

from ai_court.api import dependencies, state


class _FakeSentenceTransformer:
    """Stand-in for sentence_transformers.SentenceTransformer.

    Records how many times it was constructed so tests can assert caching
    behavior without downloading a real model.
    """

    instances_created = 0

    def __init__(self, model_name):
        _FakeSentenceTransformer.instances_created += 1
        self.model_name = model_name

    def encode(self, texts, normalize_embeddings=True):
        return [[float(len(t))] for t in texts]


@pytest.fixture(autouse=True)
def _reset_cache_and_counter(monkeypatch):
    # Clear the module-level cache/lock state between tests.
    dependencies._sentence_transformer_cache.clear()
    _FakeSentenceTransformer.instances_created = 0

    fake_module = types.ModuleType("sentence_transformers")
    fake_module.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)

    old_index = state.semantic_index
    old_preprocess = state.preprocess_fn
    yield
    state.semantic_index = old_index
    state.preprocess_fn = old_preprocess


def test_semantic_query_embeddings_reuses_cached_encoder():
    state.semantic_index = {"model_name": "all-MiniLM-L6-v2"}
    state.preprocess_fn = lambda x: x

    for _ in range(5):
        vec = dependencies.semantic_query_embeddings("bail application under section 439")
        assert vec is not None

    assert _FakeSentenceTransformer.instances_created == 1


def test_semantic_query_embeddings_returns_none_without_index():
    state.semantic_index = None
    assert dependencies.semantic_query_embeddings("anything") is None
    assert _FakeSentenceTransformer.instances_created == 0


def test_get_sentence_transformer_caches_per_model_name():
    a1 = dependencies._get_sentence_transformer("model-a")
    a2 = dependencies._get_sentence_transformer("model-a")
    b1 = dependencies._get_sentence_transformer("model-b")

    assert a1 is a2
    assert a1 is not b1
    assert _FakeSentenceTransformer.instances_created == 2
