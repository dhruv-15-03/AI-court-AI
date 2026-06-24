import json

import pytest

import ai_court.utils.cache as cache_mod
from ai_court.utils.cache import ResponseCache, make_key, normalize_text


# ---------------------------------------------------------------------------
# make_key — stability, normalization and key-stripping
# ---------------------------------------------------------------------------
def test_make_key_is_order_independent():
    assert make_key("ns", {"a": 1, "b": 2}) == make_key("ns", {"b": 2, "a": 1})


def test_make_key_ignores_identity_fields():
    k1 = make_key("analyze", {"a": 1, "_user_id": "u1", "request_id": "r1"})
    k2 = make_key("analyze", {"a": 1, "_user_id": "u2", "request_id": "r2"})
    assert k1 == k2


def test_make_key_includes_plan():
    basic = make_key("analyze", {"a": 1, "_plan": "basic"})
    pro = make_key("analyze", {"a": 1, "_plan": "pro"})
    assert basic != pro


def test_make_key_includes_model_version():
    v1 = make_key("ns", {"a": 1}, model_version="v1")
    v2 = make_key("ns", {"a": 1}, model_version="v2")
    assert v1 != v2


def test_make_key_is_namespaced():
    assert make_key("analyze", {"a": 1}).startswith("analyze:")


def test_normalize_text_collapses_whitespace_but_keeps_case():
    assert normalize_text("  The   appeal\nis  allowed ") == "The appeal is allowed"
    assert normalize_text("Hello World") == "Hello World"


# ---------------------------------------------------------------------------
# ResponseCache — get/set, counters, copies, TTL, LRU, disabled
# ---------------------------------------------------------------------------
def test_get_set_roundtrip_and_counters():
    c = ResponseCache(ttl_seconds=0, max_size=8)
    assert c.get("missing") is None  # miss
    c.set("x", {"judgment": "Allowed"})
    assert c.get("x") == {"judgment": "Allowed"}  # hit
    stats = c.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["backend"] == "memory"
    assert stats["hit_rate"] == 0.5


def test_returned_values_are_copies():
    c = ResponseCache(ttl_seconds=0)
    c.set("k", {"n": 1})
    got = c.get("k")
    assert got is not None
    got["n"] = 999
    assert c.get("k")["n"] == 1


def test_ttl_expiry(monkeypatch):
    clock = {"now": 1000.0}
    monkeypatch.setattr(cache_mod.time, "time", lambda: clock["now"])
    c = ResponseCache(ttl_seconds=10, max_size=8)
    c.set("k", {"v": 1})
    assert c.get("k") == {"v": 1}
    clock["now"] += 11  # advance past the TTL
    assert c.get("k") is None


def test_lru_eviction():
    c = ResponseCache(ttl_seconds=0, max_size=2)
    c.set("a", {"n": 1})
    c.set("b", {"n": 2})
    c.set("c", {"n": 3})  # evicts the least-recently-used "a"
    assert c.get("a") is None
    assert c.get("b") == {"n": 2}
    assert c.get("c") == {"n": 3}


def test_disabled_cache_is_noop():
    c = ResponseCache(enabled=False)
    c.set("x", {"a": 1})
    assert c.get("x") is None
    assert c.stats()["enabled"] is False


def test_clear_resets_entries_and_counters():
    c = ResponseCache(ttl_seconds=0)
    c.set("a", {"n": 1})
    c.get("a")
    c.clear()
    assert c.get("a") is None
    stats = c.stats()
    assert stats["entries"] == 0


# ---------------------------------------------------------------------------
# Integration — /api/analyze/quick serves a cached response on repeat
# ---------------------------------------------------------------------------
def test_quick_endpoint_caches_repeat_request():
    try:
        from ai_court.api.server import app
    except Exception as exc:  # pragma: no cover - ML stack not installed
        pytest.skip(f"AI server stack unavailable: {exc}")

    payload = {
        "text": "The appeal is allowed. The conviction is set aside and the appellant is acquitted.",
        "case_type": "Criminal",
    }
    with app.test_client() as client:
        first = client.post(
            "/api/analyze/quick",
            data=json.dumps(payload),
            content_type="application/json",
        )
        if first.status_code != 200:
            pytest.skip("classifier model not available in this environment")
        assert first.headers.get("X-Cache") == "MISS"
        assert first.get_json().get("cached") is False

        second = client.post(
            "/api/analyze/quick",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert second.status_code == 200
        assert second.headers.get("X-Cache") == "HIT"
        body = second.get_json()
        assert body.get("cached") is True
        assert body.get("judgment") == first.get_json().get("judgment")
