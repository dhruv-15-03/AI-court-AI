"""Auth-gate regression tests for the AI agent routes.

The expensive /api/agent/* POST routes (analyze, analyze-with-docs, chat, rag,
upload-documents, generate-document) must enforce the shared X-API-Key when
``config.API_KEY`` is set, matching the /api/analyze* routes and the Spring Boot
backend which sends the key on every server-to-server call.

The browser-facing SSE route (/api/agent/stream) and the read-only GET routes
(health, sessions, session info, document-types) are intentionally exempt: the
browser streams to them directly and cannot safely hold the shared secret, so
they rely on rate limiting instead.

``require_api_key()`` is a no-op when ``config.API_KEY`` is empty, so local dev
without a key keeps working. These tests toggle ``config.API_KEY`` at runtime via
monkeypatch (the production fail-fast guard in server.py is bypassed because the
suite runs with APP_ENV=testing — see tests/conftest.py).
"""
import json

import pytest

from ai_court.api import config
from ai_court.api.server import app

# (path, json body) for every protected POST route.
PROTECTED_POST_ROUTES = [
    ("/api/agent/analyze", {"case_type": "Civil", "parties": "A vs B"}),
    ("/api/agent/analyze-with-docs", {"case_type": "Civil"}),
    ("/api/agent/chat", {"session_id": "s1", "question": "hello"}),
    ("/api/agent/rag", {"question": "what is anticipatory bail"}),
    ("/api/agent/generate-document", {"doc_type": "case_summary"}),
]

# Read-only / browser-facing routes that must NOT be auth-gated.
EXEMPT_GET_ROUTES = [
    "/api/agent/health",
    "/api/agent/document-types",
    "/api/agent/sessions",
]

API_KEY_VALUE = "test-secret-key"


@pytest.fixture
def with_api_key(monkeypatch):
    """Enable server-to-server auth enforcement for the duration of a test."""
    monkeypatch.setattr(config, "API_KEY", API_KEY_VALUE)
    return API_KEY_VALUE


def test_protected_routes_reject_missing_key(with_api_key):
    with app.test_client() as client:
        for path, body in PROTECTED_POST_ROUTES:
            resp = client.post(
                path, data=json.dumps(body), content_type="application/json"
            )
            assert resp.status_code == 401, (
                f"{path} must be 401 without X-API-Key, got {resp.status_code}"
            )
            assert resp.get_json().get("error") == "Unauthorized"


def test_protected_route_rejects_wrong_key(with_api_key):
    with app.test_client() as client:
        resp = client.post(
            "/api/agent/rag",
            data=json.dumps({"question": "x"}),
            content_type="application/json",
            headers={"X-API-Key": "not-the-key"},
        )
        assert resp.status_code == 401
        assert resp.get_json().get("error") == "Unauthorized"


def test_upload_documents_rejects_missing_key(with_api_key):
    with app.test_client() as client:
        resp = client.post("/api/agent/upload-documents")
        assert resp.status_code == 401


def test_protected_route_accepts_correct_key(with_api_key):
    """A correct key passes the gate; the request then fails fast on a downstream
    precondition (LLM unavailable -> 503), proving the key was accepted (not 401).
    """
    with app.test_client() as client:
        resp = client.post(
            "/api/agent/generate-document",
            data=json.dumps({}),
            content_type="application/json",
            headers={"X-API-Key": API_KEY_VALUE},
        )
        assert resp.status_code != 401


def test_no_key_configured_is_noop(monkeypatch):
    """When API_KEY is empty (local dev), protected routes are not auth-gated."""
    monkeypatch.setattr(config, "API_KEY", "")
    with app.test_client() as client:
        resp = client.post(
            "/api/agent/generate-document",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code != 401


def test_exempt_routes_not_gated(with_api_key):
    """Read-only GET routes stay reachable even when a key is configured and the
    caller sends no X-API-Key header."""
    with app.test_client() as client:
        for path in EXEMPT_GET_ROUTES:
            resp = client.get(path)
            assert resp.status_code != 401, (
                f"{path} must not be auth-gated, got {resp.status_code}"
            )
