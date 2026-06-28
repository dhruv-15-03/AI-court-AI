"""Tests for the readiness/liveness probes.

The readiness probe must reflect whether the classifier model is actually loaded
(not a deprecated always-truthy module function), and must never gate on the
optional LLM agent. Liveness must always report alive.
"""
from ai_court.api.server import app


def test_health_ready_structure():
    with app.test_client() as c:
        r = c.get('/api/health/ready')
        assert r.status_code in (200, 503)
        j = r.get_json()
        assert 'ready' in j
        assert 'checks' in j
        assert 'model_loaded' in j['checks']
        # LLM availability is reported but must not gate readiness.
        assert 'llm_available' in j
        # Readiness is exactly the model_loaded gate (LLM excluded).
        assert j['ready'] == j['checks']['model_loaded']
        # The 503/200 status must agree with the `ready` flag.
        assert (r.status_code == 200) == bool(j['ready'])


def test_health_live_always_ok():
    with app.test_client() as c:
        r = c.get('/api/health/live')
        assert r.status_code == 200
        assert r.get_json().get('alive') is True
