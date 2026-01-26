"""Tests for batch and quick analyze endpoints."""

import json
from ai_court.api.server import app


class TestBatchAnalyze:
    """Tests for /api/analyze/batch endpoint."""

    def test_batch_analyze_empty(self):
        """Test batch with empty cases array."""
        with app.test_client() as c:
            resp = c.post("/api/analyze/batch", data=json.dumps({"cases": []}), content_type="application/json")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["count"] == 0
            assert data["results"] == []

    def test_batch_analyze_single_case(self):
        """Test batch with single case."""
        with app.test_client() as c:
            payload = {
                "cases": [{"case_type": "Criminal", "violence_level": "Moderate"}],
                "format": "minimal"
            }
            resp = c.post("/api/analyze/batch", data=json.dumps(payload), content_type="application/json")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["count"] == 1
            assert len(data["results"]) == 1
            assert "judgment" in data["results"][0]
            assert "confidence" in data["results"][0]
            assert "timing_ms" in data

    def test_batch_analyze_multiple_cases(self):
        """Test batch with multiple cases."""
        with app.test_client() as c:
            case = {"case_type": "Criminal", "violence_level": "Low"}
            payload = {"cases": [case, case, case], "format": "minimal"}
            resp = c.post("/api/analyze/batch", data=json.dumps(payload), content_type="application/json")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["count"] == 3
            assert "avg_ms_per_case" in data

    def test_batch_analyze_full_format(self):
        """Test batch with full response format."""
        with app.test_client() as c:
            payload = {
                "cases": [{"case_type": "Criminal"}],
                "format": "full"
            }
            resp = c.post("/api/analyze/batch", data=json.dumps(payload), content_type="application/json")
            assert resp.status_code == 200
            data = resp.get_json()
            if data["count"] > 0:
                result = data["results"][0]
                assert "confidence_info" in result or "confidence" in result

    def test_batch_analyze_invalid_case(self):
        """Test batch with invalid case (not a dict)."""
        with app.test_client() as c:
            resp = c.post("/api/analyze/batch", data=json.dumps({"cases": ["not a dict"]}), content_type="application/json")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["error_count"] == 1

    def test_batch_analyze_no_body(self):
        """Test batch with no body."""
        with app.test_client() as c:
            resp = c.post("/api/analyze/batch")
            # 400 for bad request or 415 for unsupported media type are both acceptable
            assert resp.status_code in (400, 415)


class TestQuickAnalyze:
    """Tests for /api/analyze/quick endpoint."""

    def test_quick_analyze_success(self):
        """Test quick analyze with valid text."""
        with app.test_client() as c:
            payload = {"text": "The accused was arrested for theft and bail was denied."}
            resp = c.post("/api/analyze/quick", data=json.dumps(payload), content_type="application/json")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "judgment" in data
            assert "confidence" in data
            assert "timing_ms" in data

    def test_quick_analyze_with_case_type(self):
        """Test quick analyze with case type hint."""
        with app.test_client() as c:
            payload = {"text": "Employment termination dispute", "case_type": "Labor"}
            resp = c.post("/api/analyze/quick", data=json.dumps(payload), content_type="application/json")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["case_type"] == "Labor"

    def test_quick_analyze_missing_text(self):
        """Test quick analyze without text field."""
        with app.test_client() as c:
            resp = c.post("/api/analyze/quick", data=json.dumps({}), content_type="application/json")
            assert resp.status_code == 400
            data = resp.get_json()
            assert "error" in data


class TestOutcomesEndpoint:
    """Tests for /api/outcomes endpoints."""

    def test_get_outcomes_list(self):
        """Test getting list of outcomes."""
        with app.test_client() as c:
            resp = c.get("/api/outcomes")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "outcomes" in data

    def test_get_outcome_info(self):
        """Test getting info for a specific outcome."""
        with app.test_client() as c:
            resp = c.get("/api/outcomes/Bail%20Granted")
            assert resp.status_code == 200
            data = resp.get_json()
            assert "outcome" in data
