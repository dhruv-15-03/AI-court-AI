import json
from ai_court.api.server import app

def test_health():
    with app.test_client() as c:
        r = c.get('/api/health')
        assert r.status_code in (200, 500)

def test_analyze_minimal():
    with app.test_client() as c:
        payload = {"case_type": "Civil", "parties": "A vs B"}
        r = c.post('/api/analyze', data=json.dumps(payload), content_type='application/json')
        assert r.status_code == 200
        j = r.get_json()
        assert 'judgment' in j
        assert 'confidence' in j

def test_search_validation():
    with app.test_client() as c:
        r = c.post('/api/search', data=json.dumps({}), content_type='application/json')
        # Now validation should trigger 400 (or 503 if index missing)
        assert r.status_code in (400, 503)

def test_search_valid():
    with app.test_client() as c:
        payload = {"query": "contract dispute", "k": 3}
        r = c.post('/api/search', data=json.dumps(payload), content_type='application/json')
        # If search index missing we allow 503; otherwise 200
        assert r.status_code in (200, 503)
        if r.status_code == 200:
            j = r.get_json()
            assert 'results' in j

def test_analyze_validation_failure():
    with app.test_client() as c:
        # Provide an invalid field value exceeding max length
        payload = {"case_type": "X" * 5000}
        r = c.post('/api/analyze', data=json.dumps(payload), content_type='application/json')
        assert r.status_code == 400
        j = r.get_json()
        assert j.get('error') == 'validation_failed'

def test_analyze_and_search():
    with app.test_client() as c:
        payload = {"case_type": "Civil", "parties": "A vs B", "dispute_type": "Contract"}
        r = c.post('/api/analyze_and_search', data=json.dumps(payload), content_type='application/json')
        assert r.status_code == 200
        j = r.get_json()
        assert 'judgment' in j
        assert 'case_type' in j

def test_metrics_endpoint():
    with app.test_client() as c:
        r = c.get('/metrics')
        # Prometheus text format content type check
        assert r.status_code == 200
        assert 'text/plain' in r.content_type
        body = r.get_data(as_text=True)
        assert 'ai_court_requests_total' in body
