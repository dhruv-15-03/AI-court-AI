
from ai_court.api.server import app


def test_drift_history_endpoint():
    with app.test_client() as c:
        r = c.get('/api/drift/history')
        assert r.status_code == 200
        data = r.get_json()
        assert 'events' in data
        # events list may be empty before any compare calls
        assert isinstance(data['events'], list)
