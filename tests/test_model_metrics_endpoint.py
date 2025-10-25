from ai_court.api.server import app


def test_model_metrics_endpoint():
    with app.test_client() as c:
        r = c.get('/api/metrics/model')
        assert r.status_code == 200
        data = r.get_json()
        assert 'metrics' in data
        assert 'metadata' in data
        # metrics may be empty if training not executed yet
        assert isinstance(data['metrics'], dict)
